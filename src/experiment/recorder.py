"""ExperimentRecorder：实验数据收集与落盘一体化。

Iteration 3 设计原则：
- 单类一体化：不拆 event_bus / tracer / sink 框架层
- 业务零侵入：业务代码仅 `recorder.emit(event, **fields)` 一次
- 进程级单例：`get_recorder()` 拿全局实例，未初始化时返回 `_SafeRecorder`
- 三类事件：log → experiment.log，observe_image → observer/*.png，video_frame 占位 no-op
- 异常隔离：emit 内部 try/except 吞错，业务主线不受影响
- enabled=False：所有方法 no-op
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO


class ExperimentRecorder:
    """实验数据落盘器：目录创建 + emit dispatch + 三类事件落地。

    Attributes:
        root: 实验根目录（如 `ExAct/data/experiment`）。
        enabled: 是否启用录制。False 时所有方法 no-op。
        log_to_stdout: log 事件是否同步输出到终端。
        exp_dir: 当前实验目录（`start()` 后填充）。
        observer_dir: observer/ 子目录路径。
    """

    # event 名 → handler 方法名（dispatch 表）
    EVENT_HANDLERS: dict[str, str] = {
        "log": "_handle_log",
        "observe_image": "_handle_observe_image",
        "video_frame": "_handle_video_frame",
    }

    def __init__(
        self,
        root: Path | str,
        enabled: bool = True,
        log_to_stdout: bool = True,
    ) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self.log_to_stdout = log_to_stdout
        self.exp_dir: Path | None = None
        self.observer_dir: Path | None = None
        self._log_fh: TextIO | None = None

    def start(self) -> Path:
        """创建 `{root}/{timestamp}/` 和 `observer/`，打开 `experiment.log`。

        Returns:
            exp_dir 路径。`enabled=False` 时返回 `self.root`，不创建任何目录。

        Note:
            同一秒内并发场景下自动追加 `_001` `_002` 后缀。
        """
        if not self.enabled:
            return self.root

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.root / timestamp
        suffix = 1
        while exp_dir.exists():
            exp_dir = self.root / f"{timestamp}_{suffix:03d}"
            suffix += 1

        exp_dir.mkdir(parents=True, exist_ok=False)
        observer_dir = exp_dir / "observer"
        observer_dir.mkdir(exist_ok=True)

        self.exp_dir = exp_dir
        self.observer_dir = observer_dir
        self._log_fh = open(exp_dir / "experiment.log", "a", encoding="utf-8")
        return exp_dir

    def emit(self, event: str, **fields: Any) -> None:
        """dispatch 表分派到对应 handler。

        Args:
            event: 事件名（log / observe_image / video_frame / 未知均可）
            **fields: 传递给 handler 的字段

        Note:
            - `enabled=False` 时直接 return，不报错、不产生文件
            - handler 抛异常时吞掉，stderr 输出 warning，业务代码不受影响
            - 未知 event 走 `_handle_unknown` 默认 no-op
        """
        if not self.enabled:
            return
        try:
            handler_name = self.EVENT_HANDLERS.get(event, "_handle_unknown")
            handler = getattr(self, handler_name)
            handler(**fields)
        except Exception as e:
            print(
                f"[ExperimentRecorder] emit({event}) failed: {e}",
                file=sys.stderr,
            )

    def _handle_log(self, message: str, level: str = "INFO") -> None:
        """追加 `{ts} [{level}] {message}\\n` 到 `experiment.log`。

        log_to_stdout=True 时同步打印到终端。
        """
        if self._log_fh is None:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = f"{ts} [{level}] {message}\n"
        self._log_fh.write(line)
        self._log_fh.flush()
        if self.log_to_stdout:
            print(line, end="")

    def _handle_observe_image(self, image: Any, idx: int) -> None:
        """保存 ndarray 到 `observer/{idx:03d}.png`。

        使用 PIL `Image.fromarray` 编码（要求 uint8 RGB ndarray）。
        """
        if self.observer_dir is None:
            return
        from PIL import Image
        path = self.observer_dir / f"{idx:03d}.png"
        Image.fromarray(image).save(path)

    def _handle_video_frame(self, image: Any, timestamp: float) -> None:
        """本迭代占位 no-op（视频录制未实现）。"""
        return

    def _handle_unknown(self, **fields: Any) -> None:
        """未知 event 默认 no-op。"""
        return

    def finish(self, success: bool, summary: str) -> None:
        """emit `Pipeline finished, success=..., summary=...` + 关闭 log 文件句柄。"""
        self.emit(
            "log",
            message=f"Pipeline finished, success={success}, summary={summary}",
        )
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None


# ============================================================================
# 进程级单例管理
# ============================================================================

_global_recorder: ExperimentRecorder | None = None

# 模块级缓存的 SafeRecorder（单例），未初始化时统一返回此实例
_safe_recorder_instance: _SafeRecorder | None = None


class _SafeRecorder(ExperimentRecorder):
    """未初始化时的 fallback：所有方法直接 no-op，不报错、不产生文件。

    主要用于：业务代码（含单元测试）调用 `get_recorder()` 时无需检查 None，
    避免埋点调用 try/except 污染业务。
    """

    def __init__(self) -> None:
        super().__init__(root=Path("/tmp"), enabled=False, log_to_stdout=False)

    def start(self) -> Path:
        return Path("/tmp")

    def emit(self, event: str, **fields: Any) -> None:
        return

    def finish(self, success: bool, summary: str) -> None:
        return


def _get_safe_recorder() -> _SafeRecorder:
    """返回缓存的 _SafeRecorder 单例。"""
    global _safe_recorder_instance
    if _safe_recorder_instance is None:
        _safe_recorder_instance = _SafeRecorder()
    return _safe_recorder_instance


def get_recorder() -> ExperimentRecorder:
    """返回 `_global_recorder`；未设置时返回缓存的 `_SafeRecorder()` 单例。"""
    global _global_recorder
    if _global_recorder is None:
        return _get_safe_recorder()
    return _global_recorder


def set_recorder(recorder: ExperimentRecorder | None) -> None:
    """设置 `_global_recorder`；传 None 时重置为未设置状态。"""
    global _global_recorder
    _global_recorder = recorder