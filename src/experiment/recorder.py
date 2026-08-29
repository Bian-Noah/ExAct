"""ExperimentRecorder：实验数据收集与落盘一体化。

Iteration 3 设计原则：
- 单类一体化：不拆 event_bus / tracer / sink 框架层
- 业务零侵入：业务代码仅 `recorder.emit(event, **fields)` 一次
- 进程级单例：`get_recorder()` 拿全局实例，未初始化时返回 `_SafeRecorder`
- 三类事件：log → experiment.log，observe_image → observer/*.png，video_frame → process/*.mp4
- 异常隔离：emit 内部 try/except 吞错，业务主线不受影响
- enabled=False：所有方法 no-op

Iteration 12 扩展（iter12-video-recording）：
- `__init__` 新增可选 `video: VideoConfig | None`，None = 不录视频（旧调用兼容）
- `start()` 创建 `process/` 并 open VideoWriter；`_handle_video_frame` 真实落盘；
  `finish()` close writer。视频不可用时降级 no-op（一次 warning）。
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from experiment.utils import ensure_dir, next_available_dir
from experiment.video.config import VideoConfig
from experiment.video.writer import VideoWriter


class ExperimentRecorder:
    """实验数据落盘器：目录创建 + emit dispatch + 三类事件落地。

    Attributes:
        root: 实验根目录（如 `ExAct/data/experiment`）。
        enabled: 是否启用录制。False 时所有方法 no-op。
        log_to_stdout: log 事件是否同步输出到终端。
        exp_dir: 当前实验目录（`start()` 后填充）。
        observer_dir: observer/ 子目录路径。
        video: 视频配置；None = 不录视频（iter12 新增）。
        process_dir: process/ 子目录路径（video 启用时创建）。
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
        video: VideoConfig | None = None,
    ) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self.log_to_stdout = log_to_stdout
        self.exp_dir: Path | None = None
        self.observer_dir: Path | None = None
        self._log_fh: TextIO | None = None
        # iter12-video-recording：视频配置与 writer
        self.video = video
        self.process_dir: Path | None = None
        self._video_writer: VideoWriter | None = None
        self._video_warning_emitted = False

    def start(self) -> Path:
        """创建 `{root}/{timestamp}/`、`observer/`、`process/`，打开 log 与 VideoWriter。

        Returns:
            exp_dir 路径。`enabled=False` 时返回 `self.root`，不创建任何目录。

        Note:
            - 同一秒内并发场景下自动追加 `_001` `_002` 后缀（utils.next_available_dir）
            - `video` 非 None 且启用时创建 `process/` 并 open VideoWriter；
              打开失败降级 no-op（视频不可用，不影响 log/observer）
        """
        if not self.enabled:
            return self.root

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = next_available_dir(self.root, timestamp)
        ensure_dir(exp_dir)

        observer_dir = exp_dir / "observer"
        ensure_dir(observer_dir)

        self.exp_dir = exp_dir
        self.observer_dir = observer_dir
        self._log_fh = open(exp_dir / "experiment.log", "a", encoding="utf-8")

        # iter12-video-recording：视频配置启用时打开 writer
        if self.video is not None and self.video.enabled:
            process_dir = exp_dir / "process"
            ensure_dir(process_dir)
            self.process_dir = process_dir
            writer = VideoWriter(process_dir, self.video)
            if writer.open():
                self._video_writer = writer
            else:
                self._video_writer = None
                self._warn_video_unavailable()
        return exp_dir

    def _warn_video_unavailable(self) -> None:
        """视频不可用时输出一次 warning（幂等）。"""
        if self._video_warning_emitted:
            return
        self._video_warning_emitted = True
        warnings.warn(
            "[ExperimentRecorder] video recording disabled: writer open failed "
            "(cv2 缺失 / 编码器不可用 / 分辨率非法)，视频录制降级 no-op。",
            stacklevel=2,
        )


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
        """委托 VideoWriter 写入一帧（iter12：真实落盘）。

        未配置视频 / writer 不可用时 no-op。
        """
        if self._video_writer is None:
            return
        self._video_writer.write(image)

    def _handle_unknown(self, **fields: Any) -> None:
        """未知 event 默认 no-op。"""
        return

    def finish(self, success: bool, summary: str) -> None:
        """emit `Pipeline finished, success=..., summary=...` + 关闭 log 与视频 writer。"""
        self.emit(
            "log",
            message=f"Pipeline finished, success={success}, summary={summary}",
        )
        # iter12-video-recording：close video writer（容错，重复调用安全）
        if self._video_writer is not None:
            try:
                self._video_writer.close()
            except Exception:
                pass
            finally:
                self._video_writer = None
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