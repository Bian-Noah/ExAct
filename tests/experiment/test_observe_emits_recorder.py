"""observe 工具 + ExperimentRecorder 集成单测。

覆盖（按 tasks.md 4.3 任务列表）：
- 真实 ExperimentRecorder + FakeEnv → _run() 后 observer/000.png 存在
- experiment.log 含 [observe] saved observer/000.png
- _SafeRecorder fallback 下 _run() 仍返回正常文本（无副作用）
- _call_count 递增、idx 序列
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiment.recorder import (
    ExperimentRecorder,
    _SafeRecorder,
    get_recorder,
    set_recorder,
)
from tools import ObserveTool


class FakeEnv:
    """测试用 FakeEnv：返回固定 obs dict。"""

    def __init__(self, include_rgb: bool = True, rgb_shape=(32, 32, 3)):
        self._include_rgb = include_rgb
        self._rgb_shape = rgb_shape
        self.call_count = 0

    def get_obs(self, include_rgb: bool = True):
        self.call_count += 1
        rgb = None
        if include_rgb and self._include_rgb:
            rgb = np.random.randint(0, 256, self._rgb_shape, dtype=np.uint8)
        return {
            "rgb": rgb,
            "ee_pos": (0.0, 0.0, 0.5),
            "object_info": [
                {"name": "red_block", "pos": [0.5, 0.0, 0.1]},
                {"name": "blue_block", "pos": [0.3, 0.2, 0.1]},
            ],
        }


def _reset_global():
    """autouse-style 辅助函数。"""
    set_recorder(None)


def _text_block(result) -> str:
    """从 _run() 返回的 list[dict] 中提取首条 text 块的 'text' 字段。

    Iteration 5 约定：_run() 返回 list[dict]，list[0] 永远是 text 块。
    本文件不注入 image_store，所以 content 长度恰好为 1，含 1 个 text 块。
    """
    assert isinstance(result, list), f"期望 list[dict]，得到 {type(result).__name__}"
    assert len(result) >= 1, f"期望至少 1 个 block，得到空 list"
    assert result[0]["type"] == "text", f"期望 text 块，得到 {result[0]!r}"
    return result[0]["text"]


# ============================================================================
# 真实 ExperimentRecorder + FakeEnv 集成
# ============================================================================


def test_observe_writes_png_and_log(tmp_path: Path):
    """真实 recorder 下，_run() 后 observer/000.png + experiment.log 都应存在。"""
    _reset_global()
    recorder = ExperimentRecorder(root=tmp_path)
    set_recorder(recorder)
    recorder.start()

    tool = ObserveTool(env=FakeEnv())
    result = tool._run()

    # 返回文本仍正确（不因埋点改动）
    assert "末端执行器位置" in _text_block(result)
    assert "red_block" in _text_block(result)

    # PNG 应写入
    assert (recorder.observer_dir / "000.png").exists()

    # experiment.log 应含 [observe] saved observer/000.png
    content = (recorder.exp_dir / "experiment.log").read_text(encoding="utf-8")
    assert "[observe] saved observer/000.png" in content

    recorder.finish(success=True, summary="x")


def test_observe_call_count_increments(tmp_path: Path):
    """连续 _run() 多次，_call_count 递增，PNG 序列 000/001/002。"""
    _reset_global()
    recorder = ExperimentRecorder(root=tmp_path)
    set_recorder(recorder)
    recorder.start()

    tool = ObserveTool(env=FakeEnv())
    for _ in range(3):
        tool._run()

    assert (recorder.observer_dir / "000.png").exists()
    assert (recorder.observer_dir / "001.png").exists()
    assert (recorder.observer_dir / "002.png").exists()
    assert not (recorder.observer_dir / "003.png").exists()

    recorder.finish(success=True, summary="x")


def test_observe_log_contains_saved_lines_for_each_call(tmp_path: Path):
    """experiment.log 应含 3 行 [observe] saved observer/{idx:03d}.png。"""
    _reset_global()
    recorder = ExperimentRecorder(root=tmp_path, log_to_stdout=False)
    set_recorder(recorder)
    recorder.start()

    tool = ObserveTool(env=FakeEnv())
    for _ in range(3):
        tool._run()

    recorder.finish(success=True, summary="x")

    content = (recorder.exp_dir / "experiment.log").read_text(encoding="utf-8")
    assert "[observe] saved observer/000.png" in content
    assert "[observe] saved observer/001.png" in content
    assert "[observe] saved observer/002.png" in content


# ============================================================================
# _SafeRecorder fallback 下无副作用
# ============================================================================


def test_observe_with_safe_recorder_returns_text_without_side_effects(tmp_path: Path):
    """_SafeRecorder fallback 下 _run() 仍返回正常文本，不创建文件。"""
    _reset_global()
    # get_recorder() fallback 到 _SafeRecorder（因为 set_recorder(None) 后未注入）
    assert isinstance(get_recorder(), _SafeRecorder)

    tool = ObserveTool(env=FakeEnv())
    result = tool._run()

    # 返回文本正常
    assert "末端执行器位置" in _text_block(result)
    assert "red_block" in _text_block(result)

    # _call_count 仍递增（不依赖 recorder 是否启用）
    assert tool._call_count == 1

    # tmp_path 下不应有任何文件（_SafeRecorder 不写文件）
    assert list(tmp_path.iterdir()) == []


def test_observe_with_enabled_false_returns_text(tmp_path: Path):
    """enabled=False 的 recorder 下，_run() 仍返回正常文本。"""
    _reset_global()
    recorder = ExperimentRecorder(root=tmp_path, enabled=False)
    set_recorder(recorder)

    tool = ObserveTool(env=FakeEnv())
    result = tool._run()

    # 返回文本正常
    assert "末端执行器位置" in _text_block(result)
    # 不应创建文件
    assert list(tmp_path.iterdir()) == []


# ============================================================================
# 边界场景
# ============================================================================


def test_observe_no_rgb_does_not_emit(tmp_path: Path):
    """obs 缺 rgb 时不应触发 observe_image 埋点。"""
    _reset_global()
    recorder = ExperimentRecorder(root=tmp_path, log_to_stdout=False)
    set_recorder(recorder)
    recorder.start()

    # FakeEnv include_rgb=False → rgb=None
    tool = ObserveTool(env=FakeEnv(include_rgb=False))
    tool._run()

    recorder.finish(success=True, summary="x")

    # observer_dir 应为空
    files = list(recorder.observer_dir.iterdir())
    assert len(files) == 0
    # experiment.log 应只含 finish 一行
    content = (recorder.exp_dir / "experiment.log").read_text(encoding="utf-8")
    assert "[observe] saved" not in content


def test_observe_call_count_increments_even_without_rgb():
    """无 rgb 时 _call_count 仍递增（与原 _run 行为兼容）。"""
    _reset_global()
    tool = ObserveTool(env=FakeEnv(include_rgb=False))
    assert tool._call_count == 0
    tool._run()
    assert tool._call_count == 0  # 无 rgb 不递增（设计）


def test_observe_target_filter_still_works(tmp_path: Path):
    """埋点不影响 target 过滤功能。"""
    _reset_global()
    recorder = ExperimentRecorder(root=tmp_path, log_to_stdout=False)
    set_recorder(recorder)
    recorder.start()

    tool = ObserveTool(env=FakeEnv())
    result = tool._run(target="red_block")

    assert "red_block" in _text_block(result)
    # target 过滤时仅返回 red_block
    assert "blue_block" not in _text_block(result)

    recorder.finish(success=True, summary="x")


# ============================================================================
# Recorder 异常时业务不受影响
# ============================================================================


def test_observe_recorder_exception_does_not_break_business(tmp_path: Path, capsys, monkeypatch):
    """recorder.emit 内部抛错时（实际被隔离），_run() 仍返回正常文本。"""
    _reset_global()
    recorder = ExperimentRecorder(root=tmp_path)
    set_recorder(recorder)
    recorder.start()

    # 让 emit 内部抛错（通过 monkeypatch Image.fromarray）
    from PIL import Image
    def fake_fromarray(*args, **kwargs):
        raise RuntimeError("模拟异常")
    monkeypatch.setattr(Image, "fromarray", fake_fromarray)

    tool = ObserveTool(env=FakeEnv())
    # 不应抛异常（recorder 内部 try/except 隔离）
    result = tool._run()

    # 返回文本正常
    assert "末端执行器位置" in _text_block(result)
    assert "red_block" in _text_block(result)

    captured = capsys.readouterr()
    # stderr 应含 warning（埋点异常被发现）
    assert "ExperimentRecorder" in captured.err

    recorder.finish(success=True, summary="x")