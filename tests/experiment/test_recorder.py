"""ExperimentRecorder 核心类单元测试。

覆盖（按 tasks.md 1.2 任务列表）：
- __init__ 参数存储
- start() 创建时间戳目录 + observer/ 子目录 + experiment.log
- start() 同秒冲突追加 _001/_002 后缀
- emit("log", ...) 追加带时间戳行 + log_to_stdout 同步
- emit("observe_image", ...) 写 PNG
- emit("video_frame", ...) 占位 no-op
- emit("unknown_event", ...) 默认 no-op
- enabled=False 时 start/emit/finish 全 no-op
- emit() 异常隔离（PIL 抛错时吞掉 + stderr warning）
- finish() 关闭文件句柄 + 写入收尾日志
"""

from __future__ import annotations

import io
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiment.recorder import ExperimentRecorder


# ============================================================================
# __init__ 参数存储
# ============================================================================


def test_init_stores_parameters(tmp_path: Path):
    """__init__ 参数应正确存储到字段。"""
    recorder = ExperimentRecorder(
        root=tmp_path, enabled=True, log_to_stdout=False
    )
    assert recorder.root == tmp_path
    assert recorder.enabled is True
    assert recorder.log_to_stdout is False
    assert recorder.exp_dir is None
    assert recorder.observer_dir is None
    assert recorder._log_fh is None


def test_init_default_values(tmp_path: Path):
    """默认值：enabled=True、log_to_stdout=True。"""
    recorder = ExperimentRecorder(root=tmp_path)
    assert recorder.enabled is True
    assert recorder.log_to_stdout is True


def test_init_accepts_string_root(tmp_path: Path):
    """root 接受字符串路径（自动转 Path）。"""
    recorder = ExperimentRecorder(root=str(tmp_path))
    assert recorder.root == tmp_path


# ============================================================================
# start() 目录与文件创建
# ============================================================================


def test_start_creates_timestamped_dir(tmp_path: Path):
    """start() 应创建 {root}/{YYYYMMDD_HHMMSS}/ 目录。"""
    recorder = ExperimentRecorder(root=tmp_path)
    exp_dir = recorder.start()
    assert exp_dir.exists()
    assert exp_dir.parent == tmp_path
    assert re.match(r"\d{8}_\d{6}$", exp_dir.name)


def test_start_creates_observer_subdir(tmp_path: Path):
    """start() 应创建 observer/ 子目录。"""
    recorder = ExperimentRecorder(root=tmp_path)
    exp_dir = recorder.start()
    observer_dir = exp_dir / "observer"
    assert observer_dir.is_dir()
    assert recorder.observer_dir == observer_dir


def test_start_opens_experiment_log(tmp_path: Path):
    """start() 应打开 experiment.log 文件句柄。"""
    recorder = ExperimentRecorder(root=tmp_path)
    exp_dir = recorder.start()
    assert recorder._log_fh is not None
    # emit 后文件应可见
    recorder.emit("log", message="test_after_start")
    recorder.finish(success=True, summary="x")
    log_content = (exp_dir / "experiment.log").read_text(encoding="utf-8")
    assert "test_after_start" in log_content


def test_start_creates_parent_dirs(tmp_path: Path):
    """root 多层不存在时，start() 应自动 mkdir -p。"""
    nested = tmp_path / "a" / "b" / "c"
    recorder = ExperimentRecorder(root=nested)
    exp_dir = recorder.start()
    assert exp_dir.exists()
    # exp_dir 是 nested/timestamp，parent 应是 nested（即 a/b/c）
    assert exp_dir.parent == nested
    # nested 本身也应被创建
    assert nested.is_dir()
    # 父目录链都应存在
    assert (tmp_path / "a").is_dir()
    assert (tmp_path / "a" / "b").is_dir()


def test_start_same_second_appends_suffix(tmp_path: Path, monkeypatch):
    """同一秒内多次 start() 应追加 _001/_002 后缀。"""
    # 固定时间戳避免跨秒
    fixed_dt = datetime(2026, 8, 13, 15, 30, 0)
    monkeypatch.setattr(
        "experiment.recorder.datetime",
        type("M", (), {"now": staticmethod(lambda: fixed_dt)}),
    )

    r1 = ExperimentRecorder(root=tmp_path)
    r2 = ExperimentRecorder(root=tmp_path)
    r3 = ExperimentRecorder(root=tmp_path)

    d1 = r1.start()
    d2 = r2.start()
    d3 = r3.start()

    assert d1.name == "20260813_153000"
    assert d2.name == "20260813_153000_001"
    assert d3.name == "20260813_153000_002"


# ============================================================================
# emit("log", ...) 行为
# ============================================================================


def test_emit_log_writes_timestamped_line(tmp_path: Path):
    """emit("log", message=...) 应追加 {ts} [INFO] message\\n。"""
    recorder = ExperimentRecorder(root=tmp_path, log_to_stdout=False)
    recorder.start()
    recorder.emit("log", message="hello world")
    recorder.finish(success=True, summary="x")

    content = (recorder.exp_dir / "experiment.log").read_text(encoding="utf-8")
    # 匹配时间戳格式 YYYY-MM-DD HH:MM:SS.mmm
    pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} \[INFO\] hello world\n"
    assert re.search(pattern, content), f"log 内容不符合格式: {content!r}"


def test_emit_log_custom_level(tmp_path: Path):
    """emit("log", level="WARNING") 自定义 level 应写入。"""
    recorder = ExperimentRecorder(root=tmp_path, log_to_stdout=False)
    recorder.start()
    recorder.emit("log", message="custom level test", level="WARNING")
    recorder.finish(success=True, summary="x")

    content = (recorder.exp_dir / "experiment.log").read_text(encoding="utf-8")
    assert "[WARNING] custom level test" in content


def test_emit_log_to_stdout(capsys, tmp_path: Path):
    """log_to_stdout=True 时 log 事件同步输出到终端。"""
    recorder = ExperimentRecorder(root=tmp_path, log_to_stdout=True)
    recorder.start()
    recorder.emit("log", message="stdout test")
    recorder.finish(success=True, summary="x")

    captured = capsys.readouterr()
    assert "stdout test" in captured.out


def test_emit_log_no_stdout_when_disabled(capsys, tmp_path: Path):
    """log_to_stdout=False 时 log 事件不输出到终端。"""
    recorder = ExperimentRecorder(root=tmp_path, log_to_stdout=False)
    recorder.start()
    recorder.emit("log", message="silent")
    recorder.finish(success=True, summary="x")

    captured = capsys.readouterr()
    assert "silent" not in captured.out


def test_emit_log_without_start_is_noop(tmp_path: Path):
    """未 start() 时 emit("log") 应静默 no-op（_log_fh is None）。"""
    recorder = ExperimentRecorder(root=tmp_path, log_to_stdout=False)
    # 不调 start()
    recorder.emit("log", message="orphan")
    # _log_fh 仍为 None，不报错
    assert recorder._log_fh is None


# ============================================================================
# emit("observe_image", ...) 行为
# ============================================================================


def test_emit_observe_image_writes_png(tmp_path: Path):
    """emit("observe_image", image=ndarray, idx=1) 写 observer/001.png。"""
    recorder = ExperimentRecorder(root=tmp_path)
    recorder.start()
    img = np.random.randint(0, 256, (64, 48, 3), dtype=np.uint8)
    recorder.emit("observe_image", image=img, idx=1)
    recorder.finish(success=True, summary="x")

    png_path = recorder.observer_dir / "001.png"
    assert png_path.exists()

    # 读回验证尺寸
    loaded = np.array(Image.open(png_path))
    assert loaded.shape == (64, 48, 3)


def test_emit_observe_image_pixels_match(tmp_path: Path):
    """PNG 像素应与原始 ndarray 一致（无损）。"""
    recorder = ExperimentRecorder(root=tmp_path)
    recorder.start()
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[0, 0] = [255, 0, 0]  # 左上角红色
    img[10, 20] = [0, 255, 0]  # 绿点
    recorder.emit("observe_image", image=img, idx=0)
    recorder.finish(success=True, summary="x")

    loaded = np.array(Image.open(recorder.observer_dir / "000.png"))
    assert tuple(loaded[0, 0]) == (255, 0, 0)
    assert tuple(loaded[10, 20]) == (0, 255, 0)


def test_emit_observe_image_sequential_idx(tmp_path: Path):
    """连续 emit 不同 idx 应生成 000/001/002 顺序文件。"""
    recorder = ExperimentRecorder(root=tmp_path)
    recorder.start()
    for idx in range(3):
        img = np.full((16, 16, 3), idx * 80, dtype=np.uint8)
        recorder.emit("observe_image", image=img, idx=idx)
    recorder.finish(success=True, summary="x")

    assert (recorder.observer_dir / "000.png").exists()
    assert (recorder.observer_dir / "001.png").exists()
    assert (recorder.observer_dir / "002.png").exists()
    assert not (recorder.observer_dir / "003.png").exists()


def test_emit_observe_image_without_start_is_noop(tmp_path: Path):
    """未 start() 时 emit("observe_image") 应静默 no-op（observer_dir is None）。"""
    recorder = ExperimentRecorder(root=tmp_path)
    # 不调 start()
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    recorder.emit("observe_image", image=img, idx=1)
    # observer_dir 仍为 None，不报错
    assert recorder.observer_dir is None


# ============================================================================
# emit("video_frame", ...) 占位 no-op
# ============================================================================


def test_emit_video_frame_is_noop(tmp_path: Path):
    """emit("video_frame", ...) 不报错、不产生视频文件。"""
    recorder = ExperimentRecorder(root=tmp_path)
    recorder.start()
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    recorder.emit("video_frame", image=img, timestamp=0.123)
    recorder.finish(success=True, summary="x")

    # observer_dir 下不应出现 000.png 等
    files = list(recorder.observer_dir.iterdir())
    assert len(files) == 0


def test_emit_video_frame_does_not_log(tmp_path: Path):
    """video_frame 事件不应写入 log（占位）。"""
    recorder = ExperimentRecorder(root=tmp_path, log_to_stdout=False)
    recorder.start()
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    recorder.emit("video_frame", image=img, timestamp=0.5)
    recorder.finish(success=True, summary="x")

    content = (recorder.exp_dir / "experiment.log").read_text(encoding="utf-8")
    # 只有 finish 写入的一行，无 video_frame 相关
    lines = [l for l in content.split("\n") if l.strip()]
    assert len(lines) == 1
    assert "Pipeline finished" in lines[0]


# ============================================================================
# emit 未知 event 默认 no-op
# ============================================================================


def test_emit_unknown_event_is_noop(tmp_path: Path):
    """emit("unknown_event", ...) 不报错、不产生文件、不写 log。"""
    recorder = ExperimentRecorder(root=tmp_path, log_to_stdout=False)
    recorder.start()
    recorder.emit("foo_event", x=1, y="hello")
    recorder.finish(success=True, summary="x")

    files = list(recorder.observer_dir.iterdir())
    assert len(files) == 0
    content = (recorder.exp_dir / "experiment.log").read_text(encoding="utf-8")
    assert "foo_event" not in content
    assert "x=1" not in content


# ============================================================================
# enabled=False 全 no-op
# ============================================================================


def test_enabled_false_start_noop(tmp_path: Path):
    """enabled=False 时 start() 不创建任何目录。"""
    recorder = ExperimentRecorder(root=tmp_path, enabled=False)
    result = recorder.start()
    assert result == tmp_path
    # tmp_path 下不应有任何新子目录
    assert list(tmp_path.iterdir()) == []


def test_enabled_false_emit_noop(tmp_path: Path):
    """enabled=False 时 emit() 不产生任何文件。"""
    recorder = ExperimentRecorder(root=tmp_path, enabled=False)
    recorder.emit("log", message="x")
    recorder.emit("observe_image", image=np.zeros((10, 10, 3), dtype=np.uint8), idx=1)
    # tmp_path 下不应有任何新子目录
    assert list(tmp_path.iterdir()) == []


def test_enabled_false_finish_noop(tmp_path: Path):
    """enabled=False 时 finish() 不报错、不产生文件。"""
    recorder = ExperimentRecorder(root=tmp_path, enabled=False)
    recorder.finish(success=True, summary="x")
    assert list(tmp_path.iterdir()) == []


# ============================================================================
# emit() 异常隔离
# ============================================================================


def test_emit_observe_image_exception_isolated(tmp_path: Path, capsys, monkeypatch):
    """PIL 抛错时 emit 不向上传播，stderr 含 warning。"""
    recorder = ExperimentRecorder(root=tmp_path)
    recorder.start()

    # 让 PIL Image.fromarray 抛错
    def fake_fromarray(*args, **kwargs):
        raise TypeError("模拟 PIL 异常")

    monkeypatch.setattr(Image, "fromarray", fake_fromarray)

    # emit 不应抛异常
    recorder.emit("observe_image", image=np.zeros((10, 10, 3), dtype=np.uint8), idx=1)

    captured = capsys.readouterr()
    assert "ExperimentRecorder" in captured.err
    assert "emit(observe_image) failed" in captured.err
    assert "模拟 PIL 异常" in captured.err

    # 业务代码继续（recorder 仍可用）
    recorder.emit("log", message="after error")
    recorder.finish(success=True, summary="x")


def test_emit_log_write_exception_isolated(tmp_path: Path, capsys):
    """_log_fh.write 抛错时 emit 不向上传播。"""
    recorder = ExperimentRecorder(root=tmp_path)
    recorder.start()

    # 替换 _log_fh.write 抛错
    original_write = recorder._log_fh.write
    def bad_write(s):
        raise IOError("模拟写入失败")

    recorder._log_fh.write = bad_write

    # emit 不应抛异常
    recorder.emit("log", message="write failed")

    captured = capsys.readouterr()
    assert "emit(log) failed" in captured.err

    # 还原（避免影响 finish 的 close）
    recorder._log_fh.write = original_write
    recorder.finish(success=True, summary="x")


# ============================================================================
# finish() 关闭文件句柄 + 收尾日志
# ============================================================================


def test_finish_writes_pipeline_finished_log(tmp_path: Path):
    """finish() 应写入 `Pipeline finished, success=..., summary=...`。"""
    recorder = ExperimentRecorder(root=tmp_path, log_to_stdout=False)
    recorder.start()
    recorder.finish(success=True, summary="steps=12")

    content = (recorder.exp_dir / "experiment.log").read_text(encoding="utf-8")
    assert "Pipeline finished, success=True, summary=steps=12" in content


def test_finish_closes_log_handle(tmp_path: Path):
    """finish() 应关闭 _log_fh 文件句柄。"""
    recorder = ExperimentRecorder(root=tmp_path)
    recorder.start()
    assert recorder._log_fh is not None
    recorder.finish(success=True, summary="x")
    assert recorder._log_fh is None


def test_finish_success_false(tmp_path: Path):
    """finish(success=False, ...) 应写入 success=False。"""
    recorder = ExperimentRecorder(root=tmp_path, log_to_stdout=False)
    recorder.start()
    recorder.finish(success=False, summary="error msg")

    content = (recorder.exp_dir / "experiment.log").read_text(encoding="utf-8")
    assert "success=False" in content
    assert "error msg" in content


def test_finish_can_be_called_twice(tmp_path: Path):
    """finish() 多次调用不应报错（第二次 no-op 因 _log_fh 已关闭）。"""
    recorder = ExperimentRecorder(root=tmp_path, log_to_stdout=False)
    recorder.start()
    recorder.finish(success=True, summary="first")
    # 第二次 finish 应 no-op（_log_fh is None，_handle_log 直接 return）
    recorder.finish(success=True, summary="second")
    # experiment.log 应只有一行 finish
    content = (recorder.exp_dir / "experiment.log").read_text(encoding="utf-8")
    assert content.count("Pipeline finished") == 1


# ============================================================================
# 启用后切换 enabled 不影响已 start 实例
# ============================================================================


def test_start_then_toggle_enabled_does_not_affect_existing(tmp_path: Path):
    """start() 后修改 enabled=False 不影响已创建的目录。"""
    recorder = ExperimentRecorder(root=tmp_path)
    recorder.start()
    exp_dir = recorder.exp_dir

    # 切换 enabled
    recorder.enabled = False

    # 已存在的目录不受影响
    assert exp_dir.exists()
    assert recorder.observer_dir.is_dir()

    # 但 emit 现在会早返回
    recorder.emit("log", message="after toggle")
    recorder.finish(success=True, summary="x")

    # log 文件应仍存在（start 时已打开）
    assert (exp_dir / "experiment.log").exists()