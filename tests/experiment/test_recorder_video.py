"""ExperimentRecorder 视频扩展单元测试（iter12-video-recording 任务 5）。

覆盖：
- __init__ video 参数存储
- start 创建 process/ 目录（video 非 None 且启用）
- video=None / enabled=False 时不创建 process 目录
- emit("video_frame") 委托 writer.write
- video=None 时 video_frame no-op
- finish 调 writer.close()
- 回归：原 start/emit/finish/enabled=False 行为不变（由 test_recorder.py 覆盖）
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from experiment.recorder import ExperimentRecorder
from experiment.video.config import VideoConfig


# ============================================================================
# Fake cv2（真实写入路径）
# ============================================================================


class FakeCv2Writer:
    def __init__(self, path, fourcc, fps, size):
        self.path = path
        self.released = False
        self.written = []

    def isOpened(self):
        return True

    def write(self, frame):
        self.written.append(frame)

    def release(self):
        self.released = True


@pytest.fixture
def fake_cv2(monkeypatch):
    """注入 fake cv2 模块，使 VideoWriter.open() 成功；记录创建实例。"""
    fake = types.ModuleType("cv2")
    fake.COLOR_RGB2BGR = 4
    fake.VideoWriter_fourcc = lambda *a: b"mp4v"
    fake.created_writers = []

    def video_writer(path, fourcc, fps, size):
        w = FakeCv2Writer(path, fourcc, fps, size)
        fake.created_writers.append(w)
        return w

    fake.VideoWriter = video_writer
    fake.cvtColor = lambda img, code: img[:, :, ::-1]
    monkeypatch.setitem(sys.modules, "cv2", fake)
    return fake


@pytest.fixture
def writer_paths(fake_cv2):
    """返回记录 fake writer 实例的容器。"""
    container = {"writers": []}
    return container


# ============================================================================
# __init__ video 参数
# ============================================================================


def test_init_stores_video(tmp_path: Path):
    """__init__ 存储 video 配置。"""
    cfg = VideoConfig()
    recorder = ExperimentRecorder(root=tmp_path, video=cfg)
    assert recorder.video is cfg
    assert recorder.process_dir is None
    assert recorder._video_writer is None


def test_init_default_video_none(tmp_path: Path):
    """不传 video 时默认为 None（旧调用兼容）。"""
    recorder = ExperimentRecorder(root=tmp_path)
    assert recorder.video is None


# ============================================================================
# start() process 目录
# ============================================================================


def test_start_creates_process_dir(tmp_path: Path, fake_cv2):
    """video 非 None 时 start() 创建 process/ 目录并 open writer。"""
    recorder = ExperimentRecorder(root=tmp_path, video=VideoConfig())
    exp_dir = recorder.start()
    process_dir = exp_dir / "process"
    assert process_dir.is_dir()
    assert recorder.process_dir == process_dir
    assert recorder._video_writer is not None
    assert recorder._video_writer.is_open


def test_start_no_video_no_process_dir(tmp_path: Path):
    """video=None 时不创建 process 目录。"""
    recorder = ExperimentRecorder(root=tmp_path)
    exp_dir = recorder.start()
    assert not (exp_dir / "process").exists()
    assert recorder.process_dir is None
    assert recorder._video_writer is None


def test_start_video_disabled_no_process_dir(tmp_path: Path):
    """video.enabled=False 时不创建 process 目录。"""
    recorder = ExperimentRecorder(
        root=tmp_path, video=VideoConfig(enabled=False)
    )
    exp_dir = recorder.start()
    assert not (exp_dir / "process").exists()
    assert recorder.process_dir is None
    assert recorder._video_writer is None


def test_start_video_open_failed_no_writer(tmp_path: Path, monkeypatch):
    """writer.open() 失败时降级 no-op（_video_writer None + warning）。"""
    cfg = VideoConfig()
    recorder = ExperimentRecorder(root=tmp_path, video=cfg)
    # 让 open() 失败：注入无法 import 的 cv2
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("No module named 'cv2'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.warns(UserWarning):
        recorder.start()
    assert recorder._video_writer is None
    # process 目录仍被创建（与设计一致：目录创建与 writer 打开解耦）
    assert (recorder.exp_dir / "process").is_dir()


# ============================================================================
# emit("video_frame") 行为
# ============================================================================


def test_video_frame_writes_through_writer(tmp_path: Path, fake_cv2):
    """emit("video_frame") 委托 writer.write（fake writer 收到帧）。"""
    recorder = ExperimentRecorder(root=tmp_path, video=VideoConfig())
    recorder.start()
    assert len(fake_cv2.created_writers) == 1
    fake_writer = fake_cv2.created_writers[0]

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    recorder.emit("video_frame", image=img, timestamp=1.0)
    assert len(fake_writer.written) == 1

    recorder.finish(success=True, summary="x")
    assert fake_writer.released is True


def test_video_frame_noop_without_video(tmp_path: Path):
    """video=None 时 video_frame no-op 不报错。"""
    recorder = ExperimentRecorder(root=tmp_path, log_to_stdout=False)
    recorder.start()
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    recorder.emit("video_frame", image=img, timestamp=0.5)
    recorder.finish(success=True, summary="x")
    assert recorder.process_dir is None


def test_video_frame_noop_writer_unavailable(tmp_path: Path, monkeypatch):
    """writer 不可用时 video_frame no-op 不报错。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("No module named 'cv2'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    recorder = ExperimentRecorder(root=tmp_path, video=VideoConfig())
    recorder.start()
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    recorder.emit("video_frame", image=img, timestamp=0.5)
    recorder.finish(success=True, summary="x")
    assert recorder._video_writer is None


# ============================================================================
# finish() close writer
# ============================================================================


def test_finish_closes_writer(tmp_path: Path, fake_cv2):
    """finish() 调 writer.close()。"""
    recorder = ExperimentRecorder(root=tmp_path, video=VideoConfig())
    recorder.start()
    assert recorder._video_writer is not None
    recorder.finish(success=True, summary="x")
    assert recorder._video_writer is None  # close 后置空
    # 再次 finish 不报错
    recorder.finish(success=True, summary="y")


# ============================================================================
# 回归：enabled=False 全 no-op
# ============================================================================


def test_enabled_false_video_noop(tmp_path: Path):
    """enabled=False 时 start 不创建任何目录（含 process）。"""
    recorder = ExperimentRecorder(root=tmp_path, enabled=False, video=VideoConfig())
    result = recorder.start()
    assert result == tmp_path
    assert list(tmp_path.iterdir()) == []
