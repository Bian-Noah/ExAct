"""VideoWriter 单元测试（iter12-video-recording 任务 3）。

策略：monkeypatch 注入 fake cv2 模块验证生命周期与降级路径，
不依赖真实 opencv（实现层 cv2 缺失时 open() 返回 False）。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from experiment.video.config import VideoConfig
from experiment.video.writer import VideoWriter


# ============================================================================
# Fake cv2 模块
# ============================================================================


class FakeCv2Writer:
    """模拟 cv2.VideoWriter。"""

    def __init__(self, path, fourcc, fps, size, *, fail_open=False):
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self.released = False
        self.written = []
        self.fail_open = fail_open

    def isOpened(self) -> bool:
        return not self.fail_open

    def write(self, frame) -> None:
        if self.released:
            raise RuntimeError("writer already released")
        self.written.append(frame)

    def release(self) -> None:
        self.released = True


def make_fake_cv2(*, writer_fail_open: bool = False, write_raises: bool = False):
    """构造 fake cv2 模块（含 VideoWriter / VideoWriter_fourcc / cvtColor）。"""
    fake = types.ModuleType("cv2")
    fake.COLOR_RGB2BGR = 4

    created: list[FakeCv2Writer] = []

    def fake_fourcc(*args):
        return "".join(args).encode()

    def fake_video_writer(path, fourcc, fps, size):
        w = FakeCv2Writer(path, fourcc, fps, size, fail_open=writer_fail_open)
        created.append(w)
        return w

    def fake_cvt_color(img, code):
        assert code == fake.COLOR_RGB2BGR
        return np.ascontiguousarray(img[:, :, ::-1])  # RGB → BGR

    fake.VideoWriter_fourcc = fake_fourcc
    fake.VideoWriter = fake_video_writer
    fake.cvtColor = fake_cvt_color
    return fake, created


@pytest.fixture
def fake_cv2(monkeypatch):
    """将 fake cv2 注入 sys.modules。"""
    fake, created = make_fake_cv2()
    monkeypatch.setitem(sys.modules, "cv2", fake)
    return fake, created


# ============================================================================
# open()
# ============================================================================


def test_open_success(tmp_path: Path, fake_cv2):
    """open 成功：返回 True、is_open True、路径/编码正确。"""
    _, created = fake_cv2
    writer = VideoWriter(tmp_path, VideoConfig())
    assert writer.open() is True
    assert writer.is_open is True
    assert len(created) == 1
    assert created[0].path == str(tmp_path / "demo.mp4")
    assert created[0].fps == 15.0
    assert created[0].size == (640, 480)


def test_open_cv2_missing_returns_false(tmp_path: Path, monkeypatch):
    """cv2 import 失败（sys.modules 移除）时 open 返回 False。"""
    monkeypatch.delitem(sys.modules, "cv2", raising=False)
    # 阻止 import 系统级 cv2 真实安装
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("No module named 'cv2'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    writer = VideoWriter(tmp_path, VideoConfig())
    assert writer.open() is False
    assert writer.is_open is False


def test_open_odd_resolution_returns_false(tmp_path: Path, fake_cv2):
    """奇数分辨率 open 返回 False（降级）。"""
    writer = VideoWriter(tmp_path, VideoConfig(resolution=(641, 480)))
    assert writer.open() is False
    assert writer.is_open is False


def test_open_writer_raises_returns_false(tmp_path: Path, monkeypatch):
    """cv2.VideoWriter 构造抛异常时 open 返回 False。"""
    fake, _ = make_fake_cv2()

    def boom(*args, **kwargs):
        raise OSError("cannot open")

    fake.VideoWriter = boom
    monkeypatch.setitem(sys.modules, "cv2", fake)
    writer = VideoWriter(tmp_path, VideoConfig())
    assert writer.open() is False
    assert writer.is_open is False


def test_open_fail_open_fallback_avc1(tmp_path: Path, monkeypatch):
    """mp4v 打开失败回退 avc1 并成功。"""
    fake, created = make_fake_cv2(writer_fail_open=False)

    # 第一个 VideoWriter 返回 fail_open 的 writer，第二个正常
    calls = []

    def flaky_writer(path, fourcc, fps, size):
        fail = len(calls) == 0
        calls.append(fourcc)
        return FakeCv2Writer(path, fourcc, fps, size, fail_open=fail)

    fake.VideoWriter = flaky_writer
    monkeypatch.setitem(sys.modules, "cv2", fake)

    writer = VideoWriter(tmp_path, VideoConfig())
    assert writer.open() is True
    assert len(calls) == 2
    assert calls[0] == b"mp4v"
    assert calls[1] == b"avc1"


# ============================================================================
# write()
# ============================================================================


def test_write_rgb_to_bgr(tmp_path: Path, fake_cv2):
    """write 调 cvtColor(RGB2BGR) + writer.write。"""
    _, created = fake_cv2
    writer = VideoWriter(tmp_path, VideoConfig())
    assert writer.open() is True
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[0, 0] = [255, 0, 0]  # 红
    result = writer.write(img)
    assert result is True
    assert len(created[0].written) == 1
    # BGR 转换：红 (255,0,0) → (0,0,255)
    written = created[0].written[0]
    assert tuple(written[0, 0]) == (0, 0, 255)


def test_write_not_open_returns_false(tmp_path: Path, fake_cv2):
    """未 open 时 write 返回 False 不调 writer。"""
    _, created = fake_cv2
    writer = VideoWriter(tmp_path, VideoConfig())
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    assert writer.write(img) is False
    assert len(created) == 0  # 从未创建底层 writer


def test_write_exception_returns_false(tmp_path: Path, monkeypatch):
    """writer.write 抛异常时 write 返回 False 不向上抛。"""
    fake, created = make_fake_cv2()
    monkeypatch.setitem(sys.modules, "cv2", fake)

    writer = VideoWriter(tmp_path, VideoConfig())
    assert writer.open() is True  # 先 open（会创建 created[0]）
    assert len(created) == 1

    def bad_write(frame):
        raise RuntimeError("write failed")

    created[0].write = bad_write
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    assert writer.write(img) is False


def test_write_invalid_shape_returns_false(tmp_path: Path, fake_cv2):
    """非 (H,W,3) 图像 write 返回 False。"""
    _, created = fake_cv2
    writer = VideoWriter(tmp_path, VideoConfig())
    assert writer.open() is True
    img = np.zeros((4, 4), dtype=np.uint8)  # 2D
    assert writer.write(img) is False
    assert len(created[0].written) == 0


# ============================================================================
# close()
# ============================================================================


def test_close_releases(tmp_path: Path, fake_cv2):
    """close 调 release + is_open False。"""
    _, created = fake_cv2
    writer = VideoWriter(tmp_path, VideoConfig())
    assert writer.open() is True
    writer.close()
    assert created[0].released is True
    assert writer.is_open is False


def test_close_twice_safe(tmp_path: Path, fake_cv2):
    """重复 close 不报错。"""
    writer = VideoWriter(tmp_path, VideoConfig())
    assert writer.open() is True
    writer.close()
    writer.close()  # 第二次 no-op


def test_close_not_open_safe(tmp_path: Path, fake_cv2):
    """未 open 时 close no-op。"""
    writer = VideoWriter(tmp_path, VideoConfig())
    writer.close()  # 不报错
    assert writer.is_open is False
