"""FrameCapturer 单元测试（iter12-video-recording 任务 4）。

策略：FakeEnv（暴露 env_config.cameras / _client_id / _renderer）+ FakeRecorder
（记录 emit 调用），monkeypatch pybullet.getCameraImage / 矩阵计算函数。
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field

import numpy as np
import pytest

from experiment.video.capture import FrameCapturer
from experiment.video.config import VideoConfig


# ============================================================================
# Fake 依赖
# ============================================================================


@dataclass(frozen=True)
class FakeCameraSpec:
    name: str
    target: tuple = (0.5, 0.0, 0.5)
    distance: float = 1.5
    yaw: float = 50
    pitch: float = -35
    roll: float = 0
    fov: float = 60
    resolution: tuple = (640, 480)


@dataclass
class FakeEnvConfig:
    cameras: tuple = (FakeCameraSpec(name="cam0"),)


class FakeEnv:
    """模拟 PyBulletEnv：暴露 capture 需要的属性。"""

    def __init__(self, cameras=(FakeCameraSpec(name="cam0"),)):
        self.env_config = FakeEnvConfig(cameras=tuple(cameras))
        self._client_id = 0
        self._renderer = 1  # 任意 int


class FakeRecorder:
    """模拟 ExperimentRecorder：记录 emit 调用。"""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def emit(self, event: str, **fields) -> None:
        self.events.append((event, fields))


# ============================================================================
# fake pybullet
# ============================================================================


@pytest.fixture
def fake_pybullet(monkeypatch):
    """注入 fake pybullet 模块：getCameraImage 返回可控 RGBA。"""
    fake = types.ModuleType("pybullet")
    fake._calls = {"get_camera": 0, "view": 0, "proj": 0}

    def getCameraImage(width, height, viewMatrix=None, projectionMatrix=None,
                       physicsClientId=None, renderer=None):
        fake._calls["get_camera"] += 1
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255
        rgba[0, 0] = [255, 0, 0, 255]  # 左上角红，RGB=(255,0,0)
        return (width, height, rgba, [], [])

    def computeViewMatrixFromYawPitchRoll(**kwargs):
        fake._calls["view"] += 1
        return np.eye(4)

    def computeProjectionMatrixFOV(**kwargs):
        fake._calls["proj"] += 1
        return np.eye(4)

    fake.getCameraImage = getCameraImage
    fake.computeViewMatrixFromYawPitchRoll = computeViewMatrixFromYawPitchRoll
    fake.computeProjectionMatrixFOV = computeProjectionMatrixFOV
    monkeypatch.setitem(sys.modules, "pybullet", fake)
    return fake


@pytest.fixture
def fake_pybullet_error(monkeypatch):
    """getCameraImage 抛异常的 fake pybullet。"""
    fake = types.ModuleType("pybullet")

    def getCameraImage(*args, **kwargs):
        raise RuntimeError("physics server disconnected")

    fake.getCameraImage = getCameraImage
    fake.computeViewMatrixFromYawPitchRoll = lambda **kw: np.eye(4)
    fake.computeProjectionMatrixFOV = lambda **kw: np.eye(4)
    monkeypatch.setitem(sys.modules, "pybullet", fake)
    return fake


# ============================================================================
# 抓帧节流
# ============================================================================


def test_capture_every_step_emits_each_call(fake_pybullet):
    """capture_every_n_steps=1 时每次 capture 触发 1 次 emit。"""
    env = FakeEnv()
    recorder = FakeRecorder()
    cfg = VideoConfig(capture_every_n_steps=1)
    capturer = FrameCapturer(env, cfg, recorder)
    capturer.capture()
    capturer.capture()
    assert len(recorder.events) == 2
    assert all(e[0] == "video_frame" for e in recorder.events)


def test_capture_throttle(fake_pybullet):
    """capture_every_n_steps=2 时每 2 次 capture 触发 1 次 emit。"""
    env = FakeEnv()
    recorder = FakeRecorder()
    cfg = VideoConfig(capture_every_n_steps=2)
    capturer = FrameCapturer(env, cfg, recorder)
    for _ in range(4):
        capturer.capture()
    assert len(recorder.events) == 2


def test_capture_disabled_no_emit(fake_pybullet):
    """enabled=False 时 capture no-op 不 emit。"""
    env = FakeEnv()
    recorder = FakeRecorder()
    cfg = VideoConfig(enabled=False)
    capturer = FrameCapturer(env, cfg, recorder)
    capturer.capture()
    capturer.capture()
    assert len(recorder.events) == 0


# ============================================================================
# 帧内容
# ============================================================================


def test_capture_emit_rgb_shape(fake_pybullet):
    """emit image shape=(H,W,3) uint8（RGBA 截 3 通道）。"""
    env = FakeEnv()
    recorder = FakeRecorder()
    cfg = VideoConfig(resolution=(320, 240))
    capturer = FrameCapturer(env, cfg, recorder)
    capturer.capture()
    image = recorder.events[0][1]["image"]
    assert image.shape == (240, 320, 3)
    assert image.dtype == np.uint8
    # RGBA 转 RGB：alpha 通道被丢弃
    assert image[0, 0].tolist() == [255, 0, 0]


def test_capture_emit_has_timestamp(fake_pybullet):
    """emit 携带 timestamp 字段。"""
    env = FakeEnv()
    recorder = FakeRecorder()
    capturer = FrameCapturer(env, VideoConfig(), recorder)
    capturer.capture()
    assert "timestamp" in recorder.events[0][1]
    assert isinstance(recorder.events[0][1]["timestamp"], float)


# ============================================================================
# 相机选择
# ============================================================================


def test_capture_camera_selection_named(fake_pybullet):
    """config.camera 指定名时使用对应相机矩阵。"""
    env = FakeEnv(cameras=(
        FakeCameraSpec(name="cam0", yaw=10),
        FakeCameraSpec(name="cam1", yaw=99),
    ))
    recorder = FakeRecorder()
    cfg = VideoConfig(camera="cam1")
    capturer = FrameCapturer(env, cfg, recorder)
    capturer.capture()
    # cam1 的 yaw=99 应被传入矩阵计算
    assert fake_pybullet._calls["view"] >= 1
    assert len(recorder.events) == 1


def test_capture_camera_selection_none_first(fake_pybullet):
    """config.camera=None 用第一个相机。"""
    env = FakeEnv(cameras=(
        FakeCameraSpec(name="cam0"),
        FakeCameraSpec(name="cam1"),
    ))
    recorder = FakeRecorder()
    capturer = FrameCapturer(env, VideoConfig(), recorder)
    capturer.capture()
    assert len(recorder.events) == 1


def test_capture_camera_matrix_cached(fake_pybullet):
    """矩阵缓存生效：连续 capture 只算一次 view/proj。"""
    env = FakeEnv()
    recorder = FakeRecorder()
    capturer = FrameCapturer(env, VideoConfig(), recorder)
    capturer.capture()
    capturer.capture()
    # 矩阵计算（view+proj 各一次）——只发生在第一次 capture
    assert fake_pybullet._calls["view"] == 1
    assert fake_pybullet._calls["proj"] == 1
    # 抓帧两次
    assert fake_pybullet._calls["get_camera"] == 2


def test_reset_timer_recomputes_matrix(fake_pybullet):
    """reset_timer 后重新计算矩阵。"""
    env = FakeEnv()
    recorder = FakeRecorder()
    capturer = FrameCapturer(env, VideoConfig(), recorder)
    capturer.capture()
    capturer.reset_timer()
    capturer.capture()
    assert fake_pybullet._calls["view"] == 2


# ============================================================================
# 异常隔离
# ============================================================================


def test_capture_exception_isolated(fake_pybullet_error):
    """getCameraImage 抛异常时 capture 吞掉不抛、不 emit。"""
    env = FakeEnv()
    recorder = FakeRecorder()
    capturer = FrameCapturer(env, VideoConfig(), recorder)
    capturer.capture()  # 不应抛
    assert len(recorder.events) == 0
    assert capturer._error_count == 1


def test_capture_exception_throttle_logging(fake_pybullet_error):
    """异常日志节流：首次 + 每 10 次。"""
    env = FakeEnv()
    recorder = FakeRecorder()
    capturer = FrameCapturer(env, VideoConfig(), recorder)
    with pytest.warns(UserWarning):
        capturer.capture()  # 首次 warning
    # 2..9 次不 warn
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")
        for _ in range(8):
            capturer.capture()
    with pytest.warns(UserWarning):
        capturer.capture()  # 第 10 次 warning
