"""PyBulletEnv 视频抓帧接入单元测试（iter12-video-recording 任务 9）。

策略：
- set_recorder 注入带/不带 video 配置的 recorder
- monkeypatch FrameCapturer 类，验证构造/不构造、step 传 on_substep、reset 抓初始帧
- 避免真实 pybullet 连接（测试只验证接线，不跑物理）
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from config.loader import EnvConfig, RobotConfig
from env.pybullet_env import PyBulletEnv
from experiment.recorder import ExperimentRecorder, get_recorder, set_recorder
from experiment.video.config import VideoConfig


# ============================================================================
# Fixtures
# ============================================================================


class FakeCapturer:
    """记录构造参数与调用。"""

    instances = []

    def __init__(self, env, config, recorder):
        self.env = env
        self.config = config
        self.recorder = recorder
        self.reset_count = 0
        self.capture_count = 0
        FakeCapturer.instances.append(self)

    def reset_timer(self):
        self.reset_count += 1

    def capture(self):
        self.capture_count += 1


@pytest.fixture(autouse=True)
def reset_singleton():
    """每个测试前清空全局 recorder 单例。"""
    set_recorder(None)
    FakeCapturer.instances = []
    yield
    set_recorder(None)


@pytest.fixture
def fake_frame_capturer(monkeypatch):
    """把 env.pybullet_env 模块中的 FrameCapturer 替换为 FakeCapturer。"""
    import env.pybullet_env as env_mod

    monkeypatch.setattr(env_mod, "FrameCapturer", FakeCapturer)
    return FakeCapturer


def _make_env(monkeypatch, recorder=None):
    """构造 PyBulletEnv（monkeypatch 掉 pybullet 模块，避免真实连接）。"""
    import env.pybullet_env as env_mod

    fake_pb = types.ModuleType("pybullet")
    fake_pb.DIRECT = 0
    fake_pb.GUI = 1
    fake_pb.ER_TINY_RENDERER = 2
    fake_pb.ER_BULLET_HARDWARE_OPENGL = 3
    # get_obs / _ensure_connected / render 依赖的 pybullet API（统一打桩）
    fake_pb.getConnectionInfo = lambda cid: {"isConnected": True}
    fake_pb.getLinkState = lambda *a, **k: ([0.3, 0.0, 0.2], [0, 0, 0, 1])
    fake_pb.getBasePositionAndOrientation = lambda *a, **k: ([0.5, 0, 0.1], [0, 0, 0, 1])
    # getCameraImage 返回 RGBA 数组（匹配默认相机分辨率 640x480）
    fake_pb.getCameraImage = lambda *a, **k: (
        640, 480,
        np.zeros((640, 480, 4), dtype=np.uint8),
        None, None,
    )
    fake_pb.computeViewMatrixFromYawPitchRoll = lambda **k: [0] * 16
    fake_pb.computeProjectionMatrixFOV = lambda **k: [0] * 16
    monkeypatch.setattr(env_mod, "p", fake_pb)
    if recorder is not None:
        set_recorder(recorder)
    return PyBulletEnv(EnvConfig(mode="direct", renderer="cpu"), RobotConfig())


# ============================================================================
# _init_video_capture 构造/不构造
# ============================================================================


def test_init_capture_enabled(monkeypatch, fake_frame_capturer):
    """recorder.video 启用时 self._capturer 非 None。"""
    recorder = ExperimentRecorder(root="/tmp", video=VideoConfig())
    set_recorder(recorder)
    env = _make_env(monkeypatch)
    assert env._capturer is not None
    assert isinstance(env._capturer, FakeCapturer)


def test_init_capture_video_none(monkeypatch, fake_frame_capturer):
    """recorder.video None 时 _capturer is None。"""
    recorder = ExperimentRecorder(root="/tmp", video=None)
    set_recorder(recorder)
    env = _make_env(monkeypatch)
    assert env._capturer is None


def test_init_capture_video_disabled(monkeypatch, fake_frame_capturer):
    """recorder.video.enabled=False 时 _capturer is None。"""
    recorder = ExperimentRecorder(root="/tmp", video=VideoConfig(enabled=False))
    set_recorder(recorder)
    env = _make_env(monkeypatch)
    assert env._capturer is None


def test_init_capture_no_recorder(monkeypatch, fake_frame_capturer):
    """未 set_recorder 时（SafeRecorder 无 video）_capturer is None。"""
    env = _make_env(monkeypatch)
    assert env._capturer is None


# ============================================================================
# step() 传 on_substep
# ============================================================================


def test_step_passes_on_substep_when_capturer(monkeypatch, fake_frame_capturer):
    """capturer 存在时 step() 传 on_substep 给 robot.step_action。"""
    recorder = ExperimentRecorder(root="/tmp", video=VideoConfig())
    set_recorder(recorder)
    env = _make_env(monkeypatch)

    captured_kwargs = {}

    class FakeRobot:
        ee_link_index = 11

        def step_action(self, action, robot_id, client_id, **kwargs):
            captured_kwargs.update(kwargs)

    env.robot = FakeRobot()
    env._client_id = 0  # 假装已连接（fake getConnectionInfo 返回 True）

    env.step({"dx": 0.0, "dy": 0.0, "dz": 0.0, "gripper": 1.0})
    assert "on_substep" in captured_kwargs
    assert callable(captured_kwargs["on_substep"])


def test_step_no_on_substep_without_capturer(monkeypatch, fake_frame_capturer):
    """capturer 不存在时 step() 传 on_substep=None。"""
    set_recorder(ExperimentRecorder(root="/tmp", video=None))
    env = _make_env(monkeypatch)

    captured_kwargs = {}

    class FakeRobot:
        ee_link_index = 11

        def step_action(self, action, robot_id, client_id, **kwargs):
            captured_kwargs.update(kwargs)

    env.robot = FakeRobot()
    env._client_id = 0

    env.step({"dx": 0.0, "dy": 0.0, "dz": 0.0, "gripper": 1.0})
    assert captured_kwargs.get("on_substep") is None


# ============================================================================
# reset() 抓初始帧
# ============================================================================


def test_reset_captures_initial_frame(monkeypatch, fake_frame_capturer):
    """reset() 调 capturer.reset_timer + capture。"""
    recorder = ExperimentRecorder(root="/tmp", video=VideoConfig())
    set_recorder(recorder)
    env = _make_env(monkeypatch)
    capturer = env._capturer
    assert capturer is not None

    # 打桩 reset 的 pybullet 调用（不需要真实仿真）
    import env.pybullet_env as env_mod

    fake_pb = env_mod.p
    fake_pb.connect = lambda mode: 0
    fake_pb.setAdditionalSearchPath = lambda *a, **k: None
    fake_pb.resetSimulation = lambda *a, **k: None
    fake_pb.setGravity = lambda *a, **k: None
    fake_pb.loadURDF = lambda *a, **k: 0
    fake_pb.changeVisualShape = lambda *a, **k: None
    monkeypatch.setattr(env_mod, "p", fake_pb)
    # 移除 time.sleep 等待
    monkeypatch.setattr(env_mod.time, "sleep", lambda *a, **k: None)

    env.reset(task_spec={"objects": []})
    assert capturer.reset_count >= 1
    assert capturer.capture_count >= 1


def test_reset_no_capturer_no_error(monkeypatch, fake_frame_capturer):
    """无 capturer 时 reset 不报错。"""
    set_recorder(ExperimentRecorder(root="/tmp", video=None))
    env = _make_env(monkeypatch)
    assert env._capturer is None

    import env.pybullet_env as env_mod

    fake_pb = env_mod.p
    fake_pb.connect = lambda mode: 0
    fake_pb.setAdditionalSearchPath = lambda *a, **k: None
    fake_pb.resetSimulation = lambda *a, **k: None
    fake_pb.setGravity = lambda *a, **k: None
    fake_pb.loadURDF = lambda *a, **k: 0
    fake_pb.changeVisualShape = lambda *a, **k: None
    monkeypatch.setattr(env_mod, "p", fake_pb)
    monkeypatch.setattr(env_mod.time, "sleep", lambda *a, **k: None)

    env.reset(task_spec={"objects": []})  # 不应抛
