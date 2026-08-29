"""视频录制端到端功能测试（iter12-video-recording 任务 10）。

覆盖：
- F1: 真实 PyBulletEnv 链路（reset + step ×N）→ process/demo.mp4 存在且非空、可被 cv2 打开
- F2: pipeline（video.enabled=False）→ 无 process 目录 + log 含 video_enabled=False
- F3: 无 video 段旧配置 → 不报错、无 process 目录
- F4: cv2 缺失降级 → 不抛异常、一次 warning、log/observer 正常
- F5: pipeline（video 启用）→ process 目录创建 + log 含 video_enabled=True
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agents.core import AgentResult
from config import (
    AgentConfig,
    AppConfig,
    EnvConfig,
    ExperimentConfig,
    ExploreConfig,
    ImageStoreConfig,
    LLMConfig,
    RobotConfig,
    TaskConfig,
    VLAConfig,
)
from experiment.recorder import set_recorder
from experiment.video.config import VideoConfig
from pipeline import run_pipeline

cv2 = pytest.importorskip("cv2")  # F1 需要真实 opencv


@pytest.fixture(autouse=True)
def _reset_global_recorder():
    set_recorder(None)
    yield
    set_recorder(None)


# ============================================================================
# F1: 真实 PyBulletEnv 链路 → mp4 产出
# ============================================================================


def test_f1_real_env_produces_mp4(tmp_path: Path):
    """真实 env reset + step 后 process/demo.mp4 存在且非空、可被 cv2 打开。"""
    from config.loader import EnvConfig, RobotConfig
    from env.pybullet_env import PyBulletEnv
    from experiment.recorder import ExperimentRecorder

    root = tmp_path / "exp"
    recorder = ExperimentRecorder(
        root=root,
        enabled=True,
        log_to_stdout=False,
        video=VideoConfig(resolution=(320, 240), fps=10),
    )
    set_recorder(recorder)
    exp_dir = recorder.start()

    env = PyBulletEnv(
        env_config=EnvConfig(mode="direct", renderer="cpu"),
        robot_config=RobotConfig(type="panda"),
    )
    env.reset(task_spec={"objects": [{"type": "cube", "pos": [0.5, 0, 0.1], "color": "red"}]})

    # 执行 2 步动作（每步 10 物理子步 → 每子步抓帧）
    from env.base import Action7D

    for _ in range(2):
        env.step(Action7D(dx=0.01, dy=0.0, dz=0.0, drx=0.0, dry=0.0, drz=0.0, gripper=1.0))
    env.close()

    recorder.finish(success=True, summary="f1")

    mp4 = exp_dir / "process" / "demo.mp4"
    assert mp4.exists(), f"mp4 未生成: {mp4}"
    assert mp4.stat().st_size > 0, "mp4 文件为空"

    # 可被 cv2 打开且帧数 >= 3（1 初始帧 + 至少 2 子步帧）
    cap = cv2.VideoCapture(str(mp4))
    assert cap.isOpened(), "cv2 无法打开生成的 mp4"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count >= 3, f"帧数异常: {frame_count}"


# ============================================================================
# 辅助：pipeline 级 FakeEnv + mock agent
# ============================================================================


class _FakeEnvWithRGB:
    """FakeEnv：env.get_obs() 返回含 RGB ndarray 的 obs（复用现有 e2e 模式）。"""

    def __init__(self, env_config=None, robot_config=None):
        self.env_config = env_config
        self.robot_config = robot_config
        self.use_gui = env_config.use_gui if env_config else False
        self.camera_resolution = env_config.camera_resolution if env_config else (64, 48)

    def reset(self, task_spec=None, seed=0):
        return None

    def step(self, action):
        import numpy as np
        return {"rgb": {"cam1": np.zeros((32, 32, 3), dtype=np.uint8)}}, 0.0, False, {}

    def render(self):
        import numpy as np
        return {"cam1": np.zeros((32, 32, 3), dtype=np.uint8)}

    def get_obs(self, include_rgb: bool = True):
        import numpy as np
        rgb = {"cam1": np.zeros((32, 32, 3), dtype=np.uint8)} if include_rgb else None
        return {
            "rgb": rgb,
            "object_info": [{"name": "red_block", "pos": [0.5, 0.0, 0.1]}],
            "ee_pos": (0.0, 0.0, 0.5),
        }

    def close(self):
        pass

    @property
    def input_spec(self):
        from env.base import ActionSpec
        return ActionSpec("task", ("dx", "dy", "dz", "drx", "dry", "drz", "gripper"))


def _patch_pipeline(monkeypatch, llm_response="任务完成"):
    """monkeypatch pipeline 依赖（FakeEnv + FakeListLLM + mock agent）。"""
    from langchain_core.language_models.fake import FakeListLLM

    monkeypatch.setattr("pipeline.runner.PyBulletEnv", _FakeEnvWithRGB)
    fake_llm = FakeListLLM(responses=[llm_response])
    monkeypatch.setattr("pipeline.runner.create_llm", lambda cfg: fake_llm)

    agent_result = AgentResult(
        success=True,
        trajectory=[],
        final_answer="任务完成",
        total_tool_calls=0,
    )
    monkeypatch.setattr(
        "pipeline.runner.create_exact_agent",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        "pipeline.runner.run_agent",
        MagicMock(return_value=agent_result),
    )


def _make_config(root: str, video: VideoConfig | None = None) -> AppConfig:
    """构造 AppConfig：experiment.video 可选。"""
    return AppConfig(
        env=EnvConfig(use_gui=False, camera_resolution=(64, 48)),
        vla=VLAConfig(backend="mock"),
        llm=LLMConfig(api_key="sk-dummy"),
        explore=ExploreConfig(),
        robot=RobotConfig(),
        task=TaskConfig(),
        agent=AgentConfig(max_react_rounds=5, max_tool_calls=3),
        experiment=ExperimentConfig(
            enabled=True, root=root, log_to_stdout=False, video=video
        ),
        image_store=ImageStoreConfig(),
    )


def _find_exp_dir(root: Path) -> Path:
    """返回唯一的实验目录。"""
    subdirs = sorted(d for d in root.iterdir() if d.is_dir())
    assert len(subdirs) == 1
    return subdirs[0]


# ============================================================================
# F2: video.enabled=False 不产出
# ============================================================================


def test_f2_video_disabled_no_process_dir(tmp_path: Path, monkeypatch):
    """video.enabled=False 时无 process 目录 + log 含 video_enabled=False。"""
    _patch_pipeline(monkeypatch)
    config = _make_config(
        root=str(tmp_path / "exp"),
        video=VideoConfig(enabled=False),
    )

    run_pipeline(config, user_goal="f2 disabled")

    exp_dir = _find_exp_dir(tmp_path / "exp")
    assert not (exp_dir / "process").exists(), "禁用视频不应创建 process/"

    content = (exp_dir / "experiment.log").read_text(encoding="utf-8")
    assert "video_enabled=False" in content


# ============================================================================
# F3: 无 video 段旧配置兼容
# ============================================================================


def test_f3_no_video_config_compatible(tmp_path: Path, monkeypatch):
    """无 video 段（video=None）时 pipeline 不报错、无 process 目录。"""
    _patch_pipeline(monkeypatch)
    config = _make_config(root=str(tmp_path / "exp"), video=None)

    run_pipeline(config, user_goal="f3 legacy")

    exp_dir = _find_exp_dir(tmp_path / "exp")
    assert not (exp_dir / "process").exists()
    content = (exp_dir / "experiment.log").read_text(encoding="utf-8")
    assert "video_enabled=False" in content


# ============================================================================
# F4: cv2 缺失降级
# ============================================================================


def test_f4_cv2_missing_degrades(tmp_path: Path, monkeypatch, capsys):
    """cv2 import 失败时：不抛异常、一次 warning、log/observer 正常。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("No module named 'cv2'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    _patch_pipeline(monkeypatch)

    config = _make_config(
        root=str(tmp_path / "exp"),
        video=VideoConfig(enabled=True),
    )

    # pipeline 不抛异常
    run_pipeline(config, user_goal="f4 cv2 missing")

    exp_dir = _find_exp_dir(tmp_path / "exp")
    # process/ 目录被创建（设计：目录创建与 writer 打开解耦），但无 mp4 文件
    process_dir = exp_dir / "process"
    assert process_dir.is_dir()
    assert not (process_dir / "demo.mp4").exists(), "cv2 缺失时不应产出 mp4"

    # log / observer 正常
    content = (exp_dir / "experiment.log").read_text(encoding="utf-8")
    assert "Pipeline started" in content
    assert "Pipeline finished" in content
    assert "video_enabled=True" in content
    assert (exp_dir / "observer").is_dir()


# ============================================================================
# F5: pipeline video 启用 → process 目录 + video_enabled=True
# ============================================================================


def test_f5_pipeline_video_enabled_creates_process_dir(tmp_path: Path, monkeypatch):
    """pipeline video 启用：process/ 目录创建 + log 含 video_enabled=True。"""
    _patch_pipeline(monkeypatch)
    config = _make_config(
        root=str(tmp_path / "exp"),
        video=VideoConfig(enabled=True),
    )

    run_pipeline(config, user_goal="f5 enabled")

    exp_dir = _find_exp_dir(tmp_path / "exp")
    assert (exp_dir / "process").is_dir()

    content = (exp_dir / "experiment.log").read_text(encoding="utf-8")
    assert "video_enabled=True" in content
