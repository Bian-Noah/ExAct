"""pipeline 集成 ExperimentRecorder 测试。

覆盖（按 tasks.md 5.2 任务列表）：
- pipeline 启动时构造 recorder（PipelineResult 仍纯净）
- experiment.log 含 Pipeline started / finished
- pipeline 异常时 finish 仍被调用 + set_recorder 重置
- enabled=False 不创建目录
- PipelineResult 字段不修改
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake import FakeListLLM

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
from experiment.recorder import (
    _SafeRecorder,
    get_recorder,
    set_recorder,
)
from pipeline import PipelineResult, run_pipeline


@pytest.fixture(autouse=True)
def _reset_global_recorder():
    """每个测试前后重置全局 recorder。"""
    set_recorder(None)
    yield
    set_recorder(None)


def _make_config(
    experiment_enabled: bool = True,
    experiment_root: str = "data/experiment",
    vla_backend: str = "mock",
) -> AppConfig:
    return AppConfig(
        env=EnvConfig(use_gui=False, camera_resolution=(64, 48)),
        vla=VLAConfig(backend=vla_backend, max_steps=5),
        llm=LLMConfig(api_key="sk-dummy"),
        explore=ExploreConfig(),
        robot=RobotConfig(),
        task=TaskConfig(),
        agent=AgentConfig(max_react_rounds=5, max_tool_calls=3),
        experiment=ExperimentConfig(
            enabled=experiment_enabled,
            root=experiment_root,
            log_to_stdout=False,
        ),
        image_store=ImageStoreConfig(),
    )


class _FakeEnv:
    """FakeEnv：duck typing 替代 PyBulletPandaEnv（与 test_runner.py 一致）。"""

    def __init__(self, env_config=None, robot_config=None):
        self.env_config = env_config
        self.robot_config = robot_config
        self.use_gui = env_config.use_gui if env_config else False
        self.camera_resolution = env_config.camera_resolution if env_config else (64, 48)
        self.reset_called_with = None
        self.close_called = False
        self._obs = {
            "rgb": None,
            "object_info": [
                {"id": 1, "name": "cube", "pos": [0.5, 0, 0.1], "quat": [0, 0, 0, 1]},
            ],
            "ee_pos": (0.0, 0.0, 0.5),
            "state_desc": "fake",
        }

    def reset(self, task_spec=None, seed=0):
        self.reset_called_with = task_spec
        return self._obs

    def step(self, action):
        return self._obs, 0.0, False, {}

    def render(self):
        return None

    def get_obs(self, include_rgb: bool = True):
        return self._obs

    def close(self):
        self.close_called = True


def _patch_pipeline(monkeypatch, llm_responses=None, agent_result=None, boom=False):
    """monkeypatch pipeline.runner 内的依赖。"""
    monkeypatch.setattr("pipeline.runner.PyBulletPandaEnv", _FakeEnv)
    if llm_responses is None:
        llm_responses = ["任务已完成。"]
    fake_llm = FakeListLLM(responses=llm_responses)
    monkeypatch.setattr("pipeline.runner.create_llm", lambda cfg: fake_llm)

    if boom:
        # 让 agent 抛错
        def boom_create_agent(*a, **kw):
            m = MagicMock()
            raise RuntimeError("模拟 pipeline 异常")
        monkeypatch.setattr("pipeline.runner.create_exact_agent", boom_create_agent)
    else:
        if agent_result is None:
            agent_result = AgentResult(
                success=True,
                trajectory=[],
                final_answer="任务完成",
                total_tool_calls=0,
            )
        mock_agent = MagicMock()
        mock_run_agent = MagicMock(return_value=agent_result)
        monkeypatch.setattr("pipeline.runner.create_exact_agent", lambda *a, **kw: mock_agent)
        # monkeypatch run_agent（agents.agent.run_agent），pipeline.runner 直接调用它
        monkeypatch.setattr("pipeline.runner.run_agent", mock_run_agent)


# ============================================================================
# enabled=True 时落盘
# ============================================================================


def test_pipeline_creates_experiment_dir(tmp_path: Path, monkeypatch):
    """pipeline 启动时构造 recorder，data/experiment/{timestamp}/ 被创建。"""
    _patch_pipeline(monkeypatch)
    config = _make_config(
        experiment_enabled=True,
        experiment_root=str(tmp_path / "exp"),
    )

    result = run_pipeline(config, user_goal="goal A")

    exp_root = tmp_path / "exp"
    assert exp_root.is_dir()
    subdirs = [d for d in exp_root.iterdir() if d.is_dir()]
    assert len(subdirs) == 1

    exp_dir = subdirs[0]
    assert (exp_dir / "experiment.log").exists()
    assert (exp_dir / "observer").is_dir()

    assert isinstance(result, PipelineResult)
    assert hasattr(result, "agent_result")
    assert hasattr(result, "env_closed")


def test_pipeline_experiment_log_contains_pipeline_started(tmp_path: Path, monkeypatch):
    """experiment.log 应含 `Pipeline started` 行。"""
    _patch_pipeline(monkeypatch)
    config = _make_config(
        experiment_enabled=True,
        experiment_root=str(tmp_path / "exp"),
    )

    run_pipeline(config, user_goal="test goal")

    exp_root = tmp_path / "exp"
    exp_dir = next(exp_root.iterdir())
    content = (exp_dir / "experiment.log").read_text(encoding="utf-8")
    assert "Pipeline started" in content
    assert "user_goal='test goal'" in content


def test_pipeline_experiment_log_contains_pipeline_finished(tmp_path: Path, monkeypatch):
    """experiment.log 应含 `Pipeline finished, success=..., summary=...` 行。"""
    _patch_pipeline(monkeypatch)
    config = _make_config(
        experiment_enabled=True,
        experiment_root=str(tmp_path / "exp"),
    )

    run_pipeline(config, user_goal="test goal")

    exp_root = tmp_path / "exp"
    exp_dir = next(exp_root.iterdir())
    content = (exp_dir / "experiment.log").read_text(encoding="utf-8")
    assert "Pipeline finished" in content
    assert "success=" in content
    assert "summary=" in content


# ============================================================================
# enabled=False 不创建目录
# ============================================================================


def test_pipeline_enabled_false_no_dir_created(tmp_path: Path, monkeypatch):
    """experiment.enabled=False 时 pipeline 不创建任何目录。"""
    _patch_pipeline(monkeypatch)
    config = _make_config(
        experiment_enabled=False,
        experiment_root=str(tmp_path / "exp"),
    )

    result = run_pipeline(config, user_goal="test goal")

    # start() 返回 root 但 enabled=False 时不调用 mkdir → exp_root 不应存在
    exp_root = tmp_path / "exp"
    assert not exp_root.exists()

    assert isinstance(result, PipelineResult)


# ============================================================================
# pipeline 异常时 finish 仍被调用 + set_recorder 重置
# ============================================================================


def test_pipeline_exception_finish_still_called(tmp_path: Path, monkeypatch):
    """pipeline 异常时 recorder.finish 仍被调用（success=False）。"""
    _patch_pipeline(monkeypatch, boom=True)

    config = _make_config(
        experiment_enabled=True,
        experiment_root=str(tmp_path / "exp"),
    )

    with pytest.raises(RuntimeError, match="模拟 pipeline 异常"):
        run_pipeline(config, user_goal="test goal")

    exp_root = tmp_path / "exp"
    exp_dir = next(exp_root.iterdir())
    content = (exp_dir / "experiment.log").read_text(encoding="utf-8")
    assert "Pipeline finished" in content
    assert "success=False" in content
    assert "Pipeline 异常" in content


def test_pipeline_resets_global_recorder_after_run(tmp_path: Path, monkeypatch):
    """pipeline 结束后 get_recorder() 应 fallback 到 _SafeRecorder。"""
    _patch_pipeline(monkeypatch)
    config = _make_config(
        experiment_enabled=True,
        experiment_root=str(tmp_path / "exp"),
    )

    run_pipeline(config, user_goal="test goal")

    assert isinstance(get_recorder(), _SafeRecorder)


def test_pipeline_exception_resets_global_recorder(tmp_path: Path, monkeypatch):
    """pipeline 异常结束后仍应重置全局 recorder。"""
    _patch_pipeline(monkeypatch, boom=True)

    config = _make_config(
        experiment_enabled=True,
        experiment_root=str(tmp_path / "exp"),
    )

    with pytest.raises(RuntimeError):
        run_pipeline(config, user_goal="test goal")

    assert isinstance(get_recorder(), _SafeRecorder)


# ============================================================================
# PipelineResult 字段不修改
# ============================================================================


def test_pipeline_result_fields_unchanged(tmp_path: Path, monkeypatch):
    """PipelineResult 字段保持纯净（仅 agent_result + env_closed）。"""
    _patch_pipeline(monkeypatch)
    config = _make_config(
        experiment_enabled=True,
        experiment_root=str(tmp_path / "exp"),
    )

    result = run_pipeline(config, user_goal="test goal")

    field_names = {f.name for f in result.__dataclass_fields__.values()}
    assert field_names == {"agent_result", "env_closed"}


# ============================================================================
# experiment.root 相对路径解析
# ============================================================================


def test_pipeline_experiment_root_relative_path(tmp_path: Path, monkeypatch):
    """experiment.root 相对路径应相对 CWD 解析。"""
    _patch_pipeline(monkeypatch)
    # 切到 tmp_path，让相对路径 "relative_exp" 解析到 tmp_path 下
    monkeypatch.chdir(tmp_path)
    config = _make_config(
        experiment_enabled=True,
        experiment_root="relative_exp",
    )

    run_pipeline(config, user_goal="test goal")

    exp_root = tmp_path / "relative_exp"
    assert exp_root.is_dir()
    assert any(exp_root.iterdir())