"""ExperimentRecorder 端到端功能测试。

覆盖（按 tasks.md 6.2-6.4）：
- 跑一次 pipeline → data/experiment/{timestamp}/ 含 experiment.log + observer/*.png
- enabled=False 时不创建任何目录
- 同一进程多次跑 pipeline → 独立时间戳目录
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake import FakeListLLM
from PIL import Image

from agents.core import AgentResult
from config import (
    AgentConfig,
    AppConfig,
    EnvConfig,
    ExperimentConfig,
    ExploreConfig,
    LLMConfig,
    RobotConfig,
    TaskConfig,
    VLAConfig,
)
from experiment.recorder import get_recorder, set_recorder
from pipeline import run_pipeline


@pytest.fixture(autouse=True)
def _reset_global_recorder():
    set_recorder(None)
    yield
    set_recorder(None)


class _FakeEnvWithRGB:
    """FakeEnv：env.get_obs() 返回含 RGB ndarray 的 obs。"""

    def __init__(self, env_config=None, robot_config=None):
        self.env_config = env_config
        self.robot_config = robot_config
        self.use_gui = env_config.use_gui if env_config else False
        self.camera_resolution = env_config.camera_resolution if env_config else (64, 48)
        self.close_called = False

    def reset(self, task_spec=None, seed=0):
        return None

    def step(self, action):
        import numpy as np
        return {"rgb": np.zeros((32, 32, 3), dtype=np.uint8)}, 0.0, False, {}

    def render(self):
        import numpy as np
        return np.zeros((32, 32, 3), dtype=np.uint8)

    def get_obs(self, include_rgb: bool = True):
        import numpy as np
        rgb = np.zeros((32, 32, 3), dtype=np.uint8) if include_rgb else None
        return {
            "rgb": rgb,
            "object_info": [{"name": "red_block", "pos": [0.5, 0.0, 0.1]}],
            "ee_pos": (0.0, 0.0, 0.5),
        }

    def close(self):
        self.close_called = True


def _patch_pipeline(monkeypatch, llm_response="任务完成"):
    """monkeypatch pipeline 依赖。"""
    monkeypatch.setattr("pipeline.runner.PyBulletPandaEnv", _FakeEnvWithRGB)
    fake_llm = FakeListLLM(responses=[llm_response])
    monkeypatch.setattr("pipeline.runner.create_llm", lambda cfg: fake_llm)

    agent_result = AgentResult(
        success=True,
        trajectory=[],
        final_answer="任务完成",
        total_tool_calls=0,
    )
    mock_agent = MagicMock()
    monkeypatch.setattr("pipeline.runner.create_exact_agent", lambda *a, **kw: mock_agent)
    monkeypatch.setattr("pipeline.runner.run_agent", MagicMock(return_value=agent_result))


def _make_config(enabled: bool, root: str) -> AppConfig:
    return AppConfig(
        env=EnvConfig(use_gui=False, camera_resolution=(64, 48)),
        vla=VLAConfig(backend="mock", max_steps=5),
        llm=LLMConfig(api_key="sk-dummy"),
        explore=ExploreConfig(),
        robot=RobotConfig(),
        task=TaskConfig(),
        agent=AgentConfig(max_react_rounds=5, max_tool_calls=3),
        experiment=ExperimentConfig(
            enabled=enabled, root=root, log_to_stdout=False
        ),
    )


# ============================================================================
# 场景 1：端到端落盘
# ============================================================================


def test_e2e_pipeline_creates_experiment_dir_with_log_and_observer(
    tmp_path: Path, monkeypatch
):
    """跑一次 pipeline → data/experiment/{timestamp}/ 含 experiment.log + observer/。"""
    _patch_pipeline(monkeypatch)
    config = _make_config(enabled=True, root=str(tmp_path / "exp"))

    run_pipeline(config, user_goal="把红色方块移到蓝色区域")

    exp_root = tmp_path / "exp"
    assert exp_root.is_dir()

    subdirs = sorted(d for d in exp_root.iterdir() if d.is_dir())
    assert len(subdirs) == 1

    exp_dir = subdirs[0]
    # 目录名格式 YYYYMMDD_HHMMSS
    assert re.match(r"\d{8}_\d{6}$", exp_dir.name), f"目录名格式错误: {exp_dir.name}"

    # experiment.log 存在
    assert (exp_dir / "experiment.log").exists()

    # observer/ 子目录存在
    observer_dir = exp_dir / "observer"
    assert observer_dir.is_dir()


def test_e2e_pipeline_log_contains_start_and_finish(tmp_path: Path, monkeypatch):
    """experiment.log 含 Pipeline started + Pipeline finished 两行。"""
    _patch_pipeline(monkeypatch)
    config = _make_config(enabled=True, root=str(tmp_path / "exp"))

    run_pipeline(config, user_goal="e2e goal")

    exp_dir = next((tmp_path / "exp").iterdir())
    content = (exp_dir / "experiment.log").read_text(encoding="utf-8")
    assert "Pipeline started" in content
    assert "user_goal='e2e goal'" in content
    assert "Pipeline finished" in content
    assert "success=" in content
    assert "summary=" in content


def test_e2e_pipeline_log_has_timestamp_format(tmp_path: Path, monkeypatch):
    """experiment.log 行格式 `{ts} [{level}] {message}\\n`。"""
    _patch_pipeline(monkeypatch)
    config = _make_config(enabled=True, root=str(tmp_path / "exp"))

    run_pipeline(config, user_goal="format test")

    exp_dir = next((tmp_path / "exp").iterdir())
    content = (exp_dir / "experiment.log").read_text(encoding="utf-8")
    # 至少一行符合时间戳格式
    pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} \[(INFO|ERROR|WARNING)\] .+\n"
    matches = re.findall(pattern, content)
    assert len(matches) >= 2, f"experiment.log 不符合行格式: {content!r}"


# ============================================================================
# 场景 2：enabled=False 不创建目录
# ============================================================================


def test_e2e_pipeline_disabled_creates_no_dir(tmp_path: Path, monkeypatch):
    """experiment.enabled=False 时 pipeline 不创建任何目录。"""
    _patch_pipeline(monkeypatch)
    config = _make_config(enabled=False, root=str(tmp_path / "exp"))

    # 不抛异常
    run_pipeline(config, user_goal="disabled test")

    # exp/ 不应被创建（因为 enabled=False，start() 直接 return root）
    exp_root = tmp_path / "exp"
    assert not exp_root.exists()


def test_e2e_pipeline_disabled_global_recorder_safe(tmp_path: Path, monkeypatch):
    """enabled=False 跑完后 get_recorder() 应 fallback 到 _SafeRecorder。"""
    from experiment.recorder import _SafeRecorder

    _patch_pipeline(monkeypatch)
    config = _make_config(enabled=False, root=str(tmp_path / "exp"))

    run_pipeline(config, user_goal="disabled test")

    # pipeline 结束后全局 recorder 应被重置
    assert isinstance(get_recorder(), _SafeRecorder)


# ============================================================================
# 场景 3：同一进程多次跑 pipeline → 独立目录
# ============================================================================


def test_e2e_pipeline_multiple_runs_create_independent_dirs(tmp_path: Path, monkeypatch):
    """连续两次跑 pipeline → 产生两个独立时间戳目录。"""
    _patch_pipeline(monkeypatch)
    config = _make_config(enabled=True, root=str(tmp_path / "exp"))

    run_pipeline(config, user_goal="first goal")
    # 等待至少 1 秒确保时间戳不同（YYYYMMDD_HHMMSS 精度到秒）
    time.sleep(1.1)
    run_pipeline(config, user_goal="second goal")

    exp_root = tmp_path / "exp"
    subdirs = sorted(d for d in exp_root.iterdir() if d.is_dir())
    assert len(subdirs) == 2

    # 两个目录名都符合时间戳格式 + 不同
    name1, name2 = subdirs[0].name, subdirs[1].name
    assert re.match(r"\d{8}_\d{6}$", name1)
    assert re.match(r"\d{8}_\d{6}$", name2)
    assert name1 != name2


def test_e2e_pipeline_multiple_runs_logs_are_independent(tmp_path: Path, monkeypatch):
    """两个实验的 experiment.log 内容独立（各自含不同的 user_goal）。"""
    _patch_pipeline(monkeypatch)
    config = _make_config(enabled=True, root=str(tmp_path / "exp"))

    run_pipeline(config, user_goal="goal_alpha")
    time.sleep(1.1)
    run_pipeline(config, user_goal="goal_beta")

    exp_root = tmp_path / "exp"
    subdirs = sorted(d for d in exp_root.iterdir() if d.is_dir())

    content1 = (subdirs[0] / "experiment.log").read_text(encoding="utf-8")
    content2 = (subdirs[1] / "experiment.log").read_text(encoding="utf-8")

    assert "user_goal='goal_alpha'" in content1
    assert "user_goal='goal_alpha'" not in content2

    assert "user_goal='goal_beta'" in content2
    assert "user_goal='goal_beta'" not in content1


# ============================================================================
# 场景 4：observe 工具调用产生 PNG（端到端真实链路）
# ============================================================================


def test_e2e_observe_calls_produce_pngs(tmp_path: Path, monkeypatch):
    """完整 pipeline 链路中 observe 工具调用 → observer/*.png。

    注意：本测试需要让 LLM 真的调 observe 工具。LLM mock 用一个能调 observe 的 fake。
    """
    from langchain_core.messages import AIMessage
    import numpy as np

    # 自定义 LLM：第一次调 observe 工具，第二次返回 final_answer
    class _ObserveLLM:
        def __init__(self):
            self.call_count = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                # 让 observe 被调用
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "observe",
                        "args": {},
                        "id": "call_observe_1",
                    }],
                )
            else:
                return AIMessage(content="完成")

    # Patch LLM
    monkeypatch.setattr(
        "pipeline.runner.create_llm",
        lambda cfg: _ObserveLLM(),
    )
    monkeypatch.setattr("pipeline.runner.PyBulletPandaEnv", _FakeEnvWithRGB)

    # Patch create_exact_agent 为返回的 agent 在被 run_agent 调用时执行 LLM.invoke
    # 这里简化：让 agent 直接调 observe 工具一次
    real_observe_tool_factory = None
    from tools import ObserveTool

    def fake_create_agent(llm, tools, **kwargs):
        # 模拟一次 observe 调用
        for tool in tools:
            if isinstance(tool, ObserveTool):
                # 这里直接触发一次 observe（mock LLM 已经决策要 observe）
                # 注意：实际不会执行 LLM 决策流程，仅验证 observe 埋点
                tool._run()
        # 返回 mock agent
        return MagicMock()

    monkeypatch.setattr(
        "pipeline.runner.create_exact_agent",
        fake_create_agent,
    )

    # Patch run_agent 直接返回 AgentResult
    agent_result = AgentResult(
        success=True,
        trajectory=[],
        final_answer="完成",
        total_tool_calls=1,
    )
    monkeypatch.setattr(
        "pipeline.runner.run_agent",
        MagicMock(return_value=agent_result),
    )

    config = _make_config(enabled=True, root=str(tmp_path / "exp"))

    run_pipeline(config, user_goal="e2e observe test")

    exp_dir = next((tmp_path / "exp").iterdir())
    observer_dir = exp_dir / "observer"

    # observer/000.png 应存在
    assert (observer_dir / "000.png").exists()

    # PNG 应能 PIL 读回
    img = np.array(Image.open(observer_dir / "000.png"))
    assert img.shape == (32, 32, 3)

    # experiment.log 应含 [observe] saved observer/000.png
    content = (exp_dir / "experiment.log").read_text(encoding="utf-8")
    assert "[observe] saved observer/000.png" in content


# ============================================================================
# 场景 5：异常路径（pipeline 业务异常时仍正确落盘 finish 日志）
# ============================================================================


def test_e2e_pipeline_exception_finish_log_present(tmp_path: Path, monkeypatch):
    """pipeline 异常时 experiment.log 仍含 Pipeline finished + ERROR 日志。"""
    monkeypatch.setattr("pipeline.runner.PyBulletPandaEnv", _FakeEnvWithRGB)
    monkeypatch.setattr(
        "pipeline.runner.create_llm",
        lambda cfg: FakeListLLM(responses=["x"]),
    )

    def boom_create_agent(*a, **kw):
        raise RuntimeError("e2e boom")

    monkeypatch.setattr("pipeline.runner.create_exact_agent", boom_create_agent)

    config = _make_config(enabled=True, root=str(tmp_path / "exp"))

    with pytest.raises(RuntimeError, match="e2e boom"):
        run_pipeline(config, user_goal="e2e exception test")

    exp_dir = next((tmp_path / "exp").iterdir())
    content = (exp_dir / "experiment.log").read_text(encoding="utf-8")

    # 应有 Pipeline started
    assert "Pipeline started" in content
    # 应有 ERROR 级别异常日志
    assert "[ERROR]" in content
    assert "Pipeline 异常" in content
    # 应有 Pipeline finished with success=False
    assert "Pipeline finished" in content
    assert "success=False" in content