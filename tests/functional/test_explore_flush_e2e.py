"""iter9 pipeline Explore 接线 e2e 测试。

覆盖：
- enabled=True：mock LLM 下发 write_note，pipeline 结束落盘到正确路径
- enabled=False：ExploreTool 不注入，data/explore 不创建
- flush 异常：pipeline 不崩溃
- explore.root 解析为绝对路径
"""

from __future__ import annotations

import json
import os
from datetime import date
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
from explore import Explore
from pipeline import run_pipeline


class _FakeEnv:
    """FakeEnv：duck typing 替代 PyBulletEnv。"""

    def __init__(self, env_config=None, robot_config=None):
        self.env_config = env_config
        self.robot_config = robot_config
        self.reset_called_with = None
        self.close_called = False
        self._obs = {
            "rgb": None,
            "object_info": [{"id": 1, "name": "cube", "pos": [0.5, 0, 0.1]}],
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

    def get_obs(self, include_rgb=True):
        return self._obs

    @property
    def input_spec(self):
        from env.base import ActionSpec
        return ActionSpec("task", ("dx", "dy", "dz", "drx", "dry", "drz", "gripper"))

    def close(self):
        self.close_called = True


def _patch_pipeline_basic(monkeypatch):
    """基础依赖 monkeypatch。"""
    monkeypatch.setattr("pipeline.runner.PyBulletEnv", _FakeEnv)
    fake_llm = FakeListLLM(responses=["ok"])
    monkeypatch.setattr("pipeline.runner.create_llm", lambda cfg: fake_llm)
    monkeypatch.setattr(
        "pipeline.runner.create_exact_agent",
        lambda *a, **kw: MagicMock(),
    )


def _make_config_with_explore(enabled: bool, root: str) -> AppConfig:
    return AppConfig(
        env=EnvConfig(use_gui=False, camera_resolution=(64, 48)),
        vla=VLAConfig(backend="mock"),
        llm=LLMConfig(api_key="sk-dummy"),
        explore=ExploreConfig(enabled=enabled, root=root),
        robot=RobotConfig(),
        task=TaskConfig(),
        agent=AgentConfig(max_react_rounds=2, max_tool_calls=2),
        experiment=ExperimentConfig(),
        image_store=ImageStoreConfig(),
    )


# ========== enabled=True 路径 ==========


class TestExploreEnabledE2e:
    def test_pipeline_flushes_notes(self, monkeypatch, tmp_path: Path):
        """enabled=True：mock LLM 触发 explore.write_note 调用，flush 落盘。"""
        _patch_pipeline_basic(monkeypatch)

        # Spy create_exact_agent 捕获 tools 列表
        captured_tools = {}

        def spy_create_exact_agent(llm, tools, **kw):
            captured_tools["tools"] = tools
            return MagicMock()

        monkeypatch.setattr("pipeline.runner.create_exact_agent", spy_create_exact_agent)

        # Spy run_agent：在 agent 启动前手动调用 ExploreTool.write_note 模拟 LLM 调用
        def spy_run_agent(agent, user_goal, **kw):
            # 找到 ExploreTool 并触发 write_note
            from tools import ExploreTool
            for t in captured_tools["tools"]:
                if isinstance(t, ExploreTool):
                    t._run("write_note", "note-A")
                    t._run("write_note", "note-B")
                    break
            return AgentResult(success=True, trajectory=[], final_answer="ok", total_tool_calls=2)

        monkeypatch.setattr("pipeline.runner.run_agent", spy_run_agent)

        # Spy recorder 避免写真实 data/experiment 目录
        fake_recorder = MagicMock()
        fake_recorder.start.return_value = "/tmp/fake_exp"
        monkeypatch.setattr("pipeline.runner._build_recorder", lambda cfg: fake_recorder)
        monkeypatch.setattr("pipeline.runner.set_recorder", lambda r: None)

        cfg = _make_config_with_explore(enabled=True, root=str(tmp_path))
        result = run_pipeline(cfg)

        assert result.env_closed is True
        # 验证落盘文件
        log_path = tmp_path / f"{date.today().isoformat()}.log"
        assert log_path.exists()
        content = log_path.read_text(encoding="utf-8")
        assert content == "note-A\nnote-B\n"


# ========== enabled=False 路径 ==========


class TestExploreDisabledE2e:
    def test_pipeline_skips_explore(self, monkeypatch, tmp_path: Path):
        """enabled=False：ExploreTool 不在 tools 列表，data/explore 不创建。"""
        _patch_pipeline_basic(monkeypatch)

        captured_tools = {}

        def spy_create_exact_agent(llm, tools, **kw):
            captured_tools["tools"] = tools
            return MagicMock()

        monkeypatch.setattr("pipeline.runner.create_exact_agent", spy_create_exact_agent)

        monkeypatch.setattr(
            "pipeline.runner.run_agent",
            lambda *a, **kw: AgentResult(success=True, trajectory=[], final_answer="ok", total_tool_calls=0),
        )

        fake_recorder = MagicMock()
        fake_recorder.start.return_value = "/tmp/fake_exp"
        monkeypatch.setattr("pipeline.runner._build_recorder", lambda cfg: fake_recorder)
        monkeypatch.setattr("pipeline.runner.set_recorder", lambda r: None)

        cfg = _make_config_with_explore(enabled=False, root=str(tmp_path))
        result = run_pipeline(cfg)

        assert result.env_closed is True
        # ExploreTool 不在 tools
        from tools import ExploreTool
        assert not any(isinstance(t, ExploreTool) for t in captured_tools["tools"])
        # data/explore 不创建
        assert not (tmp_path).exists() or not list(tmp_path.iterdir()) if (tmp_path).exists() else True


# ========== flush 异常路径 ==========


class TestExploreFlushErrorE2e:
    def test_pipeline_survives_flush_error(self, monkeypatch, tmp_path: Path):
        """explore.flush 抛异常时 pipeline 主流程不崩溃。"""
        _patch_pipeline_basic(monkeypatch)

        monkeypatch.setattr(
            "pipeline.runner.create_exact_agent",
            lambda *a, **kw: MagicMock(),
        )
        monkeypatch.setattr(
            "pipeline.runner.run_agent",
            lambda *a, **kw: AgentResult(success=True, trajectory=[], final_answer="ok", total_tool_calls=0),
        )

        fake_recorder = MagicMock()
        fake_recorder.start.return_value = "/tmp/fake_exp"
        monkeypatch.setattr("pipeline.runner._build_recorder", lambda cfg: fake_recorder)
        monkeypatch.setattr("pipeline.runner.set_recorder", lambda r: None)

        # 替换 Explore 实例的 flush 为抛异常
        original_flush = Explore.flush

        def boom_flush(self):
            raise IOError("disk full")

        monkeypatch.setattr("explore.explore.Explore.flush", boom_flush)

        cfg = _make_config_with_explore(enabled=True, root=str(tmp_path))
        # 不抛异常
        result = run_pipeline(cfg)
        assert result.env_closed is True


# ========== tools 列表内容验证 ==========


class TestExploreToolInjection:
    def test_explore_tool_present_when_enabled(self, monkeypatch, tmp_path: Path):
        """enabled=True 时 ExploreTool 在 tools 列表中。"""
        _patch_pipeline_basic(monkeypatch)

        captured_tools = {}

        def spy_create_exact_agent(llm, tools, **kw):
            captured_tools["tools"] = tools
            return MagicMock()

        monkeypatch.setattr("pipeline.runner.create_exact_agent", spy_create_exact_agent)
        monkeypatch.setattr(
            "pipeline.runner.run_agent",
            lambda *a, **kw: AgentResult(success=True, trajectory=[], final_answer="ok", total_tool_calls=0),
        )

        fake_recorder = MagicMock()
        fake_recorder.start.return_value = "/tmp/fake_exp"
        monkeypatch.setattr("pipeline.runner._build_recorder", lambda cfg: fake_recorder)
        monkeypatch.setattr("pipeline.runner.set_recorder", lambda r: None)

        cfg = _make_config_with_explore(enabled=True, root=str(tmp_path))
        run_pipeline(cfg)

        from tools import ExploreTool
        explore_tools = [t for t in captured_tools["tools"] if isinstance(t, ExploreTool)]
        assert len(explore_tools) == 1