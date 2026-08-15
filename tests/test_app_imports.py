"""app.py 导入验证测试（iter1-pipeline-refactor-config 后）。

T7（app.py 瘦身）尚未执行，本测试仅验证各模块的 import 路径可用。
"""

from __future__ import annotations

import importlib


def test_app_module_importable():
    """验证 src.app 模块可导入（仅解析，不执行 main）。"""
    mod = importlib.import_module("app")
    assert hasattr(mod, "main")
    assert callable(mod.main)


def test_app_dependencies_importable():
    """验证 app.py 依赖的所有子模块可导入（去 LC 前缀版）。"""
    from config import load_config
    assert callable(load_config)

    from env.pybullet_env import PyBulletEnv
    assert PyBulletEnv.__name__ == "PyBulletEnv"

    from executor import Executor, MockVLA
    assert Executor.__name__ == "Executor"
    assert MockVLA.__name__ == "MockVLA"

    from agents import (
        AgentResult,
        create_exact_agent,
        create_llm,
        run_agent,
    )
    assert callable(create_llm)
    assert callable(create_exact_agent)
    assert callable(run_agent)
    assert AgentResult.__name__ == "AgentResult"

    # 去 LC 前缀后从 tools 直接拿
    from tools import ActionTool, ObserveTool
    assert ActionTool.model_fields["name"].default == "action"
    assert ObserveTool.model_fields["name"].default == "observe"

    from utils.logging import setup_logging
    assert callable(setup_logging)


def test_dead_imports_fail():
    """LC 前缀版本与 MiniMaxClient / ExActAgent 应全部抛 ImportError。"""
    import pytest

    for path in (
        "tools.lc_action",
        "tools.lc_observe",
        "tools.base",
        "agents.llm_client",
        "agents.core.ExActAgent",
        "agents.llm_client.MiniMaxClient",
    ):
        with pytest.raises(ImportError):
            importlib.import_module(path)