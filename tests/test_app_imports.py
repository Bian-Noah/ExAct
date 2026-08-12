"""app.py 导入验证测试。

验证 src/app.py 中所有 import 语句都能成功导入，
不运行 main()（避免 PyBullet GUI 启动）。
"""

from __future__ import annotations

import importlib


def test_app_module_importable():
    """验证 src.app 模块可导入。"""
    mod = importlib.import_module("app")
    assert hasattr(mod, "main")
    assert callable(mod.main)


def test_app_dependencies_importable():
    """验证 app.py 依赖的所有子模块可导入（LangChain 版）。"""
    # config
    from config import load_config
    assert callable(load_config)

    # env.pybullet_env
    from env.pybullet_env import PyBulletPandaEnv
    assert PyBulletPandaEnv.__name__ == "PyBulletPandaEnv"

    # executor
    from executor import Executor, MockVLA
    assert Executor.__name__ == "Executor"
    assert MockVLA.__name__ == "MockVLA"

    # agents —— 新增 LangChain 工厂与适配层
    from agents import (
        create_llm,
        create_exact_agent,
        run_agent,
        AgentResult,
    )
    assert callable(create_llm)
    assert callable(create_exact_agent)
    assert callable(run_agent)
    assert AgentResult.__name__ == "AgentResult"

    # tools —— LangChain BaseTool 子类（name 是 pydantic 字段，用 model_fields 取默认值）
    from tools import LCObserveTool, LCActionTool
    assert LCObserveTool.model_fields["name"].default == "observe"
    assert LCActionTool.model_fields["name"].default == "action"

    # utils.logging
    from utils.logging import setup_logging
    assert callable(setup_logging)


def test_lc_agent_recursion_limit_is_small():
    """验证 MAX_REACT_ROUNDS 已按"几步就行"设为 5 轮。"""
    from agents.lc_agent import MAX_REACT_ROUNDS
    assert MAX_REACT_ROUNDS == 5
    # recursion_limit = MAX_REACT_ROUNDS * 2 + 1 = 11
    assert MAX_REACT_ROUNDS * 2 + 1 == 11
