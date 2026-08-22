"""ActionTool operation 字段 单元测试（iter11-reset-multicam）。

验证 ActionInput.operation 字段、_run 分发逻辑、reset 路径不走 validate_instruction。
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from executor.model.mock.mock_vla import MockVLA
from tools.action import ActionInput, ActionTool


@pytest.fixture
def mock_action_tool_env():
    """构造带 env + executor + adapter 的 ActionTool。"""
    env = MagicMock()
    env.reset_arm_to_home = MagicMock()
    env.get_obs = MagicMock(return_value={
        "ee_pos": (0.1, 0.2, 0.3),
        "object_info": [],
        "state_desc": "test",
    })

    executor = MagicMock()
    executor.vla = MockVLA(seed=0)
    executor.run_action = MagicMock(return_value=MagicMock(
        success=True, steps=5, final_obs={"ee_pos": (0.5, 0.0, 0.3)},
        message="执行 VLA 规划的 5 步"
    ))

    adapter = lambda raw, env: raw  # 直通

    tool = ActionTool(env=env, executor=executor, adapter=adapter)
    return tool, env, executor


def test_action_input_default_operation_is_vla():
    """ActionInput 默认 operation="vla"。"""
    inp = ActionInput(instruction="pick red cube")
    assert inp.operation == "vla"


def test_action_input_reset_operation_accepted():
    """ActionInput(operation="reset") 可构造。"""
    inp = ActionInput(operation="reset")
    assert inp.operation == "reset"
    assert inp.instruction == ""  # 默认空


def test_action_input_invalid_operation_rejected():
    """ActionInput(operation="invalid") pydantic 校验失败。"""
    with pytest.raises(Exception):  # ValidationError
        ActionInput(operation="invalid")


def test_action_tool_reset_calls_env_reset(mock_action_tool_env):
    """action(operation="reset") 调 env.reset_arm_to_home()。"""
    tool, env, _ = mock_action_tool_env
    result = tool._run(instruction="", operation="reset")
    env.reset_arm_to_home.assert_called_once()
    assert "home" in result.lower()


def test_action_tool_reset_skips_executor(mock_action_tool_env):
    """action(operation="reset") 不调 executor.run_action。"""
    tool, _, executor = mock_action_tool_env
    tool._run(instruction="", operation="reset")
    executor.run_action.assert_not_called()


def test_action_tool_vla_default_calls_executor(mock_action_tool_env):
    """action() 默认 operation="vla",调 executor.run_action。"""
    tool, _, executor = mock_action_tool_env
    tool._run(instruction="pick red cube")
    executor.run_action.assert_called_once()
    call_kwargs = executor.run_action.call_args[1]
    assert call_kwargs["operation"] == "vla"


def test_action_tool_vla_explicit_calls_executor(mock_action_tool_env):
    """action(operation="vla", instruction=...) 调 executor.run_action。"""
    tool, _, executor = mock_action_tool_env
    tool._run(instruction="pick red cube", operation="vla")
    executor.run_action.assert_called_once()


def test_action_tool_reset_skips_validate_instruction(mock_action_tool_env):
    """reset 路径不走 validate_instruction,即使 instruction 不合规也不被拒。"""
    tool, env, executor = mock_action_tool_env
    # 中文/超长,正常 vla 路径会被 validate_instruction 拒
    tool._run(instruction="请拿起红色方块并放到蓝色方块旁边很长很长的指令", operation="reset")
    # 应该走到 reset 路径(env.reset_arm_to_home),不走 executor
    env.reset_arm_to_home.assert_called_once()
    executor.run_action.assert_not_called()


def test_action_tool_reset_returns_ee_pos(mock_action_tool_env):
    """reset 返回字符串包含末端位置信息。"""
    tool, _, _ = mock_action_tool_env
    result = tool._run(instruction="", operation="reset")
    assert "末端位置" in result or "ee_pos" in result or "0.300" in result


def test_action_tool_description_mentions_operation():
    """ActionTool.description 描述 operation 字段用法。"""
    tool = ActionTool(env=MagicMock(), executor=MagicMock(), adapter=lambda r, e: r)
    assert "operation" in tool.description
    assert "reset" in tool.description.lower()