"""Executor.run_action operation 透传 + max_steps 删除 单元测试（iter11-reset-multicam）。

验证 operation="reset" 走 env 复位路径不走 VLA,operation="vla" 走原路径,
Executor 构造不再含 max_steps 参数。
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import numpy as np
import pytest

from executor import Executor
from executor.model.mock.mock_vla import MockVLA
from env.base import ActionSpec


@pytest.fixture
def mock_env():
    """MagicMock 模拟 BaseEnv,含 reset_arm_to_home/get_obs/get_joint_state。"""
    env = MagicMock()
    env.reset_arm_to_home = MagicMock()
    env.get_obs = MagicMock(return_value={
        "ee_pos": (0.1, 0.2, 0.3),
        "object_info": [],
        "state_desc": "test",
        "rgb": {"cam1": np.zeros((480, 640, 3), dtype=np.uint8)},
    })
    env.get_joint_state = MagicMock(return_value=np.zeros(6))
    env.step = MagicMock(return_value=(
        {"ee_pos": (0.1, 0.2, 0.3), "object_info": [], "state_desc": "", "rgb": None},
        0.0, False, {}
    ))
    env.input_spec = ActionSpec("task", ("dx", "dy", "dz", "drx", "dry", "drz", "gripper"))
    return env


@pytest.fixture
def mock_vla():
    """MockVLA 实例,verify 调用次数用。"""
    vla = MockVLA(seed=0)
    vla.predict = MagicMock(wraps=vla.predict)
    return vla


def test_executor_init_no_max_steps():
    """Executor(vla) 构造无警告,inspect.signature 不含 max_steps。"""
    sig = inspect.signature(Executor.__init__)
    assert "max_steps" not in sig.parameters


def test_run_action_reset_calls_env_reset(mock_vla, mock_env):
    """operation="reset" 调一次 env.reset_arm_to_home()。"""
    executor = Executor(mock_vla)
    executor.run_action(mock_env, instruction="", done_criteria="reached", operation="reset")
    mock_env.reset_arm_to_home.assert_called_once()


def test_run_action_reset_skips_vla_predict(mock_vla, mock_env):
    """operation="reset" 不调 vla.predict(verify 调用 0 次)。"""
    executor = Executor(mock_vla)
    executor.run_action(mock_env, instruction="", done_criteria="reached", operation="reset")
    mock_vla.predict.assert_not_called()


def test_run_action_reset_returns_zero_steps(mock_vla, mock_env):
    """operation="reset" 返回 ExecResult.steps == 0。"""
    executor = Executor(mock_vla)
    result = executor.run_action(
        mock_env, instruction="", done_criteria="reached", operation="reset"
    )
    assert result.steps == 0
    assert result.success is None  # reset 路径不判定语义成功


def test_run_action_vla_default(mock_vla, mock_env):
    """不传 operation 参数,默认 "vla",走 vla.predict。"""
    executor = Executor(mock_vla)
    executor.run_action(mock_env, instruction="test", done_criteria="reached")
    mock_vla.predict.assert_called_once()
    mock_env.reset_arm_to_home.assert_not_called()


def test_run_action_vla_explicit(mock_vla, mock_env):
    """operation="vla" 显式传,走 vla.predict。"""
    executor = Executor(mock_vla)
    executor.run_action(
        mock_env, instruction="test", done_criteria="reached", operation="vla"
    )
    mock_vla.predict.assert_called_once()
    mock_env.reset_arm_to_home.assert_not_called()


def test_run_action_vla_predict_image_is_dict(mock_vla, mock_env):
    """vla 路径下,vla.predict 收到的 image 是 dict[str, ndarray]。"""
    executor = Executor(mock_vla)
    executor.run_action(mock_env, instruction="test", done_criteria="reached")
    call_args = mock_vla.predict.call_args
    image_arg = call_args[0][0]
    assert isinstance(image_arg, dict)
    assert "cam1" in image_arg


def test_run_action_reset_message_mentions_home(mock_vla, mock_env):
    """reset 路径返回 message 含 "home" 字样。"""
    executor = Executor(mock_vla)
    result = executor.run_action(
        mock_env, instruction="", done_criteria="reached", operation="reset"
    )
    assert "home" in result.message.lower() or "home" in result.message