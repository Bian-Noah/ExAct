"""ActionTool 单元测试。

复用第二步 FakeEnvWithEePosControl + MockVLA + Executor 真实组合，
验证空指令、空白指令、正常指令等场景。
"""

from __future__ import annotations

import pytest

from env.base import BaseEnv
from executor import Executor, ExecResult, MockVLA
from tools.action import ActionTool


# ========== FakeEnv（复用第二步实现） ==========


class FakeEnvWithEePosControl(BaseEnv):
    """可控制 ee_pos 随步数推进的 FakeEnv。

    前 satisfy_at_step 步 ee_pos 位移 > 阈值（reached 不满足），
    其后 ee_pos 不再变化（reached 满足）。
    """

    def __init__(self, satisfy_at_step=3, move_step=0.05):
        self._step_count = 0
        self._satisfy_at_step = satisfy_at_step
        self._move_step = move_step
        self._ee_pos = (0.0, 0.0, 0.0)

    def reset(self, task_spec=None, seed=0):
        self._step_count = 0
        self._ee_pos = (0.0, 0.0, 0.0)
        return self._make_obs()

    def step(self, action):
        self._step_count += 1
        if self._step_count < self._satisfy_at_step:
            self._ee_pos = (
                self._ee_pos[0] + self._move_step,
                self._ee_pos[1],
                self._ee_pos[2],
            )
        return self._make_obs(), 0.0, False, {}

    def render(self):
        import numpy as np
        return np.zeros((10, 10, 3), dtype=np.uint8)

    def get_obs(self, include_rgb: bool = True):
        return self._make_obs()

    def close(self):
        pass

    def _make_obs(self):
        import numpy as np
        return {
            "rgb": np.zeros((10, 10, 3), dtype=np.uint8),
            "object_info": [{"name": "cube", "pos": [0.5, 0, 0.1]}],
            "ee_pos": self._ee_pos,
            "state_desc": f"step={self._step_count}",
        }


def _make_setup(max_steps=5, satisfy_at_step=3):
    env = FakeEnvWithEePosControl(satisfy_at_step=satisfy_at_step)
    env.reset()
    vla = MockVLA(seed=0)
    executor = Executor(vla, max_steps=max_steps)
    return env, executor


# ---------- 1. 空指令 ----------

def test_empty_instruction():
    env, executor = _make_setup()
    tool = ActionTool(env, executor)
    result = tool.run(instruction="")

    assert "不能为空" in result


def test_whitespace_only_instruction():
    env, executor = _make_setup()
    tool = ActionTool(env, executor)
    result = tool.run(instruction="   ")

    assert "不能为空" in result


# ---------- 2. 正常指令 ----------

def test_normal_instruction_returns_execresult_message():
    env, executor = _make_setup(max_steps=5, satisfy_at_step=3)
    tool = ActionTool(env, executor)
    result = tool.run(instruction="往前走")

    # 应返回 ExecResult.message 字符串
    assert isinstance(result, str)
    assert len(result) > 0
    # satisfy_at_step=3 < max_steps=5，应在第 3 步达到 reached
    # check_done reached 逻辑：ee_pos 位移小于阈值时触发 reason
    assert "ee_pos" in result or "位移" in result or "到达" in result or "成功" in result


def test_normal_instruction_calls_executor_once():
    """验证 executor.run_action 恰好被调用 1 次。"""
    env, executor = _make_setup(max_steps=5, satisfy_at_step=10)
    # 用 mock 包一层 executor.run_action 以计数
    original_run_action = executor.run_action
    call_count = {"n": 0}

    def counting_run_action(env_arg, instr, done_criteria, **kw):
        call_count["n"] += 1
        return original_run_action(env_arg, instr, done_criteria, **kw)

    executor.run_action = counting_run_action

    tool = ActionTool(env, executor)
    tool.run(instruction="往前走")

    assert call_count["n"] == 1


def test_instruction_timeout_message():
    """satisfy_at_step 大于 max_steps 时应返回超时 message。"""
    env, executor = _make_setup(max_steps=2, satisfy_at_step=10)
    tool = ActionTool(env, executor)
    result = tool.run(instruction="走很远")

    # Executor 超时 message 格式："达到最大步数 N，未完成：..."
    assert "最大步数" in result or "未完成" in result or "ee_pos" in result


# ---------- 3. run 异常捕获（BaseTool 兜底） ----------

def test_run_catches_executor_exception():
    """executor.run_action 抛异常时，ActionTool.run 应返回 ToolError 字符串。"""

    class BoomExecutor:
        def run_action(self, *a, **kw):
            raise RuntimeError("executor boom")

    env, _ = _make_setup()
    tool = ActionTool(env, BoomExecutor())
    result = tool.run(instruction="走")

    assert "ToolError[action]" in result
    assert "executor boom" in result
