"""LCActionTool 单元测试。

复用第二步 FakeEnvWithEePosControl + MockVLA + Executor 真实组合，
验证空指令、正常指令、target_pos 传递等场景。
"""

from __future__ import annotations

import pytest

from env.base import BaseEnv
from executor import Executor, ExecResult, MockVLA
from tools.lc_action import LCActionTool, ActionInput


# ========== FakeEnv（复用第二步实现） ==========


class FakeEnvWithEePosControl(BaseEnv):
    """可控制 ee_pos 随步数推进的 FakeEnv。"""

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
    tool = LCActionTool(env=env, executor=executor)
    result = tool._run(instruction="")

    assert "不能为空" in result


# ---------- 2. 正常指令（无坐标）target_pos=None ----------

def test_normal_instruction_no_coords():
    env, executor = _make_setup(max_steps=5, satisfy_at_step=3)
    tool = LCActionTool(env=env, executor=executor)
    result = tool._run(instruction="往前走")

    assert isinstance(result, str)
    assert len(result) > 0


def test_normal_instruction_target_pos_none():
    """无坐标指令时 target_pos=None 传给 executor。"""
    env, executor = _make_setup(max_steps=5, satisfy_at_step=10)
    original_run_action = executor.run_action
    captured = {}

    def capturing_run_action(env_arg, instr, done_criteria, **kw):
        captured["target_pos"] = kw.get("target_pos")
        return original_run_action(env_arg, instr, done_criteria, **kw)

    executor.run_action = capturing_run_action

    tool = LCActionTool(env=env, executor=executor)
    tool._run(instruction="往前走")

    assert captured["target_pos"] is None


# ---------- 3. 含坐标指令 target_pos 正确传递 ----------

def test_instruction_with_coords_passes_target_pos():
    """含坐标指令时 target_pos 正确传递给 executor。"""
    env, executor = _make_setup(max_steps=5, satisfy_at_step=10)
    original_run_action = executor.run_action
    captured = {}

    def capturing_run_action(env_arg, instr, done_criteria, **kw):
        captured["target_pos"] = kw.get("target_pos")
        return original_run_action(env_arg, instr, done_criteria, **kw)

    executor.run_action = capturing_run_action

    tool = LCActionTool(env=env, executor=executor)
    tool._run(instruction="移动到 x=0.5, y=0.0, z=0.4")

    assert captured["target_pos"] == (0.5, 0.0, 0.4)


# ---------- 4. 属性验证 ----------

def test_name_property():
    env, executor = _make_setup()
    tool = LCActionTool(env=env, executor=executor)
    assert tool.name == "action"


def test_description_property():
    env, executor = _make_setup()
    tool = LCActionTool(env=env, executor=executor)
    assert "动作" in tool.description


def test_args_schema_property():
    env, executor = _make_setup()
    tool = LCActionTool(env=env, executor=executor)
    assert tool.args_schema == ActionInput
