"""ActionTool 单元测试（LangChain BaseTool 版）。

ActionTool 已从手写 BaseTool（tools.action）迁移到 LangChain BaseTool（tools.action）。
本文件使用 LangChain 版的 ActionTool，验证：
- 空指令 / 正常指令 / 含坐标指令（target_pos 传递）
- ActionInput args_schema
- name / description 属性

Executor / MockVLA 与 FakeEnv 与第二步相同。
"""

from __future__ import annotations

import pytest

from env.base import ActionSpec, BaseEnv
from executor import Executor, MockVLA
from tools.action import ActionInput, ActionTool


# ========== FakeEnv ==========


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

    @property
    def input_spec(self):
        return ActionSpec("task", ("dx", "dy", "dz", "drx", "dry", "drz", "gripper"))

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

def test_action_tool_empty_instruction():
    """空指令返回错误提示，不调用 executor。"""
    env, executor = _make_setup()
    tool = ActionTool(env=env, executor=executor)
    result = tool._run(instruction="")

    assert "不能为空" in result


def test_action_tool_whitespace_instruction():
    """仅空白指令也视为空。"""
    env, executor = _make_setup()
    tool = ActionTool(env=env, executor=executor)
    result = tool._run(instruction="   ")

    assert "不能为空" in result


# ---------- 2. 正常指令 ----------

def test_action_tool_normal_instruction():
    """正常指令返回 ExecResult.message 字符串。"""
    env, executor = _make_setup(max_steps=5, satisfy_at_step=3)
    tool = ActionTool(env=env, executor=executor)
    result = tool._run(instruction="move forward")

    assert isinstance(result, str)
    assert len(result) > 0


def test_action_tool_calls_executor_once():
    """正常指令调用 executor.run_action 恰好 1 次。"""
    env, executor = _make_setup(max_steps=5, satisfy_at_step=10)
    original = executor.run_action
    captured = {}

    def capturing_run_action(env_arg, instr, done_criteria, **kw):
        captured["count"] = captured.get("count", 0) + 1
        captured["target_pos"] = kw.get("target_pos")
        return original(env_arg, instr, done_criteria, **kw)

    executor.run_action = capturing_run_action

    tool = ActionTool(env=env, executor=executor)
    tool._run(instruction="move forward")

    assert captured["count"] == 1
    assert captured["target_pos"] is None


# ---------- 3. 含坐标指令 ----------

def test_action_tool_with_coords_extracts_target_pos():
    """含坐标指令应解析 target_pos 传给 executor。"""
    env, executor = _make_setup(max_steps=5, satisfy_at_step=10)
    original = executor.run_action
    captured = {}

    def capturing_run_action(env_arg, instr, done_criteria, **kw):
        captured["target_pos"] = kw.get("target_pos")
        return original(env_arg, instr, done_criteria, **kw)

    executor.run_action = capturing_run_action

    tool = ActionTool(env=env, executor=executor)
    tool._run(instruction="move to x=0.5, y=0.0, z=0.4")

    assert captured["target_pos"] == (0.5, 0.0, 0.4)


# ---------- 4. 属性验证 ----------

def test_action_tool_name():
    env, executor = _make_setup()
    tool = ActionTool(env=env, executor=executor)
    assert tool.name == "action"


def test_action_tool_description_contains_action_keyword():
    env, executor = _make_setup()
    tool = ActionTool(env=env, executor=executor)
    # 描述现在用英文，应含 "action"
    assert "action" in tool.description.lower()


def test_action_tool_args_schema_is_actioninput():
    env, executor = _make_setup()
    tool = ActionTool(env=env, executor=executor)
    assert tool.args_schema == ActionInput


# ---------- 5. 继承关系 ----------

def test_action_tool_inherits_langchain_basetool():
    from langchain_core.tools import BaseTool
    env, executor = _make_setup()
    tool = ActionTool(env=env, executor=executor)
    assert isinstance(tool, BaseTool)


# ---------- 6. parse_target_pos 解析（来自 tools.action） ----------

def test_parse_target_pos_various_formats():
    from tools.action import parse_target_pos
    assert parse_target_pos("x=0.5, y=0.0, z=0.4") == (0.5, 0.0, 0.4)
    assert parse_target_pos("(0.5, 0, 0.4)") == (0.5, 0.0, 0.4)
    assert parse_target_pos("x:0.5, y:0, z:0.4") == (0.5, 0.0, 0.4)
    assert parse_target_pos("move to nowhere") is None
    assert parse_target_pos("") is None
    assert parse_target_pos(None) is None


# ---------- 7. iter9：smolVLA 指令校验拦截 ----------


def test_action_tool_rejects_non_verb_prefix():
    """非动词开头的英文指令返回拒绝消息，executor 不被调用。"""
    env, executor = _make_setup()
    captured = {}

    original = executor.run_action

    def capturing(env_arg, instr, done_criteria, **kw):
        captured["called"] = True
        return original(env_arg, instr, done_criteria, **kw)

    executor.run_action = capturing
    tool = ActionTool(env=env, executor=executor)
    result = tool._run(instruction="red cube")

    assert "指令不符合 smolVLA 规范" in result
    assert "不是动作动词开头" in result
    assert "called" not in captured


def test_action_tool_rejects_chinese_instruction():
    """iter9-extend：中文指令一律拒绝。"""
    env, executor = _make_setup()
    captured = {}

    original = executor.run_action

    def capturing(env_arg, instr, done_criteria, **kw):
        captured["called"] = True
        return original(env_arg, instr, done_criteria, **kw)

    executor.run_action = capturing
    tool = ActionTool(env=env, executor=executor)
    result = tool._run(instruction="夹取红色方块")

    assert "指令不符合 smolVLA 规范" in result
    assert "包含非英文字符" in result
    assert "called" not in captured


def test_action_tool_rejects_multi_sentence():
    """多句指令返回拒绝消息。"""
    env, executor = _make_setup()
    tool = ActionTool(env=env, executor=executor)
    result = tool._run(instruction="pick red cube. then drop.")

    assert "指令不符合 smolVLA 规范" in result
    assert "包含多个句子" in result


def test_action_tool_rejects_over_30_chars():
    """超过 30 字符的指令返回拒绝消息。"""
    env, executor = _make_setup()
    tool = ActionTool(env=env, executor=executor)
    long_instr = "place " + "a" * 30  # 36 字符，英文动词开头
    result = tool._run(instruction=long_instr)

    assert "指令不符合 smolVLA 规范" in result
    assert "超过 30 字符上限" in result


def test_action_tool_accepts_valid_verb():
    """合规英文指令仍正常执行。"""
    env, executor = _make_setup(max_steps=5, satisfy_at_step=3)
    captured = {}

    original = executor.run_action

    def capturing(env_arg, instr, done_criteria, **kw):
        captured["called"] = True
        return original(env_arg, instr, done_criteria, **kw)

    executor.run_action = capturing
    tool = ActionTool(env=env, executor=executor)
    result = tool._run(instruction="pick red cube")

    assert "called" in captured
    assert "指令不符合" not in result


def test_action_tool_reject_message_format():
    """拒绝消息格式严格匹配。"""
    env, executor = _make_setup()
    tool = ActionTool(env=env, executor=executor)
    result = tool._run(instruction="fly")  # 英文非动词

    assert result.startswith("指令不符合 smolVLA 规范：")


def test_action_tool_no_parse_target_pos_on_reject():
    """拒绝时 executor.run_action 完全未被调用。"""
    env, executor = _make_setup()
    call_count = {"n": 0}

    original = executor.run_action

    def counting(env_arg, instr, done_criteria, **kw):
        call_count["n"] += 1
        return original(env_arg, instr, done_criteria, **kw)

    executor.run_action = counting
    tool = ActionTool(env=env, executor=executor)
    tool._run(instruction="red cube")  # 非动词开头

    assert call_count["n"] == 0