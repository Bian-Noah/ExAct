"""iter9 ActionTool 校验拦截 e2e 测试（English-only 版）。

覆盖：LLM 下发不合规英文指令时 ActionTool 拒绝执行，VLA 不被调用。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from env.base import ActionSpec, BaseEnv
from executor import Executor, MockVLA
from tools.action import ActionTool


class FakeEnv(BaseEnv):
    def __init__(self):
        self._ee_pos = (0.0, 0.0, 0.0)

    def reset(self, task_spec=None, seed=0):
        return self._obs()

    def step(self, action):
        return self._obs(), 0.0, False, {}

    def render(self):
        return None

    def get_obs(self, include_rgb=True):
        return self._obs()

    def close(self):
        pass

    @property
    def input_spec(self):
        return ActionSpec("task", ("dx", "dy", "dz", "drx", "dry", "drz", "gripper"))

    def _obs(self):
        import numpy as np
        return {
            "rgb": np.zeros((10, 10, 3), dtype="uint8"),
            "object_info": [{"name": "cube", "pos": [0.5, 0, 0.1]}],
            "ee_pos": self._ee_pos,
            "state_desc": "fake",
        }


def _make_setup():
    env = FakeEnv()
    env.reset()
    vla = MockVLA(seed=0)
    executor = Executor(vla, max_steps=5)
    return env, executor, vla


def test_invalid_instruction_returns_rejection(monkeypatch):
    """不合规英文指令被拒绝，executor 不被调用。"""
    env, executor, vla = _make_setup()

    captured = {"calls": 0}
    original = executor.run_action

    def spy(env_arg, instr, **kw):
        captured["calls"] += 1
        return original(env_arg, instr, **kw)

    executor.run_action = spy

    tool = ActionTool(env=env, executor=executor)
    result = tool._run(instruction="red cube")  # 名词开头，不以动词开头

    assert "指令不符合 smolVLA 规范" in result
    assert "不是动作动词开头" in result
    assert captured["calls"] == 0


def test_chinese_instruction_rejected(monkeypatch):
    """iter9-extend：中文指令一律拒绝。"""
    env, executor, vla = _make_setup()
    tool = ActionTool(env=env, executor=executor)
    result = tool._run(instruction="夹取红色方块")

    assert "包含非英文字符" in result


def test_vla_not_called_on_rejection(monkeypatch):
    """拒绝路径不触发 VLA.predict。"""
    env, executor, vla = _make_setup()

    predict_calls = {"n": 0}
    original_predict = vla.predict

    def spy_predict(*a, **kw):
        predict_calls["n"] += 1
        return original_predict(*a, **kw)

    vla.predict = spy_predict

    tool = ActionTool(env=env, executor=executor)
    tool._run(instruction="red cube")

    assert predict_calls["n"] == 0


def test_valid_instruction_passes_through(monkeypatch):
    """合规英文指令仍正常执行。"""
    env, executor, vla = _make_setup()

    captured = {"calls": 0}
    original = executor.run_action

    def spy(env_arg, instr, **kw):
        captured["calls"] += 1
        return original(env_arg, instr, **kw)

    executor.run_action = spy

    tool = ActionTool(env=env, executor=executor)
    result = tool._run(instruction="pick red cube")

    assert "指令不符合" not in result
    assert captured["calls"] == 1


def test_multi_sentence_rejected(monkeypatch):
    """多句英文指令被拒绝。"""
    env, executor, vla = _make_setup()
    tool = ActionTool(env=env, executor=executor)
    result = tool._run(instruction="pick red cube. then drop.")

    assert "包含多个句子" in result


def test_over_30_chars_rejected(monkeypatch):
    """超长英文指令被拒绝。"""
    env, executor, vla = _make_setup()
    tool = ActionTool(env=env, executor=executor)
    long_instr = "place " + "a" * 30  # 36 chars
    result = tool._run(instruction=long_instr)

    assert "超过 30 字符上限" in result