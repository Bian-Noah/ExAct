"""LCObserveTool 单元测试。

使用 FakeEnv（duck typing）验证 target 过滤、大小写不敏感、包含匹配、
缺失字段容错等场景。逻辑与原 ObserveTool 测试一致。
"""

from __future__ import annotations

import pytest

from tools.lc_observe import LCObserveTool, ObserveInput


class FakeEnv:
    """duck typing 环境，可配置 get_obs 返回值。"""

    def __init__(self, obs: dict):
        self._obs = obs

    def get_obs(self, include_rgb: bool = True) -> dict:
        return self._obs


def _make_env_default():
    return FakeEnv({
        "object_info": [
            {"name": "red_block", "position": (0.1, 0.2, 0.05)},
            {"name": "blue_box", "position": (0.3, 0, 0.03)},
        ],
        "ee_pos": (0.0, 0.0, 0.5),
    })


# ---------- 1. target=None 全量 ----------

def test_target_none_returns_all_objects():
    tool = LCObserveTool(env=_make_env_default())
    result = tool._run(target=None)

    assert "red_block" in result
    assert "blue_box" in result
    assert "0.000, 0.000, 0.500" in result
    assert "场景物体列表:" in result


# ---------- 2. target 精确过滤 ----------

def test_target_exact_filter():
    tool = LCObserveTool(env=_make_env_default())
    result = tool._run(target="red_block")

    assert "red_block" in result
    assert "blue_box" not in result


# ---------- 3. 大小写不敏感 ----------

def test_target_case_insensitive():
    tool = LCObserveTool(env=_make_env_default())
    result = tool._run(target="RED_BLOCK")

    assert "red_block" in result
    assert "blue_box" not in result


# ---------- 4. pos 字段兼容（真实 env 契约） ----------

def test_pos_field_compat():
    """真实 PyBullet env 返回 pos 而非 position。"""
    env = FakeEnv({
        "object_info": [
            {"name": "cube", "pos": [0.5, 0.0, 0.1]},
        ],
        "ee_pos": (0.088, -0.000, 0.821),
    })
    tool = LCObserveTool(env=env)
    result = tool._run()

    assert "cube" in result
    assert "0.500" in result or "0.5" in result


# ---------- 5. 属性验证 ----------

def test_name_property():
    tool = LCObserveTool(env=_make_env_default())
    assert tool.name == "observe"


def test_description_property():
    tool = LCObserveTool(env=_make_env_default())
    assert "观察" in tool.description


def test_args_schema_property():
    tool = LCObserveTool(env=_make_env_default())
    assert tool.args_schema == ObserveInput
