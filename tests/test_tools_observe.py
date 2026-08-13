"""ObserveTool 单元测试（LangChain BaseTool 版）。

ObserveTool 已从手写 BaseTool（tools.observe）迁移到 LangChain BaseTool（tools.observe）。
本文件使用 LangChain 版的 ObserveTool，验证：
- target 过滤 / 大小写不敏感 / 包含匹配 / 找不到提示
- 缺失字段容错 / pos 字段兼容
- name / description / args_schema 属性 + 继承关系
"""

from __future__ import annotations

import pytest

from tools.observe import ObserveInput, ObserveTool


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

def test_observe_tool_returns_ee_pos():
    """返回字符串含 ee_pos 格式化输出。"""
    tool = ObserveTool(env=_make_env_default())
    result = tool._run()

    assert "0.000, 0.000, 0.500" in result
    assert "末端执行器位置" in result


def test_observe_tool_target_none_returns_all_objects():
    tool = ObserveTool(env=_make_env_default())
    result = tool._run(target=None)

    assert "red_block" in result
    assert "blue_box" in result
    assert "场景物体列表:" in result


# ---------- 2. 精确过滤 ----------

def test_observe_tool_with_target_filter():
    """target 精确匹配只返回对应物体。"""
    tool = ObserveTool(env=_make_env_default())
    result = tool._run(target="red_block")

    assert "red_block" in result
    assert "blue_box" not in result


def test_observe_tool_case_insensitive():
    tool = ObserveTool(env=_make_env_default())
    result = tool._run(target="RED_BLOCK")

    assert "red_block" in result
    assert "blue_box" not in result


# ---------- 3. 包含匹配 ----------

def test_observe_tool_substring_match_red():
    tool = ObserveTool(env=_make_env_default())
    result = tool._run(target="red")

    assert "red_block" in result
    assert "blue_box" not in result


def test_observe_tool_substring_match_box():
    tool = ObserveTool(env=_make_env_default())
    result = tool._run(target="box")

    assert "blue_box" in result
    assert "red_block" not in result


# ---------- 4. 找不到 ----------

def test_observe_tool_no_match_shows_hint():
    """目标不存在时显示提示。"""
    tool = ObserveTool(env=_make_env_default())
    result = tool._run(target="nonexistent")

    assert "未找到目标物体 'nonexistent'" in result


# ---------- 5. 缺失字段容错 ----------

def test_observe_tool_missing_ee_pos():
    env = FakeEnv({
        "object_info": [{"name": "cube", "position": (0.1, 0.2, 0.3)}],
    })
    tool = ObserveTool(env=env)
    result = tool._run()

    assert "末端执行器位置: unknown" in result
    assert "cube" in result


def test_observe_tool_object_info_missing_position():
    """物体元素仅含 name 无 position → pos=(None, None, None)。"""
    env = FakeEnv({
        "object_info": [{"name": "cube"}],
        "ee_pos": (0.0, 0.0, 0.5),
    })
    tool = ObserveTool(env=env)
    result = tool._run()

    assert "cube" in result
    assert "pos=(None, None, None)" in result


def test_observe_tool_object_info_empty_list():
    env = FakeEnv({
        "object_info": [],
        "ee_pos": (0.0, 0.0, 0.5),
    })
    tool = ObserveTool(env=env)
    result = tool._run()

    assert "场景物体列表:" in result


def test_observe_tool_object_info_missing_entirely():
    env = FakeEnv({
        "ee_pos": (0.0, 0.0, 0.5),
    })
    tool = ObserveTool(env=env)
    result = tool._run()

    assert "场景物体列表:" in result


# ---------- 6. pos 字段 ----------

def test_observe_tool_pos_field_compat():
    """真实 PyBulletPandaEnv 返回 pos 而非 position。"""
    env = FakeEnv({
        "object_info": [
            {"id": 1, "name": "cube", "pos": [0.5, 0.1, 0.1], "quat": [0, 0, 0, 1]},
        ],
        "ee_pos": (0.1, 0.2, 0.3),
    })
    tool = ObserveTool(env=env)
    result = tool._run()

    assert "cube" in result
    assert "0.500" in result or "0.5" in result
    assert "0.100" in result or "0.1" in result


# ---------- 7. 属性 + 继承 ----------

def test_observe_tool_name():
    tool = ObserveTool(env=_make_env_default())
    assert tool.name == "observe"


def test_observe_tool_description():
    tool = ObserveTool(env=_make_env_default())
    assert "观察" in tool.description


def test_observe_tool_args_schema_is_observeinput():
    tool = ObserveTool(env=_make_env_default())
    assert tool.args_schema == ObserveInput


def test_observe_tool_inherits_langchain_basetool():
    from langchain_core.tools import BaseTool
    tool = ObserveTool(env=_make_env_default())
    assert isinstance(tool, BaseTool)