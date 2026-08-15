"""ObserveTool 单元测试（LangChain BaseTool 版）。

ObserveTool 已从手写 BaseTool（tools.observe）迁移到 LangChain BaseTool（tools.observe）。
本文件使用 LangChain 版的 ObserveTool，验证：
- target 过滤 / 大小写不敏感 / 包含匹配 / 找不到提示
- 缺失字段容错 / pos 字段兼容
- name / description / args_schema 属性 + 继承关系

Iteration 5：_run() 返回结构化 list[dict]（LangChain 标准 content blocks），
text 块 + 可选 image 块。本文件不注入 image_store，所以 _run() 始终只返回
1 个 text 块。helper 函数 _text_block() 从 list[dict] 中提取 text 字段。
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


def _text_block(result) -> str:
    """从 _run() 返回的 list[dict] 中提取首条 text 块的 'text' 字段。

    Iteration 5 约定：_run() 返回 list[dict]，list[0] 永远是 text 块。
    本文件不注入 image_store，所以 content 长度恰好为 1，含 1 个 text 块。
    """
    assert isinstance(result, list), f"期望 list[dict]，得到 {type(result).__name__}"
    assert len(result) >= 1, f"期望至少 1 个 block，得到空 list"
    assert result[0]["type"] == "text", f"期望 text 块，得到 {result[0]!r}"
    return result[0]["text"]


# ---------- 1. target=None 全量 ----------

def test_observe_tool_returns_ee_pos():
    """返回字符串含 ee_pos 格式化输出。"""
    tool = ObserveTool(env=_make_env_default())
    result = tool._run()
    text = _text_block(result)

    assert "0.000, 0.000, 0.500" in text
    assert "末端执行器位置" in text


def test_observe_tool_target_none_returns_all_objects():
    tool = ObserveTool(env=_make_env_default())
    result = tool._run(target=None)
    text = _text_block(result)

    assert "red_block" in text
    assert "blue_box" in text
    assert "场景物体列表:" in text


# ---------- 2. 精确过滤 ----------

def test_observe_tool_with_target_filter():
    """target 精确匹配只返回对应物体。"""
    tool = ObserveTool(env=_make_env_default())
    result = tool._run(target="red_block")
    text = _text_block(result)

    assert "red_block" in text
    assert "blue_box" not in text


def test_observe_tool_case_insensitive():
    tool = ObserveTool(env=_make_env_default())
    result = tool._run(target="RED_BLOCK")
    text = _text_block(result)

    assert "red_block" in text
    assert "blue_box" not in text


# ---------- 3. 包含匹配 ----------

def test_observe_tool_substring_match_red():
    tool = ObserveTool(env=_make_env_default())
    result = tool._run(target="red")
    text = _text_block(result)

    assert "red_block" in text
    assert "blue_box" not in text


def test_observe_tool_substring_match_box():
    tool = ObserveTool(env=_make_env_default())
    result = tool._run(target="box")
    text = _text_block(result)

    assert "blue_box" in text
    assert "red_block" not in text


# ---------- 4. 找不到 ----------

def test_observe_tool_no_match_shows_hint():
    """目标不存在时显示提示。"""
    tool = ObserveTool(env=_make_env_default())
    result = tool._run(target="nonexistent")
    text = _text_block(result)

    assert "未找到目标物体 'nonexistent'" in text


# ---------- 5. 缺失字段容错 ----------

def test_observe_tool_missing_ee_pos():
    env = FakeEnv({
        "object_info": [{"name": "cube", "position": (0.1, 0.2, 0.3)}],
    })
    tool = ObserveTool(env=env)
    result = tool._run()
    text = _text_block(result)

    assert "末端执行器位置: unknown" in text
    assert "cube" in text


def test_observe_tool_object_info_missing_position():
    """物体元素仅含 name 无 position → pos=(None, None, None)。"""
    env = FakeEnv({
        "object_info": [{"name": "cube"}],
        "ee_pos": (0.0, 0.0, 0.5),
    })
    tool = ObserveTool(env=env)
    result = tool._run()
    text = _text_block(result)

    assert "cube" in text
    assert "pos=(None, None, None)" in text


def test_observe_tool_object_info_empty_list():
    env = FakeEnv({
        "object_info": [],
        "ee_pos": (0.0, 0.0, 0.5),
    })
    tool = ObserveTool(env=env)
    result = tool._run()
    text = _text_block(result)

    assert "场景物体列表:" in text


def test_observe_tool_object_info_missing_entirely():
    env = FakeEnv({
        "ee_pos": (0.0, 0.0, 0.5),
    })
    tool = ObserveTool(env=env)
    result = tool._run()
    text = _text_block(result)

    assert "场景物体列表:" in text


# ---------- 6. pos 字段 ----------

def test_observe_tool_pos_field_compat():
    """真实 PyBulletEnv 返回 pos 而非 position。"""
    env = FakeEnv({
        "object_info": [
            {"id": 1, "name": "cube", "pos": [0.5, 0.1, 0.1], "quat": [0, 0, 0, 1]},
        ],
        "ee_pos": (0.1, 0.2, 0.3),
    })
    tool = ObserveTool(env=env)
    result = tool._run()
    text = _text_block(result)

    assert "cube" in text
    assert "0.500" in text or "0.5" in text
    assert "0.100" in text or "0.1" in text


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