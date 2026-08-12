"""ObserveTool 单元测试。

使用 FakeEnv（duck typing）验证 target 过滤、大小写不敏感、包含匹配、
缺失字段容错等场景。
"""

from __future__ import annotations

import pytest

from tools.observe import ObserveTool


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
    tool = ObserveTool(_make_env_default())
    result = tool.run(target=None)

    assert "red_block" in result
    assert "blue_box" in result
    # ee_pos 格式化
    assert "0.000, 0.000, 0.500" in result
    # 位置数值
    assert "0.100" in result or "0.1" in result
    assert "0.200" in result or "0.2" in result
    assert "0.300" in result or "0.3" in result
    assert "场景物体列表:" in result


def test_target_none_explicit_no_target_arg():
    tool = ObserveTool(_make_env_default())
    result = tool.run()

    assert "red_block" in result
    assert "blue_box" in result


# ---------- 2. 精确过滤 ----------

def test_target_exact_match_red_block():
    tool = ObserveTool(_make_env_default())
    result = tool.run(target="red_block")

    assert "red_block" in result
    assert "blue_box" not in result


def test_target_case_insensitive():
    tool = ObserveTool(_make_env_default())
    result = tool.run(target="RED_BLOCK")

    assert "red_block" in result
    assert "blue_box" not in result


# ---------- 3. 包含匹配 ----------

def test_target_substring_match_red():
    tool = ObserveTool(_make_env_default())
    result = tool.run(target="red")

    assert "red_block" in result
    assert "blue_box" not in result


def test_target_substring_match_box():
    tool = ObserveTool(_make_env_default())
    result = tool.run(target="box")

    assert "blue_box" in result
    assert "red_block" not in result


# ---------- 4. 找不到 ----------

def test_target_not_found():
    tool = ObserveTool(_make_env_default())
    result = tool.run(target="green_ball")

    assert "未找到目标物体 'green_ball'" in result
    # 不抛异常
    assert isinstance(result, str)


# ---------- 5. 缺失字段容错 ----------

def test_obs_missing_ee_pos():
    env = FakeEnv({
        "object_info": [{"name": "cube", "position": (0.1, 0.2, 0.3)}],
    })
    tool = ObserveTool(env)
    result = tool.run()

    assert "末端执行器位置: unknown" in result
    assert "cube" in result


def test_object_info_missing_position():
    """物体元素仅含 name 无 position → pos=(None, None, None)，不抛 KeyError。"""
    env = FakeEnv({
        "object_info": [{"name": "cube"}],
        "ee_pos": (0.0, 0.0, 0.5),
    })
    tool = ObserveTool(env)
    result = tool.run()

    assert "cube" in result
    assert "pos=(None, None, None)" in result


def test_object_info_empty_list():
    env = FakeEnv({
        "object_info": [],
        "ee_pos": (0.0, 0.0, 0.5),
    })
    tool = ObserveTool(env)
    result = tool.run()

    assert "场景物体列表:" in result
    # 空列表时只有标题行
    assert "未找到" not in result  # target=None 不显示未找到


def test_object_info_missing_entirely():
    env = FakeEnv({
        "ee_pos": (0.0, 0.0, 0.5),
    })
    tool = ObserveTool(env)
    result = tool.run()

    assert "场景物体列表:" in result


# ---------- 6. 兼容 pos 字段（真实 PyBulletPandaEnv 契约） ----------

def test_pos_field_compatibility():
    """真实 env 返回 pos 而非 position，ObserveTool 应兼容。"""
    env = FakeEnv({
        "object_info": [
            {"id": 1, "name": "cube", "pos": [0.5, 0.1, 0.1], "quat": [0, 0, 0, 1]},
        ],
        "ee_pos": (0.1, 0.2, 0.3),
    })
    tool = ObserveTool(env)
    result = tool.run()

    assert "cube" in result
    assert "0.500" in result or "0.5" in result
    assert "0.100" in result or "0.1" in result


def test_pos_field_with_target():
    env = FakeEnv({
        "object_info": [
            {"name": "cube", "pos": [0.5, 0.1, 0.1]},
        ],
        "ee_pos": (0.0, 0.0, 0.5),
    })
    tool = ObserveTool(env)
    result = tool.run(target="cube")
    assert "cube" in result
