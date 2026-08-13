"""ObserveTool include_rgb 参数单元测试（iter2-renderer-env-mode）。

覆盖：
- ObserveTool._run() 调用 env.get_obs(include_rgb=True)
- target 过滤参数下仍调用 include_rgb=True
- 返回字符串格式不变（含末端位置、物体列表）
"""

from __future__ import annotations

import numpy as np
from unittest.mock import MagicMock

from tools.observe import ObserveTool


def _make_env_mock(include_rgb: bool = True) -> MagicMock:
    """构造 env mock，使 get_obs 返回含 ee_pos/object_info/state_desc 的 obs。"""
    env = MagicMock()
    env.get_obs.return_value = {
        "ee_pos": (0.5, 0.0, 0.4),
        "object_info": [
            {"id": 1, "name": "cube", "pos": [0.5, 0.0, 0.1], "quat": [0, 0, 0, 1]},
        ],
        "state_desc": "场景中1个物体，末端在(0.50,0.00,0.40)",
        "rgb": np.zeros((480, 640, 3), dtype=np.uint8) if include_rgb else None,
    }
    return env


# ============================================================
# include_rgb 参数传递
# ============================================================

def test_observe_calls_get_obs_with_rgb_true():
    """默认调用 → env.get_obs(include_rgb=True)。"""
    env = _make_env_mock()
    tool = ObserveTool(env=env)
    tool._run()
    env.get_obs.assert_called_once_with(include_rgb=True)


def test_observe_with_target_still_calls_rgb_true():
    """target 过滤参数下仍调用 include_rgb=True。"""
    env = _make_env_mock()
    tool = ObserveTool(env=env)
    tool._run(target="red")
    env.get_obs.assert_called_once_with(include_rgb=True)


# ============================================================
# 返回字符串格式
# ============================================================

def test_observe_return_text_contains_ee_pos():
    """返回文本包含末端执行器位置。"""
    env = _make_env_mock()
    tool = ObserveTool(env=env)
    text = tool._run()
    assert "末端执行器位置" in text
    assert "0.500" in text or "0.50" in text


def test_observe_return_text_contains_object_list():
    """返回文本包含物体列表。"""
    env = _make_env_mock()
    tool = ObserveTool(env=env)
    text = tool._run()
    assert "场景物体列表" in text
    assert "cube" in text


def test_observe_target_filter_works():
    """target 过滤后只返回匹配物体。"""
    env = MagicMock()
    env.get_obs.return_value = {
        "ee_pos": (0.5, 0.0, 0.4),
        "object_info": [
            {"id": 1, "name": "red_cube", "pos": [0.5, 0.0, 0.1], "quat": [0, 0, 0, 1]},
            {"id": 2, "name": "blue_cube", "pos": [-0.5, 0.0, 0.1], "quat": [0, 0, 0, 1]},
        ],
        "state_desc": "...",
        "rgb": None,
    }
    tool = ObserveTool(env=env)
    text = tool._run(target="red")
    assert "red_cube" in text
    assert "blue_cube" not in text
