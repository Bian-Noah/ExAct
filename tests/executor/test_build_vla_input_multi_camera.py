"""build_vla_input 适配多相机 dict 单元测试（iter11-reset-multicam）。"""
from __future__ import annotations

import numpy as np
import pytest

from executor.build_input import build_vla_input


def test_build_vla_input_rgb_dict_passes_through():
    """obs["rgb"] = dict 时,vla_input["image"] = 同样 dict(透传)。"""
    rgb = {
        "cam1": np.zeros((480, 640, 3), dtype=np.uint8),
        "cam2": np.ones((480, 640, 3), dtype=np.uint8),
    }
    obs = {
        "ee_pos": (0.1, 0.2, 0.3),
        "object_info": [],
        "state_desc": "test",
        "rgb": rgb,
    }
    result = build_vla_input(obs, "move")
    assert result["image"] is rgb  # 同一个对象,直接透传


def test_build_vla_input_rgb_dict_none():
    """obs["rgb"] = None 时,vla_input["image"] = None。"""
    obs = {
        "ee_pos": (0.1, 0.2, 0.3),
        "object_info": [],
        "state_desc": "",
        "rgb": None,
    }
    result = build_vla_input(obs, "move")
    assert result["image"] is None


def test_build_vla_input_rgb_missing():
    """obs 不含 rgb key 时,vla_input["image"] = None。"""
    obs = {
        "ee_pos": (0.1, 0.2, 0.3),
        "object_info": [],
        "state_desc": "",
    }
    result = build_vla_input(obs, "move")
    assert result["image"] is None


def test_build_vla_input_rgb_ndarray_raises():
    """obs["rgb"] 是 ndarray(iter 10 旧形态)→ 抛 ValueError。"""
    obs = {
        "ee_pos": (0.1, 0.2, 0.3),
        "object_info": [],
        "state_desc": "",
        "rgb": np.zeros((480, 640, 3), dtype=np.uint8),  # 旧 ndarray 形态
    }
    with pytest.raises(ValueError, match="dict.*或 None"):
        build_vla_input(obs, "move")


def test_build_vla_input_ee_pos_required():
    """obs 缺 ee_pos → ValueError。"""
    obs = {"object_info": [], "state_desc": ""}
    with pytest.raises(ValueError, match="ee_pos"):
        build_vla_input(obs, "move")


def test_build_vla_input_rgb_empty_dict():
    """obs["rgb"] = {} (空 dict) → image={}(透传,不校验)。"""
    obs = {
        "ee_pos": (0.1, 0.2, 0.3),
        "object_info": [],
        "state_desc": "",
        "rgb": {},
    }
    result = build_vla_input(obs, "move")
    assert result["image"] == {}