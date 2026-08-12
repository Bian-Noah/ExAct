"""executor/build_input.py 单元测试。"""

import numpy as np
import pytest

from executor.build_input import build_vla_input


def _make_rgb(h=10, w=10):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_build_vla_input_normal_full_obs():
    """完整 obs + instruction → 返回 dict 含 5 个键，类型正确。"""
    obs = {
        "rgb": _make_rgb(),
        "object_info": [{"name": "red_block"}],
        "ee_pos": (0.1, 0.2, 0.3),
        "state_desc": "ee near origin",
    }
    result = build_vla_input(obs, "move")
    assert set(result.keys()) == {
        "image", "instruction", "ee_pos", "objects", "state_desc"
    }
    assert isinstance(result["image"], np.ndarray)
    assert isinstance(result["ee_pos"], tuple)
    assert result["instruction"] == "move"
    assert result["objects"] == [{"name": "red_block"}]
    assert result["state_desc"] == "ee near origin"


def test_build_vla_input_minimal_obs():
    """obs 只含 rgb + ee_pos → 不报错，可选字段使用默认值。"""
    obs = {"rgb": _make_rgb(), "ee_pos": (0.1, 0.2, 0.3)}
    result = build_vla_input(obs, "move")
    assert result["objects"] == []
    assert result["state_desc"] == ""


def test_build_vla_input_empty_instruction():
    """instruction='' → 不报错，instruction 字段为 ''。"""
    obs = {"rgb": _make_rgb(), "ee_pos": (0.1, 0.2, 0.3)}
    result = build_vla_input(obs, "")
    assert result["instruction"] == ""


def test_build_vla_input_missing_rgb():
    """obs 缺 rgb → ValueError。"""
    obs = {"ee_pos": (0, 0, 0)}
    with pytest.raises(ValueError, match="rgb"):
        build_vla_input(obs, "move")


def test_build_vla_input_missing_ee_pos():
    """obs 缺 ee_pos → ValueError。"""
    obs = {"rgb": _make_rgb()}
    with pytest.raises(ValueError, match="ee_pos"):
        build_vla_input(obs, "move")


def test_build_vla_input_rgb_not_ndarray():
    """rgb 传 list → ValueError。"""
    obs = {"rgb": [1, 2, 3], "ee_pos": (0, 0, 0)}
    with pytest.raises(ValueError, match="np.ndarray"):
        build_vla_input(obs, "move")


def test_build_vla_input_rgb_wrong_dtype():
    """rgb 传 float32 ndarray → ValueError。"""
    obs = {
        "rgb": np.ones((5, 5, 3), dtype=np.float32),
        "ee_pos": (0, 0, 0),
    }
    with pytest.raises(ValueError, match="uint8"):
        build_vla_input(obs, "move")


def test_build_vla_input_instruction_not_str():
    """instruction 传 int → TypeError。"""
    obs = {"rgb": _make_rgb(), "ee_pos": (0, 0, 0)}
    with pytest.raises(TypeError, match="instruction"):
        build_vla_input(obs, 123)


def test_build_vla_input_instruction_none():
    """instruction 传 None → TypeError。"""
    obs = {"rgb": _make_rgb(), "ee_pos": (0, 0, 0)}
    with pytest.raises(TypeError, match="instruction"):
        build_vla_input(obs, None)


def test_build_vla_input_extra_fields_passthrough():
    """obs 含未知字段 → 不报错（宽松校验）。"""
    obs = {
        "rgb": _make_rgb(),
        "ee_pos": (0, 0, 0),
        "unknown_field": "abc",
    }
    # 不报错即通过
    result = build_vla_input(obs, "move")
    assert result["instruction"] == "move"


def test_build_vla_input_ee_pos_list_to_tuple():
    """ee_pos 传 list → 返回 dict 中 ee_pos 是 tuple。"""
    obs = {"rgb": _make_rgb(), "ee_pos": [0.1, 0.2, 0.3]}
    result = build_vla_input(obs, "move")
    assert isinstance(result["ee_pos"], tuple)
    assert result["ee_pos"] == (0.1, 0.2, 0.3)
