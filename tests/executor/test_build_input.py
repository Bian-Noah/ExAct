"""executor/build_input.py 单元测试(iter11-reset-multicam 改造)。

iter10 之前 rgb 是 ndarray,iter11 改为 dict[str, ndarray]。
本文件改造后 rgb 统一传 dict 形态,验证 build_vla_input 在新形态下行为正确。
"""

import numpy as np
import pytest

from executor.build_input import build_vla_input


def _make_rgb_dict(h=10, w=10, name="cam1"):
    """构造单相机 rgb dict(iter11 新形态)。"""
    return {name: np.zeros((h, w, 3), dtype=np.uint8)}


def test_build_vla_input_normal_full_obs():
    """完整 obs + instruction → 返回 dict 含 6 个键,image 是 dict。"""
    obs = {
        "rgb": _make_rgb_dict(),
        "object_info": [{"name": "red_block"}],
        "ee_pos": (0.1, 0.2, 0.3),
        "state_desc": "ee near origin",
    }
    result = build_vla_input(obs, "move")
    assert set(result.keys()) == {
        "image", "instruction", "ee_pos", "objects", "state_desc", "state"
    }
    assert isinstance(result["image"], dict)
    assert "cam1" in result["image"]
    assert isinstance(result["image"]["cam1"], np.ndarray)
    assert isinstance(result["ee_pos"], tuple)
    assert result["instruction"] == "move"
    assert result["objects"] == [{"name": "red_block"}]
    assert result["state_desc"] == "ee near origin"


def test_build_vla_input_minimal_obs():
    """obs 只含 rgb + ee_pos → 不报错,可选字段使用默认值。"""
    obs = {"rgb": _make_rgb_dict(), "ee_pos": (0.1, 0.2, 0.3)}
    result = build_vla_input(obs, "move")
    assert result["objects"] == []
    assert result["state_desc"] == ""


def test_build_vla_input_empty_instruction():
    """instruction='' → 不报错,instruction 字段为 ''。"""
    obs = {"rgb": _make_rgb_dict(), "ee_pos": (0.1, 0.2, 0.3)}
    result = build_vla_input(obs, "")
    assert result["instruction"] == ""


def test_build_vla_input_missing_rgb():
    """obs 缺 rgb → 兼容(rgb 可选,仅 ee_pos 必填)。"""
    obs = {"ee_pos": (0, 0, 0)}
    result = build_vla_input(obs, "move")
    assert result["image"] is None
    assert result["instruction"] == "move"


def test_build_vla_input_missing_ee_pos():
    """obs 缺 ee_pos → ValueError。"""
    obs = {"rgb": _make_rgb_dict()}
    with pytest.raises(ValueError, match="ee_pos"):
        build_vla_input(obs, "move")


def test_build_vla_input_rgb_ndarray_legacy_raises():
    """iter10 旧形态 rgb ndarray → ValueError(防御性,iter11 必须传 dict)。"""
    obs = {
        "rgb": np.zeros((10, 10, 3), dtype=np.uint8),
        "ee_pos": (0, 0, 0),
    }
    with pytest.raises(ValueError, match="dict"):
        build_vla_input(obs, "move")


def test_build_vla_instruction_not_str():
    """instruction 传 int → TypeError。"""
    obs = {"rgb": _make_rgb_dict(), "ee_pos": (0, 0, 0)}
    with pytest.raises(TypeError, match="instruction"):
        build_vla_input(obs, 123)


def test_build_vla_input_instruction_none():
    """instruction 传 None → TypeError。"""
    obs = {"rgb": _make_rgb_dict(), "ee_pos": (0, 0, 0)}
    with pytest.raises(TypeError, match="instruction"):
        build_vla_input(obs, None)


def test_build_vla_input_extra_fields_passthrough():
    """obs 含未知字段 → 不报错(宽松校验)。"""
    obs = {
        "rgb": _make_rgb_dict(),
        "ee_pos": (0, 0, 0),
        "unknown_field": "abc",
    }
    result = build_vla_input(obs, "move")
    assert result["instruction"] == "move"


def test_build_vla_input_ee_pos_list_to_tuple():
    """ee_pos 传 list → 返回 dict 中 ee_pos 是 tuple。"""
    obs = {"rgb": _make_rgb_dict(), "ee_pos": [0.1, 0.2, 0.3]}
    result = build_vla_input(obs, "move")
    assert isinstance(result["ee_pos"], tuple)
    assert result["ee_pos"] == (0.1, 0.2, 0.3)


def test_build_vla_input_multi_camera_rgb_dict():
    """rgb 是含多相机的 dict → image 透传整个 dict。"""
    rgb_dict = {
        "cam1": np.zeros((10, 10, 3), dtype=np.uint8),
        "cam2": np.ones((10, 10, 3), dtype=np.uint8),
        "cam3": np.full((10, 10, 3), 100, dtype=np.uint8),
    }
    obs = {"rgb": rgb_dict, "ee_pos": (0, 0, 0)}
    result = build_vla_input(obs, "move")
    assert result["image"] is rgb_dict
    assert set(result["image"].keys()) == {"cam1", "cam2", "cam3"}


def test_build_vla_input_rgb_none_passes_through():
    """rgb 显式传 None → image 是 None。"""
    obs = {"rgb": None, "ee_pos": (0, 0, 0)}
    result = build_vla_input(obs, "move")
    assert result["image"] is None