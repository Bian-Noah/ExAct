"""joint_to_joint_transform 单元测试（robot-vla-adapter 任务 3）。"""

import numpy as np
import pytest

from env.base import ActionSpec
from utils.adapter.adapters.joint_to_joint import joint_to_joint_transform


class _FakeEnv:
    """仅实现 input_spec 的假 env。"""

    def __init__(self, input_spec):
        self.input_spec = input_spec


def _joint_env(dim=6, with_gripper=False):
    if with_gripper:
        components = ("joint",) * (dim - 1) + ("gripper",)
    else:
        components = ("joint",) * dim
    return _FakeEnv(ActionSpec("joint", components))


def test_j2j_equal_dim_passthrough():
    """输入维度 == 目标维度 → 原样返回。"""
    env = _joint_env(dim=6)
    out = joint_to_joint_transform(np.array([1.0, 2, 3, 4, 5, 6]), env)
    np.testing.assert_array_equal(out, np.array([1.0, 2, 3, 4, 5, 6]))


def test_j2j_truncate_extra():
    """输入维度 > 目标维度 → 截取前 N 维。"""
    env = _joint_env(dim=6)
    out = joint_to_joint_transform(np.array([1.0, 2, 3, 4, 5, 6, 7, 8]), env)
    np.testing.assert_array_equal(out, np.array([1.0, 2, 3, 4, 5, 6]))


def test_j2j_under_dim_raises():
    """输入维度远小于目标 → ValueError（含实际/期望维度）。"""
    env = _joint_env(dim=6)
    with pytest.raises(ValueError) as exc_info:
        joint_to_joint_transform(np.array([1.0, 2]), env)
    msg = str(exc_info.value)
    assert "2" in msg and "6" in msg


def test_j2j_pad_gripper_when_missing():
    """目标 7 维含 gripper，输入 6 维无 gripper → 补默认占位。"""
    env = _joint_env(dim=7, with_gripper=True)
    out = joint_to_joint_transform(np.array([1.0, 2, 3, 4, 5, 6]), env)
    assert out.shape == (7,)
    np.testing.assert_array_equal(out[:6], np.array([1.0, 2, 3, 4, 5, 6]))
    assert out[6] == pytest.approx(0.5)  # _DEFAULT_GRIPPER


def test_j2j_under_dim_with_gripper_still_raises():
    """目标 7 维含 gripper，输入仅 4 维（缺 2+ 维）→ ValueError。"""
    env = _joint_env(dim=7, with_gripper=True)
    with pytest.raises(ValueError):
        joint_to_joint_transform(np.array([1.0, 2, 3, 4]), env)


def test_j2j_accepts_plain_list():
    """输入为普通 list → 也能处理。"""
    env = _joint_env(dim=6)
    out = joint_to_joint_transform([1.0, 2, 3, 4, 5, 6], env)
    np.testing.assert_array_equal(out, np.array([1.0, 2, 3, 4, 5, 6]))
