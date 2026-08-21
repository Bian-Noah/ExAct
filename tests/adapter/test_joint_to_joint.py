"""joint_to_joint_transform 单元测试（robot-vla-adapter 任务 3 + Iteration 10 chunk）。

Iteration 10 chunk 契约：adapter 输入 shape (N, action_dim)，输出 shape (N, target_dim)。
ndim=1 输入按 (1, -1) reshape 保持 chunk 一致。
"""

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


# ========== ndim=1 单步输入（兼容旧接口）==========


def test_j2j_equal_dim_passthrough():
    """ndim=1 输入维度 == 目标维度 → reshape (1, -1) 透传。"""
    env = _joint_env(dim=6)
    out = joint_to_joint_transform(np.array([1.0, 2, 3, 4, 5, 6]), env)
    assert out.shape == (1, 6)
    np.testing.assert_array_equal(out, np.array([[1.0, 2, 3, 4, 5, 6]]))


def test_j2j_truncate_extra():
    """ndim=1 输入维度 > 目标维度 → 截取前 N 维（第一维保留为 1）。"""
    env = _joint_env(dim=6)
    out = joint_to_joint_transform(np.array([1.0, 2, 3, 4, 5, 6, 7, 8]), env)
    assert out.shape == (1, 6)
    np.testing.assert_array_equal(out, np.array([[1.0, 2, 3, 4, 5, 6]]))


def test_j2j_under_dim_raises():
    """ndim=1 输入维度远小于目标 → ValueError。"""
    env = _joint_env(dim=6)
    with pytest.raises(ValueError) as exc_info:
        joint_to_joint_transform(np.array([1.0, 2]), env)
    msg = str(exc_info.value)
    assert "2" in msg and "6" in msg


def test_j2j_pad_gripper_when_missing():
    """ndim=1 目标 7 维含 gripper，输入 6 维无 gripper → 补默认占位（按 (1, 1) 拼）。"""
    env = _joint_env(dim=7, with_gripper=True)
    out = joint_to_joint_transform(np.array([1.0, 2, 3, 4, 5, 6]), env)
    assert out.shape == (1, 7)
    np.testing.assert_array_equal(out[0, :6], np.array([1.0, 2, 3, 4, 5, 6]))
    assert out[0, 6] == pytest.approx(0.5)  # _DEFAULT_GRIPPER


def test_j2j_under_dim_with_gripper_still_raises():
    """ndim=1 目标 7 维含 gripper，输入仅 4 维（缺 2+ 维）→ ValueError。"""
    env = _joint_env(dim=7, with_gripper=True)
    with pytest.raises(ValueError):
        joint_to_joint_transform(np.array([1.0, 2, 3, 4]), env)


def test_j2j_accepts_plain_list():
    """ndim=1 输入为普通 list → 也能处理。"""
    env = _joint_env(dim=6)
    out = joint_to_joint_transform([1.0, 2, 3, 4, 5, 6], env)
    assert out.shape == (1, 6)
    np.testing.assert_array_equal(out, np.array([[1.0, 2, 3, 4, 5, 6]]))


# ========== ndim=2 chunk 输入（Iteration 10 新契约）==========


def test_j2j_ndim_2_equal_dim_passthrough():
    """ndim=2 chunk 输入维度 == 目标维度 → 整 chunk 透传。"""
    env = _joint_env(dim=6)
    chunk = np.zeros((50, 6), dtype=float)
    out = joint_to_joint_transform(chunk, env)
    assert out.shape == (50, 6)
    np.testing.assert_array_equal(out, chunk)


def test_j2j_ndim_2_truncate_extra():
    """ndim=2 chunk 输入维度 > 目标维度 → 按第一维切片。"""
    env = _joint_env(dim=6)
    chunk = np.arange(50 * 8, dtype=float).reshape(50, 8)
    out = joint_to_joint_transform(chunk, env)
    assert out.shape == (50, 6)
    np.testing.assert_array_equal(out, chunk[:, :6])


def test_j2j_ndim_2_pad_gripper_chunk():
    """ndim=2 chunk 目标 7 维含 gripper，输入 6 维无 gripper → 整 chunk 补 (N, 1)。"""
    env = _joint_env(dim=7, with_gripper=True)
    chunk = np.arange(50 * 6, dtype=float).reshape(50, 6)
    out = joint_to_joint_transform(chunk, env)
    assert out.shape == (50, 7)
    np.testing.assert_array_equal(out[:, :6], chunk)
    assert np.all(out[:, 6] == pytest.approx(0.5))


def test_j2j_ndim_2_under_dim_raises():
    """ndim=2 chunk 输入维度远小于目标 → ValueError。"""
    env = _joint_env(dim=6)
    chunk = np.zeros((50, 2), dtype=float)
    with pytest.raises(ValueError) as exc_info:
        joint_to_joint_transform(chunk, env)
    msg = str(exc_info.value)
    assert "2" in msg and "6" in msg


# ========== ndim=3 非法输入 ==========


def test_j2j_ndim_3_raises():
    """ndim=3 输入 → ValueError（chunk 契约只允许 1 或 2 维）。"""
    env = _joint_env(dim=6)
    with pytest.raises(ValueError) as exc_info:
        joint_to_joint_transform(np.zeros((10, 50, 6)), env)
    assert "ndim" in str(exc_info.value)