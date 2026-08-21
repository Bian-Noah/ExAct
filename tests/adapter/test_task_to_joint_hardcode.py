"""task_to_joint_hardcode_transform 单元测试（robot-vla-adapter 任务 3 + Iteration 10 chunk）。

Iteration 10 chunk 契约：ndim=1 输入按 (1, -1) reshape；ndim=2 输入按第一维切片。
"""

import numpy as np
import pytest

from env.base import ActionSpec
from utils.adapter.adapters.task_to_joint_hardcode import (
    task_to_joint_hardcode_transform,
)


class _JointEnv:
    """仅提供 input_spec 的假 env（joint 6 维）。"""

    def __init__(self, dim=6):
        self._dim = dim

    @property
    def input_spec(self):
        return ActionSpec("joint", ("joint",) * self._dim)


# ========== ndim=1 单步输入 ==========


def test_hardcode_truncates_to_target_dim():
    """ndim=1 task 7 维输入 → joint 6 维（前 6 维截取），reshape (1, -1)。"""
    out = task_to_joint_hardcode_transform(
        np.array([1.0, 2, 3, 4, 5, 6, 7]), _JointEnv(dim=6)
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 6)
    np.testing.assert_allclose(out, np.array([[1.0, 2, 3, 4, 5, 6]]))


def test_hardcode_passthrough_same_dim():
    """ndim=1 输入维度 == 目标维度 → 原样直通。"""
    out = task_to_joint_hardcode_transform(
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), _JointEnv(dim=6)
    )
    assert out.shape == (1, 6)
    np.testing.assert_allclose(out, np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]))


def test_hardcode_tuple_input():
    """ndim=1 tuple 输入同样处理。"""
    out = task_to_joint_hardcode_transform(
        (1, 2, 3, 4, 5, 6, 7), _JointEnv(dim=6)
    )
    assert out.shape == (1, 6)
    np.testing.assert_allclose(out, np.array([[1.0, 2, 3, 4, 5, 6]]))


def test_hardcode_insufficient_dim_raises():
    """ndim=1 输入维度 < 目标维度 → ValueError。"""
    with pytest.raises(ValueError) as exc_info:
        task_to_joint_hardcode_transform(
            np.array([1.0, 2, 3]), _JointEnv(dim=6)
        )
    assert "不足" in str(exc_info.value)


# ========== ndim=2 chunk 输入 ==========


def test_hardcode_ndim_2_truncates_to_target_dim():
    """ndim=2 chunk task 7 维 → joint 6 维，第一维保留 N。"""
    chunk = np.arange(50 * 7, dtype=float).reshape(50, 7)
    out = task_to_joint_hardcode_transform(chunk, _JointEnv(dim=6))
    assert out.shape == (50, 6)
    np.testing.assert_allclose(out, chunk[:, :6])


def test_hardcode_ndim_2_passthrough_same_dim():
    """ndim=2 chunk 输入维度 == 目标维度 → 整 chunk 透传。"""
    chunk = np.arange(50 * 6, dtype=float).reshape(50, 6)
    out = task_to_joint_hardcode_transform(chunk, _JointEnv(dim=6))
    assert out.shape == (50, 6)
    np.testing.assert_allclose(out, chunk)


# ========== ndim=3 非法输入 ==========


def test_hardcode_ndim_3_raises():
    """ndim=3 输入 → ValueError。"""
    with pytest.raises(ValueError) as exc_info:
        task_to_joint_hardcode_transform(
            np.zeros((10, 50, 7)), _JointEnv(dim=6)
        )
    assert "ndim" in str(exc_info.value)