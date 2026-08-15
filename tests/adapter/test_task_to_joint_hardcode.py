"""task_to_joint_hardcode_transform 单元测试。"""

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


def test_hardcode_truncates_to_target_dim():
    """task 7 维输入 → joint 6 维（前 6 维截取）。"""
    out = task_to_joint_hardcode_transform(
        np.array([1.0, 2, 3, 4, 5, 6, 7]), _JointEnv(dim=6)
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == (6,)
    np.testing.assert_allclose(out, [1.0, 2, 3, 4, 5, 6])


def test_hardcode_passthrough_same_dim():
    """输入维度 == 目标维度 → 原样直通。"""
    out = task_to_joint_hardcode_transform(
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), _JointEnv(dim=6)
    )
    np.testing.assert_allclose(out, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def test_hardcode_tuple_input():
    """tuple 输入同样处理。"""
    out = task_to_joint_hardcode_transform(
        (1, 2, 3, 4, 5, 6, 7), _JointEnv(dim=6)
    )
    np.testing.assert_allclose(out, [1.0, 2, 3, 4, 5, 6])


def test_hardcode_insufficient_dim_raises():
    """输入维度 < 目标维度 → ValueError。"""
    with pytest.raises(ValueError) as exc_info:
        task_to_joint_hardcode_transform(
            np.array([1.0, 2, 3]), _JointEnv(dim=6)
        )
    assert "不足" in str(exc_info.value)
