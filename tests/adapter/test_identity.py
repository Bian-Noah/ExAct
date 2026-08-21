"""identity_transform 单元测试（robot-vla-adapter 任务 3 + Iteration 10 chunk）。

Iteration 10 chunk 契约：ndim=1 输入按 (1, -1) reshape；ndim=2 原样返回。
"""

import numpy as np
import pytest

from utils.adapter.adapters.identity import identity_transform


def test_identity_passthrough_raw_object():
    """非 ndarray 兜底：Action7D 等 → 转 ndarray 后 reshape。"""
    # Action7D 在 chunk 契约下应被转 ndarray + reshape (1, -1)
    arr = np.array([1.0, 2, 3], dtype=float)
    out = identity_transform(arr, env=None)
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 3)
    np.testing.assert_array_equal(out, np.array([[1.0, 2, 3]]))


def test_identity_passthrough_array_ndim_1():
    """ndim=1 ndarray → reshape (1, -1)。"""
    arr = np.array([1.0, 2, 3])
    out = identity_transform(arr, env=None)
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 3)


def test_identity_passthrough_array_ndim_2():
    """ndim=2 ndarray → 透传（值与形状一致；引用不必一致因为内部可能 .astype）。"""
    arr = np.zeros((50, 5), dtype=float)
    out = identity_transform(arr, env=None)
    assert isinstance(out, np.ndarray)
    assert out.shape == (50, 5)
    np.testing.assert_array_equal(out, arr)


def test_identity_ndim_3_raises():
    """ndim=3 输入 → ValueError。"""
    with pytest.raises(ValueError) as exc_info:
        identity_transform(np.zeros((10, 50, 5)), env=None)
    assert "ndim" in str(exc_info.value)