"""identity_transform 单元测试（robot-vla-adapter 任务 3）。

输入为 executor 解包后的裸动作值，identity 原样直通。
"""

import numpy as np

from utils.adapter.adapters.identity import identity_transform


def test_identity_passthrough_raw_object():
    """裸对象原样返回。"""
    obj = object()
    assert identity_transform(obj, env=None) is obj


def test_identity_passthrough_array():
    """数组原样透传。"""
    arr = np.array([1.0, 2, 3])
    out = identity_transform(arr, env=None)
    assert out is arr
