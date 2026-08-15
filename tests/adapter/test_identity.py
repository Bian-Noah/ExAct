"""identity_transform 单元测试（robot-vla-adapter 任务 3）。"""

import numpy as np

from env.base import Action7D, ActionSpec
from executor.model.base import VLAOutput
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


def test_identity_unwraps_vlaoutput():
    """VLAOutput 包装 → 取 .values 返回。"""
    spec = ActionSpec("task", ("dx", "dy", "dz", "drx", "dry", "drz", "gripper"))
    action = Action7D(0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.8)
    out = identity_transform(VLAOutput(values=action, spec=spec), env=None)
    assert out == action
