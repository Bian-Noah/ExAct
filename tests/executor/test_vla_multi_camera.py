"""BaseVLA.predict image=dict 适配单元测试（iter11-reset-multicam）。

验证 BaseVLA.predict 的 image 参数从 ndarray 改为 dict[str, ndarray],
并测试 LeRobotVLA / OpenVLA 改造后的取图逻辑(fallback 占位 warn)。

注意:lerobot/torch 在 M4 Mac 环境未安装(iter11 验证环境为 RTX 4060),
LerobotVLA 相关测试用 sys.modules 注入 fake torch。
"""
from __future__ import annotations

import inspect
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest


# ----- Fake torch 注入(让 lerobot_vla.py 的 `import torch` 不爆) -----

def _install_fake_torch():
    """在 sys.modules 注入最小 fake torch,使 lerobot_vla.py 的局部 import torch 通过。

    让 _FakeTensor 直接持有最终 ndarray,所有 chain 操作直接修改内部 arr,
    这样 allclose/permute/unsqueeze/float/div_ 等链式调用结果与 numpy 一致。
    """
    if "torch" in sys.modules:
        return  # 已存在(真实 torch),不覆盖

    fake = types.ModuleType("torch")

    class _FakeTensor:
        def __init__(self, arr):
            self._arr = np.asarray(arr)

        @property
        def shape(self):
            return self._arr.shape

        def permute(self, *axes):
            self._arr = np.transpose(self._arr, axes)
            return self

        def unsqueeze(self, axis):
            self._arr = np.expand_dims(self._arr, axis)
            return self

        def contiguous(self):
            self._arr = np.ascontiguousarray(self._arr)
            return self

        def float(self):
            self._arr = self._arr.astype(np.float32)
            return self

        def div_(self, scalar):
            self._arr = self._arr / scalar
            return self

        def min(self):
            return float(self._arr.min())

        def max(self):
            return float(self._arr.max())

        def __array__(self, dtype=None):
            if dtype is None:
                return self._arr
            return self._arr.astype(dtype)

    def _from_numpy(arr):
        return _FakeTensor(arr)

    def _as_tensor(arr, device=None):
        return _FakeTensor(np.asarray(arr))

    def _zeros(shape, dtype=None, device=None):
        return _FakeTensor(np.zeros(shape, dtype=dtype or np.float32))

    fake.from_numpy = _from_numpy
    fake.as_tensor = _as_tensor
    fake.zeros = _zeros
    # OpenVLA 默认参数 dtype: torch.dtype = torch.bfloat16 需要这些属性
    fake.bfloat16 = "bfloat16"
    fake.float32 = "float32"
    fake.dtype = lambda: None  # torch.dtype 占位
    fake.device = lambda *a, **kw: None

    # allclose:接受 _FakeTensor / ndarray,转 ndarray 比较
    def _allclose(a, b, atol=0.0):
        a_arr = a._arr if isinstance(a, _FakeTensor) else np.asarray(a)
        b_arr = b._arr if isinstance(b, _FakeTensor) else np.asarray(b)
        return bool(np.allclose(a_arr, b_arr, atol=atol))

    fake.allclose = _allclose

    sys.modules["torch"] = fake


@pytest.fixture
def fake_torch():
    """注入 fake torch 到 sys.modules,测试结束后移除(避免污染其他测试文件)。

    lerobot_vla.py 顶层 import torch,M4 Mac 无真实 torch。
    用 fixture 局部注入,teardown 时恢复,不影响 test_vla.py 的 importorskip。
    """
    had_real_torch = "torch" in sys.modules
    real_torch = sys.modules.get("torch")
    _install_fake_torch()
    yield sys.modules["torch"]
    if had_real_torch:
        sys.modules["torch"] = real_torch
    else:
        sys.modules.pop("torch", None)


from executor.model.base import BaseVLA, VLAOutput
from env.base import ActionSpec


class _FakeVLA(BaseVLA):
    """最小 BaseVLA 子类,用于签名验证。"""

    @property
    def output_spec(self) -> ActionSpec:
        return ActionSpec("task", ("dx", "dy", "dz", "drx", "dry", "drz", "gripper"))

    def predict(self, image, instruction, state=None):
        # 只验证 image 是 dict
        assert isinstance(image, dict)
        return VLAOutput(values=np.zeros((1, 7), dtype=np.float32), spec=self.output_spec)


def test_base_vla_predict_signature_image_is_dict():
    """BaseVLA.predict 签名 image 参数类型注解为 dict[str, np.ndarray]。"""
    sig = inspect.signature(BaseVLA.predict)
    image_param = sig.parameters["image"]
    # 类型注解可能是 "dict[str, np.ndarray]" 字符串
    assert "dict" in str(image_param.annotation)


def test_base_vla_predict_accepts_dict_image():
    """BaseVLA.predict 可接收 dict[str, np.ndarray] 类型的 image。"""
    fake = _FakeVLA()
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    out = fake.predict({"cam1": rgb, "cam2": rgb}, "test")
    assert isinstance(out, VLAOutput)
    assert out.values.shape == (1, 7)


def test_base_vla_predict_rejects_non_dict_image():
    """BaseVLA.predict 子类拒绝非 dict 的 image(防御性)。"""
    fake = _FakeVLA()
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    # 改造后 _FakeVLA 内部 assert isinstance(image, dict)
    with pytest.raises(AssertionError):
        fake.predict(rgb, "test")


# ----- LeRobotVLA _build_raw_obs 测试 -----

def test_lerobot_vla_build_raw_obs_multi_camera_distinct(fake_torch):
    """LeRobotVLA._build_raw_obs 多相机独立图 → 每个 obs[key] tensor 不同。"""
    from executor.model.lerobot.lerobot_vla import LeRobotVLA

    # 构造一个最小 LeRobotVLA 实例(不加载真 policy)
    vla = LeRobotVLA.__new__(LeRobotVLA)
    vla._resolved_image_keys = ["observation.images.cam1", "observation.images.cam2"]
    vla._resolved_state_key = "observation.state"
    vla.device = "cpu"
    vla._infer_state_dim = lambda: 6
    vla._log = type("L", (), {"warning": lambda *a, **kw: None})()

    img1 = np.ones((64, 64, 3), dtype=np.uint8) * 100  # 全部 100
    img2 = np.ones((64, 64, 3), dtype=np.uint8) * 200  # 全部 200
    obs = vla._build_raw_obs(
        image={"observation.images.cam1": img1, "observation.images.cam2": img2},
        instruction="test",
        state=np.zeros(6),
    )

    t1 = obs["observation.images.cam1"]
    t2 = obs["observation.images.cam2"]
    # 两个 tensor 应明显不同(差值 ~0.392)
    import torch
    assert not torch.allclose(t1, t2, atol=0.05)
    # 都归一化到 [0,1]
    assert t1.min() >= 0 and t1.max() <= 1
    assert t2.min() >= 0 and t2.max() <= 1


def test_lerobot_vla_build_raw_obs_fallback_warn(fake_torch):
    """policy 期望 3 路,env 只给 1 张 → fallback 用首张 + warn。"""
    from executor.model.lerobot.lerobot_vla import LeRobotVLA

    vla = LeRobotVLA.__new__(LeRobotVLA)
    vla._resolved_image_keys = ["cam1", "cam2", "cam3"]
    vla._resolved_state_key = "observation.state"
    vla.device = "cpu"
    vla._infer_state_dim = lambda: 6

    warn_called = []

    class _FakeLog:
        def warning(self, msg):
            warn_called.append(msg)

    vla._log = _FakeLog()

    img = np.zeros((64, 64, 3), dtype=np.uint8)

    obs = vla._build_raw_obs(
        image={"cam1": img},  # 只给 1 张
        instruction="test",
        state=np.zeros(6),
    )

    # 3 个 key 都填了(cam2/cam3 fallback 用 cam1)
    assert "cam1" in obs
    assert "cam2" in obs
    assert "cam3" in obs
    assert len(warn_called) == 1
    assert "fallback" in warn_called[0]


def test_lerobot_vla_build_raw_obs_empty_image_raises(fake_torch):
    """image dict 为空时抛 ValueError。"""
    from executor.model.lerobot.lerobot_vla import LeRobotVLA

    vla = LeRobotVLA.__new__(LeRobotVLA)
    vla._resolved_image_keys = ["cam1"]
    vla._resolved_state_key = "observation.state"
    vla.device = "cpu"
    vla._infer_state_dim = lambda: 6
    vla._log = type("L", (), {"warning": lambda *a, **kw: None})()

    with pytest.raises(ValueError, match="image dict 为空"):
        vla._build_raw_obs(image={}, instruction="test")


# ----- OpenVLA.predict image=dict 测试 -----

def test_openvla_vla_predict_rejects_non_dict_image(fake_torch):
    """OpenVLA.predict(image 非 dict) → TypeError。"""
    from executor.model.openvla.openvla_vla import OpenVLA

    vla = OpenVLA.__new__(OpenVLA)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="image 必须为 dict"):
        vla.predict(image=rgb, instruction="test")


def test_openvla_vla_predict_empty_dict_raises(fake_torch):
    """OpenVLA.predict(image={}) → ValueError。"""
    from executor.model.openvla.openvla_vla import OpenVLA

    vla = OpenVLA.__new__(OpenVLA)
    with pytest.raises(ValueError, match="image dict 为空"):
        vla.predict(image={}, instruction="test")


def test_openvla_vla_predict_dict_with_invalid_dtype_raises(fake_torch):
    """OpenVLA.predict(image=dict 但首张 dtype 不对) → ValueError。"""
    from executor.model.openvla.openvla_vla import OpenVLA

    vla = OpenVLA.__new__(OpenVLA)
    rgb_float = np.zeros((480, 640, 3), dtype=np.float32)  # 应是 uint8

    with pytest.raises(ValueError, match="image.dtype"):
        vla.predict(image={"cam1": rgb_float}, instruction="test")