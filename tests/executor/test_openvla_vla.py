"""OpenVLA 后端单元测试（无 torch 环境）。

OpenVLA 顶层零 torch import（决策 6），因此构造 / output_spec / 输入校验
在无 torch 的 exact env 下可直接运行；懒加载 / predict 推理路径通过
`sys.modules` 注入假 torch / transformers / bitsandbytes / flash_attn 覆盖。

假模块注入约定（helper `_make_fake_env`）：
- 假 torch：SimpleNamespace，提供 bfloat16 / float16 / float32。
- 假 transformers：SimpleNamespace，提供 AutoProcessor / AutoModelForVision2Seq /
  BitsAndBytesConfig；from_pretrained 为 classmethod，返回共享假实例。
- 假 bitsandbytes：空 SimpleNamespace（仅存在性触发）。
- flash_attn 默认不注入（delitem），保证 attn 回退 eager 分支可测。
"""

import sys
import types

import numpy as np
import pytest

from env.base import ActionSpec
from executor.model.base import VLAOutput
from executor.model.openvla.openvla_vla import OpenVLA, _parse_dtype


def _valid_image(h: int = 16, w: int = 16) -> dict:
    """构造合法的多相机 image dict（首张 (H,W,3) uint8）。"""
    return {"top": np.zeros((h, w, 3), dtype=np.uint8)}


# ========== 假模块构造 helpers ==========


class _FakeInputs:
    """假 processor 输出：`.to(device, dtype)` 返回 dict 供 predict_action **展开。"""

    def to(self, device, dtype):
        return {
            "prompt": "fake-inputs",
            "pixel_values": np.zeros((1, 3, 8, 8), dtype=np.float32),
        }


class _FakeProcessor:
    def __init__(self, state: dict):
        self._state = state

    def __call__(self, prompt, pil):
        self._state["processor_calls"].append((prompt, pil))
        return _FakeInputs()


class _FakeModel:
    """假 VLA model：记录 `.to` 与 `predict_action` 调用，返回值可配置。"""

    def __init__(self):
        self.to_calls: list = []
        self.predict_kwargs: list = []
        self.return_value: np.ndarray | None = None

    def to(self, device):
        self.to_calls.append(device)
        return self

    def predict_action(self, **kwargs):
        self.predict_kwargs.append(kwargs)
        return self.return_value


def _make_fake_env(
    monkeypatch,
    *,
    include_bnb: bool = True,
    include_bitsandbytes: bool = True,
) -> dict:
    """把假 torch/transformers/bitsandbytes 塞进 sys.modules，返回共享 state。

    flash_attn 一律不注入，保证 attn_impl="flash_attention_2" 走 eager 回退。
    """
    fake_torch = types.SimpleNamespace(
        bfloat16=object(), float16=object(), float32=object()
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)

    model = _FakeModel()
    state = {
        "model": model,
        "from_pretrained_kwargs": [],
        "processor_calls": [],
        "bnb_kwargs": None,
        "fake_torch": fake_torch,
    }

    class _FakeAutoModelForVision2Seq:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            state["from_pretrained_kwargs"].append(kwargs)
            return model

    class _FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            return _FakeProcessor(state)

    class _FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            state["bnb_kwargs"] = kwargs

    parts = {
        "AutoModelForVision2Seq": _FakeAutoModelForVision2Seq,
        "AutoProcessor": _FakeAutoProcessor,
    }
    if include_bnb:
        parts["BitsAndBytesConfig"] = _FakeBitsAndBytesConfig
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(**parts))

    if include_bitsandbytes:
        monkeypatch.setitem(sys.modules, "bitsandbytes", types.SimpleNamespace())
    else:
        monkeypatch.delitem(sys.modules, "bitsandbytes", raising=False)

    return state


# ========== A. 构造与无 torch 路径（无需 mock） ==========


def test_init_model_path_required():
    """model_path 为空 / None 时构造抛 ValueError。"""
    with pytest.raises(ValueError):
        OpenVLA(model_path="")
    with pytest.raises(ValueError):
        OpenVLA(model_path=None)


def test_init_quantization_invalid():
    """quantization 非法值构造抛 ValueError。"""
    with pytest.raises(ValueError):
        OpenVLA(model_path="x", quantization="8bit")


def test_output_spec():
    """output_spec 为 task 空间 7 维，gripper_index=6。"""
    spec = OpenVLA(model_path="x").output_spec
    assert isinstance(spec, ActionSpec)
    assert spec.space == "task"
    assert spec.dim == 7
    assert spec.gripper_index == 6


def test_predict_input_validation():
    """predict 输入校验发生在懒加载之前，无需任何 mock。"""
    v = OpenVLA(model_path="x")

    # image 非 dict
    with pytest.raises(TypeError):
        v.predict(None, "move")
    with pytest.raises(TypeError):
        v.predict(np.zeros((4, 4, 3), dtype=np.uint8), "move")

    # image 为空 dict
    with pytest.raises(ValueError):
        v.predict({}, "move")

    # 首张图非 uint8
    with pytest.raises(ValueError):
        v.predict(
            {"top": np.zeros((4, 4, 3), dtype=np.float32)}, "move"
        )

    # 首张图非 (H, W, 3)
    with pytest.raises(ValueError):
        v.predict({"top": np.zeros((4, 4), dtype=np.uint8)}, "move")


# ========== B. _parse_dtype（注入假 torch） ==========


def test_parse_dtype(monkeypatch):
    """_parse_dtype 从 sys.modules 的假 torch 取 dtype，非法值抛 ValueError。"""
    fake_torch = types.SimpleNamespace(
        bfloat16="b16", float16="f16", float32="f32"
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert _parse_dtype("bfloat16") is fake_torch.bfloat16
    assert _parse_dtype("float16") is fake_torch.float16
    assert _parse_dtype("float32") is fake_torch.float32

    with pytest.raises(ValueError) as excinfo:
        _parse_dtype("unknown")
    assert "bfloat16|float16|float32" in str(excinfo.value)


# ========== C. 懒加载路径（注入假 torch + 假 transformers） ==========


def test_ensure_loaded_missing_transformers(monkeypatch):
    """缺 transformers 时 predict 抛 ImportError，文案含 requirements-openvla.txt。"""
    fake_torch = types.SimpleNamespace(
        bfloat16=object(), float16=object(), float32=object()
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delitem(sys.modules, "transformers", raising=False)

    v = OpenVLA(model_path="x")
    with pytest.raises(ImportError) as excinfo:
        v.predict(_valid_image(), "move")
    assert "requirements-openvla.txt" in str(excinfo.value)


def test_ensure_loaded_flash_attn_fallback(monkeypatch):
    """未安装 flash_attn 时回退 eager：kwargs 无 attn_implementation；model 走 .to()。"""
    state = _make_fake_env(monkeypatch)

    v = OpenVLA(model_path="x", quantization="none")
    v._ensure_loaded()

    (load_kwargs,) = state["from_pretrained_kwargs"]
    assert "attn_implementation" not in load_kwargs
    assert state["model"].to_calls == ["cuda:0"]


def test_ensure_loaded_4bit_branch(monkeypatch):
    """4bit 分支：quantization_config 为 BitsAndBytesConfig 实例 + device_map="auto"，无 .to()。"""
    state = _make_fake_env(monkeypatch)

    v = OpenVLA(model_path="x", quantization="4bit")
    v._ensure_loaded()

    (load_kwargs,) = state["from_pretrained_kwargs"]
    qcfg = load_kwargs.get("quantization_config")
    assert qcfg is not None
    assert qcfg.__class__.__name__ == "_FakeBitsAndBytesConfig"
    assert load_kwargs.get("device_map") == "auto"
    assert state["model"].to_calls == []


def test_ensure_loaded_none_branch(monkeypatch):
    """none 分支：from_pretrained 返回的 model 被调用 .to("cuda:0")。"""
    state = _make_fake_env(monkeypatch)

    v = OpenVLA(model_path="x", quantization="none")
    v._ensure_loaded()

    assert state["model"].to_calls == ["cuda:0"]


def test_ensure_loaded_4bit_missing_bnb(monkeypatch):
    """4bit 但缺 bitsandbytes 时抛 ImportError，文案含 requirements-openvla.txt。"""
    _make_fake_env(monkeypatch, include_bitsandbytes=False)

    v = OpenVLA(model_path="x", quantization="4bit")
    with pytest.raises(ImportError) as excinfo:
        v.predict(_valid_image(), "move")
    assert "requirements-openvla.txt" in str(excinfo.value)


# ========== D. predict 正常路径 ==========


def test_predict_ok_returns_contract_values(monkeypatch):
    """predict 返回 VLAOutput：values (1,7) float / gripper=0.5 / spec=task。"""
    state = _make_fake_env(monkeypatch)
    state["model"].return_value = np.array(
        [0.1, 0.2, 0.3, 0.05, 0.05, 0.05, 0.5]
    )

    v = OpenVLA(model_path="x", quantization="none")
    result = v.predict(_valid_image(), "move to the red cube")

    assert isinstance(result, VLAOutput)
    assert result.values.shape == (1, 7)
    assert result.values.dtype == float
    assert result.values[0, 6] == 0.5
    assert result.spec.space == "task"

    (action_kwargs,) = state["model"].predict_kwargs
    assert action_kwargs["unnorm_key"] == "bridge_orig"
    assert action_kwargs["do_sample"] is False


def test_predict_action_shape_mismatch(monkeypatch):
    """predict_action 返回非 7 维时抛 RuntimeError，文案含 unnorm_key。"""
    state = _make_fake_env(monkeypatch)
    state["model"].return_value = np.array([0.1] * 6)

    v = OpenVLA(model_path="x", quantization="none")
    with pytest.raises(RuntimeError) as excinfo:
        v.predict(_valid_image(), "move")
    assert "unnorm_key" in str(excinfo.value)
