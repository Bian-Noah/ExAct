"""executor/model/factory.py 单元测试。

覆盖 4 种 VLA backend 的分派行为 + 异常路径 + llm_config 透传。
"""

import pytest

from config import LLMConfig, VLAConfig
from config.loader import OpenVLAConfig
from executor.model.base import BaseVLA
from executor.model.factory import create_vla
from executor.model.mock.mock_vla import JointMockVLA, MockVLA
from executor.model.openvla.openvla_vla import OpenVLA


def test_create_vla_mock_backend():
    """backend='mock' 返回 MockVLA 实例，且是 BaseVLA 子类。"""
    vla = create_vla(VLAConfig(backend="mock"))
    assert isinstance(vla, MockVLA)
    assert isinstance(vla, BaseVLA)
    # 默认 variant="task"，应返回 MockVLA 而非 JointMockVLA
    assert not isinstance(vla, JointMockVLA)


def test_create_vla_mock_backend_default_variant_is_task():
    """backend='mock' 不指定 mock.variant → 默认 variant='task' → MockVLA。"""
    cfg = VLAConfig(backend="mock")
    assert cfg.mock.variant == "task"  # 默认值
    vla = create_vla(cfg)
    assert isinstance(vla, MockVLA)
    assert not isinstance(vla, JointMockVLA)


def test_create_vla_mock_backend_variant_joint():
    """backend='mock' + mock.variant='joint' → 返回 JointMockVLA。"""
    from config.loader import MockConfig
    cfg = VLAConfig(backend="mock", mock=MockConfig(variant="joint"))
    vla = create_vla(cfg)
    assert isinstance(vla, JointMockVLA)
    assert isinstance(vla, BaseVLA)
    # JointMockVLA 与 MockVLA 是兄弟类（都继承 BaseVLA）
    assert not isinstance(vla, MockVLA)


def test_create_vla_mock_backend_unknown_variant_raises():
    """backend='mock' + mock.variant='invalid' → 抛 ValueError，错误信息含 '未知 mock variant'。"""
    from config.loader import MockConfig
    cfg = VLAConfig(backend="mock", mock=MockConfig(variant="invalid"))
    with pytest.raises(ValueError, match="未知 mock variant"):
        create_vla(cfg)


def test_create_vla_llm_vla_not_implemented():
    """backend='llm_vla' 抛 NotImplementedError，错误信息含 Iteration 5。"""
    with pytest.raises(NotImplementedError, match="Iteration 5"):
        create_vla(VLAConfig(backend="llm_vla"))


def test_create_vla_small_vla_not_implemented():
    """backend='small_vla' 抛 NotImplementedError，错误信息含 Iteration 9。"""
    with pytest.raises(NotImplementedError, match="Iteration 9"):
        create_vla(VLAConfig(backend="small_vla"))


def test_create_vla_openvla_returns_openvla():
    """backend='openvla' 返回 OpenVLA 实例（BaseVLA 子类），构造不抛错不加载模型。"""
    vla = create_vla(VLAConfig(backend="openvla", model_path="openvla/openvla-7b"))
    assert isinstance(vla, OpenVLA)
    assert isinstance(vla, BaseVLA)


def test_create_vla_openvla_field_passthrough():
    """backend='openvla' 透传 openvla 特化字段（quantization + dtype）。"""
    vla = create_vla(
        VLAConfig(
            backend="openvla",
            model_path="x",
            openvla=OpenVLAConfig(quantization="none"),
        )
    )
    assert vla.quantization == "none"
    assert vla._dtype_str == "bfloat16"


def test_create_vla_openvla_missing_model_path():
    """backend='openvla' 缺 model_path → 构造校验抛 ValueError。"""
    with pytest.raises(ValueError):
        create_vla(VLAConfig(backend="openvla"))


def test_create_vla_unknown_backend_raises():
    """未知 backend 抛 ValueError，错误信息含 '未知 VLA backend'。"""
    with pytest.raises(ValueError, match="未知 VLA backend"):
        create_vla(VLAConfig(backend="unknown_vla"))


def test_create_vla_empty_backend_raises():
    """空字符串 backend 抛 ValueError。"""
    with pytest.raises(ValueError, match="未知 VLA backend"):
        create_vla(VLAConfig(backend=""))


def test_create_vla_uppercase_backend_raises():
    """大写 backend（MOCK）抛 ValueError（大小写敏感）。"""
    with pytest.raises(ValueError, match="未知 VLA backend"):
        create_vla(VLAConfig(backend="MOCK"))


def test_create_vla_with_llm_config_passes_through():
    """Mock 后端忽略 llm_config 参数，不抛错。"""
    vla = create_vla(
        VLAConfig(backend="mock"),
        llm_config=LLMConfig(api_key="dummy"),
    )
    assert isinstance(vla, MockVLA)