"""executor/model/factory.py 单元测试。

覆盖 4 种 VLA backend 的分派行为 + 异常路径 + llm_config 透传。
"""

import pytest

from config import LLMConfig, VLAConfig
from executor.model.base import BaseVLA
from executor.model.factory import create_vla
from executor.model.mock.mock_vla import MockVLA


def test_create_vla_mock_backend():
    """backend='mock' 返回 MockVLA 实例，且是 BaseVLA 子类。"""
    vla = create_vla(VLAConfig(backend="mock"))
    assert isinstance(vla, MockVLA)
    assert isinstance(vla, BaseVLA)


def test_create_vla_llm_vla_not_implemented():
    """backend='llm_vla' 抛 NotImplementedError，错误信息含 Iteration 5。"""
    with pytest.raises(NotImplementedError, match="Iteration 5"):
        create_vla(VLAConfig(backend="llm_vla"))


def test_create_vla_small_vla_not_implemented():
    """backend='small_vla' 抛 NotImplementedError，错误信息含 Iteration 9。"""
    with pytest.raises(NotImplementedError, match="Iteration 9"):
        create_vla(VLAConfig(backend="small_vla"))


def test_create_vla_openvla_not_implemented():
    """backend='openvla' 抛 NotImplementedError，错误信息含 Iteration 10。"""
    with pytest.raises(NotImplementedError, match="Iteration 10"):
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