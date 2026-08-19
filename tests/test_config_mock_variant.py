"""MockConfig 加载测试。

验证 `vla.mock.variant` 字段从 yaml 加载：
- 默认值是 "task"
- 显式 "joint" 加载正确
- 与 LerobotConfig 嵌套模式一致（不影响 lerobot 字段）
- 非法值不在 _from_dict 通用校验中拒绝（_check_type 只校验类型，不校验值集合）
  注：variant 取值合法性校验在 factory.create_vla 中进行，不在 config 加载层。
"""

from __future__ import annotations

import textwrap

from config.loader import (
    LerobotConfig,
    MockConfig,
    VLAConfig,
    load_config,
)


def test_mock_config_default_variant():
    """MockConfig 默认 variant='task'。"""
    cfg = MockConfig()
    assert cfg.variant == "task"


def test_mock_config_explicit_joint():
    """MockConfig 显式 variant='joint'。"""
    cfg = MockConfig(variant="joint")
    assert cfg.variant == "joint"


def test_vla_config_default_has_mock_field():
    """VLAConfig 默认包含 mock 字段（默认 variant='task'）。"""
    cfg = VLAConfig()
    assert isinstance(cfg.mock, MockConfig)
    assert cfg.mock.variant == "task"


def test_load_yaml_default_mock_variant():
    """yaml 不写 vla.mock 块 → 默认 variant='task'。"""
    yaml = textwrap.dedent("""
        env: {}
        vla: {}
        llm: {}
        explore: {}
        robot: {}
        task: {}
        agent: {}
        experiment: {}
        image_store: {}
    """).strip()
    cfg = load_config_from_text(yaml)
    assert cfg.vla.mock.variant == "task"


def test_load_yaml_explicit_mock_variant_joint():
    """yaml vla.mock.variant='joint' 加载正确。"""
    yaml = textwrap.dedent("""
        env: {}
        vla:
          mock:
            variant: joint
        llm: {}
        explore: {}
        robot: {}
        task: {}
        agent: {}
        experiment: {}
        image_store: {}
    """).strip()
    cfg = load_config_from_text(yaml)
    assert cfg.vla.mock.variant == "joint"


def test_load_yaml_mock_and_lerobot_coexist():
    """vla.mock 与 vla.lerobot 嵌套字段可同时存在（互不影响）。"""
    yaml = textwrap.dedent("""
        env: {}
        vla:
          mock:
            variant: joint
          lerobot:
            policy_type: smolvla
            device: "cuda:0"
        llm: {}
        explore: {}
        robot: {}
        task: {}
        agent: {}
        experiment: {}
        image_store: {}
    """).strip()
    cfg = load_config_from_text(yaml)
    assert cfg.vla.mock.variant == "joint"
    assert isinstance(cfg.vla.lerobot, LerobotConfig)
    assert cfg.vla.lerobot.policy_type == "smolvla"


def test_load_yaml_mock_variant_invalid_value_passes_config():
    """variant='invalid' 能通过 config 加载（_check_type 不校验值集合），由 factory 拒绝。"""
    yaml = textwrap.dedent("""
        env: {}
        vla:
          mock:
            variant: invalid_variant
        llm: {}
        explore: {}
        robot: {}
        task: {}
        agent: {}
        experiment: {}
        image_store: {}
    """).strip()
    # config 加载层只校验类型（str），不校验值集合
    cfg = load_config_from_text(yaml)
    assert cfg.vla.mock.variant == "invalid_variant"
    # factory 层才会拒绝
    from executor.model.factory import create_vla
    import pytest
    with pytest.raises(ValueError, match="未知 mock variant"):
        create_vla(cfg.vla)


def test_load_yaml_mock_variant_wrong_type_raises():
    """variant 类型错（非 str）应在 config 加载层抛 TypeError。"""
    yaml = textwrap.dedent("""
        env: {}
        vla:
          mock:
            variant: 123
        llm: {}
        explore: {}
        robot: {}
        task: {}
        agent: {}
        experiment: {}
        image_store: {}
    """).strip()
    import pytest
    with pytest.raises(TypeError, match="MockConfig.variant"):
        load_config_from_text(yaml)


def test_load_yaml_mock_block_full_keys():
    """完整 mock 块加载（未来扩展字段时不必改 yaml）。"""
    yaml = textwrap.dedent("""
        env: {}
        vla:
          mock:
            variant: joint
        llm: {}
        explore: {}
        robot: {}
        task: {}
        agent: {}
        experiment: {}
        image_store: {}
    """).strip()
    cfg = load_config_from_text(yaml)
    assert cfg.vla.mock.variant == "joint"


# ========== helper ==========


def load_config_from_text(yaml_text: str):
    """从 yaml 字符串加载 AppConfig（避免每个用例写 tmp_path 文件）。"""
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        f.write(yaml_text)
        path = f.name
    try:
        return load_config(path)
    finally:
        os.unlink(path)