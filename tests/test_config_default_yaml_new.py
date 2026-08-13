"""configs/default.yaml 单元测试（iter2-renderer-env-mode）。

覆盖：
- load_config 成功返回 AppConfig
- config.env.mode / renderer / use_gui 字段正确
- config.env.camera_resolution 正确转换
"""

from __future__ import annotations

import os

import pytest

from config.loader import load_config


CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "default.yaml",
)


def test_default_yaml_loads_successfully():
    """load_config 不抛异常。"""
    config = load_config(CONFIG_PATH)
    assert config is not None
    assert config.env is not None


def test_default_yaml_env_mode():
    assert load_config(CONFIG_PATH).env.mode == "direct"


def test_default_yaml_env_renderer():
    assert load_config(CONFIG_PATH).env.renderer == "auto"


def test_default_yaml_env_use_gui_default_false():
    """use_gui 字段已移除，默认值为 False。"""
    assert load_config(CONFIG_PATH).env.use_gui is False


def test_default_yaml_env_camera_resolution():
    """camera_resolution 正确转换为 tuple。"""
    assert load_config(CONFIG_PATH).env.camera_resolution == (640, 480)


def test_default_yaml_env_has_no_legacy_use_gui_key():
    """YAML 中不应再含 use_gui 字段（迁移完成）。"""
    import yaml
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    assert "use_gui" not in raw["env"]
    assert "mode" in raw["env"]
    assert "renderer" in raw["env"]
