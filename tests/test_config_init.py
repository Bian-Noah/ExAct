from __future__ import annotations

import os

import src.config as config_module
from src.config import (
    AppConfig,
    EnvConfig,
    ExploreConfig,
    LLMConfig,
    VLAConfig,
    load_config,
)


def test_config_all_exports_present():
    expected = {
        "AppConfig",
        "EnvConfig",
        "VLAConfig",
        "LLMConfig",
        "ExploreConfig",
        "load_config",
    }
    assert set(config_module.__all__) == expected
    for name in expected:
        assert hasattr(config_module, name), f"src.config 缺少导出：{name}"


def test_imported_dataclasses_are_correct_types():
    # 确保导入的 dataclass 可以直接实例化
    assert EnvConfig().use_gui is True
    assert VLAConfig().backend == "mock"
    assert LLMConfig().model == "MiniMax-M3"
    assert ExploreConfig().enabled is False
    # AppConfig 需要显式传入子配置实例
    app = AppConfig(env=EnvConfig(), vla=VLAConfig(), llm=LLMConfig(), explore=ExploreConfig())
    assert isinstance(app.env, EnvConfig)


def test_load_config_via_public_import():
    """通过 src.config.load_config 公共接口调用，确保其与 loader.load_config 行为一致。"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_path = os.path.join(project_root, "configs", "default.yaml")
    cfg = load_config(default_path)
    assert isinstance(cfg, AppConfig)
    assert cfg.vla.backend == "mock"
    assert cfg.env.camera_resolution == (640, 480)
