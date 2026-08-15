"""配置驱动 env 行为端到端功能测试（iter2-renderer-env-mode）。

覆盖：
- 通过 yaml 配置文件构造 env，env 内部字段与配置一致
- 多种 mode/renderer 组合均按预期传递
"""

from __future__ import annotations

import os
import platform
import tempfile
from unittest.mock import MagicMock, patch

import pybullet as p
import pytest
import yaml

from config.loader import load_config
from env.pybullet_env import PyBulletEnv


def _write_yaml(content: dict) -> str:
    """写入临时 yaml，返回文件路径。"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        yaml.safe_dump(content, f, allow_unicode=True)
        return f.name


def test_config_drives_env_mode_renderer(tmp_path):
    """配置中 mode=direct + renderer=cpu → env._mode='direct'、env._renderer=TINY。"""
    cfg_path = tmp_path / "env.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "env": {
                    "mode": "direct",
                    "renderer": "cpu",
                    "camera_resolution": [320, 240],
                }
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    config = load_config(str(cfg_path))
    env = PyBulletEnv(env_config=config.env)
    assert env._mode == "direct"
    assert env._renderer == p.ER_TINY_RENDERER
    assert env.use_gui is False
    assert env.camera_resolution == (320, 240)


def test_config_drives_env_gui_gpu(tmp_path):
    """配置中 mode=gui + renderer=gpu → env._mode='gui'、env._renderer=OPENGL。"""
    cfg_path = tmp_path / "env.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "env": {
                    "mode": "gui",
                    "renderer": "gpu",
                    "camera_resolution": [800, 600],
                }
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    config = load_config(str(cfg_path))
    env = PyBulletEnv(env_config=config.env)
    assert env._mode == "gui"
    assert env._renderer == p.ER_BULLET_HARDWARE_OPENGL
    assert env.use_gui is True
    assert env.camera_resolution == (800, 600)


def test_config_drives_env_auto_renderer_resolved(tmp_path):
    """配置中 renderer=auto → 平台相关（Apple Silicon → TINY，其他 → OPENGL）。"""
    cfg_path = tmp_path / "env.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {"env": {"mode": "direct", "renderer": "auto"}},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    config = load_config(str(cfg_path))
    env = PyBulletEnv(env_config=config.env)
    is_apple_silicon = (
        platform.system() == "Darwin" and platform.processor() == "arm"
    )
    if is_apple_silicon:
        assert env._renderer == p.ER_TINY_RENDERER
    else:
        assert env._renderer == p.ER_BULLET_HARDWARE_OPENGL
