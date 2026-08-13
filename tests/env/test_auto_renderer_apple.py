"""M4 Mac auto 模式验证（iter2-renderer-env-mode）。

在 Apple Silicon 上：renderer='auto' 应解析为 p.ER_TINY_RENDERER，
且真实 render() 不触发段错误。
"""

from __future__ import annotations

import platform

import pybullet as p
import pytest

from config.loader import EnvConfig
from env.pybullet_env import PyBulletPandaEnv


_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.processor() == "arm"


@pytest.mark.slow
@pytest.mark.skipif(
    not _IS_APPLE_SILICON,
    reason="仅在 Apple Silicon 平台运行",
)
def test_auto_renderer_resolves_to_tiny_on_apple_silicon():
    """Apple Silicon + auto → ER_TINY_RENDERER。"""
    env = PyBulletPandaEnv(env_config=EnvConfig(mode="direct", renderer="auto"))
    assert env._renderer == p.ER_TINY_RENDERER


@pytest.mark.slow
@pytest.mark.skipif(
    not _IS_APPLE_SILICON,
    reason="仅在 Apple Silicon 平台运行",
)
def test_auto_renderer_does_not_crash_on_apple_silicon():
    """Apple Silicon + auto + DIRECT，render() 不段错误。"""
    import numpy as np
    env = PyBulletPandaEnv(env_config=EnvConfig(mode="direct", renderer="auto"))
    try:
        env.reset(task_spec={"objects": []})
        rgb = env.render()
        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (480, 640, 3)
    finally:
        env.close()
