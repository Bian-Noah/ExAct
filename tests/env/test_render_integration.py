"""渲染稳定性集成测试（iter2-renderer-env-mode）。

真实启动 PyBullet DIRECT + CPU 模式，连续 render() 不崩溃，shape/dtype 正确。
"""

from __future__ import annotations

import platform

import numpy as np
import pytest

from config.loader import EnvConfig
from env.pybullet_env import PyBulletPandaEnv


@pytest.mark.slow
def test_render_multiple_calls_no_crash():
    """连续 render() 10 次不崩溃，返回 shape (480, 640, 3)。"""
    env = PyBulletPandaEnv(env_config=EnvConfig(mode="direct", renderer="cpu"))
    try:
        env.reset(task_spec={"objects": []})
        for _ in range(10):
            rgb = env.render()
            assert isinstance(rgb, np.ndarray)
            assert rgb.shape == (480, 640, 3)
            assert rgb.dtype == np.uint8
    finally:
        env.close()
