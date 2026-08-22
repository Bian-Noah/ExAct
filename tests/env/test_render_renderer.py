"""PyBulletEnv.render 渲染器参数单元测试（iter2-renderer-env-mode）。

覆盖：
- self._renderer=ER_TINY_RENDERER 时 getCameraImage(renderer=...) 正确传入
- self._renderer=ER_BULLET_HARDWARE_OPENGL 时同上
- 返回 RGBA reshape 为 (H, W, 3) RGB
- 日志输出包含 "CPU/TINY_RENDERER" 或 "GPU/OPENGL" 标识
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pybullet as p

from config.loader import CameraSpec, EnvConfig
from env.pybullet_env import PyBulletEnv


def _make_mock_p():
    """构造 spec=pybullet 的 MagicMock。"""
    # spec=p 让 MagicMock 模仿 pybullet 的属性；常量属性赋真实值便于模块内引用
    mock_p = MagicMock(spec=p)
    mock_p.DIRECT = p.DIRECT
    mock_p.GUI = p.GUI
    mock_p.ER_TINY_RENDERER = p.ER_TINY_RENDERER
    mock_p.ER_BULLET_HARDWARE_OPENGL = p.ER_BULLET_HARDWARE_OPENGL
    mock_p.connect.return_value = 999
    mock_p.disconnect.return_value = None
    mock_p.getLinkState.return_value = ((0.5, 0.0, 0.5), (0, 0, 0, 1), None, None, None, None, None)
    mock_p.getBasePositionAndOrientation.return_value = ((0.5, 0.0, 0.1), (0, 0, 0, 1))
    return mock_p


def _make_rgba_pixels(width: int, height: int, r: int = 128, g: int = 128, b: int = 128) -> list:
    """构造 HxWx4 的 RGBA 像素（list of list of [r,g,b,a]）。"""
    return [
        [[r, g, b, 255] for _ in range(width)]
        for _ in range(height)
    ]


# ============================================================
# renderer 参数传递
# ============================================================

def test_render_uses_tiny_renderer_param(caplog):
    """_renderer=ER_TINY_RENDERER 时 getCameraImage 调用带 renderer=ER_TINY_RENDERER。"""
    env = PyBulletEnv(env_config=EnvConfig(mode="direct", renderer="cpu"))
    mock_p = _make_mock_p()
    width, height = 640, 480
    mock_p.getCameraImage.side_effect = (
        lambda *a, **kw: (width, height, _make_rgba_pixels(width, height), None, None)
    )

    with patch("env.pybullet_env.p", mock_p):
        env._client_id = 999
        with caplog.at_level(logging.INFO, logger="env.pybullet_env"):
            rgb = env.render()

    assert mock_p.getCameraImage.called
    call_kwargs = mock_p.getCameraImage.call_args.kwargs
    assert call_kwargs["renderer"] == p.ER_TINY_RENDERER
    assert any("CPU/TINY_RENDERER" in rec.message for rec in caplog.records)
    assert rgb['observation.images.top'].shape == (height, width, 3)
    assert rgb['observation.images.top'].dtype == np.uint8


def test_render_uses_opengl_renderer_param(caplog):
    """_renderer=ER_BULLET_HARDWARE_OPENGL 时同上。"""
    env = PyBulletEnv(env_config=EnvConfig(mode="direct", renderer="gpu"))
    mock_p = _make_mock_p()
    width, height = 640, 480
    mock_p.getCameraImage.side_effect = (
        lambda *a, **kw: (
            width, height, _make_rgba_pixels(width, height, 200, 100, 50), None, None
        )
    )

    with patch("env.pybullet_env.p", mock_p):
        env._client_id = 999
        with caplog.at_level(logging.INFO, logger="env.pybullet_env"):
            rgb = env.render()

    call_kwargs = mock_p.getCameraImage.call_args.kwargs
    assert call_kwargs["renderer"] == p.ER_BULLET_HARDWARE_OPENGL
    assert any("GPU/OPENGL" in rec.message for rec in caplog.records)
    assert rgb['observation.images.top'].shape == (height, width, 3)


# ============================================================
# RGBA → RGB reshape 正确性
# ============================================================

def test_render_rgba_to_rgb_reshape():
    """RGBA (H, W, 4) reshape 为 (H, W, 3)。"""
    # iter11-reset-multicam:render 用 CameraSpec.resolution 而非 camera_resolution
    env = PyBulletEnv(
        env_config=EnvConfig(
            mode="direct", renderer="cpu",
            cameras=(
                CameraSpec(
                    name="observation.images.top",
                    target=(0.5, 0.0, 0.5), distance=1.5,
                    yaw=50, pitch=-35, roll=0, resolution=(320, 240),
                ),
            ),
        )
    )
    mock_p = _make_mock_p()
    width, height = 320, 240
    mock_p.getCameraImage.side_effect = (
        lambda *a, **kw: (
            width, height, _make_rgba_pixels(width, height, 100, 150, 200), None, None
        )
    )

    with patch("env.pybullet_env.p", mock_p):
        env._client_id = 999
        rgb = env.render()

    assert rgb['observation.images.top'].shape == (height, width, 3)
    assert rgb['observation.images.top'][0, 0, 0] == 100
    assert rgb['observation.images.top'][0, 0, 1] == 150
    assert rgb['observation.images.top'][0, 0, 2] == 200


# ============================================================
# 日志格式
# ============================================================

def test_render_log_includes_renderer_value(caplog):
    """render() 入口日志包含实际 renderer 常量值。"""
    env = PyBulletEnv(env_config=EnvConfig(mode="direct", renderer="cpu"))
    mock_p = _make_mock_p()
    width, height = 640, 480
    mock_p.getCameraImage.side_effect = (
        lambda *a, **kw: (width, height, _make_rgba_pixels(width, height), None, None)
    )

    with patch("env.pybullet_env.p", mock_p):
        env._client_id = 999
        with caplog.at_level(logging.INFO, logger="env.pybullet_env"):
            env.render()

    msgs = [rec.message for rec in caplog.records]
    assert any("CPU/TINY_RENDERER" in m for m in msgs)
