"""PyBulletEnv._resolve_renderer 单元测试（iter2-renderer-env-mode）。

覆盖：
- 显式 "cpu" / "gpu" 映射
- "auto" 在不同平台下的自动判断（mock platform）
- 非法值抛 ValueError
"""

from __future__ import annotations

from unittest.mock import patch

import pybullet as p
import pytest

from env.pybullet_env import PyBulletEnv


# ============================================================
# 显式映射
# ============================================================

def test_resolve_renderer_cpu_returns_tiny():
    """'cpu' → p.ER_TINY_RENDERER。"""
    assert PyBulletEnv._resolve_renderer("cpu") == p.ER_TINY_RENDERER


def test_resolve_renderer_gpu_returns_opengl():
    """'gpu' → p.ER_BULLET_HARDWARE_OPENGL。"""
    assert PyBulletEnv._resolve_renderer("gpu") == p.ER_BULLET_HARDWARE_OPENGL


# ============================================================
# auto 模式 + 平台判断（mock）
# ============================================================

def test_resolve_renderer_auto_apple_silicon():
    """Apple Silicon (Darwin + arm) → TINY_RENDERER。"""
    with patch("env.pybullet_env.platform") as mock_platform:
        mock_platform.system.return_value = "Darwin"
        mock_platform.processor.return_value = "arm"
        assert PyBulletEnv._resolve_renderer("auto") == p.ER_TINY_RENDERER


def test_resolve_renderer_auto_linux_x86():
    """Linux + x86_64 → BULLET_HARDWARE_OPENGL。"""
    with patch("env.pybullet_env.platform") as mock_platform:
        mock_platform.system.return_value = "Linux"
        mock_platform.processor.return_value = "x86_64"
        assert PyBulletEnv._resolve_renderer("auto") == p.ER_BULLET_HARDWARE_OPENGL


def test_resolve_renderer_auto_intel_mac():
    """Intel Mac (Darwin + x86_64) → BULLET_HARDWARE_OPENGL。"""
    with patch("env.pybullet_env.platform") as mock_platform:
        mock_platform.system.return_value = "Darwin"
        mock_platform.processor.return_value = "x86_64"
        assert PyBulletEnv._resolve_renderer("auto") == p.ER_BULLET_HARDWARE_OPENGL


def test_resolve_renderer_auto_windows():
    """Windows + AMD64 → BULLET_HARDWARE_OPENGL。"""
    with patch("env.pybullet_env.platform") as mock_platform:
        mock_platform.system.return_value = "Windows"
        mock_platform.processor.return_value = "AMD64"
        assert PyBulletEnv._resolve_renderer("auto") == p.ER_BULLET_HARDWARE_OPENGL


# ============================================================
# 非法值
# ============================================================

@pytest.mark.parametrize("bad_value", ["vulkan", "fast", "null", "DIRECT", "GPU"])
def test_resolve_renderer_invalid_value_raises(bad_value: str):
    """非法值抛 ValueError，错误信息包含 'renderer' 与非法值。"""
    with pytest.raises(ValueError) as exc_info:
        PyBulletEnv._resolve_renderer(bad_value)
    msg = str(exc_info.value)
    assert "renderer" in msg.lower()
    assert bad_value in msg or repr(bad_value) in msg
