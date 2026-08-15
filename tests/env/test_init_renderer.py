"""PyBulletEnv.__init__ 字段初始化单元测试（iter2-renderer-env-mode）。

覆盖：
- _mode / _renderer / use_gui 字段在不同 EnvConfig 组合下正确
- env_config=None 时使用默认值（向后兼容）
- use_gui=True 与 mode=direct 不一致时触发 DeprecationWarning
"""

from __future__ import annotations

import platform
import warnings

import pybullet as p
import pytest

from config.loader import EnvConfig
from env.pybullet_env import PyBulletEnv


# ============================================================
# 不同 EnvConfig 组合下的字段值
# ============================================================

def test_init_mode_direct_renderer_cpu():
    cfg = EnvConfig(mode="direct", renderer="cpu")
    env = PyBulletEnv(env_config=cfg, robot_config=None)
    assert env._mode == "direct"
    assert env._renderer == p.ER_TINY_RENDERER
    assert env.use_gui is False


def test_init_mode_gui_renderer_gpu():
    cfg = EnvConfig(mode="gui", renderer="gpu")
    env = PyBulletEnv(env_config=cfg, robot_config=None)
    assert env._mode == "gui"
    assert env._renderer == p.ER_BULLET_HARDWARE_OPENGL
    assert env.use_gui is True


def test_init_mode_direct_renderer_gpu():
    """mode=direct 但 renderer=gpu 也允许（两者正交）。"""
    cfg = EnvConfig(mode="direct", renderer="gpu")
    env = PyBulletEnv(env_config=cfg, robot_config=None)
    assert env._mode == "direct"
    assert env._renderer == p.ER_BULLET_HARDWARE_OPENGL
    assert env.use_gui is False


def test_init_mode_gui_renderer_cpu():
    """mode=gui 但 renderer=cpu 也允许。"""
    cfg = EnvConfig(mode="gui", renderer="cpu")
    env = PyBulletEnv(env_config=cfg, robot_config=None)
    assert env._mode == "gui"
    assert env._renderer == p.ER_TINY_RENDERER
    assert env.use_gui is True


def test_init_defaults_auto_renderer():
    """默认 EnvConfig → mode=direct, renderer=auto（平台相关）。"""
    env = PyBulletEnv()
    assert env._mode == "direct"
    assert env.use_gui is False
    # renderer 取决于当前平台：Apple Silicon → TINY，其他 → OPENGL
    is_apple_silicon = (
        platform.system() == "Darwin" and platform.processor() == "arm"
    )
    if is_apple_silicon:
        assert env._renderer == p.ER_TINY_RENDERER
    else:
        assert env._renderer == p.ER_BULLET_HARDWARE_OPENGL


# ============================================================
# 向后兼容
# ============================================================

def test_init_env_config_none():
    """env_config=None 时使用默认值构造。"""
    env = PyBulletEnv(env_config=None, robot_config=None)
    assert env._mode == "direct"
    assert env.use_gui is False


def test_init_both_none():
    """env_config=None + robot_config=None 双默认。"""
    env = PyBulletEnv()
    assert env.env_config is not None
    assert env.robot_config is not None


# ============================================================
# use_gui 字段兼容 + DeprecationWarning
# ============================================================

def test_init_use_gui_true_mode_gui_no_warning():
    """use_gui=True + mode=gui 一致，无警告。"""
    cfg = EnvConfig(use_gui=True, mode="gui")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        env = PyBulletEnv(env_config=cfg, robot_config=None)
    assert env.use_gui is True
    assert env._mode == "gui"
    # 一致不触发 deprecation warning
    assert not any(issubclass(warning.category, DeprecationWarning) for warning in w)


def test_init_use_gui_true_mode_direct_triggers_warning():
    """use_gui=True + mode=direct 不一致，触发 DeprecationWarning。"""
    cfg = EnvConfig(use_gui=True, mode="direct")
    with pytest.warns(DeprecationWarning, match="use_gui"):
        env = PyBulletEnv(env_config=cfg, robot_config=None)
    # 兼容性：以 use_gui 为准，自动升级为 mode=gui
    assert env.use_gui is True
    assert env._mode == "gui"


def test_init_use_gui_false_mode_gui_no_warning():
    """use_gui=False + mode=gui 一致，use_gui 被 mode 覆盖，无警告。"""
    cfg = EnvConfig(use_gui=False, mode="gui")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        env = PyBulletEnv(env_config=cfg, robot_config=None)
    assert env.use_gui is True  # 由 mode=gui 推导
    assert env._mode == "gui"
    assert not any(issubclass(warning.category, DeprecationWarning) for warning in w)
