"""PyBulletEnv.reset 连接模式单元测试（iter2-renderer-env-mode）。

覆盖：
- _resolve_connection_mode 静态方法：direct / gui 映射
- mode="direct" 时 reset 调 p.connect(p.DIRECT)
- mode="gui" 时 reset 调 p.connect(p.GUI)
- 多次 reset 切换 mode 时各自使用正确模式

策略：mock 整个 env.pybullet_env.p 模块（spec=pybullet 保留所有常量与 API），
让 reset 的下游调用全部短路为 Mock，仅断言 p.connect 的调用参数。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pybullet as p
import pytest

from config.loader import EnvConfig
from env.pybullet_env import PyBulletEnv


# ============================================================
# _resolve_connection_mode 静态方法
# ============================================================

def test_resolve_connection_mode_direct():
    assert PyBulletEnv._resolve_connection_mode("direct") == p.DIRECT


def test_resolve_connection_mode_gui():
    assert PyBulletEnv._resolve_connection_mode("gui") == p.GUI


def test_resolve_connection_mode_invalid():
    with pytest.raises(ValueError) as exc_info:
        PyBulletEnv._resolve_connection_mode("headless")
    assert "headless" in str(exc_info.value)


# ============================================================
# reset 调用 p.connect 时使用正确的连接模式
# ============================================================

def _make_mock_p():
    """构造 spec=pybullet 的 MagicMock。

    注意：mock_p.DIRECT / mock_p.GUI 等常量被赋值为真实 pybullet 常量，
    这样模块内 `p.DIRECT` / `p.GUI` 引用在 mock 期间仍返回真实值。
    """
    import numpy as np
    mock_p = MagicMock(spec=p)
    # 让常量属性返回真实值
    mock_p.DIRECT = p.DIRECT
    mock_p.GUI = p.GUI
    mock_p.ER_TINY_RENDERER = p.ER_TINY_RENDERER
    mock_p.ER_BULLET_HARDWARE_OPENGL = p.ER_BULLET_HARDWARE_OPENGL
    mock_p.connect.return_value = 999  # 假 client_id
    mock_p.disconnect.return_value = None
    # 让 getLinkState 返回合法元组
    mock_p.getLinkState.return_value = ((0.5, 0.0, 0.5), (0, 0, 0, 1), None, None, None, None, None)
    mock_p.getBasePositionAndOrientation.return_value = ((0.5, 0.0, 0.1), (0, 0, 0, 1))
    # getCameraImage 返回 5 元组（width, height, rgb_pixels, depth, seg）
    width, height = 640, 480
    px = [[[128, 128, 128, 255]] * width] * height
    mock_p.getCameraImage.return_value = (width, height, px, None, None)
    return mock_p


def test_reset_mode_direct_uses_direct_connect():
    env = PyBulletEnv(env_config=EnvConfig(mode="direct", renderer="auto"))
    mock_p = _make_mock_p()
    with patch("env.pybullet_env.p", mock_p):
        env.reset(task_spec={"objects": []})
    assert mock_p.connect.called
    assert mock_p.connect.call_args[0][0] == p.DIRECT


def test_reset_mode_gui_uses_gui_connect():
    env = PyBulletEnv(env_config=EnvConfig(mode="gui", renderer="auto"))
    mock_p = _make_mock_p()
    with patch("env.pybullet_env.p", mock_p):
        env.reset(task_spec={"objects": []})
    assert mock_p.connect.called
    assert mock_p.connect.call_args[0][0] == p.GUI


def test_reset_mode_switch_uses_correct_mode():
    """先 direct 后 gui，验证两次 reset 各自使用正确模式。"""
    env = PyBulletEnv(env_config=EnvConfig(mode="direct", renderer="auto"))
    mock_p = _make_mock_p()
    with patch("env.pybullet_env.p", mock_p):
        env.reset(task_spec={"objects": []})
        assert mock_p.connect.call_args_list[0][0][0] == p.DIRECT
        # 切换 mode
        env._mode = "gui"
        env.reset(task_spec={"objects": []})
        assert mock_p.connect.call_args_list[1][0][0] == p.GUI
