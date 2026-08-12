"""PyBullet 连接检查和 close 安全断开的单元测试。"""

import pytest
from unittest.mock import patch, MagicMock

from env.pybullet_env import PyBulletPandaEnv


class TestIsConnected:
    """_is_connected 方法测试。"""

    def test_connected_returns_true(self):
        """正常连接时返回 True。"""
        env = PyBulletPandaEnv(use_gui=False)
        env._client_id = 0
        with patch("env.pybullet_env.p.getConnectionInfo", return_value={"isConnected": True}):
            assert env._is_connected() is True

    def test_client_id_negative_returns_false(self):
        """client_id < 0 时返回 False。"""
        env = PyBulletPandaEnv(use_gui=False)
        env._client_id = -1
        assert env._is_connected() is False

    def test_exception_returns_false(self):
        """getConnectionInfo 抛异常时返回 False。"""
        env = PyBulletPandaEnv(use_gui=False)
        env._client_id = 0
        with patch("env.pybullet_env.p.getConnectionInfo", side_effect=Exception("disconnected")):
            assert env._is_connected() is False

    def test_isConnected_false_returns_false(self):
        """getConnectionInfo 返回 isConnected=False 时返回 False。"""
        env = PyBulletPandaEnv(use_gui=False)
        env._client_id = 0
        with patch("env.pybullet_env.p.getConnectionInfo", return_value={"isConnected": False}):
            assert env._is_connected() is False


class TestEnsureConnected:
    """_ensure_connected 方法测试。"""

    def test_connected_does_not_raise(self):
        """正常连接时不抛异常。"""
        env = PyBulletPandaEnv(use_gui=False)
        env._client_id = 0
        with patch("env.pybullet_env.p.getConnectionInfo", return_value={"isConnected": True}):
            env._ensure_connected()  # 不应抛异常

    def test_disconnected_raises_runtime_error(self):
        """断连时抛 RuntimeError。"""
        env = PyBulletPandaEnv(use_gui=False)
        env._client_id = -1
        with pytest.raises(RuntimeError, match="PyBullet physics server not connected"):
            env._ensure_connected()


class TestClose:
    """close 方法测试。"""

    def test_close_normal_disconnect(self):
        """正常断开时 _client_id 置 -1。"""
        env = PyBulletPandaEnv(use_gui=False)
        env._client_id = 0
        with patch("env.pybullet_env.p.disconnect") as mock_disconnect:
            env.close()
            mock_disconnect.assert_called_once_with(0)
            assert env._client_id == -1

    def test_close_already_disconnected_no_error(self):
        """已断连时 close 不报错，_client_id 置 -1。"""
        env = PyBulletPandaEnv(use_gui=False)
        env._client_id = 0
        with patch("env.pybullet_env.p.disconnect", side_effect=Exception("Not connected")):
            env.close()  # 不应抛异常
            assert env._client_id == -1

    def test_close_when_client_id_negative(self):
        """client_id 已为 -1 时 close 不调用 disconnect。"""
        env = PyBulletPandaEnv(use_gui=False)
        env._client_id = -1
        with patch("env.pybullet_env.p.disconnect") as mock_disconnect:
            env.close()
            mock_disconnect.assert_not_called()
            assert env._client_id == -1
