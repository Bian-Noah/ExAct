"""PyBullet 连接检查和 close 安全断开的单元测试。

iter1-pipeline-refactor-config 后：构造签名改为
`PyBulletPandaEnv(env_config: EnvConfig, robot_config: RobotConfig)`，
模块级 ARM_JOINT_INDICES / EE_LINK_INDEX / FINGER_JOINT_INDICES 保留作
RobotConfig 默认值的别名引用。
"""

import pytest
from unittest.mock import patch, MagicMock

from config.loader import EnvConfig, RobotConfig
from env.pybullet_env import (
    PyBulletPandaEnv,
    ARM_JOINT_INDICES,
    EE_LINK_INDEX,
    FINGER_JOINT_INDICES,
)


def _make_env():
    return PyBulletPandaEnv(
        env_config=EnvConfig(use_gui=False),
        robot_config=RobotConfig(),
    )


class TestModuleLevelConstants:
    """模块级常量（与 RobotConfig 默认值一致）。"""

    def test_arm_joint_indices_matches_default(self):
        assert ARM_JOINT_INDICES == (0, 1, 2, 3, 4, 5, 6)

    def test_ee_link_index_matches_default(self):
        assert EE_LINK_INDEX == 11

    def test_finger_joint_indices_matches_default(self):
        assert FINGER_JOINT_INDICES == (9, 10)


class TestConstruction:
    """构造签名变化测试。"""

    def test_accepts_env_and_robot_config(self):
        env = PyBulletPandaEnv(
            env_config=EnvConfig(use_gui=False),
            robot_config=RobotConfig(),
        )
        assert env.use_gui is False
        assert env.camera_resolution == (640, 480)
        assert env.robot_config.urdf_path == "franka_panda/panda.urdf"
        assert env.robot_config.arm_joint_indices == (0, 1, 2, 3, 4, 5, 6)
        assert env.robot_config.ee_link_index == 11

    def test_accepts_none_uses_defaults(self):
        env = PyBulletPandaEnv()
        # iter2: 默认 mode=direct, use_gui=False
        assert env.use_gui is False
        assert env._mode == "direct"
        assert env.camera_resolution == (640, 480)
        assert env.robot_config.arm_joint_indices == (0, 1, 2, 3, 4, 5, 6)

    def test_accepts_custom_robot_config(self):
        custom = RobotConfig(urdf_path="custom/panda.urdf", ee_link_index=7)
        env = PyBulletPandaEnv(env_config=EnvConfig(), robot_config=custom)
        assert env.robot_config.urdf_path == "custom/panda.urdf"
        assert env.robot_config.ee_link_index == 7


class TestIsConnected:
    """_is_connected 方法测试。"""

    def test_connected_returns_true(self):
        env = _make_env()
        env._client_id = 0
        with patch("env.pybullet_env.p.getConnectionInfo", return_value={"isConnected": True}):
            assert env._is_connected() is True

    def test_client_id_negative_returns_false(self):
        env = _make_env()
        env._client_id = -1
        assert env._is_connected() is False

    def test_exception_returns_false(self):
        env = _make_env()
        env._client_id = 0
        with patch("env.pybullet_env.p.getConnectionInfo", side_effect=Exception("disconnected")):
            assert env._is_connected() is False

    def test_isConnected_false_returns_false(self):
        env = _make_env()
        env._client_id = 0
        with patch("env.pybullet_env.p.getConnectionInfo", return_value={"isConnected": False}):
            assert env._is_connected() is False


class TestEnsureConnected:
    """_ensure_connected 方法测试。"""

    def test_connected_does_not_raise(self):
        env = _make_env()
        env._client_id = 0
        with patch("env.pybullet_env.p.getConnectionInfo", return_value={"isConnected": True}):
            env._ensure_connected()

    def test_disconnected_raises_runtime_error(self):
        env = _make_env()
        env._client_id = -1
        with pytest.raises(RuntimeError, match="PyBullet physics server not connected"):
            env._ensure_connected()


class TestClose:
    """close 方法测试。"""

    def test_close_normal_disconnect(self):
        env = _make_env()
        env._client_id = 0
        with patch("env.pybullet_env.p.disconnect") as mock_disconnect:
            env.close()
            mock_disconnect.assert_called_once_with(0)
            assert env._client_id == -1

    def test_close_already_disconnected_no_error(self):
        env = _make_env()
        env._client_id = 0
        with patch("env.pybullet_env.p.disconnect", side_effect=Exception("Not connected")):
            env.close()
            assert env._client_id == -1

    def test_close_when_client_id_negative(self):
        env = _make_env()
        env._client_id = -1
        with patch("env.pybullet_env.p.disconnect") as mock_disconnect:
            env.close()
            mock_disconnect.assert_not_called()
            assert env._client_id == -1


class TestResetUsesRobotConfig:
    """reset() 使用 robot_config.urdf_path / ee_link_index。"""

    def test_urdf_path_from_robot_config(self):
        """robot_config.urdf_path 被读取。"""
        env = PyBulletPandaEnv(
            env_config=EnvConfig(use_gui=False),
            robot_config=RobotConfig(urdf_path="custom/panda.urdf"),
        )
        assert env.robot_config.urdf_path == "custom/panda.urdf"

    def test_ee_link_index_from_robot_config(self):
        """robot_config.ee_link_index 被读取。"""
        env = PyBulletPandaEnv(
            env_config=EnvConfig(use_gui=False),
            robot_config=RobotConfig(ee_link_index=7),
        )
        assert env.robot_config.ee_link_index == 7

    def test_arm_joint_indices_from_robot_config(self):
        """robot_config.arm_joint_indices 被读取。"""
        env = PyBulletPandaEnv(
            env_config=EnvConfig(use_gui=False),
            robot_config=RobotConfig(arm_joint_indices=(1, 2, 3, 4, 5, 6, 7)),
        )
        assert env.robot_config.arm_joint_indices == (1, 2, 3, 4, 5, 6, 7)
