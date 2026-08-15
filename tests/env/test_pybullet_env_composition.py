"""PyBulletEnv 通用化单元测试（robot-vla-adapter 任务 5）。

验证：
- PyBulletEnv(env_config, robot_config) 构造后 self.robot 是 PandaRobot
- step() 委托给 self.robot.step_action（用 FakeRobot 注入验证）
"""

from config.loader import EnvConfig, RobotConfig
from env.pybullet_env import PyBulletEnv
from env.robot.panda import PandaRobot


def test_pybullet_env_injects_panda_robot():
    """构造后 self.robot 是 PandaRobot。"""
    env = PyBulletEnv(
        env_config=EnvConfig(mode="direct"),
        robot_config=RobotConfig(),
    )
    assert isinstance(env.robot, PandaRobot)
    env.close()


def test_step_delegates_to_robot():
    """step() 委托 self.robot.step_action（FakeRobot 注入验证）。"""
    env = PyBulletEnv(
        env_config=EnvConfig(mode="direct"),
        robot_config=RobotConfig(),
    )

    calls = []

    class FakeRobot:
        def step_action(self, action, robot_id, client_id):
            calls.append((action, robot_id, client_id))

    env.robot = FakeRobot()
    env._ensure_connected = lambda: None
    env.get_obs = lambda: {"ee_pos": (0.0, 0.0, 0.0)}
    env._robot_id = 7
    env._client_id = 9

    obs, reward, done, info = env.step("dummy_action")

    assert calls == [("dummy_action", 7, 9)]
    assert obs == {"ee_pos": (0.0, 0.0, 0.0)}
    env.close()
