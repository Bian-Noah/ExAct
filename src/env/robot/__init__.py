"""机器人构建工厂：按 robot_config.type 分派构造 Robot 对象。

仅由 PyBulletEnv 内部调用（self.robot = build_robot(robot_config)），不对外暴露。
"""

from config.loader import RobotConfig
from env.robot.panda.panda_robot import PandaRobot
from env.robot.so101.so101_robot import SO101Robot

# type → Robot 类 注册表
_ROBOT_REGISTRY: dict[str, type] = {
    "panda": PandaRobot,
    "so101": SO101Robot,
}


def build_robot(robot_config: RobotConfig):
    """按 robot_config.type 分派构造对应 Robot 实例。

    Args:
        robot_config: 机器人配置（含 type 与特化配置）。

    Returns:
        对应 type 的 Robot 实例（如 PandaRobot / SO101Robot）。

    Raises:
        ValueError: type 不在注册表，列出可选类型。
    """
    robot_type = robot_config.type
    robot_cls = _ROBOT_REGISTRY.get(robot_type)
    if robot_cls is None:
        raise ValueError(
            f"未知机器人类型: {robot_type!r}，可选类型: {sorted(_ROBOT_REGISTRY)}"
        )
    return robot_cls(robot_config)


__all__ = ["build_robot"]
