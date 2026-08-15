"""机器人构建工厂：按 robot_config.type 分派构造 Robot 对象。

仅由 PyBulletEnv 内部调用（self.robot = build_robot(robot_config)），不对外暴露。
"""

from config.loader import RobotConfig
from env.robot.panda.panda_robot import PandaRobot

# type → Robot 类 注册表
_ROBOT_REGISTRY: dict[str, type] = {
    "panda": PandaRobot,
}


def build_robot(robot_config: RobotConfig):
    """按 robot_config.type 分派构造对应 Robot 实例。

    Args:
        robot_config: 机器人配置（含 type 与特化配置）。

    Returns:
        对应 type 的 Robot 实例（如 PandaRobot）。

    Raises:
        NotImplementedError: type 为已预留但未实现的类型（如 so101）。
        ValueError: type 不在注册表，列出可选类型。
    """
    robot_type = robot_config.type
    robot_cls = _ROBOT_REGISTRY.get(robot_type)
    if robot_cls is None:
        if robot_type == "so101":
            raise NotImplementedError(
                "SO101 机器人尚未实现（robot-vla-adapter 本迭代只留骨架），"
                "后续迭代填充后即可用。"
            )
        raise ValueError(
            f"未知机器人类型: {robot_type!r}，可选类型: {sorted(_ROBOT_REGISTRY)}"
        )
    return robot_cls(robot_config)


__all__ = ["build_robot"]
