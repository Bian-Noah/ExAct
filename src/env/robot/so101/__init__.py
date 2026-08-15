"""SO101 机器人：单臂 6 自由度，无独立 gripper。

在 build_robot 注册表中以 "so101" 分派。
"""

from env.robot.so101.so101_robot import SO101Robot

__all__ = ["SO101Robot"]
