"""SO101 机器人特化：单臂 5 自由度 + 1 夹爪。

真实 URDF（so101_new_calib.urdf）关节布局：
  joint[0-4]  shoulder_pan/shoulder_lift/elbow_flex/wrist_flex/wrist_roll（臂关节）
  joint[5]    gripper_frame_joint（固定，不可动）
  joint[6]    gripper（夹爪铰链）

SO101 消费 joint 空间 6 维动作（adapter 已把 VLA 输出转换为 env 原生动作）：
  - 前 5 维 → POSITION_CONTROL 驱动 5 臂关节
  - 第 6 维 → 驱动 gripper 关节
  - stepSimulation 推进物理
"""

import pybullet as p

from config.loader import RobotConfig
from env.base import ActionSpec


class SO101Robot:
    """SO101 特化机器人（由 build_robot 实例化，PyBulletEnv 持有）。

    Args:
        robot_config: 机器人配置，读取 robot_config.so101 特化字段。
    """

    def __init__(self, robot_config: RobotConfig):
        self.arm_joint_indices = tuple(robot_config.so101.arm_joint_indices)
        self.ee_link_index = robot_config.so101.ee_link_index
        self.gripper_joint_index = robot_config.so101.gripper_joint_index

    @property
    def input_spec(self) -> ActionSpec:
        """SO101 消费 joint 空间 6 维动作（5 臂 + gripper）。"""
        return ActionSpec("joint", ("joint",) * 6)

    def step_action(self, action, robot_id: int, client_id: int) -> None:
        """执行一步动作：POSITION_CONTROL 驱动 5 臂关节 + gripper + 推进物理。

        Args:
            action: joint 空间动作（np.ndarray 或类似可索引容器，前 5 维为臂关节角，
                第 6 维为 gripper 开合）。
            robot_id: pybullet 中该机器人的 body id。
            client_id: pybullet 连接 id。
        """
        joints = self.arm_joint_indices
        target_positions = list(action)[: len(joints)]

        p.setJointMotorControlArray(
            robot_id,
            joints,
            p.POSITION_CONTROL,
            targetPositions=target_positions,
            physicsClientId=client_id,
        )

        # gripper 关节（若动作有第 6 维）
        if len(action) > len(joints):
            gripper_pos = float(action[len(joints)])
            p.setJointMotorControl2(
                robot_id,
                self.gripper_joint_index,
                p.POSITION_CONTROL,
                targetPosition=gripper_pos,
                physicsClientId=client_id,
            )

        for _ in range(10):
            p.stepSimulation(physicsClientId=client_id)
