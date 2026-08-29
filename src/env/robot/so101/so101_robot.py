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

import numpy as np
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

    def home_joint_positions(self) -> tuple[float, ...]:
        """iter11-reset-multicam:SO101 5 臂关节 home pose。

        全零位姿（臂关节在 home 位姿附近,可作为复位起点）。
        仅覆盖 arm_joint_indices 对应的臂关节,不含 gripper——
        gripper 由 step_action 独立控制,复位只动臂。
        """
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def input_spec(self) -> ActionSpec:
        """SO101 消费 joint 空间 6 维动作（5 臂 + gripper）。"""
        return ActionSpec("joint", ("joint",) * 6)

    def step_action(
        self,
        action,
        robot_id: int,
        client_id: int,
        on_substep=None,
    ) -> None:
        """执行一步动作：POSITION_CONTROL 驱动 5 臂关节 + gripper + 推进物理。

        Args:
            action: joint 空间动作（np.ndarray 或类似可索引容器，前 5 维为**绝对关节角
                目标** rad，第 6 维为 gripper 开合 0~1）。
                VLA（SmolVLA / LeRobot 训练约定 action_space="joint_angle"）直接输出
                目标关节角，env 原样作为 POSITION_CONTROL 的 targetPositions 使用。
                注意：JointMockVLA 输出的 [-0.1, 0.1] rad 小幅值在绝对语义下会成为
                "目标 ≈ 0" 的近零动作，mock 不再产生大幅可见运动 — 这是为对齐真 VLA
                语义必须接受的副作用（详见 docstring 外的 TODO 注释）。
            robot_id: pybullet 中该机器人的 body id。
            client_id: pybullet 连接 id。
            on_substep: iter12-video-recording 可选回调——每个物理子步
                stepSimulation 后调用 `on_substep(i)`（i 为 0-based 子步序号）；
                默认 None 保持原行为。
        """
        # TODO(增量 vs 绝对值 语义切换):
        #     现状：step_action 把 action 当作**绝对关节角**(VLA 真模型对齐)。
        #     副作用：JointMockVLA 仍按"增量 [-0.1, 0.1] rad"输出，在绝对语义下被解读
        #     为"目标 ≈ 0 rad"的近零动作，mock 测试不再产生可见运动。
        #     后续处理方向（任选其一）：
        #       1) JointMockVLA 改成输出绝对关节角范围（与训练数据 stats 一致）
        #       2) 在 SO101Robot / joint_to_joint adapter 加"语义标志位"，区分
        #          增量源(mock)与绝对源(VLA)，各自按需解读
        #       3) 把"绝对 vs 增量"信息加入 ActionSpec(spec 加字段)，env 按 spec 分派
        #     关联问题：增量 ↔ 绝对值方向是否需要保留为 env 行为可配置项？需后续讨论。
        target_positions = list(action)[: len(self.arm_joint_indices)]

        p.setJointMotorControlArray(
            robot_id,
            self.arm_joint_indices,
            p.POSITION_CONTROL,
            targetPositions=target_positions,
            physicsClientId=client_id,
        )

        # gripper 关节（绝对位置控制；第 6 维 0~1 直接当目标）
        if len(action) > len(self.arm_joint_indices):
            gripper_pos = float(action[len(self.arm_joint_indices)])
            p.setJointMotorControl2(
                robot_id,
                self.gripper_joint_index,
                p.POSITION_CONTROL,
                targetPosition=gripper_pos,
                physicsClientId=client_id,
            )

        # 推进物理直至 5 臂关节收敛到目标（部分执行 → 全部执行）。
        # POSITION_CONTROL 靠 PD 逐步逼近，固定步数（如 10）会中途返回导致
        # 关节只走一部分、末端位移偏小；改为轮询关节误差直至收敛，超时兜底。
        # iter12-video-recording:每子步 stepSimulation 后回调 on_substep。
        converge_tol = 1e-3  # 关节误差阈值 rad（≈0.057°）
        max_steps = 300      # 兜底上限（300 步 ≈ 1.25s @240Hz），防限位/卡死死循环
        for step in range(max_steps):
            p.stepSimulation(physicsClientId=client_id)
            if on_substep is not None:
                on_substep(step)
            if step % 5 == 0:  # 每 5 步查一次误差，降低 getJointStates 开销
                states = p.getJointStates(
                    robot_id, self.arm_joint_indices, physicsClientId=client_id
                )
                err = max(abs(s[0] - t) for s, t in zip(states, target_positions))
                if err < converge_tol:
                    break

    def get_joint_state(self, robot_id: int, client_id: int) -> np.ndarray:
        """读取 SO101 当前关节角（含 gripper）作为 VLA 的 state 输入。

        Returns:
            np.ndarray, shape=(6,): 5 臂关节 + 1 gripper，按 URDF 关节索引顺序。
        """
        states = p.getJointStates(
            robot_id, self.arm_joint_indices, physicsClientId=client_id
        )
        arm = [s[0] for s in states]
        gripper_state = p.getJointState(
            robot_id, self.gripper_joint_index, physicsClientId=client_id
        )
        return np.array(list(arm) + [gripper_state[0]], dtype=float)
