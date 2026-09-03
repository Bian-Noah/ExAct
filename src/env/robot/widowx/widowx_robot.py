"""WidowX 机器人特化：5 臂 IK + 2 夹爪（add-widowx-robot）。

wx250.urdf 关节布局：
  joint[0-4]  5 臂关节（shoulder/elbow/wrist）
  joint[5-8]  其他（含 fixed）
  joint[9-10] gripper 夹爪 2 指

WidowX 消费 task 空间 7 维增量动作（与 Panda 同构）：
  - 前 3 维 (dx,dy,dz) 末端位移增量 → IK 转 5 臂关节角
  - 第 4-6 维旋转增量不参与 IK（pybullet 内置 IK 只解位置，与 Panda 一致）
  - 第 7 维 gripper 0/1 → 两指 [0,0]（闭）/[0.04,-0.04]（开）
"""

import numpy as np
import pybullet as p

from config.loader import RobotConfig
from env.base import ActionSpec
from env.robot.widowx.urdf_downloader import ensure_assets_downloaded, ensure_urdf_downloaded


class WidowxRobot:
    """WidowX 特化机器人（由 build_robot 实例化，PyBulletEnv 持有）。

    Args:
        robot_config: 机器人配置，读取 robot_config.widowx 特化字段。
    """

    def __init__(self, robot_config: RobotConfig):
        self.arm_joint_indices = tuple(robot_config.widowx.arm_joint_indices)
        self.ee_link_index = robot_config.widowx.ee_link_index
        self.gripper_joint_indices = tuple(robot_config.widowx.gripper_joint_indices)
        self.urdf_url = robot_config.widowx.urdf_url
        self.urdf_local_path = robot_config.widowx.urdf_local_path
        # 运行时缓存的夹爪两指限位 [(lower, upper), ...]；None=未读取
        self._gripper_limits: list[tuple[float, float]] | None = None
        # URDF 资产自举：首次使用时自动下载（本地已有则跳过）
        self.ensure_urdf()

    def ensure_urdf(self) -> None:
        """确保本地 URDF 及其引用的 mesh 资产存在（缺失则下载，幂等）。

        两步：① URDF 单文件（本地已有则跳过）；② 解析 URDF 内 package://
        引用并补齐缺失资产（.stl/.png）。仅下载 URDF 而缺 mesh 时，
        pybullet loadURDF 会失败（Cannot load URDF file），故两步都要执行。
        """
        ensure_urdf_downloaded(
            self.urdf_local_path,
            self.urdf_url,
        )
        ensure_assets_downloaded(
            self.urdf_local_path,
            self.urdf_url,
        )

    @property
    def urdf_url(self) -> str:
        return self._urdf_url

    @urdf_url.setter
    def urdf_url(self, value: str) -> None:
        self._urdf_url = value

    @property
    def urdf_local_path(self) -> str:
        return self._urdf_local_path

    @urdf_local_path.setter
    def urdf_local_path(self, value: str) -> None:
        self._urdf_local_path = value

    def home_joint_positions(self) -> tuple[float, ...]:
        """WidowX 5 臂关节 home pose（全零近似，复位起点）。"""
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def input_spec(self) -> ActionSpec:
        """WidowX 消费 task 空间 7 维增量动作（与 Panda 同构）。"""
        return ActionSpec(
            "task", ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")
        )

    def step_action(
        self,
        action,
        robot_id: int,
        client_id: int,
        on_substep=None,
    ) -> None:
        """执行一步动作：IK + 5 臂关节驱动 + 2 指夹爪 + 推进物理。

        Args:
            action: task 空间增量动作（Action7D 或 np.ndarray shape (7,)）。
                前 3 维 (dx,dy,dz) 末端位移增量（米）→ IK 转关节角；
                第 4-6 维旋转增量不参与求解（pybullet IK 只解位置，与 Panda 一致）；
                第 7 维 gripper（0~1，阈值 0.5：闭 [0,0] / 开 [0.04,-0.04]）。
            robot_id: pybullet 中该机器人的 body id。
            client_id: pybullet 连接 id。
            on_substep: iter12-video-recording 可选回调——每个物理子步
                stepSimulation 后调用 `on_substep(i)`（i 为 0-based 子步序号）。
        """
        # 兼容 Action7D（属性访问）与 ndarray（索引访问）
        if hasattr(action, "dx"):
            dx, dy, dz = action.dx, action.dy, action.dz
            gripper = action.gripper
        else:
            arr = np.asarray(action, dtype=float).ravel()
            dx, dy, dz = arr[0], arr[1], arr[2]
            gripper = arr[6] if len(arr) > 6 else 0.5

        # 1. 读当前末端位置
        link_state = p.getLinkState(
            robot_id, self.ee_link_index, physicsClientId=client_id
        )
        current_ee = link_state[0]

        # 2. 目标位置 = 当前位置 + 位移增量（z 下限保护，防穿桌）
        target_ee = [
            current_ee[0] + dx,
            current_ee[1] + dy,
            max(current_ee[2] + dz, 0.01),
        ]

        # 3. IK 求 5 臂关节角
        joint_angles = p.calculateInverseKinematics(
            robot_id,
            self.ee_link_index,
            target_ee,
            physicsClientId=client_id,
        )

        # 4. POSITION_CONTROL 驱动 5 臂关节
        p.setJointMotorControlArray(
            robot_id,
            self.arm_joint_indices,
            p.POSITION_CONTROL,
            targetPositions=joint_angles[: len(self.arm_joint_indices)],
            physicsClientId=client_id,
        )

        # 5. 夹爪两指：从 pybullet 读关节限位（首次缓存），开→外极限、闭→内极限。
        #    不硬编码 [0.04,-0.04]/[0,0]——manipulator_gym 的假设值与
        #    本项目 wx250.urdf 实际限位（left [0.015,0.037] / right [-0.037,-0.015]）不符。
        #    force/maxVelocity 参照 WidowXSimInterface.move_gripper，保证限位内快速到位。
        if self._gripper_limits is None:
            limits: list[tuple[float, float]] = []
            for joint_idx in self.gripper_joint_indices:
                info = p.getJointInfo(robot_id, joint_idx, physicsClientId=client_id)
                limits.append((float(info[8]), float(info[9])))  # lower, upper
            self._gripper_limits = limits
        if gripper < 0.5:
            # 闭：左指下限 + 右指上限（两指靠拢）
            grip_positions = [self._gripper_limits[0][0], self._gripper_limits[1][1]]
        else:
            # 开：左指上限 + 右指下限（两指张开）
            grip_positions = [self._gripper_limits[0][1], self._gripper_limits[1][0]]
        for joint_idx, grip_pos in zip(self.gripper_joint_indices, grip_positions):
            p.setJointMotorControl2(
                robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=grip_pos,
                force=1000.0,
                maxVelocity=50.0,
                physicsClientId=client_id,
            )

        # 6. 收敛循环推进物理（参照 SO101：轮询误差 < 1e-3 rad，max_steps 兜底）
        #    min_steps 保证：即使臂目标==当前位置（IK 误差≈0，如纯 gripper 动作），
        #    也至少推进固定步数让夹爪 POSITION_CONTROL 到位，避免 1 步即 break。
        converge_tol = 1e-3
        min_steps = 60   # 两指 0→[0.04,-0.04] 行程所需物理步数（@240Hz ≈ 250ms）
        max_steps = 300
        for step in range(max_steps):
            p.stepSimulation(physicsClientId=client_id)
            if on_substep is not None:
                on_substep(step)
            if step >= min_steps and step % 5 == 0:
                states = p.getJointStates(
                    robot_id, self.arm_joint_indices, physicsClientId=client_id
                )
                err = max(
                    abs(s[0] - t)
                    for s, t in zip(states, joint_angles[: len(self.arm_joint_indices)])
                )
                if err < converge_tol:
                    break

    def get_joint_state(self, robot_id: int, client_id: int) -> np.ndarray:
        """读取当前关节角（5 臂 + 1 gripper）作为 VLA 的 state 输入。

        Returns:
            np.ndarray, shape=(6,): 5 臂关节 + 1 gripper（取第一指关节角），
            按 URDF 关节索引顺序。
        """
        states = p.getJointStates(
            robot_id, self.arm_joint_indices, physicsClientId=client_id
        )
        arm = [s[0] for s in states]
        gripper_state = p.getJointState(
            robot_id, self.gripper_joint_indices[0], physicsClientId=client_id
        )
        return np.array(list(arm) + [gripper_state[0]], dtype=float)
