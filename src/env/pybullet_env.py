"""PyBullet Panda 仿真环境实现。

封装 PyBullet 物理引擎，提供 Franka Panda 机械臂的仿真环境。
运行方式：PYTHONPATH=src python src/app.py

iter1-pipeline-refactor-config 后：
- 构造签名改为 `PyBulletPandaEnv(env_config: EnvConfig, robot_config: RobotConfig)`
- 内部 `ARM_JOINT_INDICES` / `EE_LINK_INDEX` / `FINGER_JOINT_INDICES` 改为从 `robot_config` 读取
- 模块级常量保留作 `RobotConfig` 默认值的别名引用
"""

import time

import numpy as np
import pybullet as p
import pybullet_data

from config.loader import EnvConfig, RobotConfig
from env.base import BaseEnv, Action7D
from utils.logging import setup_logging

logger = setup_logging(__name__)

# Panda 机械臂 7 个关节索引（模块级别名，引用 RobotConfig 默认值）
ARM_JOINT_INDICES = tuple(RobotConfig().arm_joint_indices)
# 末端执行器 link 索引（panda_hand）
EE_LINK_INDEX = RobotConfig().ee_link_index
# 夹爪关节索引
FINGER_JOINT_INDICES = tuple(RobotConfig().finger_joint_indices)


class PyBulletPandaEnv(BaseEnv):
    """基于 PyBullet 的 Franka Panda 仿真环境。

    Args:
        env_config: 环境配置（use_gui / camera_resolution）。
        robot_config: 机械臂配置（URDF 路径 / 关节索引常量 / 底座位置）。
    """

    def __init__(
        self,
        env_config: EnvConfig | None = None,
        robot_config: RobotConfig | None = None,
    ):
        # 兼容旧调用：env_config 为 None 时使用默认值
        if env_config is None:
            env_config = EnvConfig()
        if robot_config is None:
            robot_config = RobotConfig()

        self.env_config = env_config
        self.robot_config = robot_config
        self.use_gui = env_config.use_gui
        self.camera_resolution = env_config.camera_resolution

        # 内部字段
        self._client_id = -1
        self._robot_id = None
        self._object_ids: list[int] = []
        self._plane_id = None

    def _is_connected(self) -> bool:
        """检查 PyBullet 物理服务器是否仍处于连接状态。"""
        if self._client_id < 0:
            return False
        try:
            info = p.getConnectionInfo(self._client_id)
            return info.get("isConnected", False)
        except Exception:
            return False

    def _ensure_connected(self) -> None:
        """确保物理服务器已连接，断连时抛出 RuntimeError。"""
        if not self._is_connected():
            raise RuntimeError("PyBullet physics server not connected")

    def reset(self, task_spec: dict, seed: int = 0) -> dict:
        """重置环境到初始状态。

        Args:
            task_spec: 任务描述字典，格式
                {"objects": [{"type":"cube","pos":[x,y,z],"color":"red"}, ...]}
            seed: 随机种子（预留，当前不使用）。

        Returns:
            obs dict，见观测字典契约。
        """
        # 断开旧连接，建立新连接
        if self._client_id >= 0:
            p.disconnect(self._client_id)

        connection_mode = p.GUI if self.use_gui else p.DIRECT
        self._client_id = p.connect(connection_mode)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self._client_id
        )

        # 重置仿真
        p.resetSimulation(physicsClientId=self._client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self._client_id)

        # 加载地面
        self._plane_id = p.loadURDF(
            "plane.urdf", physicsClientId=self._client_id
        )

        # 加载机械臂（URDF 路径与底座位置从 robot_config 读取）
        self._robot_id = p.loadURDF(
            self.robot_config.urdf_path,
            basePosition=list(self.robot_config.base_position),
            useFixedBase=True,
            physicsClientId=self._client_id,
        )

        # 加载任务物体
        self._object_ids = []
        objects = task_spec.get("objects", [])
        color_map = {
            "red": [1, 0, 0, 1],
            "green": [0, 1, 0, 1],
            "blue": [0, 0, 1, 1],
            "yellow": [1, 1, 0, 1],
        }
        for obj in objects:
            pos = obj.get("pos", [0.5, 0, 0.1])
            color = obj.get("color", "red")
            rgba = color_map.get(color, [1, 0, 0, 1])
            cube_id = p.loadURDF(
                "cube_small.urdf",
                basePosition=pos,
                physicsClientId=self._client_id,
            )
            p.changeVisualShape(
                cube_id, -1, rgbaColor=rgba, physicsClientId=self._client_id
            )
            self._object_ids.append(cube_id)

        # 等待物体稳定
        time.sleep(0.5)

        return self.get_obs()

    def step(self, action: Action7D) -> tuple[dict, float, bool, dict]:
        """执行一步动作。

        通过逆运动学计算关节角度，并推进物理仿真。

        Returns:
            (obs, reward, done, info) — 当前 reward=0.0，done=False，info={}。
        """
        self._ensure_connected()
        # 关节索引常量从 robot_config 读取（兼容字段名）
        arm_indices = self.robot_config.arm_joint_indices
        ee_link_idx = self.robot_config.ee_link_index
        finger_indices = self.robot_config.finger_joint_indices

        # 获取当前末端位置
        link_state = p.getLinkState(
            self._robot_id,
            ee_link_idx,
            physicsClientId=self._client_id,
        )
        current_ee = link_state[0]

        # 计算目标位置（当前位置 + 位移增量）
        target_ee = [
            current_ee[0] + action.dx,
            current_ee[1] + action.dy,
            current_ee[2] + action.dz,
        ]

        # 逆运动学求解关节角度
        joint_angles = p.calculateInverseKinematics(
            self._robot_id,
            ee_link_idx,
            target_ee,
            physicsClientId=self._client_id,
        )

        # 控制机械臂关节
        p.setJointMotorControlArray(
            self._robot_id,
            arm_indices,
            p.POSITION_CONTROL,
            targetPositions=joint_angles[:len(arm_indices)],
            physicsClientId=self._client_id,
        )

        # 控制夹爪（gripper: 0=闭, 1=开 → 关节角度 0~0.04）
        gripper_pos = 0.04 * action.gripper
        p.setJointMotorControlArray(
            self._robot_id,
            finger_indices,
            p.POSITION_CONTROL,
            targetPositions=[gripper_pos, gripper_pos],
            physicsClientId=self._client_id,
        )

        # 推进物理仿真
        for _ in range(10):
            p.stepSimulation(physicsClientId=self._client_id)

        return self.get_obs(), 0.0, False, {}

    def render(self) -> np.ndarray:
        """返回当前相机 RGB 图像。

        Returns:
            shape=(H, W, 3), dtype=uint8，范围 [0, 255]。
        """
        logger.info("render() 开始 — 调用 p.getCameraImage (GPU)")
        self._ensure_connected()
        width, height = self.camera_resolution

        # 相机参数
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.5, 0, 0.5],
            distance=1.5,
            yaw=50,
            pitch=-35,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width / height,
            nearVal=0.1,
            farVal=100.0,
        )

        # 获取相机图像
        (_, _, px, _, _) = p.getCameraImage(
            width,
            height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            physicsClientId=self._client_id,
        )

        # RGBA → RGB
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = rgb_array.reshape((height, width, 4))[:, :, :3]

        logger.info("render() 完成")
        return rgb_array

    def get_obs(self, include_rgb: bool = True) -> dict:
        """返回当前观测，不推进物理。

        Args:
            include_rgb: 是否包含 RGB 图像。False 时跳过 GPU 渲染，
                适用于仅需 object_info/ee_pos 的轻量调用。

        Returns:
            obs dict，包含 rgb/object_info/ee_pos/state_desc。
        """
        self._ensure_connected()
        ee_link_idx = self.robot_config.ee_link_index
        # 末端位置
        link_state = p.getLinkState(
            self._robot_id,
            ee_link_idx,
            physicsClientId=self._client_id,
        )
        ee_pos = tuple(link_state[0])

        # 物体信息
        object_info = []
        for obj_id in self._object_ids:
            pos, quat = p.getBasePositionAndOrientation(
                obj_id, physicsClientId=self._client_id
            )
            object_info.append(
                {
                    "id": obj_id,
                    "name": "cube",
                    "pos": list(pos),
                    "quat": list(quat),
                }
            )

        # 状态描述
        state_desc = (
            f"场景中{len(self._object_ids)}个物体，"
            f"末端在({ee_pos[0]:.2f},{ee_pos[1]:.2f},{ee_pos[2]:.2f})"
        )

        obs = {
            "object_info": object_info,
            "ee_pos": ee_pos,
            "state_desc": state_desc,
        }

        if include_rgb:
            obs["rgb"] = self.render()
        else:
            obs["rgb"] = None

        return obs

    def close(self) -> None:
        """释放所有资源，断开仿真连接。"""
        if self._client_id >= 0:
            try:
                p.disconnect(self._client_id)
                logger.info("PyBullet 仿真已断开")
            except Exception:
                pass  # 已断连，忽略
            finally:
                self._client_id = -1