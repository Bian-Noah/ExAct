"""通用 PyBullet 仿真环境实现（robot-vla-adapter）。

封装 PyBullet 物理引擎，提供通用仿真能力（连接/渲染/相机/物体/观测），
机器人特化逻辑（IK/关节驱动/夹爪）委托给自构造的 Robot 对象。

robot-vla-adapter 重构：
- 类名 `PyBulletPandaEnv` → `PyBulletEnv`（通用化）
- `__init__` 内 `self.robot = build_robot(robot_config)` 自我构造 Robot 注入
- `step()` 委托 `self.robot.step_action(action, robot_id, client_id)`
- URDF 加载走顶层 `robot_config.urdf_path`；索引走 `robot_config.panda.*`
- 调用方签名不变：`PyBulletEnv(env_config, robot_config)`，机器人类型对外透明
"""

import platform
import time
import warnings

import numpy as np
import pybullet as p
import pybullet_data

from config.loader import EnvConfig, RobotConfig
from env.base import BaseEnv, Action7D, ActionSpec
from env.robot import build_robot
from utils.logging import setup_logging

logger = setup_logging(__name__)

# Panda 机械臂 7 个关节索引（模块级别名，引用 RobotConfig 默认值）
ARM_JOINT_INDICES = tuple(RobotConfig().panda.arm_joint_indices)
# 末端执行器 link 索引（panda_hand）
EE_LINK_INDEX = RobotConfig().panda.ee_link_index
# 夹爪关节索引
FINGER_JOINT_INDICES = tuple(RobotConfig().panda.finger_joint_indices)

# iter2-renderer-env-mode：renderer 配置合法值
_RENDERER_VALID_VALUES: frozenset[str] = frozenset({"auto", "cpu", "gpu"})


class PyBulletEnv(BaseEnv):
    """基于 PyBullet 的通用仿真环境。

    连接/渲染/相机/物体/观测为所有机器人公用；机器人的 step 特化
    由 self.robot（build_robot 按 config.type 构造）承担。

    Args:
        env_config: 环境配置（mode / renderer / camera_resolution）。
        robot_config: 机器人配置（type + urdf_path 顶层 + 特化配置）。
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
        # ★ robot-vla-adapter：自我构造 Robot 对象注入（调用方零改动）
        self.robot = build_robot(robot_config)
        # iter2-renderer-env-mode：解析渲染器常量与连接模式
        self._renderer = self._resolve_renderer(env_config.renderer)
        self._mode = env_config.mode
        # use_gui 字段保留向后兼容：默认从 mode 推导
        self.use_gui = env_config.use_gui
        # iter6-log-throttle：render() 日志节流计数器，仅首次 + 每 10 次打摘要
        self._render_call_count: int = 0
        # 若显式传 use_gui=True 但 mode 不是 gui，发出 deprecation 警告
        if env_config.use_gui and env_config.mode != "gui":
            warnings.warn(
                "EnvConfig.use_gui 已 deprecated，请使用 mode='gui' 字段。"
                "当前 use_gui=True 与 mode='{}' 不一致，建议迁移。".format(env_config.mode),
                DeprecationWarning,
                stacklevel=2,
            )
            # 兼容性：旧 use_gui=True 仍尝试开 GUI（以 use_gui 为准）
            self.use_gui = True
            self._mode = "gui"
        else:
            # 一致：use_gui 由 mode 推导
            self.use_gui = (env_config.mode == "gui")
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
    @staticmethod
    def _resolve_connection_mode(mode: str) -> int:
        """将 mode 配置字符串解析为 pybullet 连接模式常量。

        Args:
            mode: "direct" | "gui"

        Returns:
            pybullet.DIRECT 或 pybullet.GUI

        Raises:
            ValueError: 非法 mode
        """
        if mode == "direct":
            return p.DIRECT
        if mode == "gui":
            return p.GUI
        raise ValueError(
            f"mode 配置值非法：{mode!r}，合法值为 ['direct', 'gui']"
        )

    @staticmethod
    def _resolve_renderer(config_value: str) -> int:
        """将 renderer 配置字符串解析为 pybullet 渲染器常量。

        映射关系：
          - "cpu"  → p.ER_TINY_RENDERER（CPU 软渲染，跨平台稳定）
          - "gpu"  → p.ER_BULLET_HARDWARE_OPENGL（GPU 渲染，需平台支持）
          - "auto" → Apple Silicon 用 ER_TINY_RENDERER，其他平台用 ER_BULLET_HARDWARE_OPENGL

        Args:
            config_value: "auto" | "cpu" | "gpu"

        Returns:
            pybullet.ER_TINY_RENDERER 或 pybullet.ER_BULLET_HARDWARE_OPENGL

        Raises:
            ValueError: 非法值（不在合法集合内）
        """
        if config_value not in _RENDERER_VALID_VALUES:
            raise ValueError(
                f"renderer 配置值非法：{config_value!r}，"
                f"合法值集合为 {sorted(_RENDERER_VALID_VALUES)}"
            )
        if config_value == "cpu":
            return p.ER_TINY_RENDERER
        if config_value == "gpu":
            return p.ER_BULLET_HARDWARE_OPENGL
        # "auto"：根据平台判断
        is_apple_silicon = (
            platform.system() == "Darwin" and platform.processor() == "arm"
        )
        if is_apple_silicon:
            return p.ER_TINY_RENDERER
        return p.ER_BULLET_HARDWARE_OPENGL

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

        # iter2-renderer-env-mode：连接模式由 self._mode 决定
        connection_mode = self._resolve_connection_mode(self._mode)
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

        robot-vla-adapter：委托给 self.robot.step_action（IK/关节/夹爪特化）。

        Returns:
            (obs, reward, done, info) — 当前 reward=0.0，done=False，info={}。
        """
        self._ensure_connected()
        self.robot.step_action(action, self._robot_id, self._client_id)
        return self.get_obs(), 0.0, False, {}

    def render(self) -> np.ndarray:
        """返回当前相机 RGB 图像。

        iter2-renderer-env-mode：渲染器由 self._renderer 决定（auto/cpu/gpu），
        日志中动态标注实际使用的渲染器（CPU/TINY_RENDERER 或 GPU/OPENGL）。

        iter6-log-throttle：日志节流——仅首次完整打"开始/完成"+渲染器，
        后续每 10 次打一条简短摘要，避免 executor 循环内刷屏。

        Returns:
            shape=(H, W, 3), dtype=uint8，范围 [0, 255]。
        """
        self._render_call_count += 1
        is_logged = (self._render_call_count == 1) or (self._render_call_count % 10 == 0)

        renderer_label = (
            "CPU/TINY_RENDERER"
            if self._renderer == p.ER_TINY_RENDERER
            else "GPU/OPENGL"
        )

        if is_logged:
            if self._render_call_count == 1:
                logger.info(
                    f"render() 开始 — 渲染器={self._renderer} ({renderer_label})"
                )
            else:
                logger.info(
                    f"render() 第 {self._render_call_count} 次调用 — "
                    f"渲染器={renderer_label}"
                )

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

        # 获取相机图像（iter2-renderer-env-mode：传入 renderer 参数）
        (_, _, px, _, _) = p.getCameraImage(
            width,
            height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            physicsClientId=self._client_id,
            renderer=self._renderer,
        )

        # RGBA → RGB
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = rgb_array.reshape((height, width, 4))[:, :, :3]

        if is_logged:
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
        # robot-vla-adapter：ee_link_index 从 self.robot（Robot 对象）读取，
        # 不硬编码 panda（SO101 等其它机器人的 ee 索引不同）
        ee_link_idx = self.robot.ee_link_index
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

    @property
    def input_spec(self) -> ActionSpec:
        """委托给实际加载的 robot 声明其消费 spec（不再硬编码 Panda）。"""
        return self.robot.input_spec

    def get_joint_state(self) -> np.ndarray:
        """读取当前机器人关节角作为 VLA observation.state。

        Returns:
            np.ndarray, shape 取决于 robot（SO101 6 维、Panda 7 维）。
        """
        self._ensure_connected()
        return self.robot.get_joint_state(self._robot_id, self._client_id)
