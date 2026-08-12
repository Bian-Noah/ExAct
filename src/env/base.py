"""环境抽象基类模块。

定义 Action7D 动作数据结构和 BaseEnv 抽象环境接口。
"""

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np


class Action7D(NamedTuple):
    """7 自由度动作向量。

    前 3 个为末端平移增量（米），中间 3 个为末端旋转增量（弧度），
    最后 1 个为夹爪开合（0.0=完全闭，1.0=完全开）。
    """

    dx: float       # 末端 X 轴位移增量（米）
    dy: float       # 末端 Y 轴位移增量（米）
    dz: float       # 末端 Z 轴位移增量（米）
    drx: float      # 末端 Roll 增量（弧度）
    dry: float      # 末端 Pitch 增量（弧度）
    drz: float      # 末端 Yaw 增量（弧度）
    gripper: float  # 夹爪开合，0.0=完全闭，1.0=完全开


class BaseEnv(ABC):
    """仿真环境抽象基类。

    所有具体环境后端（如 PyBulletPandaEnv）需继承此类并实现全部抽象方法。
    统一的观测字典契约：
        - rgb: np.ndarray (H,W,3) uint8，相机 RGB 图像
        - object_info: list[dict]，每个物体 {"id":int, "name":str, "pos":[x,y,z], "quat":[x,y,z,w]}
        - ee_pos: tuple[float, float, float]，末端执行器世界坐标（米）
        - state_desc: str，简短文字描述当前状态
    """

    @abstractmethod
    def reset(self, task_spec: dict, seed: int = 0) -> dict:
        """重置环境到初始状态。

        Args:
            task_spec: 任务描述字典，当前接受
                {"objects": [{"type":"cube","pos":[x,y,z],"color":"red"}, ...]}
            seed: 随机种子，当前不使用（预留）。

        Returns:
            obs dict，见观测字典契约。
        """

    @abstractmethod
    def step(self, action: Action7D) -> tuple[dict, float, bool, dict]:
        """执行一步动作。

        Returns:
            (obs, reward, done, info) — 当前 reward=0.0，done=False，info={}。
        """

    @abstractmethod
    def render(self) -> np.ndarray:
        """返回当前相机 RGB 图像。

        Returns:
            shape=(H, W, 3), dtype=uint8, 范围 [0, 255]。
        """

    @abstractmethod
    def get_obs(self, include_rgb: bool = True) -> dict:
        """返回当前观测，不推进物理。

        Args:
            include_rgb: 是否包含 RGB 图像。False 时跳过 GPU 渲染，
                适用于仅需 object_info/ee_pos 的轻量调用（如 observe 工具）。

        Returns:
            obs dict，见观测字典契约。
        """

    @abstractmethod
    def close(self) -> None:
        """释放所有资源，断开仿真连接。"""
