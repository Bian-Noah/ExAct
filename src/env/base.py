"""环境抽象基类模块。

定义 Action7D 动作数据结构、ActionSpec 动作语义规格和 BaseEnv 抽象环境接口。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

# ActionSpec.space 合法值
_ACTION_SPACES: frozenset[str] = frozenset({"task", "joint"})


@dataclass(frozen=True)
class ActionSpec:
    """动作空间语义规格（适配层分派依据）。

    Attributes:
        space: 动作空间，"task"（任务空间）| "joint"（关节空间）。
        components: 每维语义名，如 ("dx","dy","dz","drx","dry","drz","gripper")。
        dim: 派生字段，= len(components)。
        gripper_index: 派生字段，含 "gripper" 的维索引；无夹爪则 None。
    """

    space: str
    components: tuple[str, ...]
    dim: int = field(init=False)
    gripper_index: int | None = field(init=False)

    def __post_init__(self) -> None:
        """校验 space/components 合法性，派生 dim 与 gripper_index。"""
        if self.space not in _ACTION_SPACES:
            raise ValueError(
                f"space 必须为 {sorted(_ACTION_SPACES)} 之一，得到 {self.space!r}"
            )
        seen: set[str] = set()
        for c in self.components:
            if not c:
                raise ValueError("components 不能包含空串")
            # "joint" 是关节空间的占位语义名，允许多维重复；其余语义名必须互异
            if c != "joint" and c in seen:
                raise ValueError(f"components 不能包含重复元素: {c!r}")
            seen.add(c)
        object.__setattr__(self, "dim", len(self.components))
        gripper_idxs = [i for i, c in enumerate(self.components) if c == "gripper"]
        object.__setattr__(self, "gripper_index", gripper_idxs[0] if gripper_idxs else None)


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

    所有具体环境后端（如 PyBulletEnv）需继承此类并实现全部抽象方法。
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

    # ---- 动作语义声明（robot-vla-adapter 新增） ----

    @property
    @abstractmethod
    def input_spec(self) -> ActionSpec:
        """声明该 env 消费的动作 spec（空间/维度/每维含义）。"""

    def ik(self, target_pos: tuple) -> Any:
        """可选能力：任务空间目标位置 → 关节角度（task→joint adapter 依赖）。

        Raises:
            NotImplementedError: 该机器人未实现 ik（任务空间→关节空间）。
        """
        raise NotImplementedError("该机器人未实现 ik（任务空间→关节空间）")

    def fk(self, joint_angles: Any) -> tuple:
        """可选能力：关节角度 → 任务空间目标位置。

        Raises:
            NotImplementedError: 该机器人未实现 fk（关节空间→任务空间）。
        """
        raise NotImplementedError("该机器人未实现 fk（关节空间→任务空间）")

    def get_joint_state(self) -> np.ndarray:
        """返回当前关节角向量（按 robot 关节顺序，可被 VLA 当作 observation.state）。

        默认未实现，env 子类按需 override。VLA 拿到后传给 predict(image, instruction, state)
        让基于 proprioception 的 IL policy（ACT/Diffusion 等）不再以零向量占位推理。

        Returns:
            np.ndarray, shape=(self.action_dim,), dtype=float。

        Raises:
            NotImplementedError: 该 env 未实现 joint state 暴露。
        """
        raise NotImplementedError("该 env 未实现 get_joint_state")
