"""VLA 抽象基类模块。

定义 BaseVLA 抽象基类约束所有 VLA 后端（MockVLA / OpenVLA 等）的接口契约。
"""

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from env.base import Action7D, ActionSpec


@dataclass
class VLAOutput:
    """VLA 输出包装：值 + 语义 spec（适配层据此分派）。

    Attributes:
        values: 原始输出（np.ndarray / torch.Tensor / tuple / Action7D 等）。
        spec: 该输出的语义 spec（空间/维度/每维含义）。
    """

    values: Any
    spec: ActionSpec


class BaseVLA(abc.ABC):
    """VLA 抽象基类。

    所有 VLA 后端（MockVLA / OpenVLA 等）必须继承此类并实现 predict 方法。
    方法签名固定为 (image, instruction) -> VLAOutput。
    """

    @abc.abstractmethod
    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        state: np.ndarray | None = None,
    ) -> VLAOutput:
        """输入图片 + 自然语言指令，输出带 spec 的动作。

        Args:
            image: np.ndarray (H, W, 3) uint8，当前场景截图。
            instruction: 自然语言指令字符串，如 "移动到红色方块上方"。
            state: 可选 np.ndarray，当前机器人关节角/本体感知状态。
                None 时各后端按自身默认（ACT 默认零向量，Mock 忽略）。
                真实部署时 executor 应从 env.get_joint_state() 注入。

        Returns:
            VLAOutput：values 为动作值，spec 声明其语义。
        """

    @property
    @abc.abstractmethod
    def output_spec(self) -> ActionSpec:
        """声明该 VLA 输出的动作 spec（空间/维度/每维含义）。

        每个后端在类内部自声明（spec 是模型的固有属性，不落入 config）。
        """