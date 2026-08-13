"""VLA 抽象基类模块。

定义 BaseVLA 抽象基类约束所有 VLA 后端（MockVLA / OpenVLA 等）的接口契约。
"""

import abc

import numpy as np

from env.base import Action7D


class BaseVLA(abc.ABC):
    """VLA 抽象基类。

    所有 VLA 后端（MockVLA / OpenVLA 等）必须继承此类并实现 predict 方法。
    方法签名固定为 (image, instruction) -> Action7D，不得修改。
    """

    @abc.abstractmethod
    def predict(self, image: np.ndarray, instruction: str) -> Action7D:
        """输入图片 + 自然语言指令，输出 7D 动作。

        Args:
            image: np.ndarray (H, W, 3) uint8，当前场景截图。
            instruction: 自然语言指令字符串，如 "移动到红色方块上方"。

        Returns:
            Action7D NamedTuple，7 个字段依次为
            dx/dy/dz（位移米）/ drx/dry/drz（旋转弧度）/ gripper（夹爪开合 [0,1]）。
        """