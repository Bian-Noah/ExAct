"""Mock VLA 实现：基于 instruction 哈希生成伪随机 7D 动作。

用于执行器主循环验证（不依赖真实模型权重）。
相同 seed + 相同 instruction 下输出完全一致，便于 pytest 断言可复现。
故意忽略 image 参数。
"""

import hashlib
import random

import numpy as np

from env.base import Action7D, ActionSpec
from executor.model.base import BaseVLA, VLAOutput


class MockVLA(BaseVLA):
    """Mock VLA：基于 instruction 哈希生成伪随机 7D 动作。"""

    def __init__(self, seed: int = 0):
        """初始化 MockVLA。

        Args:
            seed: 伪随机种子基数，与 instruction 哈希组合决定输出。
        """
        self.seed = seed

    @property
    def output_spec(self) -> ActionSpec:
        """Mock 输出 task 空间 7 维动作（Action7D 语义）。"""
        return ActionSpec(
            "task", ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")
        )

    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        state: np.ndarray | None = None,
    ) -> VLAOutput:
        """生成伪随机 7D 动作。

        位移分量范围 [-0.02, 0.02] 米，旋转分量范围 [-0.05, 0.05] 弧度，
        夹爪固定 0.5（半开半合）。相同 seed + instruction 输出一致。

        Args:
            image: 故意忽略，可为任意值（含 None）。
            instruction: 用于哈希生成种子，影响输出。
            state: 故意忽略（Mock 不依赖本体感知）。

        Returns:
            VLAOutput：values 为 Action7D 实例，spec 为 task 空间 7 维。
        """
        # 基于 instruction 哈希与 seed 组合生成种子，保证可复现
        instruction_bytes = instruction.encode("utf-8")
        hash_digest = hashlib.md5(instruction_bytes).digest()
        hash_int = int.from_bytes(hash_digest[:4], byteorder="big")
        combined_seed = hash_int ^ self.seed

        rng = random.Random(combined_seed)

        # 生成 7D 动作分量
        dx = rng.uniform(-0.02, 0.02)
        dy = rng.uniform(-0.02, 0.02)
        dz = rng.uniform(-0.02, 0.02)
        drx = rng.uniform(-0.05, 0.05)
        dry = rng.uniform(-0.05, 0.05)
        drz = rng.uniform(-0.05, 0.05)
        gripper = 0.5

        return VLAOutput(
            values=Action7D(dx, dy, dz, drx, dry, drz, gripper),
            spec=self.output_spec,
        )