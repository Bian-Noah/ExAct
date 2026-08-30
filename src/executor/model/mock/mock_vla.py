"""Mock VLA 实现：基于 instruction 哈希生成伪随机 7D 动作。

用于执行器主循环验证（不依赖真实模型权重）。
相同 seed + 相同 instruction 下输出完全一致，便于 pytest 断言可复现。
故意忽略 image 参数。

返回值遵循 chunk 契约：`VLAOutput.values` shape `(N, action_dim)`。
单步 VLA（Mock / LLMVLA）shape 为 `(1, action_dim)`，N=1，调用方按第一维
迭代执行即可。
"""

import hashlib
import random

import numpy as np

from env.base import Action7D, ActionSpec
from executor.model.base import BaseVLA, VLAOutput


class MockVLA(BaseVLA):
    """Mock VLA：基于 instruction 哈希生成伪随机 7D 动作。

    幅度设计（用户 2026-08-30 要求"非常明显"）：
      - 位移分量范围 [-0.15, 0.15] 米/步（单步 15cm，肉眼明显可见）
      - 旋转分量范围 [-0.15, 0.15] 弧度/步
      - gripper 固定 0.5（半开半合）
    注：widowx 工作空间约 ±0.3m，单步 0.15m 多次随机游走可能撞限位，
    但单步动作在范围内，链路验证目的达成。
    """

    # 位移/旋转分量范围（类常量，便于测试引用）
    TRANSLATION_RANGE: float = 0.15   # 米/步
    ROTATION_RANGE: float = 0.15      # 弧度/步

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
        """生成伪随机 7D 动作（chunk shape `(1, 7)`）。

        位移分量范围 [-0.15, 0.15] 米，旋转分量范围 [-0.15, 0.15] 弧度，
        夹爪固定 0.5（半开半合）。相同 seed + instruction 输出一致。

        Args:
            image: 故意忽略，可为任意值（含 None）。
            instruction: 用于哈希生成种子，影响输出。
            state: 故意忽略（Mock 不依赖本体感知）。

        Returns:
            VLAOutput：values 为 np.ndarray shape=(1, 7)，spec 为 task 空间 7 维。
        """
        # 基于 instruction 哈希与 seed 组合生成种子，保证可复现
        instruction_bytes = instruction.encode("utf-8")
        hash_digest = hashlib.md5(instruction_bytes).digest()
        hash_int = int.from_bytes(hash_digest[:4], byteorder="big")
        combined_seed = hash_int ^ self.seed

        rng = random.Random(combined_seed)

        # 生成 7D 动作分量（幅度加大：位移 ±0.15m/步，旋转 ±0.15rad/步）
        r = self.TRANSLATION_RANGE
        rot = self.ROTATION_RANGE
        dx = rng.uniform(-r, r)
        dy = rng.uniform(-r, r)
        dz = rng.uniform(-r, r)
        drx = rng.uniform(-rot, rot)
        dry = rng.uniform(-rot, rot)
        drz = rng.uniform(-rot, rot)
        gripper = 0.5

        values = np.array(
            [[dx, dy, dz, drx, dry, drz, gripper]], dtype=float
        )  # shape (1, 7)
        return VLAOutput(values=values, spec=self.output_spec)


class JointMockVLA(BaseVLA):
    """Joint 空间 mock VLA：5 关节增量 + gripper = 6 维输出。

    用于 joint 输入空间的 robot 后端（so101）。JointMockVLA 与 MockVLA 共享
    相同的伪随机种子机制（基于 instruction MD5 哈希 + seed 异或），便于 pytest
    断言可复现。故意忽略 image 参数。

    关节增量范围 [-0.1, 0.1] rad（约 ±5.7°/步），与 MockVLA 的 ±0.15 m 位移
    在视觉/数值上等量级，便于 mock 模拟产生可见运动。
    """

    OUTPUT_DIM: int = 6
    GRIPPER_INDEX: int = 5  # 最后一维固定为 gripper
    GRIPPER_DEFAULT: float = 0.5
    JOINT_DELTA_MIN: float = -0.1
    JOINT_DELTA_MAX: float = 0.1

    def __init__(self, seed: int = 0):
        self.seed = seed

    @property
    def output_spec(self) -> ActionSpec:
        """Joint 空间 6 维（与 so101 的 input_spec 等维，触发 joint_to_joint 等维直通）。"""
        return ActionSpec("joint", ("joint",) * self.OUTPUT_DIM)

    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        state: np.ndarray | None = None,
    ) -> VLAOutput:
        """生成伪随机 6D joint 动作（chunk shape `(1, 6)`）。

        前 5 维：关节角增量（弧度），范围 [-0.1, 0.1]。
        第 6 维：gripper 开合，固定 0.5。

        Args:
            image: 故意忽略。
            instruction: 用于哈希生成种子，影响输出。
            state: 故意忽略。

        Returns:
            VLAOutput：values 为 np.ndarray shape=(1, 6)，spec 为 joint 空间 6 维。
        """
        # 基于 instruction 哈希与 seed 组合生成种子，保证可复现
        instruction_bytes = instruction.encode("utf-8")
        hash_digest = hashlib.md5(instruction_bytes).digest()
        hash_int = int.from_bytes(hash_digest[:4], byteorder="big")
        combined_seed = hash_int ^ self.seed

        rng = random.Random(combined_seed)

        # 生成 6 维 joint 动作分量
        joint_dim = self.OUTPUT_DIM - 1
        joints = [
            rng.uniform(self.JOINT_DELTA_MIN, self.JOINT_DELTA_MAX)
            for _ in range(joint_dim)
        ]
        values = np.array([joints + [self.GRIPPER_DEFAULT]], dtype=float)  # (1, 6)

        return VLAOutput(values=values, spec=self.output_spec)