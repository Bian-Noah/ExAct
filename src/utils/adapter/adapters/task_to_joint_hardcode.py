"""task→joint 硬编码转换 adapter：把 task 空间动作前 N 维当关节角直通。

无真实 IK 时的占位实现：task 空间动作的前 env.input_spec.dim 维直接作为
joint 关节角返回（丢弃 gripper 等多余维度）。物理语义不对齐（机械臂会乱动），
但保证 task→joint 链路不崩，用于验证仿真链路可跑通。

替代方案：真实场景用 env.ik()（见 task_to_joint.py，依赖 env 实现 ik 能力）。
"""

import numpy as np

from utils.logging import setup_logging

logger = setup_logging("adapter.task_to_joint_hardcode")


def _to_flat(vla_output):
    """把动作值（数组/张量/tuple/Action7D）转为一维 numpy。"""
    if isinstance(vla_output, np.ndarray):
        return vla_output.reshape(-1)
    if hasattr(vla_output, "detach"):  # torch.Tensor
        return vla_output.detach().cpu().numpy().reshape(-1)
    return np.asarray(list(vla_output)).reshape(-1)


def task_to_joint_hardcode_transform(vla_output, env):
    """task 空间动作前 N 维硬编码为 joint 关节角。

    Args:
        vla_output: task 空间动作（executor 已解包，裸值）。
        env: 目标 env，其 input_spec 提供目标关节维度。

    Returns:
        np.ndarray，长度 = env.input_spec.dim，值为输入前 N 维。
    """
    arr = _to_flat(vla_output)
    target_dim = env.input_spec.dim
    if arr.size < target_dim:
        raise ValueError(
            f"task→joint 硬编码维度不足：实际 {arr.size} 维，"
            f"目标需要 {target_dim} 维"
        )
    mapped = arr[:target_dim].astype(float)
    logger.info(f"[adapter] task→joint hardcode selected: {arr.size} → {target_dim} 维")
    return mapped
