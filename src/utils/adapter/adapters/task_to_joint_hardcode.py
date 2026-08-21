"""task→joint 硬编码转换 adapter：把 task 空间动作前 N 维当关节角直通。

无真实 IK 时的占位实现：task 空间动作的前 env.input_spec.dim 维直接作为
joint 关节角返回（丢弃 gripper 等多余维度）。物理语义不对齐（机械臂会乱动），
但保证 task→joint 链路不崩，用于验证仿真链路可跑通。

替代方案：真实场景用 env.ik()（见 task_to_joint.py，依赖 env 实现 ik 能力）。

Iteration 10 chunk 契约：adapter 接收 shape (N, action_dim)，输出 shape (N, target_dim)。
ndim=1 输入按 (1, -1) reshape。
"""

import numpy as np

from utils.logging import setup_logging

logger = setup_logging("adapter.task_to_joint_hardcode")


def _to_2d(vla_output) -> np.ndarray:
    """把动作值转 shape (N, D) 二维 numpy。

    - ndim=1 → reshape (1, -1)
    - ndim=2 → 保持
    - 其他 → ValueError
    """
    if isinstance(vla_output, np.ndarray):
        arr = vla_output.astype(float)
    elif hasattr(vla_output, "detach"):
        arr = vla_output.detach().cpu().numpy().astype(float)
    else:
        arr = np.asarray(list(vla_output)).astype(float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim != 2:
        raise ValueError(
            f"adapter 输入 ndim={arr.ndim}，期望 1 或 2"
        )
    return arr


def task_to_joint_hardcode_transform(vla_output, env):
    """task 空间动作前 N 维硬编码为 joint 关节角。

    Args:
        vla_output: task 空间动作（executor 已解包，裸值）。
            shape (N, action_dim) 或 (action_dim,)；后者按 (1, -1) reshape。
        env: 目标 env，其 input_spec 提供目标关节维度。

    Returns:
        np.ndarray shape (N, target_dim)，值为输入前 N 维。第一维 N 与输入一致。
    """
    arr = _to_2d(vla_output)
    target_dim = env.input_spec.dim
    if arr.shape[1] < target_dim:
        raise ValueError(
            f"task→joint 硬编码维度不足：实际 {arr.shape[1]} 维（chunk_size={arr.shape[0]}），"
            f"目标需要 {target_dim} 维"
        )
    mapped = arr[:, :target_dim].astype(float)
    logger.info(f"[adapter] task→joint hardcode selected: {arr.shape[1]} → {target_dim} 维（chunk={arr.shape[0]}）")
    return mapped