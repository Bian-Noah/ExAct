"""joint→joint 同空间数值映射 adapter。

VLA 输出与 env 输入都是 joint 空间，但维度/夹爪存在可能不同。
规则（自底向上）：
  1. 输入维度 == 目标维度 → 原样返回。
  2. 输入维度 > 目标维度 → 截取前 N 维。
  3. 输入维度 == 目标维度 - 1 且目标含 gripper → 截取 + 补默认 gripper 占位。
  4. 其余输入维度 < 目标所需 → ValueError（含实际/期望维度）。
"""

import numpy as np

from utils.logging import setup_logging

logger = setup_logging("adapter.joint_to_joint")

# 缺 gripper 时补的默认开合值（0.5 = 半开）
_DEFAULT_GRIPPER: float = 0.5


def _to_numpy(vla_output) -> np.ndarray:
    """把 VLA 输出（数组/张量/tuple）转为一维 numpy。"""
    if isinstance(vla_output, np.ndarray):
        arr = vla_output
    elif hasattr(vla_output, "detach"):  # torch.Tensor
        arr = vla_output.detach().cpu().numpy()
    else:
        arr = np.asarray(vla_output)
    return arr.reshape(-1).astype(float)


def joint_to_joint_transform(vla_output, env):
    """把 joint 空间 VLA 输出映射为 env 原生动作。

    Args:
        vla_output: VLA 输出（joint 空间，可为数组/张量/tuple）。
        env: 目标 env，其 input_spec 提供目标维度与 gripper 位置。

    Returns:
        np.ndarray，长度 = env.input_spec.dim。

    Raises:
        ValueError: 输出维度 < 目标所需（缺的不只是 gripper 维）。
    """
    arr = _to_numpy(vla_output)
    target = env.input_spec
    target_dim = target.dim
    has_gripper = target.gripper_index is not None

    # 缺的不只是 gripper 维（输入比目标至少少 2 维，或目标无 gripper 却仍不足）→ 报错
    min_required = target_dim - (1 if has_gripper else 0)
    if arr.size < min_required:
        raise ValueError(
            f"joint→joint 维度不匹配：实际输出 {arr.size} 维，"
            f"目标 env 需要 {target_dim} 维"
        )

    # 截取前 target_dim 维
    mapped = arr[:target_dim].copy()

    # 仅缺 gripper 维 → 先补足维度，再填默认占位
    if has_gripper and arr.size == target_dim - 1:
        mapped = np.concatenate([mapped, np.zeros(1, dtype=float)])
        mapped[target.gripper_index] = _DEFAULT_GRIPPER

    logger.info(
        f"[adapter] joint→joint selected: {arr.size} → {target_dim} 维"
    )
    return mapped
