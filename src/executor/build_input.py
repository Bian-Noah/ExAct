"""build_vla_input 纯函数：obs + 指令 → VLA 输入格式。

iter11-reset-multicam:image 字段类型从 np.ndarray 改为 dict[str, np.ndarray],
透传 obs["rgb"] 的多相机 dict(单相机配置时 dict 长度 1)。
不再校验 dtype/ndim(由各 VLA 子类按 key 取图时校验)。
"""
from __future__ import annotations

import numpy as np


def build_vla_input(obs: dict, instruction: str) -> dict:
    """组装 VLA 输入。

    iter11-reset-multicam:image 字段类型从 np.ndarray 改为
    dict[str, np.ndarray](来自 obs["rgb"])。

    Args:
        obs: env.get_obs() 返回的 dict。
            必需字段：ee_pos（tuple/list of 3 floats）。
            可选字段：rgb（dict[str, np.ndarray], dtype=uint8, 可为 None）、
                object_info、state_desc、joint_state（缺失不报错）。
            iter10 之前 rgb 是 np.ndarray;iter11 改为 dict。
        instruction: 自然语言指令字符串（允许空字符串）。

    Returns:
        dict 含 6 个键：
        - image: dict[str, np.ndarray] 或 None（来自 obs["rgb"],iter11 改为 dict）
        - instruction: str
        - ee_pos: tuple
        - objects: list
        - state_desc: str
        - state: np.ndarray 或 None

    Raises:
        TypeError: instruction 非 str 类型。
        ValueError: obs 缺 ee_pos。
    """
    # 校验 instruction 类型（允许空字符串）
    if not isinstance(instruction, str):
        raise TypeError(
            f"instruction 必须为 str 类型，收到 {type(instruction).__name__}"
        )

    # ee_pos 必需
    if "ee_pos" not in obs:
        raise ValueError("obs 缺少必需字段 'ee_pos'")
    ee_pos = tuple(obs["ee_pos"])

    # iter11-reset-multicam:rgb 是 dict[str, np.ndarray] 或 None,不再校验 dtype/ndim
    rgb = obs.get("rgb", None)
    if rgb is not None and not isinstance(rgb, dict):
        # iter 10 之前的遗留调用点(传入 ndarray),做防御性提示
        raise ValueError(
            f"obs['rgb'] 必须是 dict[str, np.ndarray] 或 None(iter11 多相机),"
            f"收到 {type(rgb).__name__}"
        )

    # 可选字段：缺失使用默认值
    objects = obs.get("object_info", [])
    state_desc = obs.get("state_desc", "")

    return {
        "image": rgb,
        "instruction": instruction,
        "ee_pos": ee_pos,
        "objects": objects,
        "state_desc": state_desc,
        "state": obs.get("joint_state", None),
    }
