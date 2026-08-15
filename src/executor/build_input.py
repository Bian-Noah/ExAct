"""build_vla_input 纯函数：obs + 指令 → VLA 输入格式。

宽松校验：rgb 可为 None（MockVLA 等不依赖图像的 VLA 后端），
ee_pos 仍是必需字段。
"""

import numpy as np


def build_vla_input(obs: dict, instruction: str) -> dict:
    """组装 VLA 输入。

    Args:
        obs: env.get_obs() 返回的 dict。
            必需字段：ee_pos（tuple/list of 3 floats）。
            可选字段：rgb（np.ndarray, dtype=uint8，可为 None）、
                object_info、state_desc（缺失不报错，使用默认值）。
        instruction: 自然语言指令字符串（允许空字符串）。

    Returns:
        dict 含 6 个键：
        - image: np.ndarray 或 None（来自 obs["rgb"]）
        - instruction: str
        - ee_pos: tuple（来自 obs["ee_pos"]，list 会被转为 tuple）
        - objects: list（来自 obs["object_info"]，缺失默认 []）
        - state_desc: str（来自 obs["state_desc"]，缺失默认 ""）
        - state: np.ndarray 或 None（来自 obs["joint_state"]，缺失默认 None）。
          VLA 拿到 None 时回退到自身默认行为（如 ACT 喂零向量），
          真实部署时 executor 应从 env.get_joint_state() 注入。

    Raises:
        TypeError: instruction 非 str 类型。
        ValueError: obs 缺 ee_pos、rgb 存在但非 np.ndarray、rgb.dtype 非 uint8。
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

    # rgb 可选：缺失或 None 时 image=None，存在时校验类型
    rgb = obs.get("rgb", None)
    if rgb is not None:
        if not isinstance(rgb, np.ndarray):
            raise ValueError(
                f"rgb 必须为 np.ndarray 或 None，收到 {type(rgb).__name__}"
            )
        if rgb.dtype != np.uint8:
            raise ValueError(f"rgb.dtype 必须为 uint8，收到 {rgb.dtype}")

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
