"""同空间直通 adapter：VLA 输出原样作为 env 原生动作（如 Mock→Panda task→task）。

输入为裸动作值（executor 已统一解包 VLAOutput），原样返回，不转换。
"""

from utils.logging import setup_logging

logger = setup_logging("adapter.identity")


def identity_transform(vla_output, env):
    """原样返回 VLA 输出值，不转换。

    Args:
        vla_output: VLA 输出裸动作值（executor 已解包 VLAOutput；数组/张量/tuple/Action7D 等）。
        env: 目标 env（本转换不使用）。

    Returns:
        vla_output 原样。
    """
    logger.info("[adapter] identity selected (task→task 同空间直通)")
    return vla_output
