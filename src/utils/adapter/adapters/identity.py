"""同空间直通 adapter：VLA 输出原样作为 env 原生动作（如 Mock→Panda task→task）。"""

from utils.logging import setup_logging

logger = setup_logging("adapter.identity")


def identity_transform(vla_output, env):
    """原样返回 VLA 输出值，不转换。

    Args:
        vla_output: VLA 输出（VLAOutput 包装 或 裸动作）。
        env: 目标 env（本转换不使用）。

    Returns:
        vla_output 的原始动作值（VLAOutput 则取 .values）。
    """
    logger.info("[adapter] identity selected (task→task 同空间直通)")
    if hasattr(vla_output, "values") and hasattr(vla_output, "spec"):
        return vla_output.values
    return vla_output
