"""task→joint 跨空间 adapter：把任务空间动作通过 env.ik 转为关节动作。

当前无真实 backend 走此路径（OpenVLA→Panda 预留），接口先行。
"""

from utils.logging import setup_logging

logger = setup_logging("adapter.task_to_joint")


def task_to_joint_transform(vla_output, env):
    """把任务空间 VLA 输出转换为 joint 动作（调用 env.ik）。

    Args:
        vla_output: 任务空间 VLA 输出（含目标位置信息）。
        env: 目标 env，需实现 ik() 能力。

    Returns:
        env.ik 返回的关节动作。

    Raises:
        NotImplementedError: env 未实现 ik()，提示该组合需要 ik 能力。
    """
    ik = getattr(env, "ik", None)
    if ik is None:
        raise NotImplementedError(
            "task→joint 需要 env 实现 ik()（任务空间→关节空间），当前 env 未实现"
        )
    logger.info("[adapter] task→joint selected（依赖 env.ik）")
    return ik(vla_output)
