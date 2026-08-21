"""同空间直通 adapter：VLA 输出原样作为 env 原生动作（如 Mock→Panda task→task）。

输入为裸动作值（executor 已统一解包 VLAOutput），原样返回，不转换。

Iteration 10 chunk 契约：ndim=1 输入按 (1, -1) reshape 保持 chunk 一致；
ndim=2 输入原样返回（shape (N, dim)）。其他 ndim 抛 ValueError。
"""

import numpy as np

from utils.logging import setup_logging

logger = setup_logging("adapter.identity")


def identity_transform(vla_output, env):
    """原样返回 VLA 输出值，不转换。

    Args:
        vla_output: VLA 输出裸动作值（executor 已解包 VLAOutput）。
        env: 目标 env（本转换不使用）。

    Returns:
        np.ndarray shape (N, dim)。ndim=1 输入 reshape 为 (1, -1)；
        ndim=2 原样返回；其他 ndim 抛 ValueError。
    """
    if isinstance(vla_output, np.ndarray):
        arr = vla_output.astype(float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(
                f"identity 输入 ndim={arr.ndim}，期望 1 或 2"
            )
        logger.info(f"[adapter] identity selected (task→task 同空间直通，chunk={arr.shape[0]})")
        return arr
    # 兜底：非 ndarray（如 Action7D / tuple）→ 转 ndarray 后 reshape
    arr = np.asarray(vla_output, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim != 2:
        raise ValueError(
            f"identity 输入 ndim={arr.ndim}，期望 1 或 2"
        )
    logger.info(f"[adapter] identity selected (task→task 同空间直通，chunk={arr.shape[0]})")
    return arr