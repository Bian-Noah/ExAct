"""VLA 模型工厂：按 vla_config.backend 字段分派，返回 BaseVLA 子类实例。

已实现后端：mock / lerobot / openvla。
llm_vla / small_vla 在对应 Iteration 启用前调用 create_vla 会抛 NotImplementedError。

openvla 后端延迟导入（openvla_vla 顶层零 torch import，构造安全），
M4 Mac（无 torch）环境调用 create_vla(backend="openvla") 不会报 ImportError。
"""

from config.loader import LLMConfig, VLAConfig
from executor.model.base import BaseVLA
from executor.model.mock.mock_vla import JointMockVLA, MockVLA


def create_vla(vla_config: VLAConfig, llm_config: LLMConfig | None = None) -> BaseVLA:
    """按 vla_config.backend 字段分派，返回 BaseVLA 子类实例。

    Args:
        vla_config: VLAConfig 实例（backend 字段决定分派目标；lerobot 后端还
            会读取 vla_config.lerobot 嵌套配置，openvla 后端读取 vla_config.openvla）。
        llm_config: 可选，LLMVLA 后端（Iteration 5）会用到；当前所有已实现后端
            均忽略此参数。

    Returns:
        BaseVLA 子类实例。

    Raises:
        NotImplementedError: backend 为 llm_vla / small_vla 时。
        ValueError: backend 为未知字符串时，或 backend=mock 但 variant 不识别时，
            或 backend=openvla 但 model_path 缺失（OpenVLA 构造校验兜住）。
    """
    backend = vla_config.backend

    if backend == "mock":
        variant = vla_config.mock.variant
        if variant == "task":
            return MockVLA(seed=0)
        if variant == "joint":
            return JointMockVLA(seed=0)
        raise ValueError(
            f"未知 mock variant: {variant!r}，期望 'task' | 'joint'"
        )

    if backend == "lerobot":
        # 延迟导入：lerobot_vla 顶层 import torch，阶段一（M4，无 torch）不应被拖垮
        from executor.model.lerobot.lerobot_vla import LeRobotVLA

        lc = vla_config.lerobot
        return LeRobotVLA(
            model_path=vla_config.model_path,
            policy_type=lc.policy_type,
            device=lc.device,
            quantization=lc.quantization,
            image_key=lc.image_key,
            action_dim=lc.action_dim,
        )

    if backend == "llm_vla":
        raise NotImplementedError(
            "VLA backend 'llm_vla' 在 Iteration 5 才实现，当前迭代（iter1）仅预留接口"
        )
    if backend == "small_vla":
        raise NotImplementedError(
            "VLA backend 'small_vla' 在 Iteration 9 才实现，当前迭代（iter1）仅预留接口"
        )
    if backend == "openvla":
        # 延迟导入：openvla_vla 顶层零 torch import（决策 6），构造不加载模型
        from executor.model.openvla.openvla_vla import OpenVLA

        oc = vla_config.openvla
        return OpenVLA(
            model_path=vla_config.model_path,
            unnorm_key=oc.unnorm_key,
            attn_impl=oc.attn_impl,
            device=oc.device,
            dtype=oc.dtype,          # str 直传；首次加载时 _ensure_loaded 内解析
            quantization=oc.quantization,
        )

    raise ValueError(f"未知 VLA backend: {backend!r}")