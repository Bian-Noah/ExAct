"""量化工具模块。

提供加载后的 post-quantization 工具，把 nn.Linear 替换为 bitsandbytes
量化线性层，用于在有限显存（如 8GB）下运行 VLA 类 policy。

背景：lerobot 的 from_pretrained 不支持 bnb 量化参数（Pi0 是自定义
cached_file+load_state_dict 加载，SmolVLA 走 PreTrainedPolicy 基类），
没有 load_in_4bit 注入点。因此采用加载后对 VLM 骨干做模块替换的方案。
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

_log = logging.getLogger("quantize")

# 需量化的 VLM 骨干子模块名前缀（按 policy 结构匹配）
VLM_PREFIXES: tuple[str, ...] = ("paligemma_with_expert", "vlm_with_expert", "vlm")


def _find_vlm_submodule(policy: nn.Module) -> nn.Module | None:
    """从 policy 中定位需量化的 VLM 骨干子模块。

    命中返回第一个匹配前缀的子模块（如 paligemma_with_expert / vlm_with_expert），
    未命中返回 None（调用方回退为整模型量化 + 日志警告）。
    """
    for name, module in policy.named_modules():
        for prefix in VLM_PREFIXES:
            if name.startswith(prefix):
                _log.info(f"量化目标 VLM 骨干：{name}（{type(module).__name__}）")
                return module
    return None


def _replace_linear_with_bnb(model: nn.Module, bits: int, compute_dtype: torch.dtype) -> int:
    """遍历模型，把 nn.Linear 替换为 bnb 量化 Linear。

    优先使用 bitsandbytes 官方的 replace_with_bnb_linear（若当前版本导出），
    否则回退到手写替换（构造 bnb.nn.Linear4bit/Linear8bit 并迁移权重）。

    Returns:
        成功替换的线性层数量。
    """
    import bitsandbytes as bnb

    replaced = 0

    # 优先官方工具：bitsandbytes>=0.43 提供 utils.utils 或 quant_state 相关，
    # 官方导出入口为 bitsandbytes.utils.replace_with_bnb_linear（旧版本）或
    # bitsandbytes.nn.modules 内的替换逻辑。这里先尝试官方入口。
    try:
        from bitsandbytes.utils import replace_with_bnb_linear  # 部分版本可用

        new_modules = replace_with_bnb_linear(
            model,
            modules_to_not_convert=["lm_head"],
            quantization_config=_bnb_config(bits, compute_dtype),
        )
        # 返回的是被替换层名集合（dict/iterable），数一下作为替换数
        try:
            replaced = len(new_modules)
        except TypeError:
            replaced = sum(1 for _ in new_modules)
        _log.info(f"bitsandbytes 官方 replace_with_bnb_linear 替换 {replaced} 个线性层")
        return replaced
    except (ImportError, AttributeError, TypeError) as e:
        _log.debug(f"官方 replace_with_bnb_linear 不可用，回退手写替换：{e}")

    # 手写回退：构造 bnb Linear 并迁移权重
    bnb_linear_cls = bnb.nn.Linear4bit if bits == 4 else bnb.nn.Linear8bit
    quant_type = "nf4" if bits == 4 else "int8"
    device = next(model.parameters()).device

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if name == "lm_head":
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        quant_linear = bnb_linear_cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            quant_type=quant_type,
            compute_dtype=compute_dtype,
            device=device,
        )
        # 迁移权重：Linear4bit.from_linear / 手动拷贝（bnb 版本差异）
        try:
            quant_linear = bnb_linear_cls.from_linear(
                module, quant_type=quant_type, compute_dtype=compute_dtype
            )
        except (AttributeError, TypeError):
            quant_linear.weight.data = module.weight.data.to(device)
            if module.bias is not None:
                quant_linear.bias = nn.Parameter(module.bias.data.to(device))
        setattr(parent, child_name, quant_linear)
        replaced += 1

    _log.info(f"手写 bnb 替换完成：{replaced} 个线性层")
    return replaced


def _bnb_config(bits: int, compute_dtype: torch.dtype) -> Any:
    """构造 BitsAndBytesConfig（若 bitsandbytes 提供）。"""
    try:
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=(bits == 4),
            load_in_8bit=(bits == 8),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
        )
    except ImportError:
        return None


def quantize_model_4bit(
    model: nn.Module,
    bits: int = 4,
    compute_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    skip_modules: set[str] | None = None,
) -> nn.Module:
    """对 pytorch 模型做 bnb 量化（加载后模块替换，原地修改）。

    Args:
        model: 待量化模型（通常为 policy 的 VLM 骨干）。
        bits: 量化位宽，4 或 8。
        compute_dtype: 反量化计算 dtype，默认 bfloat16。
        device_map: 设备映射，默认 "auto"（兼容参数，本次实现不做多设备分片）。
        skip_modules: 跳过量化的模块名集合，默认 {"lm_head"}。

    Returns:
        量化后的 model（原地修改）。

    Raises:
        ImportError: bitsandbytes 未安装。
        ValueError: bits 不是 4 或 8。
    """
    if bits not in (4, 8):
        raise ValueError(f"bits 必须为 4 或 8，得到 {bits}")

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "4bit/8bit 量化需要 bitsandbytes。请安装：pip install bitsandbytes>=0.43；"
            "或配置 lerobot.quantization: none 禁用量化。"
        ) from e

    vlm = _find_vlm_submodule(model)
    target: nn.Module = vlm if vlm is not None else model
    if vlm is None:
        _log.warning(
            "未在 policy 中命中 VLM 骨干前缀 %s，将对整个模型做量化，"
            "可能包含不适用 bnb 的层，请谨慎。", VLM_PREFIXES
        )

    n = _replace_linear_with_bnb(target, bits, compute_dtype)
    _log.info(f"量化完成：{bits}bit，替换 {n} 个线性层")
    return model
