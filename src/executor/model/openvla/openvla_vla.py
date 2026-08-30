"""OpenVLA 后端实现。

通过 HuggingFace AutoModelForVision2Seq / AutoProcessor 加载 OpenVLA-7B 远端权重，
封装 predict(image, instruction) -> VLAOutput(values=(1,7))。

仅推理（不覆盖 LoRA 微调、REST 部署）。依赖见 requirements-openvla.txt。

设计约定（决策 6，测试零依赖）：
- 模块顶层【不】import torch：torch 仅在 `_ensure_loaded` / `_parse_dtype`
  首次使用时 import，因此无 torch 环境（开发机/CI）也能构造、取 output_spec、
  跑输入校验；predict 推理路径由单测注入假 torch/transformers 覆盖。

构造仅保存参数，首次调用 predict 时才下载权重并移至 device（懒加载）。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

from env.base import ActionSpec
from executor.model.base import BaseVLA, VLAOutput

# 默认提示模板（与 OpenVLA 官方 README 完全一致）
DEFAULT_PROMPT_TEMPLATE: str = (
    "In: What action should the robot take to {instruction}?\nOut:"
)


def _parse_dtype(dtype_str: str) -> "torch.dtype":
    """把配置字符串 dtype 解析为 torch.dtype（非法值抛 ValueError）。

    仅在 `_ensure_loaded` 首次加载时调用（决策 6：顶层零 torch import，
    本函数内部才 import torch，测试可注入假 torch 覆盖）。

    Args:
        dtype_str: "bfloat16" | "float16" | "float32"（小写）。

    Returns:
        torch.dtype 实例。

    Raises:
        ValueError: dtype_str 不在映射表中。
    """
    import torch  # noqa: F401  # 决策 6：顶层不 import torch

    mapping: dict[str, torch.dtype] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[dtype_str]
    except KeyError:
        raise ValueError(
            f"dtype 必须为 bfloat16|float16|float32 之一，得到 {dtype_str!r}"
        ) from None


class OpenVLA(BaseVLA):
    """OpenVLA-7B 后端（仅推理）。

    Attributes:
        model_path: HF Hub 远端 ID（如 "openvla/openvla-7b"）或本地 checkpoint 路径。
        unnorm_key: 反归一化键名，决定动作的物理量纲。常见值：
            "bridge_orig"（BridgeData V2 训练版）；不同微调版可能不同。
        attn_impl: attention 实现，"flash_attention_2" 需安装 flash-attn；
            无 GPU / 装不上时退回到 "eager"。
        device: 推理设备，默认 "cuda:0"（quantization="4bit" 时被 device_map 覆盖）。
        dtype: 推理精度字符串，默认 "bfloat16"（"bfloat16"|"float16"|"float32"），
            首次加载时由 `_parse_dtype` 解析为 torch.dtype。
        quantization: 量化等级，"4bit"（bnb nf4，4060 8GB 默认，~4GB 显存）
            | "none"（bf16 全精度，需 16GB+ 显存）。
    """

    def __init__(
        self,
        model_path: str,
        unnorm_key: str = "bridge_orig",
        attn_impl: str = "flash_attention_2",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        quantization: str = "4bit",
    ):
        if not model_path or not isinstance(model_path, str):
            raise ValueError(f"model_path 必须为非空字符串，得到 {model_path!r}")
        if quantization not in ("4bit", "none"):
            raise ValueError(
                f"quantization 必须为 '4bit' | 'none'，得到 {quantization!r}"
            )
        self.model_path = model_path
        self.unnorm_key = unnorm_key
        self.attn_impl = attn_impl
        self.device = device
        self._dtype_str = dtype          # 字符串；首次加载时解析为 torch.dtype
        self.quantization = quantization
        self.dtype = None                # torch.dtype，_ensure_loaded 内赋值

        # 懒加载占位
        self._processor = None
        self._vla = None

        self._log = logging.getLogger("openvla_vla")

    @property
    def output_spec(self) -> ActionSpec:
        """OpenVLA 输出 task 空间 7 维动作（Action7D 语义）。

        spec 是模型的固有属性（unnorm_key 决定量纲），不落入 config。
        """
        return ActionSpec(
            "task", ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")
        )

    def _ensure_loaded(self) -> None:
        """首次调用 predict 时加载 processor + model。

        决策 6（测试零依赖）：torch 只在【本方法内】首次 import，
        模块顶层不 import torch，因此无 torch 环境也可构造/校验/output_spec。

        量化分支：
          - quantization="4bit"：BitsAndBytesConfig(nf4) + device_map="auto"
            （bnb 量化要求，不能 .to(device)）
          - quantization="none"：bf16 全精度 + .to(device)（原路径）

        单独 try/except 让 ImportError（缺 torch/transformers/bnb）
        与 HF 下载失败分开上报，便于上层做分支处理。
        """
        if self._vla is not None and self._processor is not None:
            return

        import torch  # noqa: F401  # 决策 6：顶层不 import torch

        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "OpenVLA 后端需要 transformers / timm / tokenizers。"
                "请安装：pip install -r requirements-openvla.txt"
            ) from e

        # 首次加载才把 dtype 字符串解析为 torch.dtype
        self.dtype = _parse_dtype(self._dtype_str)

        self._log.info(
            f"OpenVLA 懒加载开始 model_path={self.model_path} "
            f"attn_impl={self.attn_impl} quantization={self.quantization} "
            f"device={self.device} dtype={self.dtype}"
        )

        self._processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        load_kwargs: dict = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        # flash_attn 可选；缺包时静默回退 eager，避免 import 失败连带整个推理失败
        if self.attn_impl == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
                load_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                self._log.warning(
                    "未安装 flash_attn，回退到 eager attention。可选安装："
                    "pip install flash-attn==2.5.5 --no-build-isolation"
                )

        if self.quantization == "4bit":
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # noqa: F401  # 触发 ImportError 以便给出指引
            except ImportError as e:
                raise ImportError(
                    "OpenVLA 4bit 量化需要 bitsandbytes + accelerate（openvla env）。"
                    "请安装：pip install -r requirements-openvla.txt"
                ) from e
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs.update(
                {"quantization_config": bnb_cfg, "device_map": "auto"}
            )
            self._vla = AutoModelForVision2Seq.from_pretrained(
                self.model_path, **load_kwargs
            )
        else:  # "none"：bf16 全精度路径（原行为）
            self._vla = AutoModelForVision2Seq.from_pretrained(
                self.model_path, **load_kwargs
            ).to(self.device)

        self._log.info("OpenVLA 懒加载完成")

    def predict(
        self,
        image: dict[str, np.ndarray],
        instruction: str,
        state: np.ndarray | None = None,
    ) -> VLAOutput:
        """输入多相机图片 dict + 自然语言指令，输出 7D 动作（chunk shape (1,7)）。

        iter11-reset-multicam：image 为 dict[str, np.ndarray]，key 为相机名，
        OpenVLA 只消费单图，取首张图。

        Args:
            image: dict[str, np.ndarray] 多相机 RGB 图（iter11 契约），
                取首张 (H, W, 3) uint8，范围 [0, 255]。
            instruction: 自然语言指令字符串。
            state: 暂未使用（OpenVLA 走 prompt 链路，不在 state 通道做 proprio）；
                保留参数位以与 BaseVLA 接口对齐。

        Returns:
            VLAOutput：values 为 np.ndarray shape (1, 7)，7 字段依次为
            dx/dy/dz（米）/ drx/dry/drz（弧度）/ gripper（[0,1]）；
            spec 为 task 空间 7 维（iter10 chunk 契约：调用方按第一维迭代）。

        Raises:
            TypeError: image 非 dict 或首张图非 np.ndarray。
            ValueError: image dict 为空 / 首张图 dtype 非 uint8 / 形状不是 (H, W, 3)。
        """
        if not isinstance(image, dict):
            raise TypeError(
                f"image 必须为 dict[str, np.ndarray]，得到 {type(image).__name__}"
            )
        # iter11-reset-multicam:OpenVLA 只消费单图,取 image dict 的首张图
        if not image:
            raise ValueError("image dict 为空,至少需 1 张相机图")
        first_key = next(iter(image))
        first_img = image[first_key]
        if not isinstance(first_img, np.ndarray):
            raise TypeError(
                f"image['{first_key}'] 不是 np.ndarray，得到 {type(first_img).__name__}"
            )
        if first_img.dtype != np.uint8:
            raise ValueError(f"image.dtype 必须为 uint8，得到 {first_img.dtype}")
        if first_img.ndim != 3 or first_img.shape[2] != 3:
            raise ValueError(
                f"image 形状必须为 (H, W, 3)，得到 {first_img.shape}"
            )

        self._ensure_loaded()

        pil_image: Image.Image = Image.fromarray(first_img)
        prompt: str = DEFAULT_PROMPT_TEMPLATE.format(instruction=instruction)

        # processor 输入 → 移到 device + dtype
        inputs: dict = self._processor(prompt, pil_image).to(
            self.device, dtype=self.dtype
        )

        # OpenVLA predict_action 返回 np.ndarray (7,)，顺序：
        # [dx, dy, dz, drx, dry, drz, gripper]
        # 备注：unnorm_key="bridge_orig" 训练版下顺序与 Action7D 一致；
        # 其他 unnorm_key（如 LIBERO 微调版）顺序可能不同，使用前需核对。
        action_arr: np.ndarray = self._vla.predict_action(
            **inputs, unnorm_key=self.unnorm_key, do_sample=False
        )

        if action_arr.shape != (7,):
            raise RuntimeError(
                f"OpenVLA 预期返回 7D 动作，实际形状 {action_arr.shape}。"
                f"unnorm_key={self.unnorm_key!r} 可能与训练数据集不匹配。"
            )

        dx, dy, dz, drx, dry, drz, gripper = (float(x) for x in action_arr.tolist())
        # iter10 chunk 契约：values 为 np.ndarray shape (N, action_dim)，单步 N=1
        values = np.array(
            [[dx, dy, dz, drx, dry, drz, gripper]], dtype=float
        )  # shape (1, 7)
        return VLAOutput(values=values, spec=self.output_spec)
