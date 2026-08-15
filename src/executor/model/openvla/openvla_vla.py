"""OpenVLA 后端实现。

通过 HuggingFace AutoModelForVision2Seq / AutoProcessor 加载 OpenVLA-7B 远端权重，
封装 predict(image, instruction) -> Action7D。

仅推理（不覆盖 LoRA 微调、REST 部署）。依赖见 requirements-stage2.txt。

构造仅保存参数，首次调用 predict 时才下载权重并移至 device（懒加载）。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image

from env.base import Action7D, ActionSpec
from executor.model.base import BaseVLA, VLAOutput

# 默认提示模板（与 OpenVLA 官方 README 完全一致）
DEFAULT_PROMPT_TEMPLATE: str = (
    "In: What action should the robot take to {instruction}?\nOut:"
)


class OpenVLA(BaseVLA):
    """OpenVLA-7B 后端（仅推理）。

    Attributes:
        model_path: HF Hub 远端 ID（如 "openvla/openvla-7b"）或本地 checkpoint 路径。
        unnorm_key: 反归一化键名，决定动作的物理量纲。常见值：
            "bridge_orig"（BridgeData V2 训练版）；不同微调版可能不同。
        attn_impl: attention 实现，"flash_attention_2" 需安装 flash-attn；
            无 GPU / 装不上时退回到 "eager"。
        device: 推理设备，默认 "cuda:0"。
        dtype: 推理精度，默认 torch.bfloat16。
    """

    def __init__(
        self,
        model_path: str,
        unnorm_key: str = "bridge_orig",
        attn_impl: str = "flash_attention_2",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        if not model_path or not isinstance(model_path, str):
            raise ValueError(f"model_path 必须为非空字符串，得到 {model_path!r}")
        self.model_path = model_path
        self.unnorm_key = unnorm_key
        self.attn_impl = attn_impl
        self.device = device
        self.dtype = dtype

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

        单独 try/except 让 ImportError（缺 transformers/timm/flash-attn）
        与 HF 下载失败分开上报，便于上层做分支处理。
        """
        if self._vla is not None and self._processor is not None:
            return

        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "OpenVLA 后端需要 transformers / timm / tokenizers。"
                "请安装：pip install -r requirements-stage2.txt"
            ) from e

        self._log.info(
            f"OpenVLA 懒加载开始 model_path={self.model_path} "
            f"attn_impl={self.attn_impl} device={self.device}"
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

        self._vla = AutoModelForVision2Seq.from_pretrained(
            self.model_path, **load_kwargs
        ).to(self.device)

        self._log.info("OpenVLA 懒加载完成")

    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        state: np.ndarray | None = None,
    ) -> VLAOutput:
        """输入图片 + 自然语言指令，输出 7D 动作。

        Args:
            image: np.ndarray (H, W, 3) uint8，范围 [0, 255]。
            instruction: 自然语言指令字符串。
            state: 暂未使用（OpenVLA 走 prompt 链路，不在 state 通道做 proprio）；
                保留参数位以与 BaseVLA 接口对齐。

        Returns:
            Action7D NamedTuple，7 字段依次为
            dx/dy/dz（米）/ drx/dry/drz（弧度）/ gripper（[0,1]）。

        Raises:
            TypeError: image 非 np.ndarray。
            ValueError: image.dtype 非 uint8 或维度不是 3。
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"image 必须为 np.ndarray，得到 {type(image).__name__}"
            )
        if image.dtype != np.uint8:
            raise ValueError(f"image.dtype 必须为 uint8，得到 {image.dtype}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"image 形状必须为 (H, W, 3)，得到 {image.shape}"
            )

        self._ensure_loaded()

        pil_image: Image.Image = Image.fromarray(image)
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
        return VLAOutput(
            values=Action7D(dx, dy, dz, drx, dry, drz, gripper),
            spec=self.output_spec,
        )
