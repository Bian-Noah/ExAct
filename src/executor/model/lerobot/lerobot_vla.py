"""LeRobot 后端实现。

通过 pip install lerobot 后从 lerobot.common.policies 加载预置 policy（默认 ACT），
封装 predict(image, instruction) -> Action7D。

仅推理（不覆盖训练、LeRobotDataset 加载、硬件遥操作、lerobot CLI）。

⚠️ 动作语义演示说明：
ACT 在 ALOHA 等数据集上训练时输出维度是 14（双臂 × 7 维）。本实现仅截取前 7 维
按 [dx, dy, dz, drx, dry, drz, gripper] 顺序填充为 Action7D。剩余 7 维丢弃。
该映射仅作"接口对齐"演示，不保证物理语义一致。真实部署前需根据所用数据集与
policy 输出维度做映射修正。

构造仅保存参数，首次调用 predict 时才下载权重并移至 device（懒加载）。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch

from env.base import Action7D
from executor.model.base import BaseVLA

# 支持的 policy 类型（与 lerobot.common.policies 子模块名一致）
SUPPORTED_POLICY_TYPES: frozenset[str] = frozenset(
    {"act", "diffusion", "vqbet", "smolvla", "pi0", "pi0fast"}
)


class LeRobotVLA(BaseVLA):
    """LeRobot policy 后端（仅推理）。

    Attributes:
        model_path: HF Hub repo_id（如 "lerobot/act_aloha_sim_transfer_cube_human"）
            或本地 policy 目录路径。
        policy_type: policy 类型，决定从 lerobot.common.policies.<type> 导入哪个类。
            常见值：act（轻量、默认）、diffusion、smolvla、pi0。
        device: 推理设备，默认 "cuda:0"。
        image_key: obs dict 中图像键名。默认 "observation.images.top"；
            实际部署时建议从 policy.config 推导（见 predict 中的自动探测）。
        action_dim: policy 输出维度；超过 7 时只截前 7 维。
            默认 14（ACT ALOHA 训练版常见值）。
    """

    def __init__(
        self,
        model_path: str,
        policy_type: str = "act",
        device: str = "cuda:0",
        image_key: str = "observation.images.top",
        action_dim: int = 14,
    ):
        if not model_path or not isinstance(model_path, str):
            raise ValueError(f"model_path 必须为非空字符串，得到 {model_path!r}")
        if policy_type not in SUPPORTED_POLICY_TYPES:
            raise ValueError(
                f"policy_type={policy_type!r} 不受支持，"
                f"可选：{sorted(SUPPORTED_POLICY_TYPES)}"
            )

        self.model_path = model_path
        self.policy_type = policy_type
        self.device = device
        self.image_key = image_key
        self.action_dim = action_dim

        # 懒加载占位
        self._policy: Any = None
        # 从 policy.config 自动推导出来的 obs 键
        self._resolved_image_key: Optional[str] = None
        self._resolved_state_key: Optional[str] = None

        self._log = logging.getLogger("lerobot_vla")

    # ------------------------------------------------------------------
    # 内部：懒加载 + obs 键探测
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """首次调用 predict 时加载 policy。

        通过 lerobot.common.policies.<type>.modeling_<type>.<Type>Policy.from_pretrained
        加载。若 lerobot 未安装抛 ImportError，让上层能分支处理。
        """
        if self._policy is not None:
            return

        try:
            from lerobot.common.policies import (
                act as _act_mod,  # noqa: F401
            )
        except ImportError as e:
            raise ImportError(
                "LeRobot 后端需要 lerobot 包。请安装：pip install lerobot>=0.4"
            ) from e

        self._log.info(
            f"LeRobot 懒加载开始 model_path={self.model_path} "
            f"policy_type={self.policy_type} device={self.device}"
        )

        # 按 policy_type 动态导入对应 Policy 类
        policy_cls = self._import_policy_class(self.policy_type)
        self._policy = policy_cls.from_pretrained(self.model_path)
        try:
            self._policy.to(self.device)
        except Exception as e:  # pragma: no cover - 设备迁移失败通常无 CUDA
            self._log.warning(f"policy.to({self.device}) 失败：{e}")
        try:
            self._policy.eval()
        except AttributeError:
            pass

        # 从 policy.config 自动探测 obs 键（仅当用户给了默认 top 才覆盖）
        self._resolve_obs_keys()

        self._log.info("LeRobot 懒加载完成")

    @staticmethod
    def _import_policy_class(policy_type: str):
        """根据 policy_type 字符串动态导入 lerobot Policy 类。

        映射表（与 SUPPORTED_POLICY_TYPES 一致）：
            act       -> ACTPolicy
            diffusion -> DiffusionPolicy
            vqbet     -> VQBeTPolicy
            smolvla   -> SmolVLAPolicy
            pi0       -> PI0Policy
            pi0fast   -> PI0FastPolicy
        """
        try:
            if policy_type == "act":
                from lerobot.common.policies.act.modeling_act import ACTPolicy
                return ACTPolicy
            if policy_type == "diffusion":
                from lerobot.common.policies.diffusion.modeling_diffusion import (
                    DiffusionPolicy,
                )
                return DiffusionPolicy
            if policy_type == "vqbet":
                from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy
                return VQBeTPolicy
            if policy_type == "smolvla":
                from lerobot.common.policies.smolvla.modeling_smolvla import (
                    SmolVLAPolicy,
                )
                return SmolVLAPolicy
            if policy_type == "pi0":
                from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
                return PI0Policy
            if policy_type == "pi0fast":
                from lerobot.common.policies.pi0fast.modeling_pi0fast import (
                    PI0FastPolicy,
                )
                return PI0FastPolicy
        except ImportError as e:
            raise ImportError(
                f"导入 lerobot policy={policy_type!r} 失败：{e}。"
                f"请检查 lerobot 版本是否包含该 policy。"
            ) from e
        raise ValueError(f"未知 policy_type: {policy_type!r}")

    def _resolve_obs_keys(self) -> None:
        """从 policy.config 自动探测图像与状态 obs 键名。

        若 self._policy 有 config.input_shapes / output_shapes dict，
        则从中挑选第一个图像键（值含 'image'）和第一个状态键（值含 'state'）。
        探测失败时保留构造时给定的 image_key。
        """
        self._resolved_image_key = self.image_key
        self._resolved_state_key = "observation.state"

        try:
            cfg = getattr(self._policy, "config", None)
            if cfg is None:
                return
            input_shapes = getattr(cfg, "input_shapes", None) or {}
            if not isinstance(input_shapes, dict):
                return
            image_keys = [k for k in input_shapes if "image" in k.lower()]
            state_keys = [k for k in input_shapes if "state" in k.lower()]
            if image_keys:
                self._resolved_image_key = image_keys[0]
            if state_keys:
                self._resolved_state_key = state_keys[0]
        except Exception as e:  # pragma: no cover - 探测失败不影响主流程
            self._log.debug(f"obs 键自动探测失败，保留默认：{e}")

    # ------------------------------------------------------------------
    # 内部：obs 构造 + 动作后处理
    # ------------------------------------------------------------------

    def _build_obs(self, image: np.ndarray, instruction: str) -> dict:
        """把 (image, instruction) 包装为 LeRobot policy 期望的 obs dict。

        Returns:
            dict 至少含:
              - {image_key}: torch.Tensor (1, C, H, W) float32，范围 [0,1]
              - {state_key}: torch.Tensor (1, state_dim) float32（占位零向量）
              - "observation.language_instruction": str（部分 policy 会用到）
        """
        # image: (H, W, 3) uint8 → (1, 3, H, W) float32 in [0, 1]
        if image.ndim == 3:
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        else:  # 已是 (C, H, W)
            tensor = torch.from_numpy(image).unsqueeze(0)
        tensor = tensor.float() / 255.0
        tensor = tensor.to(self.device)

        # state 用零向量占位；真实部署需传入当前关节角
        try:
            state_dim = self._infer_state_dim()
        except Exception:
            state_dim = 14
        state = torch.zeros((1, state_dim), dtype=torch.float32, device=self.device)

        obs: dict = {
            self._resolved_image_key or self.image_key: tensor,
            self._resolved_state_key or "observation.state": state,
            "observation.language_instruction": instruction,
        }
        return obs

    def _infer_state_dim(self) -> int:
        """从 policy.config 推断 state 维度；失败时返回 14。"""
        try:
            cfg = getattr(self._policy, "config", None)
            input_shapes = getattr(cfg, "input_shapes", None) or {}
            for k in input_shapes:
                if "state" in k.lower():
                    shape = input_shapes[k]
                    if isinstance(shape, (list, tuple)) and len(shape) >= 1:
                        return int(shape[0])
        except Exception:
            pass
        return 14

    # ------------------------------------------------------------------
    # BaseVLA.predict
    # ------------------------------------------------------------------

    def predict(self, image: np.ndarray, instruction: str) -> Action7D:
        """输入图片 + 自然语言指令，输出 7D 动作。

        Args:
            image: np.ndarray (H, W, 3) uint8，范围 [0, 255]。
            instruction: 自然语言指令字符串。

        Returns:
            Action7D NamedTuple，7 字段依次为
            dx/dy/dz（米）/ drx/dry/drz（弧度）/ gripper（[0,1]）。

        注意：LeRobot policy 原始输出维度由训练数据集决定（ACT ALOHA 为 14）。
        本实现截取前 7 维当作 Action7D，剩余维度丢弃；语义未对齐属已知风险。

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

        obs = self._build_obs(image, instruction)

        with torch.no_grad():
            action = self._policy.select_action(obs)

        # select_action 多种返回类型：torch.Tensor / np.ndarray
        if isinstance(action, torch.Tensor):
            action_np: np.ndarray = action.detach().cpu().numpy()
        else:
            action_np = np.asarray(action)

        # select_action 在 ACT 下会返回 (action_dim,)；也可能是 (1, action_dim)
        if action_np.ndim == 2:
            action_np = action_np[0]

        if action_np.shape[0] < 7:
            raise RuntimeError(
                f"LeRobot policy 输出维度 {action_np.shape[0]} 不足 7 维，"
                f"无法映射到 Action7D。请检查 model_path 与 action_dim 设置。"
            )

        # 截前 7 维（已知语义不对齐风险，docstring 已说明）
        first7 = action_np[:7].astype(float).tolist()
        dx, dy, dz, drx, dry, drz, gripper = first7
        return Action7D(dx, dy, dz, drx, dry, drz, gripper)
