"""LeRobot 后端实现。

通过 pip install lerobot 后从 lerobot.common.policies 加载预置 policy（默认 ACT），
封装 predict(image, instruction) -> Action7D。

仅推理（不覆盖训练、LeRobotDataset 加载、硬件遥操作、lerobot CLI）。

⚠️ 动作语义演示说明：
ACT 在 ALOHA 等数据集上训练时输出维度是 14（双臂 × 7 维）。本实现仅截取前 7 维
按 [dx, dy, dz, drx, dry, drz, gripper] 顺序填充为 Action7D。剩余 7 维丢弃。
该映射仅作"接口对齐"演示，不保证物理语义一致。真实部署前需根据所用数据集与
policy 输出维度做映射修正。

obs 预处理 / 动作后处理统一走 lerobot 官方 PolicyProcessorPipeline：
  preprocessor(raw_obs) -> select_action(preprocessed) -> postprocessor(action)
不手写 obs 拼装，避免与 lerobot 版本 API 脱节。

构造仅保存参数，首次调用 predict 时才加载权重并移至 device（懒加载）。
权重离线优先：本地路径直接用；HF repo_id 仅从缓存读取，绝不自动联网下载。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import torch

from env.base import Action7D
from executor.model.base import BaseVLA

# 支持的 policy 类型（与 lerobot.common.policies 子模块名一致）
SUPPORTED_POLICY_TYPES: frozenset[str] = frozenset(
    {"act", "diffusion", "vqbet", "smolvla", "pi0", "pi0fast"}
)

# 依赖归一化统计（dataset_stats）的 policy：纯推理无数据集时无法归一化。
# Pi0/SmolVLA 图像归一化写死、action 直接输出，可脱离数据集推理；ACT/Diffusion
# 等 IL policy 依赖 obs/action 归一化，无 stats 时推理结果可能不准（不报错）。
STATS_FREE_POLICY_TYPES: frozenset[str] = frozenset({"smolvla", "pi0", "pi0fast"})


class LeRobotVLA(BaseVLA):
    """LeRobot policy 后端（仅推理）。

    Attributes:
        model_path: HF Hub repo_id（如 "lerobot/act_aloha_sim_transfer_cube_human"）
            或本地 policy 目录路径。
        policy_type: policy 类型，决定从 lerobot.common.policies.<type> 导入哪个类。
            常见值：act（轻量、默认）、diffusion、smolvla、pi0。
        device: 推理设备，默认 "cuda:0"。
        quantization: 量化等级，"none" | "8bit" | "4bit"。仅对 VLA 类
            policy（smolvla / pi0 / pi0fast）生效；ACT / Diffusion 不量化。
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
        quantization: str = "none",
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
        if quantization not in ("none", "8bit", "4bit"):
            raise ValueError(
                f"quantization={quantization!r} 不受支持，可选：none / 8bit / 4bit"
            )

        self.model_path = model_path
        self.policy_type = policy_type
        self.device = device
        self.quantization = quantization
        self.image_key = image_key
        self.action_dim = action_dim

        # 懒加载占位
        self._policy: Any = None
        self._preprocessor: Any = None
        self._postprocessor: Any = None
        # 从 policy.config 自动推导出来的 obs 键
        self._resolved_image_key: Optional[str] = None
        self._resolved_state_key: Optional[str] = None

        self._log = logging.getLogger("lerobot_vla")

    # ------------------------------------------------------------------
    # 内部：policy 类型判定
    # ------------------------------------------------------------------

    def _is_vla_policy(self) -> bool:
        """判断当前 policy 是否基于 transformers VLM（可量化）。

        返回 True 的类型：smolvla / pi0 / pi0fast。
        ACT / Diffusion / VQ-BeT 是纯 PyTorch IL policy，无量化概念。
        """
        return self.policy_type in {"smolvla", "pi0", "pi0fast"}

    # ------------------------------------------------------------------
    # 内部：懒加载 + obs 键探测
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """首次调用 predict 时加载 policy + 处理器管线。

        通过 lerobot.common.policies.<type>.modeling_<type>.<Type>Policy.from_pretrained
        加载 policy，再用官方 make_pre_post_processors 生成 obs 预处理与动作后处理管线。
        若 lerobot 未安装抛 ImportError，让上层能分支处理。
        """
        if self._policy is not None:
            return

        try:
            import lerobot  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "LeRobot 后端需要 lerobot 包。请安装：pip install lerobot>=0.6"
            ) from e

        self._log.info(
            f"LeRobot 懒加载开始 model_path={self.model_path} "
            f"policy_type={self.policy_type} device={self.device}"
        )

        # 离线优先：确认权重在本地，否则抛明确提示（禁止自动联网下载）
        self._check_model_available_locally()

        # 按 policy_type 动态导入对应 Policy 类
        policy_cls = self._import_policy_class(self.policy_type)
        # 离线优先：尝试 local_files_only=True 只读缓存，禁止联网下载
        try:
            self._policy = policy_cls.from_pretrained(
                self.model_path, local_files_only=True
            )
        except TypeError:
            # 个别 policy（如 Pi0 自定义签名）不接受 local_files_only，
            # 回退普通加载。本地权重已由 _check_model_available_locally 校验，
            # 不会触发真实联网下载；HF repo_id 场景已在检查时拦截。
            self._policy = policy_cls.from_pretrained(self.model_path)
        try:
            self._policy.to(self.device)
        except Exception as e:  # pragma: no cover - 设备迁移失败通常无 CUDA
            self._log.warning(f"policy.to({self.device}) 失败：{e}")
        try:
            self._policy.eval()
        except AttributeError:
            pass

        # 新增：VLA 类 policy + 量化配置 → 加载后 post-quantization
        if self.quantization in ("8bit", "4bit") and self._is_vla_policy():
            self._apply_quantization()

        # 创建官方 pre/post 处理器管线（离线优先，不需要外部数据集 stats）
        self._build_processors()

        # 从 policy.config 自动探测 obs 键
        self._resolve_obs_keys()

        self._log.info("LeRobot 懒加载完成")

    def _apply_quantization(self) -> None:
        """对 VLA 类 policy 的 VLM 骨干做 bnb 量化（加载后模块替换）。

        量化目标是 policy.model（真正的 nn.Module），而非 policy wrapper。
        bitsandbytes 缺失时抛 ImportError（提示可关闭量化）。
        """
        from utils.quantize import quantize_model_4bit

        bits = 4 if self.quantization == "4bit" else 8
        self._log.info(f"对 {self.policy_type} policy 应用 {bits}bit 量化")
        target = getattr(self._policy, "model", self._policy)
        quantize_model_4bit(target, bits=bits, compute_dtype=torch.bfloat16)

    def _build_processors(self) -> None:
        """用官方 make_pre_post_processors 创建 obs 预处理与动作后处理管线。

        纯推理场景不传 dataset_stats（None）：依赖归一化的 policy（ACT/Diffusion）
        会跳过归一化（可能精度下降但可运行）；Pi0/SmolVLA 图像归一化写死、可离线推理。

        Raises:
            RuntimeError: 无法从 lerobot 找到 make_pre_post_processors 入口。
        """
        try:
            from lerobot.policies import make_pre_post_processors
        except ImportError:
            raise RuntimeError(
                "当前 lerobot 版本找不到 make_pre_post_processors 入口。"
                "请升级 lerobot（pip install -U lerobot）或改用 policy_type 直接加载。"
            )

        cfg = getattr(self._policy, "config", None)
        if cfg is None:
            raise RuntimeError("policy 缺少 config，无法构造处理器管线")

        try:
            self._preprocessor, self._postprocessor = make_pre_post_processors(
                cfg, pretrained_path=self.model_path, dataset_stats=None
            )
        except Exception as e:  # pragma: no cover - API 差异
            raise RuntimeError(
                f"make_pre_post_processors 构造失败：{e}。"
                f"请检查 lerobot 版本与 policy_type={self.policy_type!r} 是否匹配。"
            ) from e
        self._log.info(
            f"处理器管线已创建：preprocessor={type(self._preprocessor).__name__}, "
            f"postprocessor={type(self._postprocessor).__name__}"
        )

    # ------------------------------------------------------------------
    # 内部：离线检查（本地权重优先，禁止自动联网下载）
    # ------------------------------------------------------------------

    def _is_hub_repo_id(self) -> bool:
        """判断 model_path 是否为 HF Hub repo_id（形如 owner/repo），而非本地路径。

        判定规则：
          - 本地已存在路径 → 不是 repo_id
          - 形如 "owner/repo"：owner 是合法 HF 用户名段（不以 /、. 、~ 开头，
            不含路径分隔符），repo 段同理 → 是 repo_id
          - 其余（绝对路径 /tmp/...、相对路径 ../../x、带后缀 x.pt 等）→ 本地路径
        """
        if os.path.isdir(self.model_path) or os.path.isfile(self.model_path):
            return False
        parts = self.model_path.split("/")
        if len(parts) != 2:
            return False
        owner, repo = parts
        return (
            len(owner) > 0
            and len(repo) > 0
            and not owner.startswith((".", "~", "/"))
            and not repo.startswith((".", "~", "/"))
            and "." not in owner
            and "/" not in owner
            and "/" not in repo
        )

    def _check_model_available_locally(self) -> None:
        """确认权重在本地可用，否则抛明确提示。

        离线优先策略：model_path 为本地路径时直接使用；为 HF repo_id 时
        仅允许从已下载缓存读取，缓存不存在则报错并给出手动下载命令，
        绝不自动联网下载占用缓存。

        Raises:
            FileNotFoundError: 本地路径不存在。
            RuntimeError: HF repo_id 但本地缓存中没有该权重。
        """
        if not self._is_hub_repo_id():
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"model_path 指向的本地路径不存在：{self.model_path}"
                )
            self._log.info(f"使用本地权重：{self.model_path}")
            return

        cached = self._cached_hub_path(self.model_path)
        if cached is None:
            raise RuntimeError(
                f"HuggingFace 权重 '{self.model_path}' 未在本地缓存中找到。\n"
                f"本项目为离线优先，不会自动联网下载。请手动下载到缓存后再运行：\n"
                f"  python -c \"from huggingface_hub import snapshot_download; "
                f"snapshot_download('{self.model_path}')\"\n"
                f"缓存目录：{os.path.expanduser('~/.cache/huggingface/hub')}"
            )
        self._log.info(f"使用 HuggingFace 本地缓存权重：{self.model_path}（{cached}）")

    def _cached_hub_path(self, repo_id: str) -> Optional[str]:
        """返回 repo_id 的本地 HF 缓存目录路径；未缓存返回 None。

        HF 缓存布局：~/.cache/huggingface/hub/models--{owner}--{repo}
        （repo 名中的点替换为下划线）。
        """
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
        except ImportError:
            return None
        repo_cache = os.path.join(
            HF_HUB_CACHE, "models--" + repo_id.replace("/", "--").replace(".", "_")
        )
        return repo_cache if os.path.isdir(repo_cache) else None

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
    # 内部：原始 obs 构造（交给官方 preprocessor 处理）
    # ------------------------------------------------------------------

    def _build_raw_obs(self, image: np.ndarray, instruction: str) -> dict:
        """把 (image, instruction) 包装为 preprocessor 期望的原始 obs dict。

        注意：这里只提供"原始"观测（HWC uint8 图像 + 状态向量 + 语言指令字符串），
        batch、归一化、语言 tokenize 等由官方 preprocessor 管线完成，不在此手拼。
        """
        # image 保持 (H, W, 3) uint8 numpy，交给 preprocessor 处理
        obs: dict = {
            self._resolved_image_key or self.image_key: image,
            "observation.language_instruction": instruction,
        }
        # state 用零向量占位；真实部署需传入当前关节角
        state_dim = self._infer_state_dim()
        obs[self._resolved_state_key or "observation.state"] = np.zeros(
            (state_dim,), dtype=np.float32
        )
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

        推理链路（官方管线）：
          raw_obs = _build_raw_obs(image, instruction)   # 原始 obs dict
          preprocessed = preprocessor(raw_obs)           # batch/归一化/tokenize
          action = policy.select_action(preprocessed)    # 单步 (1, action_dim)
          action = postprocessor(action)                 # 反归一化
          → 截前 7 维 → Action7D

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

        raw_obs = self._build_raw_obs(image, instruction)
        preprocessed = self._preprocessor(raw_obs)

        with torch.no_grad():
            action = self._policy.select_action(preprocessed)

        action = self._postprocessor(action)

        # postprocessor 输出为 (1, action_dim) 或 (action_dim,)
        if isinstance(action, torch.Tensor):
            action_np: np.ndarray = action.detach().cpu().numpy()
        else:
            action_np = np.asarray(action)
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
