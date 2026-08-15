"""LeRobot 后端实现。

通过 pip install lerobot 后从 lerobot.policies 加载预置 policy（默认 ACT），
封装 predict(image, instruction) -> joint 空间动作（VLAOutput，values 为
np.ndarray float，维度由 action_dim 决定）。

lerobot 路径变更：
  - 0.5 及更早：policy 类位于 lerobot.common.policies.<type>.modeling_<type>
  - 0.6+     ：policy 类位于 lerobot.policies.<type>.modeling_<type>（common 层被移除），
    同时 pi0fast 子包改名为 pi0_fast。本项目 requirements-stage2.txt 锁 >=0.6，
    _import_policy_class 优先用新路径，失败时回退旧路径以便兼容。

仅推理（不覆盖训练、LeRobotDataset 加载、硬件遥操作、lerobot CLI）。

动作语义：模型输出即关节角（joint 空间），原样返回，维度裁剪/补维交给
(joint, joint) adapter（按 env.input_spec 对齐）。不在此处映射为 task 空间。

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

from env.base import Action7D, ActionSpec
from executor.model.base import BaseVLA, VLAOutput

# 支持的 policy 类型（与 lerobot.policies 子模块名一致；pi0fast 在 0.6+ 改名为 pi0_fast）
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
        policy_type: policy 类型，决定从 lerobot.policies.<type> 导入哪个类（0.5 及
            更早为 lerobot.common.policies.<type>）。常见值：act（轻量、默认）、
            diffusion、smolvla、pi0。
        device: 推理设备，默认 "cuda:0"。
        quantization: 量化等级，"none" | "8bit" | "4bit"。仅对 VLA 类
            policy（smolvla / pi0 / pi0fast）生效；ACT / Diffusion 不量化。
        image_key: obs dict 中图像键名。默认 "observation.images.top"；
            实际部署时建议从 policy.config 推导（见 predict 中的自动探测）。
        action_dim: policy 输出维度（joint 空间）。spec 与 predict 返回值维度
            均由此决定，不硬编码。默认 14（ACT ALOHA 训练版常见值；
            SO101 单臂常见 6）。
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
        # policy 输出 <7 维时补到 Action7D 的 gripper 占位值；0.5=半开
        self._default_gripper: float = 0.5

        # 懒加载占位
        self._policy: Any = None
        self._preprocessor: Any = None
        self._postprocessor: Any = None
        # 从 policy.config 自动推导出来的 obs 键
        # lerobot 0.6+ 用 input_features（dict[PolicyFeature]），老版本用 input_shapes
        # 图像键可能有多个（多相机 SO101 就有 overhead+wrist）
        self._resolved_image_keys: list[str] = []
        self._resolved_state_key: Optional[str] = None

        self._log = logging.getLogger("lerobot_vla")

    @property
    def output_spec(self) -> ActionSpec:
        """LeRobot 输出 joint 空间动作，维度由 action_dim 决定（不硬编码 7）。

        spec 是模型的固有属性（训练数据集决定动作维度），不落入 config。
        """
        return ActionSpec("joint", ("joint",) * self.action_dim)

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

        通过 lerobot.policies.<type>.modeling_<type>.<Type>Policy.from_pretrained
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
        # 0.6+ 新路径：去掉 common. 层；pi0fast 改名为 pi0_fast
        new_paths: dict[str, tuple[str, str]] = {
            "act":       ("lerobot.policies.act.modeling_act",             "ACTPolicy"),
            "diffusion": ("lerobot.policies.diffusion.modeling_diffusion", "DiffusionPolicy"),
            "vqbet":     ("lerobot.policies.vqbet.modeling_vqbet",         "VQBeTPolicy"),
            "smolvla":   ("lerobot.policies.smolvla.modeling_smolvla",     "SmolVLAPolicy"),
            "pi0":       ("lerobot.policies.pi0.modeling_pi0",             "PI0Policy"),
            "pi0fast":   ("lerobot.policies.pi0_fast.modeling_pi0_fast",   "PI0FastPolicy"),
        }
        # 0.5- 旧路径：保留以便 lerobot<0.6 环境也能跑（超出 requirements 范围，但便宜）
        old_paths: dict[str, tuple[str, str]] = {
            "act":       ("lerobot.common.policies.act.modeling_act",             "ACTPolicy"),
            "diffusion": ("lerobot.common.policies.diffusion.modeling_diffusion", "DiffusionPolicy"),
            "vqbet":     ("lerobot.common.policies.vqbet.modeling_vqbet",         "VQBeTPolicy"),
            "smolvla":   ("lerobot.common.policies.smolvla.modeling_smolvla",     "SmolVLAPolicy"),
            "pi0":       ("lerobot.common.policies.pi0.modeling_pi0",             "PI0Policy"),
            "pi0fast":   ("lerobot.common.policies.pi0fast.modeling_pi0fast",     "PI0FastPolicy"),
        }

        last_err: Optional[Exception] = None
        for paths in (new_paths, old_paths):
            if policy_type not in paths:
                continue
            mod_path, cls_name = paths[policy_type]
            try:
                import importlib
                module = importlib.import_module(mod_path)
                return getattr(module, cls_name)
            except ImportError as e:
                last_err = e
                continue

        if last_err is not None:
            raise ImportError(
                f"导入 lerobot policy={policy_type!r} 失败：{last_err}。"
                f"请检查 lerobot 版本是否包含该 policy。"
            ) from last_err
        raise ValueError(f"未知 policy_type: {policy_type!r}")

    def _resolve_obs_keys(self) -> None:
        """从 policy.config 自动探测所有图像键与状态 obs 键名。

        lerobot 0.6+ 把特征元数据放在 config.input_features（dict[str, PolicyFeature]）；
        每个 PolicyFeature 有 .type（VISUAL/STATE/...）和 .shape。
        多相机 policy（如 SO101 overhead+wrist）需要把所有 VISUAL 键都填到 obs dict，
        否则下游 predict_action_chunk 会因 KeyError 中断。

        探测失败时回退到构造时给定的 image_key（单相机占位）。
        """
        self._resolved_image_keys = []
        self._resolved_state_key = "observation.state"

        try:
            cfg = getattr(self._policy, "config", None)
            if cfg is None:
                return
            input_features = getattr(cfg, "input_features", None) or {}
            if not isinstance(input_features, dict):
                return

            from lerobot.configs.types import FeatureType

            # 优先按 FeatureType 取：所有 VISUAL 是图像键，第一个 STATE 是状态键
            for k, ft in input_features.items():
                if getattr(ft, "type", None) == FeatureType.VISUAL:
                    self._resolved_image_keys.append(k)
            if not self._resolved_image_keys:
                # 退化：按 key 名匹配（兼容老 lerobot 或自定义 feature）
                for k in input_features:
                    if "image" in k.lower():
                        self._resolved_image_keys.append(k)

            for k, ft in input_features.items():
                if getattr(ft, "type", None) == FeatureType.STATE:
                    self._resolved_state_key = k
                    break
            if self._resolved_state_key == "observation.state":
                # 退化：按 key 名匹配
                for k in input_features:
                    if "state" in k.lower():
                        self._resolved_state_key = k
                        break
        except Exception as e:  # pragma: no cover - 探测失败不影响主流程
            self._log.debug(f"obs 键自动探测失败，保留默认：{e}")

        # 最终 fallback：没探测到图像键时用构造时的 image_key 占位（单相机）
        if not self._resolved_image_keys:
            self._resolved_image_keys = [self.image_key]

    # ------------------------------------------------------------------
    # 内部：原始 obs 构造（交给官方 preprocessor 处理）
    # ------------------------------------------------------------------

    def _build_raw_obs(
        self,
        image: np.ndarray,
        instruction: str,
        state: np.ndarray | None = None,
    ) -> dict:
        """把 (image, instruction[, state]) 包装为 preprocessor 期望的原始 obs dict。

        图像预处理（关键）：
          输入  image : numpy uint8 (H, W, 3)
          输出  tensor : torch float32 (1, 3, H, W)，值域 [0, 1]

        必须做这一步，原因是这个权重保存的 preprocessor.json 里漏装了
        "uint8 HWC → float32 BCHW / 255" 的转换步骤。直接喂 uint8 HWC 会
        让 normalizer 尝试把 float32 stats 转 uint8 → overflow；并且即使
        dtype 修好，stats 形状 (3,1,1) 与 HWC 布局广播会算错维度。

        多相机 policy：同一张 image 复制到所有 VISUAL obs 键（占位用；
        真实部署需 env 端传入各相机各自帧）。

        注意：这里只做"原始"观测的 dtype/layout 转换，batch、归一化、tokenize
        等由官方 preprocessor 管线完成，不在此手拼。
        """
        import torch  # 仅此处局部 import，避免模块级 torch 依赖被无谓触发
        img_tensor = torch.from_numpy(image)                          # (H, W, 3) uint8
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)         # (1, 3, H, W) uint8
        img_tensor = img_tensor.contiguous().float().div_(255.0)       # (1, 3, H, W) float32 [0,1]

        obs: dict = {}
        for key in self._resolved_image_keys or [self.image_key]:
            obs[key] = img_tensor
        obs["observation.language_instruction"] = instruction
        # state：优先用真实关节角（由 env.get_joint_state() 注入）；
        # None 或维度不匹配时回退到零向量占位（向后兼容）。
        # 防御性：直接放到 self.device，不依赖 device_processor 是否正确处理。
        # 否则若 saved preprocessor 的 device_processor 配置与 self.device 不一致，
        # 或某步对 state 走了 numpy 兜底分支，state 会留 CPU，与 GPU 模型权重 matmul 时
        # 触发 device mismatch（cuda:0 vs cpu）。
        state_dim = self._infer_state_dim()
        state_tensor: torch.Tensor
        if state is not None and len(state) == state_dim:
            state_tensor = torch.as_tensor(
                np.asarray(state, dtype=np.float32), device=self.device
            )
        else:
            if state is not None:
                self._log.warning(
                    f"state 维度 {len(state) if hasattr(state, '__len__') else '?'} "
                    f"与 policy.state_dim {state_dim} 不匹配，回退到零向量"
                )
            state_tensor = torch.zeros(
                (state_dim,), dtype=torch.float32, device=self.device
            )
        obs[self._resolved_state_key or "observation.state"] = state_tensor
        return obs

    def _infer_state_dim(self) -> int:
        """从 policy.config 推断 state 维度；失败时返回 6（SO101 单臂 6 自由度）。"""
        try:
            cfg = getattr(self._policy, "config", None)
            input_features = getattr(cfg, "input_features", None) or {}
            if not isinstance(input_features, dict):
                return 6

            from lerobot.configs.types import FeatureType

            # 优先按 FeatureType 取 STATE 特征的 shape
            for k, ft in input_features.items():
                if getattr(ft, "type", None) == FeatureType.STATE:
                    shape = getattr(ft, "shape", None)
                    if shape and len(shape) >= 1:
                        return int(shape[0])
            # 退化：按 key 名匹配
            for k, ft in input_features.items():
                if "state" in k.lower():
                    shape = getattr(ft, "shape", None)
                    if shape and len(shape) >= 1:
                        return int(shape[0])
        except Exception:
            pass
        return 6

    # ------------------------------------------------------------------
    # BaseVLA.predict
    # ------------------------------------------------------------------

    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        state: np.ndarray | None = None,
    ) -> VLAOutput:
        """输入图片 + 自然语言指令，输出 joint 空间动作。

        推理链路（官方管线）：
          raw_obs = _build_raw_obs(image, instruction, state)  # 原始 obs dict
          preprocessed = preprocessor(raw_obs)                 # batch/归一化/tokenize
          action = policy.select_action(preprocessed)          # 单步 (1, action_dim)
          action = postprocessor(action)                       # 反归一化
          → 原样返回 joint 空间动作（维度由 action_dim 决定，不硬编码）

        Args:
            image: np.ndarray (H, W, 3) uint8，范围 [0, 255]。
            instruction: 自然语言指令字符串。
            state: np.ndarray 或 None。当前机器人关节角，作为
                observation.state 喂给 policy。None 时回退到零向量
                （旧行为，向后兼容）。

        Returns:
            VLAOutput：values 为 np.ndarray float（joint 空间，dim=action_dim），
            spec 为 joint 空间同维度。

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

        raw_obs = self._build_raw_obs(image, instruction, state)
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

        # joint 空间直通：模型输出即为关节角，原样返回。
        # 维度裁剪/补维交给 (joint, joint) adapter（按 env.input_spec 对齐）。
        # 旧 task 空间映射已废弃（见 git 历史 1bb7cf6 之前的 task 语义处理）。
        return VLAOutput(
            values=action_np.astype(float),
            spec=self.output_spec,
        )

        # ==== 以下旧后处理已注释（task 空间硬映射，与 joint spec 矛盾） ====
        # # 适配不同 action 维度的 policy：
        # #   7 维 → 原样填入 7 字段（含真实 gripper）
        # #   6 维 → 用配置的默认 gripper 补足第 7 维（SO101 等单臂无独立 gripper 输出）
        # #   其他 <6 → 抛错（连 6 维位置/姿态都不够，没法映射）
        # if action_np.shape[0] < 6:
        #     raise RuntimeError(
        #         f"LeRobot policy 输出维度 {action_np.shape[0]} 不足 6 维，"
        #         f"无法映射到 Action7D。请检查 model_path 与 action_dim 设置。"
        #     )
        #
        # if action_np.shape[0] >= 7:
        #     first7 = action_np[:7].astype(float).tolist()
        #     dx, dy, dz, drx, dry, drz, gripper = first7
        # else:
        #     # 6 维：截 6 个位置/姿态维度 + gripper 用默认占位（模型没输出夹爪信号）
        #     first6 = action_np[:6].astype(float).tolist()
        #     dx, dy, dz, drx, dry, drz = first6
        #     gripper = self._default_gripper
        # return VLAOutput(
        #     values=Action7D(dx, dy, dz, drx, dry, drz, gripper),
        #     spec=self.output_spec,
        # )
