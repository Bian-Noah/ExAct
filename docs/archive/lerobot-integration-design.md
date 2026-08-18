# lerobot-integration 设计文档（轻量模式）

## 设计目标

在 ExAct 项目中接入 HuggingFace **LeRobot** 库的**推理路径**，使其作为 `vla.backend == "lerobot"` 时的真实后端，复用现有 `BaseVLA.predict(image, instruction) -> Action7D` 契约。范围限定为**仅推理**（不含训练、数据集录制、硬件遥操作），默认 policy 选 **ACT**（轻量、纯 PyTorch、纯模仿学习），policy 实现走 **`pip install lerobot`** 后从 `lerobot.common.policies` 导入预置类，不自实现模型结构。目标平台为阶段二的 Linux/macOS + CUDA（RTX 4060 起步）。

## 文件架构设计

```
项目根/
├── src/
│   ├── executor/
│   │   └── model/
│   │       ├── (M) factory.py              # 新增 backend=="lerobot" 分派
│   │       └── lerobot/                    # (N) 新增子包
│   │           ├── (N) __init__.py         # re-export LeRobotVLA
│   │           └── (N) lerobot_vla.py      # LeRobotVLA(BaseVLA) 实现
│   └── config/
│       └── (M) loader.py                   # VLAConfig 增加 lerobot 字段
├── configs/
│   └── (M) default.yaml                    # vla 节补充 lerobot 字段注释
├── requirements-stage2.txt
│   └── (M)                                # 追加 lerobot>=0.4
├── docs/
│   └── (N) lerobot-integration-notes.md    # 落地笔记（可选）
└── tests/
    └── (N) test_lerobot_vla.py             # mock lerobot.policy 的关键分支单测
```

### 关键文件说明

| 文件 | 职责 |
|------|------|
| `src/executor/model/lerobot/lerobot_vla.py` | `LeRobotVLA(BaseVLA)`：构造时按 `vla_config` 加载 LeRobot 预置 policy（默认 `act`），暴露 `predict(image, instruction) -> Action7D` |
| `src/executor/model/lerobot/__init__.py` | `from .lerobot_vla import LeRobotVLA` |
| `src/executor/model/factory.py` | 新增 `backend == "lerobot"` 分派，传 `model_path / policy_type / device` |
| `src/config/loader.py` | `VLAConfig` 新增 `lerobot_policy_type: str`、`lerobot_device: str` 字段 |
| `configs/default.yaml` | `vla.backend` 仍默认 `"mock"`；注释说明 lerobot 用法 |
| `requirements-stage2.txt` | 追加 `lerobot>=0.4` |
| `tests/test_lerobot_vla.py` | monkeypatch `lerobot.common.policies.act.ACTPolicy` 不依赖真实权重 |

## 改动情况

| 操作 | 文件路径 | 说明 |
|------|----------|------|
| 新增 | `src/executor/model/lerobot/lerobot_vla.py` | `LeRobotVLA(BaseVLA)`，封装 ACT policy 的 `select_action(obs)` 为单步 `predict(image, instruction)` |
| 新增 | `src/executor/model/lerobot/__init__.py` | re-export |
| 新增 | `tests/test_lerobot_vla.py` | 关键分支单测（懒加载、obs 构造、7D 映射） |
| 修改 | `src/executor/model/factory.py` | 新增 `lerobot → LeRobotVLA(...)` 分派 |
| 修改 | `src/config/loader.py` | `VLAConfig` 增加 `lerobot_policy_type`、`lerobot_device` 字段（带默认值） |
| 修改 | `configs/default.yaml` | `vla` 节补注释说明 lerobot backend 用法（默认仍 mock） |
| 修改 | `requirements-stage2.txt` | 追加 `lerobot>=0.4`（与官方 PyPI 同步） |

## 数据流

```
env.get_obs()["rgb"] (uint8 HxWx3)
   │
   ▼
ActionTool.run → Executor.run_action → VLAInputBuilder.build_vla_input
   │                                                       │
   │            obs.image + instruction                    │
   ▼                                                       │
LeRobotVLA.predict(image, instruction)                     │
   │                                                       │
   │ 1) 构造 LeRobot 风格 obs dict:                        │
   │    {                                                   │
   │      "observation.images.front": torch tensor(C,H,W), │
   │      "observation.state": torch tensor([...]),         │（可选，关节状态）
   │      "observation.language_instruction": str,           │
   │    }                                                   │
   │ 2) self.policy.select_action(obs) → np.ndarray         │
   │ 3) 截取前 7 维 → Action7D(dx,dy,dz,drx,dry,drz,gripper)│
   ▼                                                       │
Executor.run_action 拿到 Action7D → env.step(action) ──────�
```

数据流关键约束：
- LeRobot policy 的 `select_action` 期望**完整 obs dict**（含 `observation.state` 关节态、`observation.images.<cam_key>` 图像键名），不是裸 `np.ndarray`。`LeRobotVLA` 内部需要**包装层**把 `(image, instruction)` → LeRobot 风格的 obs dict。
- ACT policy 训练时用 `lerobot/aloha_sim_transfer_cube_human` 等数据集，obs 键名因数据集而异。本次轻量设计**仅接入推理**，默认 obs 键取 ACT 通用约定（`observation.images.top`、`observation.state` 长度 14、`observation.language_instruction`）；具体键名从 `policy.config.input_shapes` 自动推导，避免硬编码。
- ACT 输出维度由数据集决定（常见 14 = 6 关节 + 1 夹爪 × 2 臂），与本项目 `Action7D` (7 维) **不完全一致**。本次设计**只取前 7 维并按对应语义填充**，剩余维度丢弃；docstring 标注"语义可能不匹配，仅作接口演示"。

## 关键函数签名

```
LeRobotVLA(model_path: str,
           policy_type: str = "act",
           device: str = "cuda:0",
           image_key: str = "observation.images.top") -> None
- 继承 BaseVLA；构造仅保存参数，不立刻加载模型（lazy）。
- model_path 是 HF Hub repo_id（如 "lerobot/act_aloha_sim_transfer_cube_human"）或本地路径。

LeRobotVLA.predict(image: np.ndarray, instruction: str) -> Action7D
- 首次调用时执行：
    from lerobot.common.policies.act.modeling_act import ACTPolicy
    self.policy = ACTPolicy.from_pretrained(self.model_path).to(self.device).eval()
- 包装 image → tensor、构造 obs dict、调 policy.select_action(obs)。
- 截取动作数组前 7 维，组装为 Action7D。
```

```
create_vla(vla_config: VLAConfig, llm_config: LLMConfig | None = None) -> BaseVLA
- backend == "lerobot" 时返回 LeRobotVLA(
      model_path=vla_config.model_path,
      policy_type=vla_config.lerobot_policy_type,  # "act"
      device=vla_config.lerobot_device,
  )。
```

## 关键设计决策

- **决策点：接入范围**
  - 选择：仅推理（不接 `lerobot-train` / `lerobot-record` / LeRobotDataset 加载 / 硬件抽象）
  - 原因：用户明确选择；与 OpenVLA 接入保持一致的"接口演示"定位。后续 Iteration 可独立扩展数据采集与训练回路。
  - 与官方 README 关系：覆盖 `SoTA Models` 章节的推理用法；不覆盖 `LeRobot Dataset`、`Robots & Control`、`Inference & Evaluation`（CLI）等。

- **决策点：默认 policy**
  - 选择：ACT（Action Chunking Transformer）
  - 原因：ACT 是 LeRobot 中**最轻量的 IL policy**，纯 PyTorch 实现，依赖少，模型权重小（约 80MB），在 RTX 4060 上推理快速；适合作为"接入流程验证"的最小可行选择。Diffusion / SmolVLA / Pi0 需要更大显存或额外依赖。

- **决策点：实现路径**
  - 选择：依赖 `pip install lerobot` 后从 `lerobot.common.policies.<type>.modeling_<type>` 导入预置类
  - 原因：LeRobot 官方仓库本身就是 policy 的官方实现，自实现 ACT 会重复造轮子且难以与上游权重兼容。引入 lerobot PyPI 包是轻量设计的最低代价。

- **决策点：运行环境**
  - 选择：Linux/macOS + CUDA（RTX 4060 起步）
  - 原因：LeRobot 在 Windows 上安装受限；policy 推理默认需要 GPU。Mac (MPS) 理论上可行但 lerobot 兼容性差，本次不覆盖。

- **决策点：动作语义映射**
  - 选择：从 LeRobot 动作数组**截取前 7 维**当作 `Action7D`，剩余维度丢弃
  - 原因：ACT 训练数据集（Aloha 等）输出维度是 14（双臂 × 7），而本项目是单臂 7 维。本设计仅做"接口对齐"演示，不做真实语义修正；docstring 明确标注"演示用，真实部署需根据数据集调整"。

- **决策点：obs 键名推导**
  - 选择：从 `policy.config.input_shapes` 自动推导，不硬编码
  - 原因：不同 policy / 数据集用不同 obs 键名（`top` / `wrist` / `front`），硬编码会限制通用性。`select_action` 需要的 obs 键可以从 policy 配置对象读取。

- **决策点：延迟加载**
  - 选择：`__init__` 仅保存参数；首次 `predict` 才下载权重并移到 device
  - 原因：与 OpenVLA 接入保持一致；让工厂启动零开销。

## 参考资料

- 官方仓库：https://github.com/huggingface/lerobot
- 本地文档：`docs/2026-08-13-github-com-lerobot.md`
- PyPI 包：https://pypi.org/project/lerobot/
- 文档站：https://huggingface.co/docs/lerobot
- 现有相关：与 openvla-integration-design.md 共享 `BaseVLA` 契约与工厂入口
- `BaseVLA` 契约：`src/executor/model/base.py`
- `Action7D` 契约：`src/env/base.py`
