# openvla-integration 设计文档（轻量模式）

## 设计目标

在 ExAct 项目中接入 OpenVLA-7B 模型的**推理路径**，使其作为 `vla.backend == "openvla"` 时的真实后端，复用现有 `BaseVLA.predict(image, instruction) -> Action7D` 契约。范围限定为**仅推理**（不包含 LoRA 微调与 REST 部署），目标平台为阶段二的 RTX 4060 + CUDA。模型权重通过 HuggingFace Hub 远端 ID 直接加载，采用 bf16 + Flash-Attention 2，输出取首 token 作为 7D 增量动作。

## 文件架构设计

```
项目根/
├── src/
│   ├── executor/
│   │   └── model/
│   │       ├── (M) factory.py              # 删 openvla 抛错分支，改成真正分派
│   │       └── openvla/
│   │           ├── (M) README.md           # 占位 README 改为本设计摘要链接
│   │           ├── (N) __init__.py         # 导出 OpenVLA 类
│   │           └── (N) openvla_vla.py      # OpenVLA(BaseVLA) 实现
│   └── config/
│       └── (M) loader.py                   # VLAConfig 增加 openvla 字段（unnorm_key）
├── configs/
│   └── (M) default.yaml                    # vla 节补充 unnorm_key、attn_impl 字段
├── requirements-stage2.txt
│   └── (M)                                # 追加 timm、tokenizers、flash-attn 固定版本
├── docs/
│   └── (N) openvla-integration-notes.md    # 落地后的官方对照笔记（可选）
└── tests/
    └── (N) test_openvla_vla.py             # 关键分支单测：输入校验 / 未加载时报错 / Mock fallback
```

### 关键文件说明

| 文件 | 职责 |
|------|------|
| `src/executor/model/openvla/openvla_vla.py` | `OpenVLA` 类：构造时按 HF ID 加载 `AutoProcessor` + `AutoModelForVision2Seq`，暴露 `predict(image, instruction) -> Action7D` |
| `src/executor/model/openvla/__init__.py` | `from .openvla_vla import OpenVLA`，方便工厂导入 |
| `src/executor/model/factory.py` | 移除 `openvla → NotImplementedError`，改为构造 `OpenVLA(vla_config.model_path, ...)` |
| `src/config/loader.py` | `VLAConfig` 增字段：`unnorm_key: str`、`attn_impl: str`、`device: str` |
| `configs/default.yaml` | `vla.backend` 仍默认 `"mock"`；新增可选字段注释说明 openvla 用法 |
| `requirements-stage2.txt` | 追加 `timm==0.9.10`、`tokenizers==0.19.1`、`flash-attn==2.5.5`（官方 README 锁定的版本） |
| `tests/test_openvla_vla.py` | 走"未安装 transformers → ImportError"等关键失败路径；正常路径用 monkeypatch 模拟 processor/model，避免 CI 拉权重 |

## 改动情况

| 操作 | 文件路径 | 说明 |
|------|----------|------|
| 新增 | `src/executor/model/openvla/openvla_vla.py` | `OpenVLA(BaseVLA)`，负责加载 HF 远端 ID、构造 prompt、`predict_action(**inputs, unnorm_key=..., do_sample=False)` → 7 元组 → `Action7D` |
| 新增 | `src/executor/model/openvla/__init__.py` | re-export `OpenVLA` |
| 新增 | `tests/test_openvla_vla.py` | 不依赖真实权重的单元测试 |
| 修改 | `src/executor/model/factory.py` | 删除 `openvla → NotImplementedError` 分支，新增 `OpenVLA` 分派（传入 model_path / unnorm_key / device） |
| 修改 | `src/executor/model/openvla/README.md` | 占位 → 指向本设计文档的简短说明 |
| 修改 | `src/config/loader.py` | `VLAConfig` 新增 `unnorm_key`、`attn_impl`、`device` 三个字段，全部带默认值 |
| 修改 | `configs/default.yaml` | `vla` 节加注释（新增字段保持默认；用户启用 openvla 时手动指定 model_path） |
| 修改 | `requirements-stage2.txt` | 追加 `timm`、`tokenizers`、`flash-attn` 固定版本（与官方 README 一致） |

## 数据流

```
env.get_obs()["rgb"] (uint8 HxWx3)
   │
   ▼
ActionTool.run → Executor.run_action → VLAInputBuilder.build_vla_input
   │                                                       │
   │            obs.image (np.uint8) + instruction (str)   │
   ▼                                                       │
OpenVLA.predict(image, instruction)                        │
   │                                                       │
   │ 1) PIL.Image.fromarray(image)                         │
   │ 2) prompt = f"In: What action should the robot take to {instruction}?\nOut:"
   │ 3) inputs = processor(prompt, pil_image).to(device, dtype=bf16)
   │ 4) action_arr = vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False)
   │ 5) 转 (dx,dy,dz,drx,dry,drz,gripper) Action7D          │
   ▼                                                       │
Executor.run_action 拿到 Action7D → env.step(action) ─────┘
```

数据流关键约束：
- `image` 必须为 `np.ndarray` (H,W,3) uint8，否则 `PIL.Image.fromarray` 抛错。
- `processor` / `vla` 仅在第一次调用 `predict` 时延迟加载（lazy init），构造 `OpenVLA(model_path)` 不立即联网，方便工厂快速返回。
- `predict_action` 返回 `np.ndarray` 长度 7，顺序由 OpenVLA 模型本身定义；本设计硬编码为 `[dx, dy, dz, drx, dry, drz, gripper]`，并在 docstring 中标注。

## 关键函数签名

```
OpenVLA(model_path: str,
        unnorm_key: str = "bridge_orig",
        attn_impl: str = "flash_attention_2",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16) -> None
- 继承 BaseVLA；构造仅保存参数，不立刻加载模型（lazy）。
- model_path 是 HF Hub ID（如 "openvla/openvla-7b"）。

OpenVLA.predict(image: np.ndarray, instruction: str) -> Action7D
- 第一次调用时加载 processor + vla（trust_remote_code=True）。
- prompt 模板硬编码 "In: What action should the robot take to {instruction}?\nOut:"。
- do_sample=False 取 argmax。

create_vla(vla_config: VLAConfig, llm_config: LLMConfig | None = None) -> BaseVLA
- backend == "openvla" 时返回 OpenVLA(vla_config.model_path,
  unnorm_key=vla_config.unnorm_key,
  attn_impl=vla_config.attn_impl,
  device=vla_config.device)。
- backend 仍为 mock / llm_vla / small_vla 时行为不变。
```

## 关键设计决策

- **决策点：本次接入范围**
  - 选择：仅推理（不含 LoRA 微调、不含 REST 部署）
  - 原因：用户明确选择；减少首次接入的代码量与依赖面（不需要 PEFT / fastapi），后续 Iteration 可独立扩展。
  - 与官方文档关系：完全覆盖 README 中"Getting Started"段落；不覆盖 "Finetune via LoRA" 与 "Evaluating OpenVLA" 章节。

- **决策点：模型权重来源**
  - 选择：HF Hub 远端 ID（如 `openvla/openvla-7b`）
  - 原因：与官方 README 示例一致；`AutoModelForVision2Seq.from_pretrained("openvla/...")` 自动处理 download + trust_remote_code，不需要本地手动注册。
  - 后续扩展：本地 checkpoint 场景可在 `OpenVLA.__init__` 增加路径探测分支，但不纳入本次轻量设计。

- **决策点：推理精度与 attention 实现**
  - 选择：bf16 + `attn_implementation="flash_attention_2"`
  - 原因：官方 README 唯一示例就是这套；显存与速度都优于 eager。RTX 4060 8GB 跑 bf16 7B 模型需要约 14GB 显存，会 OOM，需提醒用户**实际最低显存 16GB**（与官方一致），或后续 Iteration 接入 4/8-bit 量化作为选项。
  - 依赖：`flash-attn==2.5.5`，需要 ninja + `--no-build-isolation` 安装，已在 README 注明。

- **决策点：动作语义映射**
  - 选择：直接返回 7 维动作向量（取首 token）
  - 原因：`do_sample=False` 时 `predict_action` 返回确定性 7 元组，与 `BaseVLA.predict` 的 `Action7D` 契约完全对齐；不引入动作块（chunk）的额外抽象，保持单步接口简洁。Action chunk 留作后续 Iteration（与 executor 主循环改造一起做）。

- **决策点：延迟加载**
  - 选择：`OpenVLA.__init__` 仅保存参数；首次 `predict` 才真正下载权重 + `to(device)`
  - 原因：`factory.create_vla` 在 `app.py` 启动时被调用，构造时联网会拖慢启动并增加失败点。延迟加载让"未启用 openvla backend 时零开销"。

- **决策点：动作维度对应关系**
  - 选择：硬编码 `[dx, dy, dz, drx, dry, drz, gripper]`，并在 docstring 中标注
  - 原因：OpenVLA 输出的 7 元组语义由其训练数据集决定（不同 `unnorm_key` 可能顺序不同）；本次设计限定 `unnorm_key="bridge_orig"`，该数据集下顺序与 `Action7D` 一致。后续接入 LIBERO 微调版时需要在 docstring 中提示顺序可能不同。

## 参考资料

- 官方仓库：https://github.com/openvla/openvla
- 本地文档：`docs/2026-08-13-github-com-openvla.md`
- 现有占位：`src/executor/model/openvla/README.md`、`src/executor/model/factory.py` 第 41-43 行
- `BaseVLA` 契约：`src/executor/model/base.py`
- `Action7D` 契约：`src/env/base.py`
- 配置入口：`src/config/loader.py`（`VLAConfig` dataclass）
