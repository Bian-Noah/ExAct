# LeRobot 后端 4060 首跑检查清单

> 场景：在阶段二（RTX 4060, CUDA）机器上，从 `configs/local.yaml` 把 `vla.backend` 切到 `lerobot`，首次跑通 LeRobot 后端。
>
> 关联代码：
> - `src/executor/model/lerobot/lerobot_vla.py`（后端实现，官方 preprocessor 管线）
> - `src/utils/quantize.py`（4bit/8bit 量化工具）
> - `configs/local.yaml`（配置入口）

---

## 0. 一句话结论

**改 `local.yaml` + 装 `lerobot>=0.6` 依赖 + 权重放在本地**，代码链路已通（mock 全链路验证过）。但真实 lerobot API 的个别参数细节需在 4060 上首跑时确认一次。

---

## 1. 权重准备（必须先做，离线优先）

本项目**离线优先：绝不自动联网下载**。`model_path` 必须是：

- **本地权重目录**（推荐），或
- **已在 HF 缓存中的 repo_id**（手动下载过）

若 `model_path` 填 HF repo_id 但缓存里没有，代码会报错并给出提示命令，**不会静默下载**。

**本地权重怎么来（任选其一）**：

```bash
# 方式 A：手动下载到 HF 缓存（之后 config 填 repo_id 即可）
python -c "from huggingface_hub import snapshot_download; snapshot_download('lerobot/act_aloha_sim_transfer_cube_human')"

# 方式 B：已有本地权重目录，config 直接指向它
# model_path: "/path/to/act_policy"
```

权重示例：
| repo_id | policy_type | 说明 |
|---------|-------------|------|
| `lerobot/act_aloha_sim_transfer_cube_human` | `act` | ACT 默认演示权重（最轻量，先跑通用这个） |
| `lerobot/smolvla_base` | `smolvla` | ~450M 参数 VLA |
| `lerobot/pi0_libero_finetuned` | `pi0` | 7B 级 VLA，8GB 卡必须 4bit 量化 |

---

## 2. 安装依赖（阶段二，CUDA）

```bash
# 1. 先装 PyTorch CUDA 版（官方命令，按你的 CUDA 版本）
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. 再装本项目阶段二依赖
pip install -r requirements-stage2.txt
```

`requirements-stage2.txt` 关键项：
- `lerobot>=0.6` —— **>=0.6 才包含 `make_pre_post_processors` 处理器管线 API**（0.4 没有，会报错）
- `bitsandbytes>=0.43` —— 4bit/8bit 量化依赖
- `transformers>=4.40` —— Pi0/SmolVLA 的 VLM 骨干依赖

---

## 3. 配置（configs/local.yaml）

```yaml
vla:
  backend: lerobot                     # "mock" | "lerobot"
  model_path: "lerobot/act_aloha_sim_transfer_cube_human"   # 本地目录 或 已缓存的 repo_id
  max_steps: 50
  lerobot:
    policy_type: act                   # act | diffusion | smolvla | pi0 | vqbet | pi0fast
    device: "cuda:0"
    quantization: none                 # none | 8bit | 4bit（仅 VLA 类生效）
```

**量化建议（8GB 显卡）**：
| policy_type | 建议 quantization | 推理显存 |
|-------------|-------------------|----------|
| `act` | `none` | 2-6GB（无需量化） |
| `smolvla` | `none` 或 `4bit` | fp16 ~2-3GB；4bit ~1.5-2GB |
| `pi0` | **必须 `4bit`** | fp16 ~14GB 装不下；4bit ~4GB |

---

## 4. 首跑步骤

```bash
cd ExAct
python src/app.py
```

或跑单测验证配置链路（M4 / 4060 均可，不加载模型）：

```bash
python -m pytest tests/test_config_loader.py -q
```

---

## 5. 可能遇到的报错及对策

### 5.1 `HuggingFace 权重 'xxx' 未在本地缓存中找到`
- **原因**：repo_id 未下载，离线优先策略拦截。
- **对策**：按提示执行 `snapshot_download(...)`，或改用本地目录路径。

### 5.2 `导入 lerobot policy='act' 失败 ... 请检查 lerobot 版本`
- **原因**：lerobot 未装 或 版本 < 0.6（模块路径是旧版）。
- **对策**：`pip install -U lerobot>=0.6`。

### 5.3 `当前 lerobot 版本找不到 make_pre_post_processors 入口`
- **原因**：lerobot 版本过旧，无处理器管线 API。
- **对策**：升级 `lerobot>=0.6`。

### 5.4 `make_pre_post_processors 构造失败：...`
- **原因**：真实 0.6.x 的参数名可能与文档示例不同（如 `policy_cfg` vs `cfg`）。
- **对策**：这是**最可能需要实测微调的地方**。看报错信息里暴露的签名，改 `lerobot_vla.py` 的 `_build_processors()` 传参。

### 5.5 `4bit/8bit 量化需要 bitsandbytes`
- **原因**：量化路径缺 `bitsandbytes`。
- **对策**：`pip install bitsandbytes>=0.43`，或配置 `quantization: none`。

### 5.6 首次 predict 时 `select_action` 报键缺失 / 类型错误
- **原因**：手写 obs 已废弃，改由官方 preprocessor 生成输入。若 preprocessor 输出与真实 policy 期望不符，会在这一步暴露。
- **对策**：检查 `policy.config` 的 `input_shapes` / 图像键名是否被 `_resolve_obs_keys()` 正确探测；必要时在 `_build_raw_obs` 调整图像键。

### 5.7 量化后 `select_action` 报 dtype / 设备错误
- **原因**：bnb 替换后部分层 dtype 与 policy 内部假设不一致（已知风险，mock 测不出）。
- **对策**：这是**第二可能需要实测微调的地方**。先 `quantization: none` 确认无量化时能跑，再逐步开量化；报错时调整 `quantize_model_4bit` 的 `compute_dtype` 或 `skip_modules`。

---

## 6. 已验证项 vs 待实测项

### ✅ 已在当前环境验证（mock / 单测）
- `local.yaml` 解析 lerobot 配置
- `create_vla` 分派 → `LeRobotVLA` 构造
- `predict` 全链路调用顺序（离线检查 → 加载 → preprocessor → select_action → postprocessor → Action7D）
- 量化分支（目标 `policy.model`、位宽正确）
- 离线检查 6 场景（本地路径 / 缓存判定 / 提示下载）
- 全量单测 494 个通过

### ⚠️ 必须在 4060 上实测
1. `make_pre_post_processors` 真实签名（§5.4）
2. `select_action` 对 preprocessor 输出的真实接受度（§5.6）
3. Pi0 自定义 `from_pretrained` 对 `local_files_only=True` 的兼容性
4. 4bit 量化后真实推理的 dtype/设备兼容（§5.7）

> 这些是真实 lerobot API 行为，mock 只能验证调用链顺序正确，验证不了真实行为。代码在遇到问题时都会抛出**带对策的明确提示**，按 §5 对照处理即可。
