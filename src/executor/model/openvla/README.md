# openvla 后端

OpenVLA-7B 真实后端（仅推理），通过 HuggingFace `AutoModelForVision2Seq` / `AutoProcessor`
加载远端权重，输出 task 空间 7 维动作 `[dx, dy, dz, drx, dry, drz, gripper]`。

## 启用方式

```yaml
# configs/local.yaml
vla:
  backend: openvla
  model_path: "openvla/openvla-7b"   # HF Hub ID 或本地 checkpoint 路径
  openvla:
    unnorm_key: "bridge_orig"        # 反归一化键；LIBERO 微调版需核对动作顺序
    attn_impl: "flash_attention_2"   # flash_attention_2 | eager（缺 flash-attn 自动回退）
    device: "cuda:0"
    dtype: "bfloat16"                # bfloat16 | float16 | float32
    quantization: "4bit"             # 4bit（bnb nf4，4060 ~4GB）| none（bf16 需 16GB+ 显存）
```

运行链路验证脚本：

```bash
python src/pipeline/script/verify_full_link.py --vla openvla --max-chunks 1
```

模型级检测脚本（加载 + 多图推理，独立脚本不依赖项目代码，输出 `data/script/openvla-summary.json`）：

```bash
python scripts/models_detect/openvla-7b/verify_inference.py                 # 默认 4bit：自动下载 10 张 BridgeData 样本图并逐张推理（仅 CUDA）
python scripts/models_detect/openvla-7b/verify_inference.py --quantization none   # bf16 全精度
python scripts/models_detect/openvla-7b/verify_inference.py --image a.png --image b.png   # 显式指定多图
```

- 样本图自动从 HF `VyoJ/BridgeData-V2-Scripted-Images`（BridgeData V2 scripted 子集图像版）获取到 `data/images/bridge_samples/`（幂等），用 `--no-fetch` 可离线；国内环境设 `HF_ENDPOINT=https://hf-mirror.com`。
- 每张图一次推理输出一个 7D 动作（单步，非 chunk）；`--samples N` 可对每图重复 N 次并做稳定性检查。

## 构造参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_path` | 必填 | HF Hub ID（`openvla/openvla-7b`）或本地 checkpoint 路径 |
| `unnorm_key` | `"bridge_orig"` | 反归一化键，决定动作量纲与顺序 |
| `attn_impl` | `"flash_attention_2"` | 缺 flash-attn 自动回退 eager |
| `device` | `"cuda:0"` | 非量化路径的推理设备 |
| `dtype` | `"bfloat16"` | 推理精度字符串，首次加载时解析为 torch.dtype |
| `quantization` | `"4bit"` | `4bit`（bnb nf4）\| `none`（bf16 需 16GB+ 显存） |

## 行为要点

- **懒加载**：构造仅保存参数，不联网不加载模型；首次 `predict` 才下载权重（openvla-7b 约 14GB）并移至设备。4060 上建议先手动触发一次下载，避免实验中断。
- **顶层零 torch**：模块顶层不 `import torch`（决策 6），无 torch 环境（M4 / CI）也可构造、取 `output_spec`、跑输入校验；推理路径由单测注入假 torch/transformers 覆盖。
- **量化与 device_map**：`quantization="4bit"` 走 bnb nf4 + `device_map="auto"`（不可 `.to(device)`）；`"none"` 走 bf16 + `.to(device)`。
- **契约**：`predict(image_dict, instruction)` 返回 `VLAOutput(values=np.ndarray (1,7))`，符合 iter10 chunk 契约；image 为多相机 dict（iter11），取首张图。
- **输入校验**：image 非 dict / 空 dict / 非 uint8 / 非 (H,W,3) 分别抛 `TypeError` / `ValueError`；`predict_action` 返回 shape ≠ (7,) 抛 `RuntimeError`（提示 unnorm_key 可能不匹配）。

## 依赖

依赖在 `requirements-openvla.txt`（全项目单文件，含 bitsandbytes/accelerate 等），独立 conda env（Python 3.10/3.11，torch 2.2.0 wheel 仅到 cp311）：

```bash
conda create -n openvla python=3.11 -y && conda activate openvla
pip install -r requirements-openvla.txt
pip check
```

> ⚠️ 显存：4bit 加载约 4GB（RTX 4060 8GB 可跑）；bf16 全精度约 14GB，需 16GB+ 显存。
