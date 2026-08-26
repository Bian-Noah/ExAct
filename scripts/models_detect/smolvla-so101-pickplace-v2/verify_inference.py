#!/usr/bin/env python3
"""验证 smolvla-so101-pickplace-v2 能否正常预测(独立脚本,不依赖项目代码)。

推理链路与 src/executor/model/lerobot/lerobot_vla.py 对齐:
pre(batch) -> policy.predict_action_chunk -> post,一次拿整 chunk 再取第 0 步。
不使用 select_action —— lerobot>=0.6 中它有内部动作队列缓存(队列空时才推理、
每次调用只吐 1 步),跨样本会残留上一 chunk 的动作,统计失真。
数据集 front/wrist 两相机映射为 camera1/camera2(缺失的 camera3 跳过,README rename_map 证实)。

可选 8bit 量化(bitsandbytes,仅 CUDA):对 VLM 骨干的 nn.Linear 做替换,
实现与 src/utils/quantize.py 的 8bit 路径一致但内联,无 src/ 依赖。
bitsandbytes 缺失时降级跳过 + 警告,继续以原 dtype 推理。

用法:
  python verify_inference.py                                  # 默认 8bit 量化 + 自动探测 CUDA
  python verify_inference.py --quantization none              # 禁用量化
  python verify_inference.py --model /local/policy --samples 20

注意:本脚本不支持 CPU 推理;无 GPU 或显式 --device cpu 均直接报错退出。
      --quantization 8bit(默认)仅 CUDA 可用(MPS 上 bitsandbytes 不可用)。
"""
import argparse
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

DEFAULT_MODEL = "ahmedsohail2003/smolvla-so101-pickplace-v2"
DEFAULT_DATASET = "ahmedsohail2003/so101-sim-pickplace-v2"
FALLBACK_INSTRUCTION = "Pick up the red block and place it in the blue tray."

# 输出锚定到项目根(<repo>/data/script/summary.json),与 cwd 无关,便于跨机器/CI 复用。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = PROJECT_ROOT / "data" / "script" / "summary.json"

# VLM 骨干子模块名前缀(对齐 src/utils/quantize.py:VLM_PREFIXES,SmolVLA 走 vlm_with_expert)
VLM_PREFIXES = ("paligemma_with_expert", "vlm_with_expert", "vlm")


def pick_device() -> str:
    """自动选择可用 GPU;无 GPU 时直接报错退出(本脚本不跑 CPU)。"""
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    raise SystemExit(
        "[env] 未检测到 CUDA/MPS 可用设备,本脚本不支持 CPU 推理。"
        "请在带 GPU 的环境运行。"
    )


def _quantize_policy_8bit(policy: torch.nn.Module) -> int:
    """对 policy 的 VLM 骨干做 bitsandbytes 8bit 量化(内联,无 src/ 依赖)。

    行为对齐 src/utils/quantize.py:quantize_model_4bit(bits=8) 的手写分支,
    仅保留 8bit 路径:遍历 VLM 骨干子模块的 nn.Linear,替换为 bnb.nn.Linear8bitLt,
    跳过 lm_head。bitsandbytes 缺失时降级为跳过 + 警告,不阻断推理。

    Returns:
        成功替换的 Linear 层数量;未命中 VLM 前缀时对整 model 做替换并打 warning。
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        print(
            "[quant] bitsandbytes 未安装,跳过 8bit 量化(继续以原 dtype 推理)。"
            "如需启用:pip install bitsandbytes>=0.43",
            file=sys.stderr,
        )
        return 0

    target = None
    for name, module in policy.named_modules():
        if name.startswith(VLM_PREFIXES):
            print(f"[quant] 命中 VLM 骨干:{name} ({type(module).__name__})")
            target = module
            break
    if target is None:
        print(
            "[quant] 未命中 VLM 骨干前缀 "
            f"{VLM_PREFIXES},将对整个模型做 8bit 量化(可能误伤非 VLM 层)",
            file=sys.stderr,
        )
        target = policy

    device = next(target.parameters()).device
    replaced = 0
    for name, module in list(target.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if name == "lm_head":
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = target.get_submodule(parent_name) if parent_name else target
        # 行为对齐 src/utils/quantize.py:quantize_model_4bit(bits=8) 的手写分支:
        # Linear8bitLt 不传 has_fp16_weights(默认 True);后续 forward 中 bnb.matmul 会
        # 每次重新量化 B 并保持 self.weight.data 为 fp,避免 weight.data 被替换为 int8 后
        # 下游 attention 的 baddbmm 撞到 Char dtype。bnb 没有 Linear8bitLt.from_linear,
        # try 块必然 AttributeError 走 except 分支做手动 weight/bias 拷贝。
        quant_linear = bnb.nn.Linear8bitLt(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=device,
        )
        quant_linear.weight.data = module.weight.data.to(device)
        if module.bias is not None:
            quant_linear.bias = torch.nn.Parameter(module.bias.data.to(device))
        setattr(parent, child_name, quant_linear)
        replaced += 1

    print(
        f"[quant] 8bit 量化完成,替换 {replaced} 个 Linear 层"
        f"(target={type(target).__name__})"
    )
    return replaced


def load_policy(model: str, device: str, quantization: str):
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(model).to(device).eval()
    if quantization == "8bit":
        _quantize_policy_8bit(policy)
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=model,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def image_to_tensor(img, device: str) -> torch.Tensor:
    """{bytes/path}/PIL/ndarray/torch.Tensor -> (1,3,H,W) float32 [0,1] tensor。

    LeRobotDataset.__getitem__ 默认 return_uint8=False 返回 (C,H,W) float32 in [0,1];
    return_uint8=True 时返回 (C,H,W) uint8 in [0,255],此时除 255。其他来源
    (PIL/ndarray) 走原 convert + permute 路径。
    """
    if isinstance(img, torch.Tensor):
        t = img.to(torch.float32)
        if img.dtype == torch.uint8:
            t = t / 255.0
        # LeRobotDataset 默认已是 CHW;仅当外部传入 HWC(uint8 ndarray 透传)时转置
        if t.ndim == 3 and t.shape[-1] == 3 and t.shape[0] != 3:
            t = t.permute(2, 0, 1)
        return t.unsqueeze(0).to(device)
    if isinstance(img, dict):
        img = Image.open(io.BytesIO(img["bytes"])) if img.get("bytes") else Image.open(img["path"])
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def get_instruction(row: dict, column_names: list[str], tasks_lookup: dict | None = None) -> str:
    """语言指令查找。

    LeRobotDataset 的 frame 只携带 task_index,实际文本在 meta.tasks 里
    (pandas DataFrame,index 名 = task,列含 task_index)。tasks_lookup 是
    {task_index: text} 的一次性预查表;旧 row 直传 'task'/'language_instruction'
    的本地行场景仍兼容。
    """
    for key in ("task", "language_instruction"):
        if key in column_names and row.get(key):
            return str(row[key])
    if tasks_lookup:
        idx_val = row.get("task_index")
        if idx_val is not None:
            idx_int = int(idx_val.item() if hasattr(idx_val, "item") else idx_val)
            text = tasks_lookup.get(idx_int)
            if text:
                return text
    return FALLBACK_INSTRUCTION


def predict_sample(policy, pre, post, row: dict, column_names: list[str], device: str, tasks_lookup: dict | None = None):
    """单帧推理: 返回 (6,) 动作 ndarray 与耗时(秒)。

    lerobot>=0.6 中 select_action 有动作队列缓存(队列空时才推理、每次调用只吐 1 步),
    跨样本会残留动作导致统计失真;改用 predict_action_chunk 一次拿整 chunk(无状态),
    再按 ndim 分派取第 0 步,与数据集当前帧 action 对比。
    """
    state = torch.as_tensor(
        np.asarray(row["observation.state"], np.float32), device=device
    ).unsqueeze(0)
    batch = {
        "observation.state": state,
        "observation.images.camera1": image_to_tensor(row["observation.images.front"], device),
        "observation.images.camera2": image_to_tensor(row["observation.images.wrist"], device),
        "task": get_instruction(row, column_names, tasks_lookup),
    }
    t0 = time.perf_counter()
    with torch.no_grad():
        action = post(policy.predict_action_chunk(pre(batch)))
    # 输出 ndim 分派(与 src lerobot_vla.py 一致,不赌输出形状):
    #   (1, N, D) -> 取 chunk 第 0 步 (D,)；(N, D) -> 取第 0 步 (D,)；(D,) 保持
    if isinstance(action, torch.Tensor):
        arr = action.detach().cpu().numpy()
    else:
        arr = np.asarray(action)
    if arr.ndim == 3:
        arr = arr[0, 0]
    elif arr.ndim == 2:
        arr = arr[0]
    elif arr.ndim == 1:
        pass
    else:
        raise ValueError(f"predict_action_chunk 输出 ndim={arr.ndim}，期望 1/2/3")
    return arr.astype(float), time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description="smolvla-so101-pickplace-v2 推理验证")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="policy repo_id 或本地路径")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="数据集 repo_id 或本地路径")
    ap.add_argument("--device", default=None, help="auto 探测 cuda>mps;显式传 cuda:0 或 mps")
    ap.add_argument(
        "--quantization", choices=("none", "8bit"), default="8bit",
        help="VLM 骨干量化等级,仅 8bit(对齐 src/utils/quantize.py 8bit 路径,内联无 src/ 依赖);默认 8bit",
    )
    ap.add_argument("--samples", type=int, default=10, help="抽样帧数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="结果 JSON 输出路径,默认 <项目根>/data/script/summary.json")
    args = ap.parse_args()

    if args.device == "cpu":
        raise SystemExit("[env] 本脚本不支持 --device cpu。请改为 cuda:0 / mps / 自动探测。")
    device = args.device or pick_device()
    if args.quantization == "8bit" and not device.startswith("cuda"):
        raise SystemExit(
            f"[env] --quantization 8bit 仅支持 CUDA(bitsandbytes 在 {device} 上不可用)。"
        )
    print(f"[env] torch={torch.__version__} device={device} quant={args.quantization}")

    print(f"[data] 加载数据集 {args.dataset} (LeRobotDataset) ...")
    ds = LeRobotDataset(repo_id=args.dataset)
    cols = list(ds.features.keys())
    # meta.tasks 是 pandas DataFrame(index='task', 含列 'task_index');
    # 用 task_index -> 文本 的查表,get_instruction 中按帧的 task_index 取语言指令。
    tasks_lookup: dict[int, str] = {}
    if getattr(ds, "meta", None) is not None and ds.meta.tasks is not None:
        # meta.tasks: index='task'(文本), 列='task_index'(int)
        for task_text, task_row in ds.meta.tasks["task_index"].items():
            tasks_lookup[int(task_row)] = str(task_text)
    print(f"[data] {len(ds)} 帧 | episode={ds.num_episodes} | 列: {cols}")

    print(f"[model] 加载 {args.model} ...")
    policy, pre, post = load_policy(args.model, device, args.quantization)
    print(f"[model] input_features: {list(policy.config.input_features or {})}")

    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(ds), args.samples, replace=False))

    rows, checks = [], {"pass": 0, "fail": 0, "errors": []}
    for i in idx:
        row = ds[int(i)]
        try:
            action, dt = predict_sample(policy, pre, post, row, cols, device, tasks_lookup)
        except Exception as e:
            checks["fail"] += 1
            checks["errors"].append({"index": int(i), "error": f"{type(e).__name__}: {e}"})
            continue
        gt = np.asarray(row["action"], np.float32)
        ok = action.shape == gt.shape and bool(np.isfinite(action).all())
        checks["pass" if ok else "fail"] += 1
        rows.append({
            "index": int(i),
            "episode": int(row["episode_index"]),
            "time_s": round(dt, 3),
            "mse": float(np.mean((action - gt) ** 2)),
            "mae": float(np.mean(np.abs(action - gt))),
            "predicted": action.tolist(),
            "ground_truth": gt.tolist(),
        })

    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "device": device,
        "quantization": args.quantization,
        "samples": len(idx),
        "checks": checks,
        "avg_mse": float(np.mean([r["mse"] for r in rows])) if rows else None,
        "avg_mae": float(np.mean([r["mae"] for r in rows])) if rows else None,
        "avg_time_s": float(np.mean([r["time_s"] for r in rows])) if rows else None,
        "results": rows,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    brief = {k: v for k, v in summary.items() if k != "results"}
    print(json.dumps(brief, ensure_ascii=False, indent=2))
    print(f"\n通过 {checks['pass']}/{args.samples} | 结果已写入 {out_path}")
    return 0 if checks["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
