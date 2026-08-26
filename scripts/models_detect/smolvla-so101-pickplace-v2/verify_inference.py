#!/usr/bin/env python3
"""验证 smolvla-so101-pickplace-v2 能否正常预测(独立脚本,不依赖项目代码)。

推理链路与 src/executor/model/lerobot/lerobot_vla.py 对齐:
pre(batch) -> policy.predict_action_chunk -> post,一次拿整 chunk 再取第 0 步。
不使用 select_action —— lerobot>=0.6 中它有内部动作队列缓存(队列空时才推理、
每次调用只吐 1 步),跨样本会残留上一 chunk 的动作,统计失真。
数据集 front/wrist 两相机映射为 camera1/camera2(缺失的 camera3 跳过,README rename_map 证实)。

用法:
  python verify_inference.py
  python verify_inference.py --model /local/path/to/policy --device cpu --samples 10
"""
import argparse
import io
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image

DEFAULT_MODEL = "ahmedsohail2003/smolvla-so101-pickplace-v2"
DEFAULT_DATASET = "ahmedsohail2003/so101-sim-pickplace-v2"
FALLBACK_INSTRUCTION = "Pick up the red block and place it in the blue tray."


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_policy(model: str, device: str):
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(model).to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=model,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def image_to_tensor(img, device: str) -> torch.Tensor:
    """{bytes/path}/PIL/ndarray -> (1,3,H,W) float32 [0,1] tensor(README 示例同格式)。"""
    if isinstance(img, dict):
        img = Image.open(io.BytesIO(img["bytes"])) if img.get("bytes") else Image.open(img["path"])
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def get_instruction(row: dict, column_names: list[str]) -> str:
    for key in ("task", "language_instruction"):
        if key in column_names and row.get(key):
            return str(row[key])
    return FALLBACK_INSTRUCTION


def predict_sample(policy, pre, post, row: dict, column_names: list[str], device: str):
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
        "task": get_instruction(row, column_names),
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
    ap.add_argument("--device", default=None, help="auto 探测 cuda>mps>cpu,或显式 cuda:0/mps/cpu")
    ap.add_argument("--samples", type=int, default=10, help="抽样帧数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="summary.json", help="结果 JSON 输出路径")
    args = ap.parse_args()

    device = args.device or pick_device()
    print(f"[env] torch={torch.__version__} device={device}")

    print(f"[data] 加载数据集 {args.dataset} ...")
    try:
        ds = load_dataset(args.dataset, split="train")
    except ValueError:
        ds = next(iter(load_dataset(args.dataset).values()))
    cols = ds.column_names
    print(f"[data] {len(ds)} 帧 | 列: {cols}")

    print(f"[model] 加载 {args.model} ...")
    policy, pre, post = load_policy(args.model, device)
    print(f"[model] input_features: {list(policy.config.input_features or {})}")

    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(ds), args.samples, replace=False))

    rows, checks = [], {"pass": 0, "fail": 0, "errors": []}
    for i in idx:
        row = ds[int(i)]
        try:
            action, dt = predict_sample(policy, pre, post, row, cols, device)
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
        "samples": len(idx),
        "checks": checks,
        "avg_mse": float(np.mean([r["mse"] for r in rows])) if rows else None,
        "avg_mae": float(np.mean([r["mae"] for r in rows])) if rows else None,
        "avg_time_s": float(np.mean([r["time_s"] for r in rows])) if rows else None,
        "results": rows,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    brief = {k: v for k, v in summary.items() if k != "results"}
    print(json.dumps(brief, ensure_ascii=False, indent=2))
    print(f"\n通过 {checks['pass']}/{args.samples} | 结果已写入 {args.out}")
    return 0 if checks["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
