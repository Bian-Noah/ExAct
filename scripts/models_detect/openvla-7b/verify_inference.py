#!/usr/bin/env python3
"""验证 OpenVLA-7B 能否正常加载与推理（独立脚本，不依赖项目代码）。

推理链路与 src/executor/model/openvla/openvla_vla.py 对齐：
  AutoProcessor(trust_remote_code) -> AutoModelForVision2Seq(trust_remote_code,
  torch_dtype, low_cpu_mem_usage) -> model.predict_action(**inputs,
  unnorm_key, do_sample=False) -> (7,) 动作 [dx, dy, dz, drx, dry, drz, gripper]。

多图模式（默认）：
  - 无 --image 时，自动从 HF 数据集 VyoJ/BridgeData-V2-Scripted-Images（BridgeData V2
    scripted 子集的第三方图像版，640x480 PNG）下载 N 张初始帧到
    <repo>/data/images/bridge_samples/（幂等：已存在跳过），然后逐张推理。
  - 说明：该图集是 BridgeData V2 的 scripted 子集（无指令/无 GT），与 OpenVLA 训练用的
    teleoperated 子集（OXE bridge_orig）同源但不同份——视觉分布内、非训练帧，适合做
    「链路 + 输出合法性」检测；量纲断言在此类真实图上才有意义（PyBullet 仿真图会误报）。

与 smolvla 检测脚本（scripts/models_detect/smolvla-so101-pickplace-v2/）的差异：
  - 无数据集/无 ground truth：用合理性断言（shape/finite/量纲/稳定性）代替 MSE 对比；
  - 动作 7D（非 6D）；trust_remote_code=True 必需；
  - 4bit 量化（bnb nf4）仅 CUDA；MPS 不支持 bfloat16，自动回退 float16；
  - CPU 可跑但极慢，仅建议 --samples 1 冒烟。

用法：
  python verify_inference.py --fetch 10                          # 默认：自动下载 10 张 BridgeData 样本图并逐张推理
  python verify_inference.py                                      # 同默认（fetch=10）
  python verify_inference.py --image a.png --image b.png          # 显式指定多张图，跳过自动获取
  python verify_inference.py --no-fetch                           # 不联网：用本地 data/images/bridge_samples/（若为空回退单张观测图）
  python verify_inference.py --quantization none --device cuda:0  # bf16 全精度（需 16GB+ 显存）
  python verify_inference.py --model /local/checkpoint --samples 2

注意：
  - 首次加载会从 HF Hub 下载 openvla-7b 权重（约 14GB），建议先手动跑一次预热；
  - 国内环境若无法直连 HF，设置 HF_ENDPOINT=https://hf-mirror.com 后重跑即可。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL = "openvla/openvla-7b"
DEFAULT_UNNORM_KEY = "bridge_orig"
DEFAULT_INSTRUCTION = "pick up the red block"
# 与 src/executor/model/openvla/openvla_vla.py:DEFAULT_PROMPT_TEMPLATE 完全一致
DEFAULT_PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"

# 输出与样本图锚定到项目根（与 cwd 无关）；独立文件名避免覆盖 smolvla 的 summary.json。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = PROJECT_ROOT / "data" / "script" / "openvla-summary.json"
DEFAULT_IMAGE = PROJECT_ROOT / "data" / "images" / "observations" / "20260813_173441_001.png"

# BridgeData V2 scripted 子集的第三方图像版（HF 数据集，ImageFolder，640x480 PNG）
BRIDGE_REPO = "VyoJ/BridgeData-V2-Scripted-Images"
BRIDGE_SUBDIR = "initial_images"
BRIDGE_DIR = PROJECT_ROOT / "data" / "images" / "bridge_samples"
DEFAULT_FETCH = 10


def pick_device() -> str:
    """自动选择可用设备；无 GPU 时回退到 CPU（仅建议冒烟）。"""
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    print("[env] 未检测到 CUDA/MPS，回退 CPU——OpenVLA-7B 推理极慢，仅建议 --samples 1 冒烟。", file=sys.stderr)
    return "cpu"


def resolve_dtype(dtype_str: str, device: str) -> "torch.dtype":
    """解析 dtype；MPS 不支持 bfloat16，自动回退 float16 并警告。"""
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    try:
        dtype = mapping[dtype_str]
    except KeyError:
        raise SystemExit(f"[env] dtype 必须为 bfloat16|float16|float32，得到 {dtype_str!r}")
    if device.startswith("mps") and dtype == torch.bfloat16:
        print("[env] MPS 不支持 bfloat16，自动回退 float16。", file=sys.stderr)
        dtype = torch.float16
    return dtype


def fetch_bridge_samples(count: int = DEFAULT_FETCH) -> list[Path]:
    """从 HF 下载 N 张 BridgeData 样本图到 BRIDGE_DIR（幂等，已存在跳过）。

    用 huggingface_hub（requirements-openvla.txt 已锁定 huggingface_hub<1.0，
    openvla env 必有）。hf_hub_download 落 HF 缓存，再 copy 平铺到 BRIDGE_DIR，
    保证缓存可复用、项目目录内干净。国内环境可设 HF_ENDPOINT=https://hf-mirror.com。
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    files = sorted(
        f for f in list_repo_files(BRIDGE_REPO, repo_type="dataset")
        if f.startswith(BRIDGE_SUBDIR + "/") and f.endswith(".png")
    )
    if not files:
        raise SystemExit(f"[fetch] {BRIDGE_REPO}/{BRIDGE_SUBDIR} 下未找到 png")
    print(f"[fetch] {BRIDGE_REPO}/{BRIDGE_SUBDIR} 共 {len(files)} 张，均匀采样 {count} 张 ...")
    step = max(1, len(files) // count)
    picked = files[::step][:count]

    BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    outs: list[Path] = []
    for p in picked:
        name = Path(p).name
        local = BRIDGE_DIR / name
        if local.is_file() and local.stat().st_size > 0:
            outs.append(local)
            continue
        print(f"[fetch] 下载 {name} ...")
        cached = hf_hub_download(BRIDGE_REPO, p, repo_type="dataset")
        shutil.copy2(cached, local)
        outs.append(local)
    print(f"[fetch] 完成：{len(outs)} 张 -> {BRIDGE_DIR}")
    return outs


def load_model(model: str, device: str, dtype: "torch.dtype", quantization: str, attn_impl: str):
    """加载 processor + 模型，链路对齐 src/executor/model/openvla/openvla_vla.py。

    4bit 分支：BitsAndBytesConfig(nf4) + device_map="auto"（不可 .to(device)）；
    none 分支：全精度 + .to(device)。
    flash_attn 缺失时静默回退 eager，不阻断。
    """
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print(f"[model] 加载 {model} (quantization={quantization}, dtype={dtype}, device={device}) ...")
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)

    load_kwargs: dict = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    effective_attn = "eager"
    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
            load_kwargs["attn_implementation"] = "flash_attention_2"
            effective_attn = "flash_attention_2"
        except ImportError:
            print("[attn] 未安装 flash_attn，回退 eager attention。", file=sys.stderr)

    if quantization == "4bit":
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401  # 触发 ImportError 以便给出指引

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs.update({"quantization_config": bnb_cfg, "device_map": "auto"})
        vla = AutoModelForVision2Seq.from_pretrained(model, **load_kwargs)
    else:  # "none"：全精度路径
        vla = AutoModelForVision2Seq.from_pretrained(model, **load_kwargs).to(device)
    vla.eval()
    print(f"[model] 加载完成 (attn={effective_attn})")
    return processor, vla, effective_attn


def predict_once(
    processor, vla, pil_image: "Image.Image", prompt: str,
    unnorm_key: str, device: str, dtype: "torch.dtype",
) -> tuple[np.ndarray, float]:
    """单次推理：processor 编码 -> predict_action -> (7,) 动作 ndarray + 耗时(秒)。

    一张图 + 一条指令 -> 一个 7D 动作 [dx,dy,dz,drx,dry,drz,gripper]（单步，非 chunk）。
    """
    inputs: dict = processor(prompt, pil_image).to(device, dtype=dtype)
    t0 = time.perf_counter()
    with torch.no_grad():
        action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    arr = np.asarray(action, dtype=float)
    return arr, time.perf_counter() - t0


def check_action(arr: np.ndarray) -> list[str]:
    """合理性断言（无 ground truth）。返回错误描述列表，空 = 通过。

    硬性检查：shape==(7,)、全 finite；
    量纲检查：平移分量米级（|·|<2.0 宽松阈值）、gripper 接近 [0,1]。
    注意：仿真/域外图上量纲断言可能误报（模型输出无物理意义），属预期。
    """
    errors: list[str] = []
    if arr.shape != (7,):
        errors.append(f"shape={arr.shape} != (7,)")
    if not np.isfinite(arr).all():
        errors.append(f"含非有限值: {arr.tolist()}")
    if arr.shape == (7,) and np.isfinite(arr).all():
        dx, dy, dz, drx, dry, drz, gripper = arr
        if max(abs(dx), abs(dy), abs(dz)) > 2.0:
            errors.append(f"平移分量量纲异常（|dx,dy,dz|={arr[:3].tolist()}）")
        if not (-0.5 <= gripper <= 1.5):
            errors.append(f"gripper 超出合理范围 [-0.5,1.5]: {gripper}")
    return errors


def collect_image_paths(args) -> list[Path]:
    """确定推理图列表：显式 --image > 本地 bridge_samples > 自动下载 > 回退单张观测图。"""
    if args.image:
        paths = [Path(p) for p in args.image]
    elif not args.no_fetch:
        try:
            paths = fetch_bridge_samples(args.fetch)
        except Exception as e:
            raise SystemExit(
                f"[fetch] 自动获取样本图失败: {type(e).__name__}: {e}\n"
                "国内环境可设 HF_ENDPOINT=https://hf-mirror.com 后重试；"
                "或 --no-fetch --image <本地图> 离线检测。"
            )
    else:
        paths = sorted(BRIDGE_DIR.rglob("*.png")) if BRIDGE_DIR.is_dir() else []
        if not paths:
            print(f"[warn] {BRIDGE_DIR} 无样本图且 --no-fetch，回退默认观测图", file=sys.stderr)
            paths = [DEFAULT_IMAGE]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"[data] 测试图不存在: {missing[0]}（请用 --image 指定）")
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description="openvla-7b 加载与推理检测（模型级，不跑仿真链路）")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF Hub ID 或本地 checkpoint 路径")
    ap.add_argument("--image", action="append", default=None, metavar="PATH",
                    help="测试图片路径，可多次指定（多图逐张推理）；不指定则自动获取 BridgeData 样本图")
    ap.add_argument("--fetch", type=int, default=DEFAULT_FETCH,
                    help="自动下载的 BridgeData 样本图数量（默认 %(default)s，仅 --image 未指定时生效）")
    ap.add_argument("--no-fetch", action="store_true", help="禁用自动下载（离线模式）")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="语言指令（所有图共用）")
    ap.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY, help="反归一化键（决定动作量纲与顺序）")
    ap.add_argument("--device", default=None, help="auto 探测 cuda>mps>cpu；显式传 cuda:0 / mps / cpu")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16",
                    help="推理精度；MPS 上 bfloat16 自动回退 float16")
    ap.add_argument("--quantization", choices=("4bit", "none"), default="4bit",
                    help="4bit（bnb nf4，仅 CUDA）| none（bf16 全精度）")
    ap.add_argument("--attn", choices=("flash_attention_2", "eager"), default="flash_attention_2",
                    help="attention 实现；缺 flash-attn 自动回退 eager")
    ap.add_argument("--samples", type=int, default=1,
                    help="每张图的重复推理次数（>=2 时额外做该图稳定性检查；默认 1）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="结果 JSON 输出路径")
    args = ap.parse_args()

    if args.samples < 1:
        raise SystemExit("[args] --samples 必须 >= 1")

    device = args.device or pick_device()
    if args.quantization == "4bit" and not device.startswith("cuda"):
        raise SystemExit(
            f"[env] --quantization 4bit 仅支持 CUDA（bitsandbytes 在 {device} 上不可用）；"
            "请改用 --quantization none。"
        )
    dtype = resolve_dtype(args.dtype, device)
    print(f"[env] torch={torch.__version__} device={device} dtype={dtype} quant={args.quantization}")

    image_paths = collect_image_paths(args)
    print(f"[data] 推理图 {len(image_paths)} 张（每张 {args.samples} 次）")

    processor, vla, effective_attn = load_model(args.model, device, dtype, args.quantization, args.attn)

    prompt = DEFAULT_PROMPT_TEMPLATE.format(instruction=args.instruction)
    print(f"[data] prompt: {prompt!r}")

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results: list[dict] = []
    check_fail = 0
    all_times: list[float] = []
    for img_path in image_paths:
        pil_image = Image.open(img_path).convert("RGB")
        img_actions: list[list[float]] = []
        img_errors: list[str] = []
        img_times: list[float] = []
        for i in range(args.samples):
            try:
                action, dt = predict_once(processor, vla, pil_image, prompt, args.unnorm_key, device, dtype)
            except Exception as e:
                img_errors.append(f"sample#{i}: {type(e).__name__}: {e}")
                continue
            errs = check_action(action)
            img_errors.extend(errs)
            img_actions.append(action.tolist())
            img_times.append(dt)
            all_times.append(dt)
            status = "OK" if not errs else "; ".join(errs)
            print(f"[infer] {Path(img_path).name} #{i} time={dt:.3f}s "
                  f"action={np.round(action, 4).tolist()} {status}")

        stability_max_diff = None
        if len(img_actions) >= 2:
            arrs = [np.asarray(a) for a in img_actions]
            diffs = [float(np.max(np.abs(a - arrs[0]))) for a in arrs[1:]]
            stability_max_diff = max(diffs)
            if stability_max_diff > 1e-3:
                print(f"[warn] {Path(img_path).name} 多次输出差异较大 "
                      f"max_diff={stability_max_diff:.2e}（核对 unnorm_key/权重）", file=sys.stderr)

        ok = (len(img_actions) == args.samples) and not img_errors
        if not ok:
            check_fail += 1
        results.append({
            "image": str(img_path),
            "errors": img_errors,
            "actions": img_actions,
            "times_s": [round(t, 3) for t in img_times],
            "stability_max_diff": stability_max_diff,
        })

    max_mem_gb = None
    if device.startswith("cuda") and all_times:
        max_mem_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 2)

    summary = {
        "model": args.model,
        "image_count": len(image_paths),
        "samples_per_image": args.samples,
        "instruction": args.instruction,
        "prompt": prompt,
        "unnorm_key": args.unnorm_key,
        "device": device,
        "dtype": str(dtype),
        "quantization": args.quantization,
        "attn_impl": effective_attn,
        "checks": {
            "pass": len(image_paths) - check_fail,
            "fail": check_fail,
            "errors": [r["errors"] for r in results if r["errors"]],
        },
        "avg_time_s": round(float(np.mean(all_times)), 3) if all_times else None,
        "max_mem_gb": max_mem_gb,
        "results": results,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    brief = {k: v for k, v in summary.items() if k != "results"}
    print(json.dumps(brief, ensure_ascii=False, indent=2))
    print(f"\n通过 {summary['checks']['pass']}/{len(image_paths)} 张图 | 结果已写入 {out_path}")
    return 0 if check_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
