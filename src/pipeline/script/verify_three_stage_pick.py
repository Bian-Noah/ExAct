"""三阶段抓取验证脚本：把单段 chunk 拆成 approach → descend → close gripper。

与 verify_full_link.py 的区别：
  - 一次跑三个独立的 VLA 推理（3 个不同指令），不靠 chunk 自带完整轨迹
  - 每个 stage 单独打印 + 落盘模型的原始输出（VLAOutput.values，**未经 adapter**），
    方便定位"模型输出问题 vs 执行链路问题"
  - run_dir 命名带前缀 `three_stage_pick_`，与 verify_full_link 区分

用法：
    python src/pipeline/script/verify_three_stage_pick.py
    python src/pipeline/script/verify_three_stage_pick.py --vla mock --no-gui

数据落盘结构：
  data/scriptData/three_stage_pick_{YYYYMMDD_HHMMSS}/
    ├── experiment.log             # 全程文本日志（含每 stage 模型输出摘要）
    ├── input/                     # 每个 stage 喂给 VLA 的多视角输入图
    │    ├── stage0_chunk00_*.png
    │    └── ...
    ├── step_000.png               # reset 后初始观测
    ├── step_{stage0_end}.png      # stage0 结束画面（hover above）
    ├── step_{stage1_end}.png      # stage1 结束画面（descend）
    ├── step_{stage2_end}.png      # stage2 结束画面（close gripper）
    ├── model_output_stage0.txt    # stage0 原始 VLAOutput.values 完整 dump
    ├── model_output_stage1.txt
    └── model_output_stage2.txt
"""

from __future__ import annotations

import argparse
import inspect
import math
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

# 让脚本能 import src/ 下的模块（与 verify_full_link.py 一致）
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent.parent  # .../src
sys.path.insert(0, str(SRC_DIR))

# ROOT 用于解析 configs/local.yaml 等相对路径
ROOT = SRC_DIR.parent

import numpy as np

from config.loader import (
    AppConfig,
    EnvConfig,
    LLMConfig,
    RobotConfig,
    VLAConfig,
    load_config,
)
from env.pybullet_env import PyBulletEnv
from executor import Executor
from executor.model.factory import create_vla
from utils.adapter import get_adapter


DEFAULT_CONFIG_PATHS = (
    str(ROOT / "configs" / "local.yaml"),
    str(ROOT / "configs" / "default.yaml"),
)
SCRIPT_DATA_ROOT = ROOT / "data" / "scriptData"
# run_dir 前缀——与 verify_full_link 区分
RUN_DIR_PREFIX = "three_stage_pick"

# 三个 stage 的固定指令（smolVLA 规范：动词开头 / 全英文 / ≤30 字符）
DEFAULT_STAGES: tuple[tuple[str, str], ...] = (
    ("approach", "move above yellow cube"),
    ("descend",  "descend to yellow cube"),
    ("grasp",    "close gripper on cube"),
)


def _load_config(path_arg: str | None) -> tuple[AppConfig, str]:
    """按优先级加载：CLI > local.yaml > default.yaml。"""
    if path_arg:
        return load_config(path_arg), path_arg
    for p in DEFAULT_CONFIG_PATHS:
        if Path(p).is_file():
            return load_config(p), p
    raise FileNotFoundError(
        f"未找到任何配置文件，期望: {DEFAULT_CONFIG_PATHS}"
    )


def parse_args() -> argparse.Namespace:
    """CLI 解析。"""
    p = argparse.ArgumentParser(
        description=("三阶段抓取验证：approach → descend → close gripper，"
                     "把模型原始输出 dump 到 model_output_stageN.txt 便于排查")
    )
    p.add_argument("--config", default=None,
                   help=f"配置文件路径（默认自动选 local.yaml > default.yaml）")
    p.add_argument("--vla", default=None,
                   choices=["mock", "lerobot", "openvla", "llm_vla"],
                   help="VLA 后端（默认读取 config.vla.backend）")
    p.add_argument("--no-gui", action="store_true",
                   help="强制 DIRECT 模式（默认按 config.env.mode）")
    p.add_argument("--data-root", default=None,
                   help=f"脚本数据根目录（默认 {SCRIPT_DATA_ROOT}）")
    p.add_argument("--prefix", default=RUN_DIR_PREFIX,
                   help=f"run_dir 前缀（默认 {RUN_DIR_PREFIX!r}）")
    p.add_argument(
        "--stages", default=None,
        help=("逗号分隔的 (label, instruction) 对，覆盖默认三阶段。"
              "格式: 'approach:move above cube,descend:descend to cube,grasp:close gripper'"
              "（冒号或 ' =' 分隔 label 与 instruction）"),
    )
    return p.parse_args()


def _parse_stages(arg: str | None) -> tuple[tuple[str, str], ...]:
    """解析 --stages CLI 参数为 (label, instruction) 元组列表。

    格式："label1:inst1,label2:inst2,..." 或 "label1 = inst1, label2 = inst2,..."
    返回的 label 经 sanitize（只保留 [a-zA-Z0-9_]），用于文件名安全。
    """
    if not arg:
        return DEFAULT_STAGES
    stages: list[tuple[str, str]] = []
    for chunk in arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        # 兼容 ":" 和 " = " 分隔
        if ":" in chunk:
            label, instr = chunk.split(":", 1)
        elif " = " in chunk:
            label, instr = chunk.split(" = ", 1)
        elif "=" in chunk:
            label, instr = chunk.split("=", 1)
        else:
            raise ValueError(
                f"--stages 条目缺少 label:instruction 分隔符: {chunk!r}"
            )
        label_safe = re.sub(r"[^a-zA-Z0-9_]", "_", label.strip())
        stages.append((label_safe, instr.strip()))
    if not stages:
        raise ValueError("--stages 解析后为空")
    return tuple(stages)


def _make_run_dir(root: Path, prefix: str) -> Path:
    """创建 {root}/{prefix}_{YYYYMMDD_HHMMSS}/ 目录，同秒追加 _001 后缀。

    Args:
        root: 数据根目录（默认 data/scriptData）。
        prefix: run_dir 前缀（默认 'three_stage_pick'）。

    Returns:
        新建的 run_dir 路径。
    """
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{prefix}_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{prefix}_{timestamp}_{suffix:03d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _save_step_image(run_dir: Path, rgb_dict: dict | None, idx: int) -> str | None:
    """把 obs['rgb'] dict 的首个相机保存为 step_{idx:03d}.png。"""
    from PIL import Image
    if rgb_dict is None or not isinstance(rgb_dict, dict) or len(rgb_dict) == 0:
        return None
    first_key = next(iter(rgb_dict))
    rgb = rgb_dict[first_key]
    path = run_dir / f"step_{idx:03d}.png"
    Image.fromarray(rgb).save(path)
    return str(path)


def _save_vla_input_images(input_dir: Path, rgb_dict: dict | None,
                           stage_label: str, chunk_idx: int) -> list[str]:
    """把喂给 VLA 的多视角输入图保存到 input/ 子文件夹。

    文件名: {stage_label}_chunk{XX:02d}_{cam_name_safe}.png
    """
    if rgb_dict is None or not isinstance(rgb_dict, dict) or len(rgb_dict) == 0:
        return []
    from PIL import Image
    saved: list[str] = []
    for cam_name, rgb in rgb_dict.items():
        if not isinstance(rgb, np.ndarray):
            continue
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", cam_name)
        fname = f"{stage_label}_chunk{chunk_idx:02d}_{safe}.png"
        path = input_dir / fname
        Image.fromarray(rgb).save(path)
        saved.append(fname)
    return saved


def _dump_model_output(run_dir: Path, stage_label: str, raw_values: np.ndarray,
                       spec) -> str:
    """把 VLA 输出的原始 ndarray 完整 dump 到 model_output_{stage}.txt。

    Args:
        run_dir: 实验目录。
        stage_label: stage 标签（用于文件名）。
        raw_values: VLAOutput.values, shape (N, action_dim)。
        spec: VLAOutput.spec（用于记录 spec.space / dim）。

    Returns:
        写入的文件路径。
    """
    path = run_dir / f"model_output_{stage_label}.txt"
    arr = np.asarray(raw_values)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# raw VLAOutput.values dump (BEFORE adapter)\n")
        fh.write(f"# shape: {arr.shape}\n")
        fh.write(f"# dtype: {arr.dtype}\n")
        fh.write(f"# spec.space: {spec.space}\n")
        fh.write(f"# spec.components: {spec.components}\n")
        fh.write(f"# per-column min: {arr.min(axis=0).tolist()}\n")
        fh.write(f"# per-column max: {arr.max(axis=0).tolist()}\n")
        fh.write(f"# per-column mean: {arr.mean(axis=0).tolist()}\n")
        fh.write(f"# per-column std:  {arr.std(axis=0).tolist()}\n")
        fh.write("# --- full chunk ---\n")
        # 用 %.6f 保证精度，逗号分隔方便 grep/awk 处理
        for row in arr:
            fh.write(",".join(f"{v:.6f}" for v in row) + "\n")
    return str(path)


def main() -> int:
    args = parse_args()
    stages = _parse_stages(args.stages)

    # 1. 加载配置
    cfg, cfg_path = _load_config(args.config)
    print("=" * 60)
    print("ExAct 三阶段抓取验证脚本（approach → descend → close gripper）")
    print("=" * 60)
    print(f"配置文件:    {cfg_path}")

    # 2. 解析 effective 配置
    vla_backend = args.vla or cfg.vla.backend
    env_mode = "direct" if args.no_gui else cfg.env.mode
    data_root = Path(args.data_root) if args.data_root else SCRIPT_DATA_ROOT
    print(f"VLA 后端:    {vla_backend}")
    print(f"Env 模式:    {env_mode}")
    print(f"数据目录:    {data_root}")
    print(f"Run 前缀:    {args.prefix}")
    print(f"Stage 数:    {len(stages)}")
    for i, (lbl, ins) in enumerate(stages):
        print(f"  [{i}] {lbl:12s} → {ins!r}")

    # 3. 创建实验目录 + input/ + experiment.log
    run_dir = _make_run_dir(data_root, args.prefix)
    input_dir = run_dir / "input"
    input_dir.mkdir(exist_ok=True)
    log_fh = open(run_dir / "experiment.log", "a", encoding="utf-8")

    def log_line(msg: str, level: str = "INFO") -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = f"{ts} [{level}] {msg}\n"
        log_fh.write(line)
        log_fh.flush()
        print(line, end="")

    log_line(f"verify_three_stage_pick started, prefix={args.prefix!r}, "
             f"vla_backend={vla_backend}, stages={[s[0] for s in stages]}")

    # 4. 构造 env
    env_cfg = EnvConfig(
        mode=env_mode,
        renderer=cfg.env.renderer,
        camera_resolution=cfg.env.camera_resolution,
        cameras=cfg.env.cameras,
    )
    robot_cfg = RobotConfig(
        type=cfg.robot.type,
        urdf_path=cfg.robot.urdf_path,
        base_position=cfg.robot.base_position,
        panda=cfg.robot.panda,
        so101=cfg.robot.so101,
    )
    env = PyBulletEnv(env_config=env_cfg, robot_config=robot_cfg)

    try:
        # 5. reset 环境
        task_spec = {"objects": [dict(obj) for obj in cfg.task.objects]}
        obs0 = env.reset(task_spec=task_spec, seed=0)
        log_line(f"[step 0] env.reset 完成, ee_pos={tuple(round(x,4) for x in obs0['ee_pos'])}")
        log_line(f"[step 0] object_info={obs0.get('object_info', [])}")

        # 6. 首帧截图
        rgb0 = obs0.get("rgb")
        p0 = _save_step_image(run_dir, rgb0, 0)
        if p0:
            log_line(f"[step 0] saved {Path(p0).name} "
                     f"(camera={next(iter(rgb0))}, "
                     f"shape={rgb0[next(iter(rgb0))].shape})")

        # 7. 构造 VLA（走工厂）
        vla_cfg = VLAConfig(
            backend=vla_backend,
            model_path=cfg.vla.model_path,
            mock=cfg.vla.mock,
            lerobot=cfg.vla.lerobot,
        )
        try:
            vla = create_vla(vla_cfg, LLMConfig(api_key=cfg.llm.api_key))
        except Exception as e:
            log_line(f"[vla] ❌ create_vla 失败: {type(e).__name__}: {e}", "ERROR")
            return 1
        log_line(f"[vla] {type(vla).__name__} 已构造")

        # 8. 构造 Executor + adapter（adapter 仍由 Executor 使用；
        #    我们手动调 vla.predict 以便拿原始 VLAOutput.values）
        executor = Executor(vla)
        adapter = get_adapter(vla.output_spec, env.input_spec)
        log_line(f"[adapter] {vla.output_spec.space} → {env.input_spec.space} "
                 f"({adapter.__name__})")

        # 9. 顺序执行 3 个 stage
        from executor.build_input import build_vla_input
        from executor.check_done import check_done

        obs_before = obs0
        cumulative_disp = 0.0
        total_steps = 0
        stage_results: list[dict] = []

        for stage_idx, (stage_label, instruction) in enumerate(stages):
            log_line(f"\n{'='*60}\n[stage {stage_idx}/{len(stages)}] "
                     f"label={stage_label!r} instruction={instruction!r}")
            log_line(f"{'='*60}")

            # 9.1 拿 obs + state，保存 VLA 输入图
            obs_input = env.get_obs(include_rgb=True)
            input_fnames = _save_vla_input_images(
                input_dir, obs_input.get("rgb"), stage_label, chunk_idx=0
            )
            if input_fnames:
                log_line(f"[stage {stage_label}] VLA 输入图 → "
                         f"input/{', '.join(input_fnames)}")

            state_vec = None
            try:
                state_vec = env.get_joint_state()
            except (NotImplementedError, AttributeError):
                state_vec = None

            vla_input = build_vla_input(obs_input, instruction)

            # 9.2 调 vla.predict 拿原始 VLAOutput
            try:
                # 兼容不同后端：部分 VLA.predict 接受 state 参数
                _accepts_state = "state" in inspect.signature(
                    vla.predict
                ).parameters
                if _accepts_state:
                    vla_output = vla.predict(
                        vla_input["image"], instruction, state=state_vec
                    )
                else:
                    vla_output = vla.predict(vla_input["image"], instruction)
            except Exception as e:
                log_line(f"[stage {stage_label}] ❌ vla.predict 失败: "
                         f"{type(e).__name__}: {e}", "ERROR")
                traceback.print_exc()
                stage_results.append({"label": stage_label, "error": str(e)})
                break

            # 9.3 dump 原始输出
            raw_values = vla_output.values
            dump_path = _dump_model_output(
                run_dir, stage_label, raw_values, vla_output.spec
            )
            arr = np.asarray(raw_values)
            log_line(f"[stage {stage_label}] VLA 原始输出: shape={arr.shape}, "
                     f"dtype={arr.dtype}, "
                     f"min={arr.min():.4f}, max={arr.max():.4f}, "
                     f"mean={arr.mean():.4f}")
            log_line(f"[stage {stage_label}] 每列 mean: "
                     f"{arr.mean(axis=0).round(4).tolist()}")
            log_line(f"[stage {stage_label}] 每列 std:  "
                     f"{arr.std(axis=0).round(4).tolist()}")
            log_line(f"[stage {stage_label}] 第 0 步: {arr[0].round(4).tolist()}")
            log_line(f"[stage {stage_label}] 最后 1 步: "
                     f"{arr[-1].round(4).tolist()}")
            log_line(f"[stage {stage_label}] 完整 dump → {Path(dump_path).name}")

            # 9.4 应用 adapter
            actions = adapter(raw_values, env) if adapter is not None else raw_values

            # 9.5 顺序执行 actions（沿用 executor 的 chunk 契约：中间不 check_done）
            obs_after = obs_input
            for action in actions:
                obs_after, _, _, _ = env.step(action)

            # 9.6 事后报告（不影响循环）
            try:
                target_pos = None
                if obs_input.get("object_info"):
                    for obj in obs_input["object_info"]:
                        if obj.get("name") == "cube":
                            target_pos = tuple(obj["pos"])
                            break
                done, reason = check_done(
                    obs_input, obs_after, "reached", target_pos=target_pos
                )
            except Exception as e:
                done, reason = None, f"check_done 异常: {e}"

            chunk_steps = len(actions) if hasattr(actions, "__len__") else 0
            total_steps += chunk_steps
            step_disp = 0.0
            if obs_before.get("ee_pos") and obs_after.get("ee_pos"):
                step_disp = math.dist(obs_before["ee_pos"], obs_after["ee_pos"])
                cumulative_disp += step_disp

            ee_after = obs_after.get("ee_pos", (None, None, None))
            log_line(f"[stage {stage_label}] steps={chunk_steps}, "
                     f"disp={step_disp:.4f}m, done={done}")
            log_line(f"[stage {stage_label}] "
                     f"ee_before={tuple(round(x,4) for x in obs_before['ee_pos'])}, "
                     f"ee_after={tuple(round(x,4) for x in ee_after)}, "
                     f"reason={reason}")

            # 9.7 截图
            rgb_after = obs_after.get("rgb")
            p = _save_step_image(run_dir, rgb_after, total_steps)
            if p:
                log_line(f"[stage {stage_label}] saved {Path(p).name}")

            stage_results.append({
                "label": stage_label,
                "instruction": instruction,
                "chunk_steps": chunk_steps,
                "ee_before": tuple(round(x, 4) for x in obs_before["ee_pos"]),
                "ee_after": tuple(round(x, 4) for x in ee_after),
                "disp_m": round(step_disp, 4),
                "done": done,
                "reason": reason,
                "dump_path": dump_path,
            })
            obs_before = obs_after

        # 10. 总结
        log_line(f"\nverify_three_stage_pick finished, total_stages={len(stage_results)}, "
                 f"total_steps={total_steps}, cumulative_disp={cumulative_disp:.4f}m")
        for r in stage_results:
            log_line(f"  [{r['label']:12s}] steps={r['chunk_steps']:3d} "
                     f"disp={r['disp_m']:.4f}m "
                     f"ee_after={r.get('ee_after')}")

        print("=" * 60)
        print(f"运行目录: {run_dir}")
        print(f"累计位移: {cumulative_disp:.4f} m | 总步数: {total_steps}")
        print("=" * 60)

    finally:
        try:
            log_fh.close()
        except Exception:
            pass
        env.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
