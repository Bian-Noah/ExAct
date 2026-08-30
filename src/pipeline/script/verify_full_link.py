"""全链路验证脚本：绕开 Agent/LLM，直接把「指令 + 图片」喂给 VLA 执行。

用途：验证 env → VLA → executor → env.step 的完整链路是否工作，不依赖 LLM。
与 diagnose_arm_movement.py 的区别：
  - 本脚本走 Executor.run_action（iter 10 chunk 契约），不是自己逐步 step
  - 指令硬编码（不读 config.task.default_user_goal），默认「捡起方块」
  - 喂给 VLA 的**多视角输入图**单独存到 input/ 子文件夹，便于复盘 VLA 看到了什么
  - 每 chunk 结束画面存 step_XXX.png

数据落盘结构：
  data/scriptData/
    └── {YYYYMMDD_HHMMSS}/
         ├── experiment.log          # 全程文本日志
         ├── input/                  # 喂给 VLA 的输入图（每 chunk 前，每相机一张）
         │    ├── chunk00_observation_images_top.png
         │    └── ...
         ├── step_000.png            # 初始观测
         └── step_00N.png            # 每 chunk 结束画面

用法：
  python src/pipeline/script/verify_full_link.py
  python src/pipeline/script/verify_full_link.py --vla mock --max-chunks 2
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

# 让脚本能 import src/ 下的模块（与 pytest 收集测试时的 sys.path 保持一致）
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
# 脚本数据根目录（绝对路径，与 experiment 独立）
SCRIPT_DATA_ROOT = ROOT / "data" / "scriptData"
# 硬编码默认指令（捡起方块，smolVLA 规范：动词 pick 开头 / 全英文 / ≤30 字符）
DEFAULT_INSTRUCTION = "pick up the cube"


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
    p = argparse.ArgumentParser(
        description="全链路验证：指令 + 图片 → VLA → executor → env（绕开 Agent/LLM）"
    )
    p.add_argument("--config", default=None,
        help=f"配置文件路径（默认自动选 local.yaml > default.yaml）")
    p.add_argument("--instruction", default=None,
        help=f"传给 VLA 的英文指令（默认硬编码 {DEFAULT_INSTRUCTION!r}，不读 config）")
    p.add_argument("--vla", default=None,
        choices=["mock", "lerobot", "openvla", "llm_vla"],
        help="VLA 后端（默认读取 config.vla.backend）")
    p.add_argument("--max-chunks", type=int, default=None,
        help="最多执行几个 chunk（VLA 每次 predict 出一个 chunk；默认 1）")
    p.add_argument("--no-gui", action="store_true",
        help="强制 DIRECT 模式（默认按 config.env.mode）")
    p.add_argument("--data-root", default=None,
        help=f"脚本数据根目录（默认 {SCRIPT_DATA_ROOT}）")
    return p.parse_args()


def _make_run_dir(root: Path) -> Path:
    """创建 {root}/{YYYYMMDD_HHMMSS}/ 目录，同秒追加 _001 后缀。"""
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{timestamp}_{suffix:03d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _save_step_image(run_dir: Path, rgb_dict: dict | None, idx: int) -> str | None:
    """把 obs['rgb'] dict 的首个相机保存为 step_{idx:03d}.png，返回保存路径或 None。

    iter11-reset-multicam:rgb 是 dict[str, ndarray],取第一个相机保存。
    """
    from PIL import Image
    if rgb_dict is None or not isinstance(rgb_dict, dict) or len(rgb_dict) == 0:
        return None
    first_key = next(iter(rgb_dict))
    rgb = rgb_dict[first_key]
    path = run_dir / f"step_{idx:03d}.png"
    Image.fromarray(rgb).save(path)
    return str(path)


def _save_vla_input_images(input_dir: Path, rgb_dict: dict | None, chunk_idx: int) -> list[str]:
    """把喂给 VLA 的多视角输入图保存到 input/ 子文件夹。

    每个相机存一张,文件名含 chunk 序号 + 相机名（sanitize 掉非 [a-zA-Z0-9_] 字符）。

    Returns:
        已保存的文件名列表（便于 log）。
    """
    if rgb_dict is None or not isinstance(rgb_dict, dict) or len(rgb_dict) == 0:
        return []
    from PIL import Image
    saved: list[str] = []
    for cam_name, rgb in rgb_dict.items():
        if not isinstance(rgb, np.ndarray):
            continue
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", cam_name)
        fname = f"chunk{chunk_idx:02d}_{safe}.png"
        path = input_dir / fname
        Image.fromarray(rgb).save(path)
        saved.append(fname)
    return saved


def main() -> int:
    args = parse_args()

    # 1. 加载配置
    cfg, cfg_path = _load_config(args.config)
    print("=" * 60)
    print("ExAct 全链路验证脚本（指令 + 图片 → VLA → executor → env）")
    print("=" * 60)
    print(f"配置文件:    {cfg_path}")

    # 2. 解析 effective 配置
    # 指令硬编码,不读 config.task.default_user_goal（用户要求）
    instruction = args.instruction or DEFAULT_INSTRUCTION
    vla_backend = args.vla or cfg.vla.backend
    max_chunks = args.max_chunks if args.max_chunks is not None else 1
    env_mode = "direct" if args.no_gui else cfg.env.mode
    data_root = Path(args.data_root) if args.data_root else SCRIPT_DATA_ROOT

    print(f"指令:        {instruction!r}（硬编码,不读 config）")
    print(f"VLA 后端:    {vla_backend}")
    print(f"最大 chunk:  {max_chunks}")
    print(f"Env 模式:    {env_mode}")
    print(f"数据目录:    {data_root}")

    # 3. 创建实验目录 + input/ 子目录 + experiment.log
    run_dir = _make_run_dir(data_root)
    input_dir = run_dir / "input"
    input_dir.mkdir(exist_ok=True)
    log_fh = open(run_dir / "experiment.log", "a", encoding="utf-8")

    def log_line(msg: str, level: str = "INFO") -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = f"{ts} [{level}] {msg}\n"
        log_fh.write(line)
        log_fh.flush()
        print(line, end="")

    log_line(f"verify_full_link started, instruction={instruction!r}（硬编码）, "
             f"vla_backend={vla_backend}, max_chunks={max_chunks}")

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
        widowx=cfg.robot.widowx,
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
                     f"(camera={next(iter(rgb0))}, shape={rgb0[next(iter(rgb0))].shape})")

        # 7. 构造 VLA（走工厂）
        vla_cfg = VLAConfig(
            backend=vla_backend,
            model_path=cfg.vla.model_path,
            mock=cfg.vla.mock,
            lerobot=cfg.vla.lerobot,
            openvla=cfg.vla.openvla,
        )
        try:
            vla = create_vla(vla_cfg, LLMConfig(api_key=cfg.llm.api_key))
        except Exception as e:
            log_line(f"[vla] ❌ create_vla 失败: {type(e).__name__}: {e}", "ERROR")
            return 1
        log_line(f"[vla] {type(vla).__name__} 已构造")

        # 8. 构造 Executor + adapter
        executor = Executor(vla)
        adapter = get_adapter(vla.output_spec, env.input_spec)
        log_line(f"[adapter] {vla.output_spec.space} → {env.input_spec.space} ({adapter.__name__})")

        # 9. 循环执行 chunk
        import math
        obs_before = obs0
        cumulative_disp = 0.0
        total_steps = 0
        for chunk_idx in range(max_chunks):
            # 9.1 保存本次喂给 VLA 的多视角输入图（每相机一张）到 input/
            obs_input = env.get_obs(include_rgb=True)
            input_fnames = _save_vla_input_images(input_dir, obs_input.get("rgb"), chunk_idx)
            if input_fnames:
                log_line(f"[chunk {chunk_idx + 1}/{max_chunks}] VLA 输入图 → "
                         f"input/{', '.join(input_fnames)}")

            # 9.2 VLA.predict（图片 + 指令直接喂入，绕开 Agent）
            vla_out = executor.run_action(
                env,
                instruction,
                done_criteria="reached",
                target_pos=None,
                adapter=adapter,
                operation="vla",
            )
            chunk_steps = vla_out.steps
            total_steps += chunk_steps
            final_obs = vla_out.final_obs

            # 9.3 位移累计
            if obs_before.get("ee_pos") and final_obs.get("ee_pos"):
                step_disp = math.dist(obs_before["ee_pos"], final_obs["ee_pos"])
                cumulative_disp += step_disp

            log_line(f"[chunk {chunk_idx + 1}/{max_chunks}] steps={chunk_steps}, "
                     f"success={vla_out.success}, message={vla_out.message}")
            log_line(f"[chunk {chunk_idx + 1}/{max_chunks}] "
                     f"ee_before={tuple(round(x,4) for x in obs_before['ee_pos'])}, "
                     f"ee_after={tuple(round(x,4) for x in final_obs['ee_pos'])}")

            # 9.4 每 chunk 结束截图（每步截图会太多，按 chunk 粒度截）
            rgb_after = final_obs.get("rgb")
            p = _save_step_image(run_dir, rgb_after, total_steps)
            if p:
                log_line(f"[chunk {chunk_idx + 1}/{max_chunks}] saved {Path(p).name}")

            obs_before = final_obs
            # 结束条件：VLA 输出 0 步（异常）或到达目标
            if chunk_steps == 0:
                log_line(f"[chunk {chunk_idx + 1}/{max_chunks}] VLA 输出 0 步，提前终止", "WARN")
                break

        # 10. 总结
        log_line(f"verify_full_link finished, total_chunks={max_chunks}, "
                 f"total_steps={total_steps}, cumulative_disp={cumulative_disp:.4f}m")
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
