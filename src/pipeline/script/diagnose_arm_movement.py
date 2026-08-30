"""诊断脚本：直接调 Executor 跑命令，看机械臂到底有没有移动。

问题：每次 LLM 都说机械臂没移动。是真的没动还是判定问题？本脚本：
  1. 构造 PyBulletEnv + Executor + (Mock 或真实)VLA
  2. 跑一条简单指令
  3. **逐步打印**每一步：
     - VLA 输出的 action（dx/dy/dz/dr/dgripper）
     - env.step 前后 ee_pos 的差异
     - check_done 的判定
  4. 最后总结：机械臂到底有没有动？动了多少？

可选参数：
  --instruction "move forward"   指令（默认 "move forward"）
  --vla mock|lerobot|smolvla    VLA 后端（默认 mock，避开 GPU）
  --max-steps 5                 步数（默认 5，便于人工看差异）
  --use-gui                     开 GUI 窗口（默认不开，打印数值即可）
  --print-image                 每步打印图像 shape（默认不打印，避免刷屏）

用法：
  python src/pipeline/script/diagnose_arm_movement.py
  python src/pipeline/script/diagnose_arm_movement.py --use-gui --max-steps 3
  python src/pipeline/script/diagnose_arm_movement.py --vla mock --max-steps 10
"""

from __future__ import annotations

import argparse
import math
import sys
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
from executor.model.base import VLAOutput
from executor.model.factory import create_vla
from utils.adapter import get_adapter


DEFAULT_CONFIG_PATHS = (
    str(ROOT / "configs" / "local.yaml"),
    str(ROOT / "configs" / "default.yaml"),
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
    p = argparse.ArgumentParser(
        description="诊断机械臂是否真的移动（绕开 LLM 与 ActionTool）"
    )
    p.add_argument("--config", default=None,
        help=f"配置文件路径（默认自动选 local.yaml > default.yaml）")
    p.add_argument("--instruction", default=None,
        help="传给 VLA 的英文指令（默认读取 config.task.default_user_goal）")
    p.add_argument("--vla", default=None,
        choices=["mock", "lerobot", "smolvla"],
        help="VLA 后端（默认读取 config.vla.backend）")
    p.add_argument("--max-steps", type=int, default=None,
        help="executor 步数（默认读取 config.vla.max_steps）")
    p.add_argument("--use-gui", action="store_true",
        help="开 PyBullet GUI 窗口（默认按 config.env.mode）")
    p.add_argument("--no-gui", action="store_true",
        help="强制 DIRECT 模式（覆盖 config.env.mode）")
    p.add_argument("--print-image", action="store_true",
        help="每步打印 obs rgb shape（默认不打印）")
    return p.parse_args()


def _print_action(action) -> None:
    """把 action 拆成 7 维打印，便于看 VLA 是否真的产生位移。"""
    if isinstance(action, VLAOutput):
        action = action.values
    if hasattr(action, "__len__") and len(action) == 7:
        dx, dy, dz, drx, dry, drz, gripper = action
        print(
            f"    action = "
            f"dx={dx:+.4f} dy={dy:+.4f} dz={dz:+.4f} "
            f"drx={drx:+.4f} dry={dry:+.4f} drz={drz:+.4f} "
            f"gripper={gripper:.2f}"
        )
        disp = math.sqrt(dx * dx + dy * dy + dz * dz)
        print(f"    ||translation|| = {disp:.4f} m")
    else:
        print(f"    action = {action!r}")


def main() -> int:
    args = parse_args()

    # 加载配置（CLI > local.yaml > default.yaml）
    cfg, cfg_path = _load_config(args.config)
    print("=" * 60)
    print("ExAct 机械臂移动诊断脚本")
    print("=" * 60)
    print(f"配置文件:    {cfg_path}")

    # 解析 effective 配置（CLI 参数覆盖 config 默认值）
    instruction = args.instruction or cfg.task.default_user_goal
    vla_backend = args.vla or cfg.vla.backend
    max_steps = args.max_steps if args.max_steps is not None else cfg.vla.max_steps
    if args.no_gui:
        env_mode = "direct"
    elif args.use_gui:
        env_mode = "gui"
    else:
        env_mode = cfg.env.mode

    print(f"指令:        {instruction!r}")
    print(f"VLA 后端:    {vla_backend}")
    print(f"最大步数:    {max_steps}")
    print(f"Env 模式:    {env_mode}")
    print()

    # 1. 构造 env
    env_cfg = EnvConfig(
        mode=env_mode,
        renderer=cfg.env.renderer,
        camera_resolution=cfg.env.camera_resolution,
    )
    robot_cfg = RobotConfig(
        type=cfg.robot.type,
        urdf_path=cfg.robot.urdf_path,
        base_position=cfg.robot.base_position,
        panda=cfg.robot.panda,
        so101=cfg.robot.so101,
    )
    env = PyBulletEnv(env_config=env_cfg, robot_config=robot_cfg)

    # 2. reset（必须，否则 env 没物理世界）
    task_spec = {
        "objects": [dict(obj) for obj in cfg.task.objects],
    }
    obs0 = env.reset(task_spec=task_spec, seed=0)
    print(f"[step 0] env.reset 完成")
    print(f"    ee_pos (initial) = {tuple(round(x, 4) for x in obs0['ee_pos'])}")
    print(f"    object_info     = {obs0.get('object_info', [])}")
    print(f"    rgb shape       = {obs0.get('rgb', np.zeros(1)).shape}")
    print()

    # 3. 构造 VLA：走 factory，自动按 vla.mock.variant 分派（MockVLA / JointMockVLA）
    from executor.model.mock.mock_vla import JointMockVLA
    vla_cfg = VLAConfig(
        backend=vla_backend,
        model_path=cfg.vla.model_path,
        max_steps=max_steps,
        mock=cfg.vla.mock,
        lerobot=cfg.vla.lerobot,
    )
    if vla_backend == "mock":
        try:
            vla = create_vla(vla_cfg, LLMConfig(api_key="dummy"))
        except Exception as e:
            print(f"[vla] ❌ create_vla 失败：{type(e).__name__}: {e}")
            env.close()
            return 1
        if isinstance(vla, JointMockVLA):
            print(f"[vla] JointMockVLA(seed=0)（joint 空间 6 维，关节增量 [-0.1, 0.1] rad）")
        else:
            print(f"[vla] MockVLA(seed=0)（task 空间 7 维，位移 [-0.15, 0.15] m）")
    else:
        try:
            vla = create_vla(vla_cfg, LLMConfig(api_key="dummy"))
        except Exception as e:
            print(f"[vla] ❌ create_vla 失败：{type(e).__name__}: {e}")
            print("[vla] 提示：用 --vla mock 先确认链路通，再换真实 VLA")
            env.close()
            return 1
        print(f"[vla] {type(vla).__name__} 已构造")

    # 4. 构造 Executor
    executor = Executor(vla, max_steps=max_steps)

    # 5. 构造 adapter（task 空间 → task 空间，identity 直通；其他空间应报错）
    adapter = get_adapter(vla.output_spec, env.input_spec)
    print(f"[adapter] vla.output_spec.space={vla.output_spec.space} → "
          f"env.input_spec.space={env.input_spec.space}（{adapter.__name__}）")
    print()

    # 6. 逐步执行（绕开 executor 主循环，自己逐步 step + check_done，
    #    这样能清楚看到每步的 action 与 ee_pos 变化）
    #
    # Iteration 10 chunk 契约：循环边界 = VLA 输出的 chunk size（N），
    # **不**是硬编码 max_steps。VLA predict 一次返回整 chunk (N, action_dim)，
    # 我们按第一维迭代执行；max_steps 仅作为兜底上限（N > max_steps 时截断）。
    obs_before = env.get_obs()
    cumulative_disp = 0.0
    rgb = obs_before.get("rgb")
    if args.print_image and rgb is not None:
        print(f"    [debug] obs_before.rgb shape = {rgb.shape}")

    vla_out = vla.predict(rgb, instruction)
    if isinstance(vla_out, VLAOutput):
        vla_out = vla_out.values
    chunk = np.asarray(vla_out)  # shape (N, action_dim)
    if chunk.ndim == 1:
        chunk = chunk.reshape(1, -1)

    chunk_n = len(chunk)
    print(f"[vla] predict 输出 chunk_size={chunk_n}")
    print()

    for step in range(chunk_n):
        # 6.1 adapter（task 空间直通，identity）
        single_action = chunk[step]
        action_for_env = adapter(single_action, env) if False else single_action
        # 这里 adapter 整 chunk 转效率更高，但诊断脚本要每步打印——按单步调用
        if adapter.__name__ != "identity_transform":
            # adapter 设计是按整 chunk 输入；这里 wrap 成 (1, dim) 给 adapter
            wrapped = single_action.reshape(1, -1)
            action_for_env = adapter(wrapped, env)
            # adapter 输出 (1, dim)，env.step 要 (dim,)
            if isinstance(action_for_env, np.ndarray) and action_for_env.ndim == 2:
                action_for_env = action_for_env[0]

        # 6.2 env.step
        obs_after, _, _, _ = env.step(action_for_env)

        # 6.3 位移计算
        delta_pos = tuple(
            round(obs_after["ee_pos"][i] - obs_before["ee_pos"][i], 4)
            for i in range(3)
        )
        step_disp = math.dist(obs_before["ee_pos"], obs_after["ee_pos"])
        cumulative_disp += step_disp

        # 6.4 check_done 判定（事后报告，不影响循环）
        from executor.check_done import check_done
        done, reason = check_done(
            obs_before, obs_after, "reached",
            target_pos=None,
        )

        print(f"[step {step + 1}/{chunk_n}]")
        _print_action(single_action)
        print(f"    ee_pos (before) = {tuple(round(x, 4) for x in obs_before['ee_pos'])}")
        print(f"    ee_pos (after)  = {tuple(round(x, 4) for x in obs_after['ee_pos'])}")
        print(f"    delta ee_pos    = {delta_pos}（本步位移 {step_disp:.4f} m）")
        print(f"    cumulative      = {cumulative_disp:.4f} m（累计位移）")
        print(f"    check_done      = {done} | {reason}")
        print()

        obs_before = obs_after

    # 7. 总结
    print("=" * 60)
    print("诊断结论")
    print("=" * 60)
    if cumulative_disp < 0.01:
        print(f"❌ 机械臂基本没动（累计位移 {cumulative_disp:.4f} m < 0.01 m）。")
        print("   可能原因：")
        print("   (a) VLA 输出的位移太小（mock 是 ±0.15 m 内随机）")
        print("   (b) check_done 阈值 0.01 m 过严，单步位移可能 ≥ 但 < 阈值")
        print("   (c) adapter 路径错误：VLA 输出未正确喂给 env.step")
        print("   (d) env.step 内部没推进物理（time.sleep 缺失、IK 求解失败）")
    elif cumulative_disp < 0.05:
        print(f"⚠️ 机械臂动了，但幅度很小（累计 {cumulative_disp:.4f} m）。")
        print("   → LLM 报告『没动』是因为相对目标位置（如 (0.5,0,0.3)）距离仍很大")
    else:
        print(f"✅ 机械臂确实移动了（累计 {cumulative_disp:.4f} m）。")
        print("   → LLM 报告『没动』可能是误判：")
        print("     · LLM 看的目标位置不在本次指令范围内")
        print("     · LLM 期望的位移比 VLA 实际产生的大")
        print("     · VLA 输出的 direction 与 LLM 期望的方向不一致")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())