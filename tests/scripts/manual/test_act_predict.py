"""ACT policy 端到端冒烟脚本。

直接绕开 LLM agent 循环，按以下链路验证 ACT 模型能否在本地权重上
跑通 predict：

  config/local.yaml
    → PyBulletEnv（启动仿真，拿一帧 RGB）
    → LeRobotVLA.predict(image, instruction)
    → 打印动作或完整 traceback

运行：
    python tests/scripts/manual/test_act_predict.py

可选参数：
    --instruction "..."     自定义指令（默认："把机械臂末端移动到 (0.5, 0, 0.3)"）
    --steps 1               一帧动作最多跑几步（默认 1，只看链路）
    --no-reset              跳过 env.reset()（默认会 reset 到初始状态）

设计目标：每一层抛错都给出 traceback + 关键诊断信息，方便定位问题。
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

# 把 src/ 加进 sys.path（与 smoke_observe_multimodal.py 一致）
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def banner(s: str) -> None:
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", type=str,
                        default="把机械臂末端移动到 (0.5, 0.0, 0.3)")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()

    # 离线优先：禁止任何远程下载/上传
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    # 1) 加载配置
    banner("STEP 1: 加载配置 configs/local.yaml")
    try:
        from config import load_config
        cfg = load_config(str(PROJECT_ROOT / "configs" / "local.yaml"))
        print(f"  backend        : {cfg.vla.backend}")
        print(f"  model_path     : {cfg.vla.model_path}")
        print(f"  policy_type    : {cfg.vla.lerobot.policy_type}")
        print(f"  device         : {cfg.vla.lerobot.device}")
        print(f"  image_key      : {cfg.vla.lerobot.image_key}")
        print(f"  action_dim cfg : {cfg.vla.lerobot.action_dim}")
        print(f"  quantization   : {cfg.vla.lerobot.quantization}")
    except Exception:
        print("✗ 配置加载失败：")
        traceback.print_exc()
        return 1

    # 2) 实例化 VLA（直接，不走 Executor）
    banner("STEP 2: 实例化 LeRobotVLA（懒加载权重，首批日志会在这里冒出来）")
    try:
        from executor.model.lerobot.lerobot_vla import LeRobotVLA
        vla = LeRobotVLA(
            model_path=str(PROJECT_ROOT / cfg.vla.model_path),
            policy_type=cfg.vla.lerobot.policy_type,
            device=cfg.vla.lerobot.device,
            image_key=cfg.vla.lerobot.image_key,
            action_dim=cfg.vla.lerobot.action_dim,
            quantization=cfg.vla.lerobot.quantization,
        )
        print(f"✓ LeRobotVLA 实例化成功（policy 仍为 None，懒加载）")
    except Exception:
        print("✗ VLA 实例化失败：")
        traceback.print_exc()
        return 2

    # 3) 启动 PyBullet 仿真（与 src/pipeline/runner.py 一致：传 env_config / robot_config）
    banner("STEP 3: 启动 PyBullet 仿真 env")
    try:
        from env.pybullet_env import PyBulletEnv
        env = PyBulletEnv(
            env_config=cfg.env,
            robot_config=cfg.robot,
        )
        if not args.no_reset:
            # 与 pipeline.runner._to_task_spec_dict 同款：objects 转 dict
            task_spec = {"objects": [dict(obj) for obj in cfg.task.objects]}
            env.reset(task_spec=task_spec, seed=0)
        print(f"✓ env 启动成功，camera_resolution={cfg.env.camera_resolution}")
    except Exception:
        print("✗ env 启动失败：")
        traceback.print_exc()
        return 3

    # 4) 拿到一帧 RGB
    banner("STEP 4: env.get_obs() 拿一帧 RGB")
    try:
        obs = env.get_obs(include_rgb=True)
        rgb = obs.get("rgb")
        if rgb is None:
            print("✗ obs['rgb'] 是 None")
            return 4
        print(f"✓ rgb shape={rgb.shape}, dtype={rgb.dtype}, "
              f"min={rgb.min()}, max={rgb.max()}")
        if rgb.dtype != "uint8":
            print(f"⚠️  rgb 不是 uint8（实际 {rgb.dtype}），predict 会校验失败")
    except Exception:
        print("✗ get_obs 失败：")
        traceback.print_exc()
        return 4

    # 5) 调 VLA predict（关键步骤）
    banner(f"STEP 5: vla.predict(rgb, instruction={args.instruction!r})")
    try:
        action = vla.predict(rgb, args.instruction)
        print(f"✓ predict 成功：action = {action}")
        return 0
    except Exception as e:
        print(f"✗ predict 失败：{type(e).__name__}: {e}")
        print("\n--- full traceback ---")
        traceback.print_exc()
        print("--- end traceback ---\n")

        # 6) 额外诊断：dump 当前 VLA 解析出的 key/dim
        banner("DIAG: VLA 自动探测结果（事后取证）")
        try:
            vla._ensure_loaded()
            print(f"  policy loaded: {type(vla._policy).__name__}")
            cfg_obj = vla._policy.config
            print(f"  resolved_image_keys: {vla._resolved_image_keys}")
            print(f"  resolved_state_key : {vla._resolved_state_key}")
            print(f"  infer_state_dim    : {vla._infer_state_dim()}")
            if hasattr(cfg_obj, "image_features"):
                for k, ft in cfg_obj.image_features.items():
                    print(f"  image_features[{k}].shape = {ft.shape}")
            if hasattr(cfg_obj, "robot_state_feature") and cfg_obj.robot_state_feature:
                print(f"  robot_state_feature.shape = {cfg_obj.robot_state_feature.shape}")
            if hasattr(cfg_obj, "action_feature") and cfg_obj.action_feature:
                print(f"  action_feature.shape      = {cfg_obj.action_feature.shape}")
        except Exception:
            print("  （事后诊断也失败，policy 可能根本没加载成功）")
            traceback.print_exc()

        return 5
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
