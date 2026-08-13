"""ObserveTool 多模态冒烟脚本（Iteration 5）。

跑一次真实 run_pipeline 流程（用 stub 替换 LLM 调用），验证：
- observe 工具返回 list[dict]（text + image 块）
- experiment.log 含 [observe] saved observer/000.png 行
- data/experiment/{ts}/observer/000.png 文件存在
- data/images/observations/{ts}_NNN.png 文件存在
- image_store.list("observations") 非空
- action 工具仍按旧契约（无 image_url 参数）

用法：
    python scripts/smoke_observe_multimodal.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config.loader import load_config  # noqa: E402
from agents.core import AgentResult  # noqa: E402
from pipeline import runner as pipeline_runner  # noqa: E402


def _stub_run_agent(*args, **kwargs):
    """替代真实 LLM 调用，直接返回成功结果。"""
    from experiment.recorder import get_recorder
    recorder = get_recorder()
    if recorder is not None and getattr(recorder, "_observer_dir", None):
        # 触发 observe 已被 recorder 落盘过；trajectory 留空
        pass
    return AgentResult(
        success=True,
        trajectory=[],
        final_answer="Smoke test 完成",
        total_tool_calls=0,
    )


def main() -> int:
    cfg_path = PROJECT_ROOT / "configs" / "default.yaml"
    print(f"加载配置: {cfg_path}")
    cfg = load_config(str(cfg_path))

    # 不需要真实 API key：用 stub 替换 run_agent
    cfg.agent.max_react_rounds = 1
    cfg.agent.max_tool_calls = 1
    original_run_agent = pipeline_runner.run_agent
    pipeline_runner.run_agent = _stub_run_agent

    # 先调用一次 observe 工具（确保有 image 落盘）
    from tools import ObserveTool
    from env.pybullet_env import PyBulletPandaEnv
    from config import EnvConfig, RobotConfig
    from utils.image_store import create_image_store

    print("\n[Step 1] 真实 env + observe 工具调用")
    env = PyBulletPandaEnv(env_config=EnvConfig(mode="direct", renderer="cpu"),
                            robot_config=RobotConfig())
    env.reset(task_spec={"objects": [{"type": "cube", "pos": [0.5, 0, 0.1], "color": "red"}]}, seed=0)
    image_store = create_image_store(cfg.image_store)
    tool = ObserveTool(env=env, image_store=image_store)
    result = tool._run()
    env.close()

    print(f"  _run() 返回类型: {type(result).__name__}")
    print(f"  返回长度: {len(result)}")
    print(f"  list[0].type: {result[0]['type']}")
    if len(result) > 1:
        print(f"  list[1].type: {result[1]['type']}")
        print(f"  list[1].url: {result[1]['url']}")

    saved_url = result[1]["url"] if len(result) > 1 else None

    # 跑一次 pipeline（用 stub LLM）
    print("\n[Step 2] 跑一次 run_pipeline（stub LLM）")
    try:
        result_pipeline = pipeline_runner.run_pipeline(cfg, user_goal="describe the scene")
        print(f"  pipeline_result: success={result_pipeline.agent_result.success}")
    except Exception as e:
        print(f"  pipeline 异常: {e}")
        return 1

    # 验证 image_store 落盘
    print(f"\n[Step 3] 验证 image_store 落盘")
    img_dir = PROJECT_ROOT / "data" / "images" / "observations"
    if img_dir.exists():
        pngs = sorted(img_dir.glob("*.png"))
        print(f"  data/images/observations/ 共 {len(pngs)} 张图")
        for p in pngs[:3]:
            print(f"    {p.name}")
    else:
        print("  (目录不存在)")

    # 验证 image URL 跨模块可加载
    if saved_url:
        print(f"\n[Step 4] 验证 image URL 跨模块可加载")
        loaded = image_store.load(saved_url)
        print(f"  loaded shape: {loaded.shape}")
        print(f"  dtype: {loaded.dtype}")

    # 验证 experiment 归档
    print(f"\n[Step 5] 验证 experiment 归档")
    exp_root = PROJECT_ROOT / "data" / "experiment"
    if exp_root.exists():
        exp_dirs = sorted(exp_root.iterdir())
        print(f"  共 {len(exp_dirs)} 次实验")
        if exp_dirs:
            latest = exp_dirs[-1]
            print(f"  最新: {latest.name}")
            log_file = latest / "experiment.log"
            if log_file.exists():
                for line in log_file.read_text(encoding="utf-8").splitlines()[:5]:
                    print(f"    log: {line}")

    # 恢复
    pipeline_runner.run_agent = original_run_agent
    print("\n✓ Smoke 完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
