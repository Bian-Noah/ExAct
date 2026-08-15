"""robot-vla-adapter 功能场景 B：adapter 分派 + Mock→Panda 全链路 + 异常路径。

验证：
  1. get_adapter(vla.output_spec, env.input_spec) → identity_transform（task→task 直通）
  2. run_pipeline 完整流程（真实 PyBulletEnv direct 模式 + 打桩 LLM/agent）
     在 data/experiment/ 落盘 experiment.log 且含 Pipeline finished
  3. 异常路径：未知 robot.type → ValueError；未注册 spec 组合 → AdapterNotFoundError

强制 EnvConfig(mode="direct")，不开 GUI 弹窗。PASS/FAIL 通过退出码表达。
LLM 部分打桩（避免真实 MiniMax API 调用）：镜像 tests/functional/test_app_runs.py。
运行：PYTHONPATH=src python tests/scripts/smoke_robot_vla_adapter.py
"""

import glob
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, "src")

from config.loader import RobotConfig, load_config
from env.base import ActionSpec
from env.robot import build_robot
from utils.adapter import get_adapter
from utils.adapter.main import AdapterNotFoundError


def _check_adapter_dispatch(config) -> None:
    """验证 Mock→Panda 的 adapter 分派为 identity_transform（task→task 直通）。"""
    from env.pybullet_env import PyBulletEnv
    from executor.model.mock.mock_vla import MockVLA
    from utils.adapter.adapters.identity import identity_transform

    env = PyBulletEnv(
        env_config=config.env,
        robot_config=config.robot,
    )
    vla = MockVLA(seed=0)
    adapter = get_adapter(vla.output_spec, env.input_spec)

    assert adapter is identity_transform, (
        f"Mock→Panda 应走 identity_transform，实际 {adapter}"
    )
    print("[smoke] ✓ adapter 分派 = identity_transform（task→task 直通）")
    env.close()


def _check_pipeline_writes_experiment(config) -> None:
    """跑完整 pipeline（真实 PyBulletEnv direct 模式 + 打桩 LLM/agent），检查落盘。"""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    import pipeline.runner as R
    from agents.core import AgentResult, ToolCallRecord

    before = set(glob.glob(os.path.join(config.experiment.root, "*")))

    def fake_create_exact_agent(*a, **kw):
        return object()

    def fake_run_agent(agent, user_goal, **kw):
        return AgentResult(
            success=True,
            trajectory=[
                ToolCallRecord(tool_name="observe", args={}, result_text="obs1", step_index=1),
            ],
            final_answer="任务已完成",
            total_tool_calls=1,
        )

    with patch.object(R, "create_llm", return_value=FakeListChatModel(responses=["ok"])), \
         patch.object(R, "create_exact_agent", side_effect=fake_create_exact_agent), \
         patch.object(R, "run_agent", side_effect=fake_run_agent):
        result = R.run_pipeline(config, user_goal="把机械臂移到红色方块上方")

    after = set(glob.glob(os.path.join(config.experiment.root, "*")))
    new_dirs = sorted(after - before)
    assert new_dirs, "run_pipeline 后应有新的 data/experiment/ 目录"
    latest = new_dirs[-1]

    log_file = Path(latest) / "experiment.log"
    assert log_file.is_file(), f"最新实验目录缺少 experiment.log: {log_file}"
    content = log_file.read_text(encoding="utf-8")
    assert "Pipeline finished" in content, (
        f"experiment.log 应含 Pipeline finished，实际: {content[-500:]}"
    )
    assert result.env_closed is True
    print(f"[smoke] ✓ run_pipeline 完成，落盘 {latest}/experiment.log（含 Pipeline finished）")


def _check_exception_paths() -> None:
    """验证异常路径被显式暴露。"""
    try:
        build_robot(RobotConfig(type="ghost"))
        raise AssertionError("build_robot 未知 type 应抛 ValueError")
    except ValueError as e:
        print(f"[smoke] ✓ build_robot 未知 type 抛 ValueError: {e}")

    try:
        get_adapter(
            ActionSpec("joint", ("joint",) * 6),
            ActionSpec("task", ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")),
        )
        raise AssertionError("未注册组合应抛 AdapterNotFoundError")
    except AdapterNotFoundError as e:
        print(f"[smoke] ✓ 未注册组合抛 AdapterNotFoundError: {e}")


def main() -> int:
    config = load_config("configs/default.yaml")
    # 强制 direct 模式（防 GUI 弹窗）
    config.env.mode = "direct"

    _check_adapter_dispatch(config)
    _check_pipeline_writes_experiment(config)
    _check_exception_paths()

    print("\n=== PASS ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n=== FAIL: {type(e).__name__}: {e} ===")
        sys.exit(1)
