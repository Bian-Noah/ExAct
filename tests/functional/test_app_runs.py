"""功能场景 H：app.py 端到端跑通。

组合 monkeypatch：
- pipeline.runner.PyBulletPandaEnv → FakeEnv
- pipeline.runner.create_llm → FakeListLLM
- pipeline.runner.create_exact_agent → 返回 mock agent（避免 bind_tools 调用）
- pipeline.runner.run_agent → 返回 AgentResult

验证 app.main() 能完整跑完（pipeline 入口 + 打印结果 + 退出码 0）。
"""

from __future__ import annotations

import os
from io import StringIO
from unittest.mock import MagicMock, patch

from agents.core import AgentResult, ToolCallRecord
from langchain_core.language_models.fake import FakeListLLM


class _FakeEnv:
    def __init__(self, env_config=None, robot_config=None):
        self.use_gui = env_config.use_gui if env_config else False
        self.close_called = False

    def reset(self, task_spec=None, seed=0):
        return {"ee_pos": (0, 0, 0.5), "object_info": [], "state_desc": "fake", "rgb": None}

    def step(self, action):
        return self._obs(), 0.0, False, {}

    def render(self):
        return None

    def get_obs(self, include_rgb=True):
        return self._obs()

    def _obs(self):
        return {"ee_pos": (0, 0, 0.5), "object_info": [], "state_desc": "fake", "rgb": None}

    def close(self):
        self.close_called = True


def test_app_py_runs_end_to_end():
    """验证 app.main() 能跑通：加载配置 → 调用 pipeline → 打印结果。"""
    captured = {}

    def fake_create_exact_agent(*a, **kw):
        captured["create_exact_agent_called_with"] = (a, kw)
        return MagicMock()

    def fake_run_agent(agent, user_goal, **kw):
        captured["run_agent_user_goal"] = user_goal
        return AgentResult(
            success=True,
            trajectory=[
                ToolCallRecord(tool_name="observe", args={}, result_text="obs1", step_index=1),
            ],
            final_answer="任务已完成",
            total_tool_calls=1,
        )

    import pipeline.runner as R
    with patch.object(R, "PyBulletPandaEnv", _FakeEnv), \
         patch.object(R, "create_llm", return_value=FakeListLLM(responses=["ok"])), \
         patch.object(R, "create_exact_agent", side_effect=fake_create_exact_agent), \
         patch.object(R, "run_agent", side_effect=fake_run_agent):
        import app
        # 抑制 logger 输出
        app.logger.handlers = [logging.StreamHandler(StringIO())]  # noqa: F821
        app.main()

    args, kwargs = captured["create_exact_agent_called_with"]
    assert "max_react_rounds" in kwargs
    assert "max_tool_calls" in kwargs
    assert captured["run_agent_user_goal"] == "把机械臂移到红色方块上方"


def test_app_py_line_count_under_30():
    """app.py 总行数 < 30（按 wc -l 计算）。"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    app_path = os.path.join(project_root, "src", "app.py")
    with open(app_path, encoding="utf-8") as f:
        content = f.read()
    # wc -l 计的是 newline 数量；splitlines 计的是行数（含末尾无换行）
    wc_line_count = content.count("\n")
    assert wc_line_count < 30, f"app.py wc 行数 {wc_line_count} >= 30"


def test_app_py_calls_run_pipeline():
    """app.py 必须调用 run_pipeline。"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    app_path = os.path.join(project_root, "src", "app.py")
    content = open(app_path, encoding="utf-8").read()
    assert "run_pipeline" in content


def test_app_py_uses_load_config():
    """app.py 必须调用 load_config。"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    app_path = os.path.join(project_root, "src", "app.py")
    content = open(app_path, encoding="utf-8").read()
    assert "load_config" in content
    assert "from config import" in content


# 添加 logging import 用于测试中的 logger 抑制
import logging  # noqa: E402