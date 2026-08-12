"""功能场景 B：Agent 端到端 Mock 链路测试。

组合真实组件：
- FakeEnv（内存 env，含 get_obs/step）
- 真实 MockVLA
- 真实 Executor
- 真实 ObserveTool + ActionTool
- FakeLLM（顺序返回 observe 调用 → action 调用 → 文本回答）

验证 Agent 端到端能跑通 observe → action → final_answer 闭环。
"""

from __future__ import annotations

import numpy as np

from env.base import BaseEnv
from executor import Executor, MockVLA
from agents.core import ExActAgent
from tools.observe import ObserveTool
from tools.action import ActionTool


# ========== FakeEnv（真实 obs 契约） ==========


class FakeEnvForE2E(BaseEnv):
    """端到端测试用 FakeEnv，含真实 object_info 和 ee_pos。"""

    def __init__(self):
        self._ee_pos = (0.0, 0.0, 0.5)
        self._step_count = 0

    def reset(self, task_spec=None, seed=0):
        self._ee_pos = (0.0, 0.0, 0.5)
        self._step_count = 0
        return self._make_obs()

    def step(self, action):
        self._step_count += 1
        # 模拟机械臂移动（前 2 步位移，之后停止触发 reached）
        if self._step_count < 3:
            self._ee_pos = (
                self._ee_pos[0] + 0.05,
                self._ee_pos[1],
                self._ee_pos[2],
            )
        return self._make_obs(), 0.0, False, {}

    def render(self):
        return np.zeros((10, 10, 3), dtype=np.uint8)

    def get_obs(self, include_rgb: bool = True):
        return self._make_obs()

    def close(self):
        pass

    def _make_obs(self):
        return {
            "rgb": np.zeros((10, 10, 3), dtype=np.uint8),
            "object_info": [
                {"id": 1, "name": "red_block", "pos": [0.5, 0, 0.1], "quat": [0, 0, 0, 1]},
            ],
            "ee_pos": self._ee_pos,
            "state_desc": f"step={self._step_count}",
        }


# ========== FakeLLM ==========


class FakeLLMForE2E:
    """按预设顺序返回响应的假 LLM。"""

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.call_count = 0

    def chat(self, messages, tools=None):
        if self.call_count >= len(self.responses):
            self.call_count += 1
            return {"role": "assistant", "content": "兜底", "tool_calls": [], "raw": {}}
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp


def _make_tool_call(name: str, arguments: str = "{}") -> dict:
    return {
        "id": f"call_{name}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


# ========== 端到端测试 ==========


def test_agent_e2e_observe_action_final_answer():
    """功能场景 B：真实 ObserveTool + ActionTool + Executor + MockVLA + FakeEnv + FakeLLM。"""
    # 真实组件
    env = FakeEnvForE2E()
    env.reset()
    vla = MockVLA(seed=0)
    executor = Executor(vla, max_steps=5)
    observe_tool = ObserveTool(env)
    action_tool = ActionTool(env, executor)

    # FakeLLM 顺序：1=observe 调用 → 2=action 调用 → 3=文本回答
    fake_llm = FakeLLMForE2E([
        {"role": "assistant", "content": None, "tool_calls": [_make_tool_call("observe")], "raw": {}},
        {"role": "assistant", "content": None, "tool_calls": [_make_tool_call("action", "{\"instruction\": \"移动到红色方块上方\"}")], "raw": {}},
        {"role": "assistant", "content": "任务已完成：机械臂已移动到红色方块上方", "tool_calls": [], "raw": {}},
    ])

    agent = ExActAgent(fake_llm, [observe_tool, action_tool])
    result = agent.run("把机械臂移到红色方块上方")

    # 断言
    assert result.success is True
    assert result.total_tool_calls == 2
    assert [t.tool_name for t in result.trajectory] == ["observe", "action"]
    assert result.final_answer == "任务已完成：机械臂已移动到红色方块上方"

    # observe 结果应包含真实 env 的物体信息
    observe_result = result.trajectory[0].result_text
    assert "red_block" in observe_result
    assert "末端执行器位置" in observe_result

    # action 结果应包含 executor 的 message（reached 或步数相关）
    action_result = result.trajectory[1].result_text
    assert isinstance(action_result, str) and len(action_result) > 0


def test_agent_e2e_observe_with_target_filter():
    """带 target 参数的 observe 调用端到端验证。"""
    env = FakeEnvForE2E()
    env.reset()
    vla = MockVLA(seed=0)
    executor = Executor(vla, max_steps=3)
    observe_tool = ObserveTool(env)
    action_tool = ActionTool(env, executor)

    fake_llm = FakeLLMForE2E([
        # observe 带 target 参数
        {"role": "assistant", "content": None, "tool_calls": [_make_tool_call("observe", "{\"target\": \"red_block\"}")], "raw": {}},
        # 直接回答
        {"role": "assistant", "content": "看到红色方块", "tool_calls": [], "raw": {}},
    ])

    agent = ExActAgent(fake_llm, [observe_tool, action_tool])
    result = agent.run("看看红色方块在哪")

    assert result.success is True
    assert result.total_tool_calls == 1
    assert result.trajectory[0].tool_name == "observe"
    assert result.trajectory[0].args == {"target": "red_block"}
    assert "red_block" in result.trajectory[0].result_text
