"""ExActAgent 单元测试。

使用 FakeLLM（按预设顺序返回响应）+ FakeTool 验证 ReAct 主循环：
observe → action → final_answer、max_tool_calls 超限、未知工具、
tool 异常捕获、arguments 非法 JSON 等场景。
"""

from __future__ import annotations

import pytest

from agents.core import ExActAgent, ToolCallRecord, AgentResult, SYSTEM_PROMPT
from tools.base import BaseTool


# ========== FakeLLM ==========


class FakeLLM:
    """按预设顺序返回响应的假 LLM。

    responses: list[dict]，每个 dict 是 chat() 的返回值。
    """

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.call_count = 0
        self.received_messages_list: list[list[dict]] = []
        self.received_tools_list: list[list[dict]] = []

    def chat(self, messages, tools=None):
        if self.call_count >= len(self.responses):
            # 兜底：返回文本回答避免死循环
            self.call_count += 1
            return {"role": "assistant", "content": "兜底回答", "tool_calls": [], "raw": {}}
        resp = self.responses[self.call_count]
        self.call_count += 1
        self.received_messages_list.append(list(messages))
        self.received_tools_list.append(list(tools) if tools else [])
        return resp


# ========== FakeTool ==========


class FakeTool(BaseTool):
    """可配置 name 和 _run 返回值的假工具。"""

    def __init__(self, name: str, description: str = "", return_value: str = "ok"):
        self.name = name
        self.description = description or f"fake tool {name}"
        self._return_value = return_value
        self.call_args_list: list[dict] = []

    def _run(self, **kwargs):
        self.call_args_list.append(kwargs)
        if isinstance(self._return_value, Exception):
            raise self._return_value
        return self._return_value


# ========== 1. dataclass 字段 ==========


def test_tool_call_record_fields():
    record = ToolCallRecord(
        tool_name="observe",
        args={"target": "red"},
        result_text="物体列表",
        step_index=1,
    )
    assert record.tool_name == "observe"
    assert record.args == {"target": "red"}
    assert record.result_text == "物体列表"
    assert record.step_index == 1


def test_agent_result_fields():
    result = AgentResult(
        success=True,
        trajectory=[],
        final_answer="完成",
        total_tool_calls=0,
    )
    assert result.success is True
    assert result.trajectory == []
    assert result.final_answer == "完成"
    assert result.total_tool_calls == 0


# ========== 2. _build_tool_schema ==========


def test_build_tool_schema_observe():
    agent = ExActAgent(FakeLLM([]), [])
    tool = FakeTool(name="observe", description="观察场景")
    schema = agent._build_tool_schema(tool)

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "observe"
    assert schema["function"]["description"] == "观察场景"
    assert "target" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["required"] == []


def test_build_tool_schema_action():
    agent = ExActAgent(FakeLLM([]), [])
    tool = FakeTool(name="action", description="执行动作")
    schema = agent._build_tool_schema(tool)

    assert schema["function"]["name"] == "action"
    assert "instruction" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["required"] == ["instruction"]


def test_build_tool_schema_unknown_name_fallback():
    agent = ExActAgent(FakeLLM([]), [])
    tool = FakeTool(name="custom_tool", description="自定义")
    schema = agent._build_tool_schema(tool)
    assert schema["function"]["name"] == "custom_tool"
    assert schema["function"]["parameters"]["properties"] == {}


# ========== 3. 主循环 observe → action → final_answer ==========


def _make_tool_call(name: str, arguments: str = "{}") -> dict:
    return {
        "id": f"call_{name}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def test_main_loop_observe_then_action_then_final():
    observe_tool = FakeTool(name="observe", return_value="物体列表: red_block")
    action_tool = FakeTool(name="action", return_value="动作成功")

    fake_llm = FakeLLM([
        # 第 1 轮：调用 observe
        {"role": "assistant", "content": None, "tool_calls": [_make_tool_call("observe", "{\"target\": null}")], "raw": {}},
        # 第 2 轮：调用 action
        {"role": "assistant", "content": None, "tool_calls": [_make_tool_call("action", "{\"instruction\": \"走\"}")], "raw": {}},
        # 第 3 轮：文本回答
        {"role": "assistant", "content": "完成", "tool_calls": [], "raw": {}},
    ])

    agent = ExActAgent(fake_llm, [observe_tool, action_tool])
    result = agent.run("目标")

    assert result.success is True
    assert result.final_answer == "完成"
    assert result.total_tool_calls == 2
    assert [t.tool_name for t in result.trajectory] == ["observe", "action"]
    assert result.trajectory[0].result_text == "物体列表: red_block"
    assert result.trajectory[1].result_text == "动作成功"
    # FakeLLM 被 call 3 次
    assert fake_llm.call_count == 3
    # observe_tool 收到 target=None
    assert observe_tool.call_args_list[0] == {"target": None}
    # action_tool 收到 instruction="走"
    assert action_tool.call_args_list[0] == {"instruction": "走"}


def test_tools_schema_passed_to_llm():
    observe_tool = FakeTool(name="observe")
    fake_llm = FakeLLM([
        {"role": "assistant", "content": "直接回答", "tool_calls": [], "raw": {}},
    ])
    agent = ExActAgent(fake_llm, [observe_tool])
    agent.run("目标")

    # 第一次 chat 调用应传入 tool_schemas
    assert len(fake_llm.received_tools_list[0]) == 1
    assert fake_llm.received_tools_list[0][0]["function"]["name"] == "observe"


# ========== 4. max_tool_calls 超限退出 ==========


def test_max_tool_calls_exceeded():
    observe_tool = FakeTool(name="observe", return_value="ok")
    # 每次都返回 tool_calls，永不给文本回答
    fake_llm = FakeLLM([
        {"role": "assistant", "content": None, "tool_calls": [_make_tool_call("observe")], "raw": {}},
        {"role": "assistant", "content": None, "tool_calls": [_make_tool_call("observe")], "raw": {}},
        {"role": "assistant", "content": None, "tool_calls": [_make_tool_call("observe")], "raw": {}},
    ])

    agent = ExActAgent(fake_llm, [observe_tool])
    result = agent.run("目标", max_tool_calls=3)

    assert result.success is False
    assert result.total_tool_calls == 3
    assert "已达工具调用上限 3 次" in result.final_answer


# ========== 5. 未知工具名 ==========


def test_unknown_tool_name():
    fake_llm = FakeLLM([
        {"role": "assistant", "content": None, "tool_calls": [_make_tool_call("unknown_tool")], "raw": {}},
        {"role": "assistant", "content": "结束", "tool_calls": [], "raw": {}},
    ])
    agent = ExActAgent(fake_llm, [FakeTool(name="observe")])
    result = agent.run("目标")

    assert result.total_tool_calls == 1
    assert "未知工具 'unknown_tool'" in result.trajectory[0].result_text
    # 主循环不抛异常
    assert result.success is True


# ========== 6. tool 异常被 BaseTool.run 捕获 ==========


def test_tool_exception_caught_by_base_run():
    observe_tool = FakeTool(name="observe", return_value=RuntimeError("tool boom"))
    fake_llm = FakeLLM([
        {"role": "assistant", "content": None, "tool_calls": [_make_tool_call("observe")], "raw": {}},
        {"role": "assistant", "content": "结束", "tool_calls": [], "raw": {}},
    ])
    agent = ExActAgent(fake_llm, [observe_tool])
    result = agent.run("目标")

    assert result.total_tool_calls == 1
    assert "ToolError[observe]" in result.trajectory[0].result_text
    assert "tool boom" in result.trajectory[0].result_text


# ========== 7. arguments 非法 JSON ==========


def test_invalid_json_arguments():
    observe_tool = FakeTool(name="observe", return_value="ok")
    fake_llm = FakeLLM([
        # arguments 不是合法 JSON
        {"role": "assistant", "content": None, "tool_calls": [_make_tool_call("observe", "not json")], "raw": {}},
        {"role": "assistant", "content": "结束", "tool_calls": [], "raw": {}},
    ])
    agent = ExActAgent(fake_llm, [observe_tool])
    result = agent.run("目标")

    # 不抛异常，args 是 dict 含 raw 键
    assert result.total_tool_calls == 1
    assert result.trajectory[0].args == {"raw": "not json"}


def test_arguments_already_dict():
    observe_tool = FakeTool(name="observe", return_value="ok")
    fake_llm = FakeLLM([
        # arguments 直接传 dict（非字符串）
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "observe", "arguments": {"target": "red"}},
        }], "raw": {}},
        {"role": "assistant", "content": "结束", "tool_calls": [], "raw": {}},
    ])
    agent = ExActAgent(fake_llm, [observe_tool])
    result = agent.run("目标")

    assert result.trajectory[0].args == {"target": "red"}
    assert observe_tool.call_args_list[0] == {"target": "red"}


# ========== 8. LLM 直接回答（无工具调用） ==========


def test_llm_direct_answer_without_tools():
    fake_llm = FakeLLM([
        {"role": "assistant", "content": "直接回答，无需工具", "tool_calls": [], "raw": {}},
    ])
    agent = ExActAgent(fake_llm, [FakeTool(name="observe")])
    result = agent.run("简单问题")

    assert result.success is True
    assert result.final_answer == "直接回答，无需工具"
    assert result.total_tool_calls == 0
    assert result.trajectory == []


def test_llm_empty_content_no_tool_calls():
    """LLM 返回 content=None 且无 tool_calls → success=False。"""
    fake_llm = FakeLLM([
        {"role": "assistant", "content": None, "tool_calls": [], "raw": {}},
    ])
    agent = ExActAgent(fake_llm, [FakeTool(name="observe")])
    result = agent.run("目标")

    assert result.success is False
    assert result.total_tool_calls == 0
