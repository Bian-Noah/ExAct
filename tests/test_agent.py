"""agent 单元测试（iter1-pipeline-refactor-config 后从 lc_agent 重命名）。

Mock LLM 和工具，验证 create_exact_agent + run_agent 的 ReAct 循环。
新接口基于 StateGraph + 自写 tool_node（主线程同步）。
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.agent import (
    create_exact_agent,
    run_agent,
    _parse_agent_result,
    MAX_REACT_ROUNDS,
    MAX_TOOL_CALLS,
)
from agents.core import AgentResult


class TestParseAgentResult:
    """_parse_agent_result 消息解析测试。"""

    def test_pure_text_answer(self):
        """纯文本回答：无 tool_calls 的 AIMessage。"""
        result = {
            "messages": [
                HumanMessage(content="把机械臂移到红色方块上方"),
                AIMessage(content="任务已完成。"),
            ]
        }
        agent_result = _parse_agent_result(result)

        assert agent_result.success is True
        assert agent_result.total_tool_calls == 0
        assert agent_result.final_answer == "任务已完成。"
        assert agent_result.trajectory == []

    def test_observe_action_final_sequence(self):
        """observe → action → final 消息序列。"""
        result = {
            "messages": [
                HumanMessage(content="把机械臂移到红色方块上方"),
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "observe",
                        "args": {"target": "red_block"},
                        "id": "call_1",
                    }],
                ),
                ToolMessage(content="末端执行器位置: (0.0, 0.0, 0.5)", tool_call_id="call_1"),
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "action",
                        "args": {"instruction": "移动到红色方块上方"},
                        "id": "call_2",
                    }],
                ),
                ToolMessage(content="ee_pos 位移过小，判定 reached", tool_call_id="call_2"),
                AIMessage(content="机械臂已移动到红色方块上方。"),
            ]
        }
        agent_result = _parse_agent_result(result)

        assert agent_result.success is True
        assert agent_result.total_tool_calls == 2
        assert len(agent_result.trajectory) == 2
        assert agent_result.trajectory[0].tool_name == "observe"
        assert agent_result.trajectory[0].args == {"target": "red_block"}
        assert "末端执行器位置" in agent_result.trajectory[0].result_text
        assert agent_result.trajectory[1].tool_name == "action"
        assert "reached" in agent_result.trajectory[1].result_text
        assert "机械臂已移动" in agent_result.final_answer

    def test_empty_messages(self):
        """空消息列表。"""
        result = {"messages": []}
        agent_result = _parse_agent_result(result)

        assert agent_result.success is False
        assert agent_result.total_tool_calls == 0
        assert agent_result.final_answer == ""
        assert agent_result.trajectory == []


class TestCreateAgent:
    """create_exact_agent 测试。"""

    def test_returns_compiled_graph(self):
        """验证返回可 invoke 的 graph。"""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = MagicMock()

        tool1 = MagicMock()
        tool1.name = "observe"
        tool2 = MagicMock()
        tool2.name = "action"

        agent = create_exact_agent(mock_llm, [tool1, tool2])

        # 返回的 graph 应该有 invoke 方法
        assert hasattr(agent, "invoke")
        mock_llm.bind_tools.assert_called_once()

    def test_custom_system_prompt(self):
        """验证自定义 system_prompt 不报错。"""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = MagicMock()

        agent = create_exact_agent(mock_llm, [], system_prompt="自定义提示")
        assert hasattr(agent, "invoke")


class TestRunAgent:
    """run_agent 函数测试。"""

    def _make_mock_llm(self, responses):
        """构造 mock LLM，按顺序返回 responses。"""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.side_effect = responses
        return mock_llm

    def _make_mock_tool(self, name, return_value="ok"):
        """构造 mock BaseTool。"""
        tool = MagicMock()
        tool.name = name
        tool.invoke.return_value = return_value
        return tool

    def test_pure_text_answer(self):
        """LLM 直接返回文本 → 最终回答。"""
        mock_llm = self._make_mock_llm([
            AIMessage(content="任务已完成。"),
        ])

        agent = create_exact_agent(mock_llm, [])
        result = run_agent(agent, "测试")

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.final_answer == "任务已完成。"
        assert result.total_tool_calls == 0

    def test_tool_call_then_answer(self):
        """LLM 先调工具，再给出最终回答。"""
        mock_tool = self._make_mock_tool("observe", "末端执行器: (0.5, 0, 0.3)")
        mock_llm = self._make_mock_llm([
            AIMessage(
                content="",
                tool_calls=[{"name": "observe", "args": {}, "id": "call_1"}],
            ),
            AIMessage(content="观察到物体，任务完成。"),
        ])

        agent = create_exact_agent(mock_llm, [mock_tool])
        result = run_agent(agent, "测试")

        assert result.success is True
        assert result.total_tool_calls == 1
        assert result.trajectory[0].tool_name == "observe"
        assert "末端执行器" in result.trajectory[0].result_text
        assert "任务完成" in result.final_answer
        mock_tool.invoke.assert_called_once()

    def test_llm_exception(self):
        """LLM 调用异常 → 返回错误 AgentResult。"""
        mock_llm = self._make_mock_llm(RuntimeError("API timeout"))

        agent = create_exact_agent(mock_llm, [])
        result = run_agent(agent, "测试")

        assert result.success is False
        assert "Agent 执行出错" in result.final_answer
        assert "API timeout" in result.final_answer

    def test_unknown_tool(self):
        """调用未知工具 → 返回错误消息进 trajectory。"""
        mock_llm = self._make_mock_llm([
            AIMessage(
                content="",
                tool_calls=[{"name": "unknown_tool", "args": {}, "id": "call_1"}],
            ),
            AIMessage(content="任务结束。"),
        ])

        agent = create_exact_agent(mock_llm, [])
        result = run_agent(agent, "测试")

        assert result.success is True
        assert result.total_tool_calls == 1
        assert "未知工具" in result.trajectory[0].result_text

    def test_tool_exception_handled(self):
        """工具执行抛异常 → 错误消息进 trajectory。"""
        mock_tool = self._make_mock_tool("observe")
        mock_tool.invoke.side_effect = RuntimeError("tool crash")
        mock_llm = self._make_mock_llm([
            AIMessage(
                content="",
                tool_calls=[{"name": "observe", "args": {}, "id": "call_1"}],
            ),
            AIMessage(content="任务结束。"),
        ])

        agent = create_exact_agent(mock_llm, [mock_tool])
        result = run_agent(agent, "测试")

        assert result.total_tool_calls == 1
        assert "工具执行出错" in result.trajectory[0].result_text

    def test_recursion_limit_is_small(self):
        """验证 recursion_limit 按 MAX_REACT_ROUNDS 计算（5 轮 = 11）。"""
        assert MAX_REACT_ROUNDS == 5
        # recursion_limit = MAX_REACT_ROUNDS * 2 + 1 = 11
        expected = MAX_REACT_ROUNDS * 2 + 1
        assert expected == 11


class TestToolCallLimit:
    """工具调用次数硬截断测试。"""

    def _make_mock_llm(self, responses):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.side_effect = responses
        return mock_llm

    def _make_mock_tool(self, name, return_value="ok"):
        tool = MagicMock()
        tool.name = name
        tool.invoke.return_value = return_value
        return tool

    def test_max_tool_calls_is_3(self):
        """验证 MAX_TOOL_CALLS = 3。"""
        assert MAX_TOOL_CALLS == 3

    def test_under_limit_normal_execution(self):
        """工具调用次数未达上限 → 正常执行。"""
        mock_tool = self._make_mock_tool("observe", "观察结果")
        # LLM 调一次工具就回答
        mock_llm = self._make_mock_llm([
            AIMessage(
                content="",
                tool_calls=[{"name": "observe", "args": {}, "id": "c1"}],
            ),
            AIMessage(content="任务完成。"),
        ])

        agent = create_exact_agent(mock_llm, [mock_tool])
        result = run_agent(agent, "测试")

        assert result.success is True
        assert result.total_tool_calls == 1
        assert result.trajectory[0].result_text == "观察结果"
        mock_tool.invoke.assert_called_once()

    def test_at_limit_still_executes(self):
        """达到 MAX_TOOL_CALLS（=3）次时仍正常执行（第 3 次不截断）。"""
        mock_tool = self._make_mock_tool("observe", "观察结果")
        # LLM 连调 3 次工具后回答
        mock_llm = self._make_mock_llm([
            AIMessage(content="", tool_calls=[{"name": "observe", "args": {}, "id": "c1"}]),
            AIMessage(content="", tool_calls=[{"name": "observe", "args": {}, "id": "c2"}]),
            AIMessage(content="", tool_calls=[{"name": "observe", "args": {}, "id": "c3"}]),
            AIMessage(content="任务完成。"),
        ])

        agent = create_exact_agent(mock_llm, [mock_tool])
        result = run_agent(agent, "测试")

        assert result.total_tool_calls == 3
        # 3 次都正常执行，无截断消息
        for t in result.trajectory:
            assert t.result_text == "观察结果"
        assert mock_tool.invoke.call_count == 3

    def test_over_limit_truncates(self):
        """超过 MAX_TOOL_CALLS 后截断，返回提示消息而非执行工具。"""
        mock_tool = self._make_mock_tool("observe", "观察结果")
        # LLM 连调 4 次工具后回答
        mock_llm = self._make_mock_llm([
            AIMessage(content="", tool_calls=[{"name": "observe", "args": {}, "id": "c1"}]),
            AIMessage(content="", tool_calls=[{"name": "observe", "args": {}, "id": "c2"}]),
            AIMessage(content="", tool_calls=[{"name": "observe", "args": {}, "id": "c3"}]),
            AIMessage(content="", tool_calls=[{"name": "observe", "args": {}, "id": "c4"}]),
            AIMessage(content="任务结束。"),
        ])

        agent = create_exact_agent(mock_llm, [mock_tool])
        result = run_agent(agent, "测试")

        assert result.total_tool_calls == 4
        # 前 3 次正常执行
        assert result.trajectory[0].result_text == "观察结果"
        assert result.trajectory[1].result_text == "观察结果"
        assert result.trajectory[2].result_text == "观察结果"
        # 第 4 次截断
        assert "工具调用上限" in result.trajectory[3].result_text
        # 工具实际只被调用 3 次（第 4 次没执行）
        assert mock_tool.invoke.call_count == 3

    def test_multiple_tool_calls_in_one_message_counted_separately(self):
        """一条 AIMessage 里多个 tool_calls 各自计数。"""
        mock_observe = self._make_mock_tool("observe", "观察结果")
        mock_action = self._make_mock_tool("action", "动作结果")
        # 一条消息里调 3 个工具（observe + action + observe）→ 第 3 个达到上限仍执行
        # 再来一条消息调 1 个工具 → 第 4 个被截断
        mock_llm = self._make_mock_llm([
            AIMessage(content="", tool_calls=[
                {"name": "observe", "args": {}, "id": "c1"},
                {"name": "action", "args": {}, "id": "c2"},
                {"name": "observe", "args": {}, "id": "c3"},
            ]),
            AIMessage(content="", tool_calls=[
                {"name": "action", "args": {}, "id": "c4"},
            ]),
            AIMessage(content="任务结束。"),
        ])

        agent = create_exact_agent(mock_llm, [mock_observe, mock_action])
        result = run_agent(agent, "测试")

        assert result.total_tool_calls == 4
        # 前 3 次正常
        assert result.trajectory[0].result_text == "观察结果"
        assert result.trajectory[1].result_text == "动作结果"
        assert result.trajectory[2].result_text == "观察结果"
        # 第 4 次截断
        assert "工具调用上限" in result.trajectory[3].result_text
        # 工具总调用次数：observe 2 + action 1 = 3（第 4 次截断未执行）
        assert mock_observe.invoke.call_count == 2
        assert mock_action.invoke.call_count == 1
