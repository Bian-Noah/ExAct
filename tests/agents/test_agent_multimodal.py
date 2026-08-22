"""Agent.call_tools 透传 list content 单元测试（Iteration 5）。

覆盖（按 tasks.md 2.3.1~2.3.6）：
- 2.3.1 test_call_tools_passes_list_content_through
- 2.3.2 test_call_tools_passes_str_content_unchanged
- 2.3.3 test_call_tools_catches_tool_exception
- 2.3.4 test_call_tools_truncates_at_max_tool_calls
- 2.3.5 test_call_tools_handles_unknown_tool
- 2.3.6 test_default_system_prompt_no_longer_says_fallback

约定：create_exact_agent 编译后无法直接调内部 call_tools 节点；
改用 monkeypatch 替换 llm_with_tools + 直接构造 AIMessage + tool_calls
驱动一次 agent.invoke 跑通；然后从 messages 中找 ToolMessage 断言。
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from agents.agent import (
    DEFAULT_SYSTEM_PROMPT,
    create_exact_agent,
)


# ============================================================================
# Mock 工具
# ============================================================================


class _MultimodalTool(BaseTool):
    """测试 mock 工具：_run() 返回 list[dict]（多模态 content blocks）。"""

    name: str = "multimodal_tool"
    description: str = "返回 list[dict] 多模态内容"

    def _run(self) -> list[dict]:
        return [
            {"type": "text", "text": "这是文本"},
            {"type": "image", "url": "img://observations/abc.png"},
        ]


class _StrTool(BaseTool):
    """测试 mock 工具：_run() 返回 str（兼容旧路径）。"""

    name: str = "str_tool"
    description: str = "返回纯文本"

    def _run(self) -> str:
        return "纯文本结果"


class _BoomTool(BaseTool):
    """测试 mock 工具：_run() 抛 RuntimeError。"""

    name: str = "boom_tool"
    description: str = "抛异常"

    def _run(self) -> str:
        raise RuntimeError("boom")


# ============================================================================
# Helpers
# ============================================================================


def _stub_llm_with_tool_call(llm: MagicMock, tool_name: str, tool_id: str = "tc-1") -> MagicMock:
    """让 stub llm.bind_tools(...).invoke(messages) 返回一条带 tool_calls 的 AIMessage。"""
    ai = AIMessage(
        content="",
        tool_calls=[{"id": tool_id, "name": tool_name, "args": {}}],
    )
    llm_with_tools = MagicMock()
    llm_with_tools.invoke = MagicMock(return_value=ai)
    llm.bind_tools = MagicMock(return_value=llm_with_tools)
    return llm_with_tools


def _stub_llm_with_final_answer(llm: MagicMock, text: str = "完成") -> None:
    """让 stub llm.bind_tools(...).invoke(messages) 第二次返回纯文本 AIMessage。"""
    final = AIMessage(content=text)
    llm_with_tools = MagicMock()
    # 第一次：tool_calls；第二次：final
    llm_with_tools.invoke = MagicMock(side_effect=[AIMessage(
        content="", tool_calls=[{"id": "tc-1", "name": "str_tool", "args": {}}],
    ), final])
    llm.bind_tools = MagicMock(return_value=llm_with_tools)


def _last_tool_message(messages):
    """从 messages 列表中找最后一条 ToolMessage。"""
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            return m
    raise AssertionError("messages 中未找到 ToolMessage")


# ============================================================================
# 2.3.1: list content 透传
# ============================================================================


def test_call_tools_passes_list_content_through():
    """mock 工具返回 list[dict] → ToolMessage.content 是 list[dict]，非 str。"""
    llm = MagicMock()
    _stub_llm_with_tool_call(llm, "multimodal_tool")

    agent = create_exact_agent(llm, [_MultimodalTool()])
    result = agent.invoke({"messages": [HumanMessage(content="hi")]})

    tm = _last_tool_message(result["messages"])
    assert isinstance(tm.content, list), f"期望 list[dict]，得到 {type(tm.content).__name__}"
    assert len(tm.content) == 2
    assert tm.content[0] == {"type": "text", "text": "这是文本"}
    assert tm.content[1] == {"type": "image", "url": "img://observations/abc.png"}


# ============================================================================
# 2.3.2: str content 兼容
# ============================================================================


def test_call_tools_passes_str_content_unchanged():
    """mock 工具返回 str → ToolMessage.content 是 str。"""
    llm = MagicMock()
    _stub_llm_with_tool_call(llm, "str_tool")

    agent = create_exact_agent(llm, [_StrTool()])
    result = agent.invoke({"messages": [HumanMessage(content="hi")]})

    tm = _last_tool_message(result["messages"])
    assert isinstance(tm.content, str)
    assert tm.content == "纯文本结果"


# ============================================================================
# 2.3.3: 工具抛异常
# ============================================================================


def test_call_tools_catches_tool_exception():
    """mock 工具抛 RuntimeError → ToolMessage.content 是 str 错误提示。"""
    llm = MagicMock()
    _stub_llm_with_tool_call(llm, "boom_tool")

    agent = create_exact_agent(llm, [_BoomTool()])
    result = agent.invoke({"messages": [HumanMessage(content="hi")]})

    tm = _last_tool_message(result["messages"])
    assert isinstance(tm.content, str)
    assert "boom" in tm.content
    assert "工具执行出错" in tm.content


# ============================================================================
# 2.3.4: max_tool_calls 截断
# ============================================================================


def test_call_tools_truncates_at_max_tool_calls():
    """max_tool_calls=1：第 2 个 tool_call（来自同一 AIMessage）被截断为 str 提示。

    简化策略：构造一个 AIMessage 含 2 个 tool_calls（同一 AIMessage 内多个 tc）
    + max_tool_calls=1。call_tools 循环里第 2 个会被截断。
    """
    from langchain_core.messages import ToolMessage

    llm = MagicMock()
    llm_with_tools = MagicMock()
    # AIMessage 含 2 个 tool_calls：观察这 2 个会不会截断
    ai = AIMessage(
        content="",
        tool_calls=[
            {"id": "tc-1", "name": "str_tool", "args": {}},
            {"id": "tc-2", "name": "str_tool", "args": {}},
        ],
    )
    llm_with_tools.invoke = MagicMock(return_value=ai)
    llm.bind_tools = MagicMock(return_value=llm_with_tools)

    agent = create_exact_agent(llm, [_StrTool()], max_tool_calls=1)
    result = agent.invoke({"messages": [HumanMessage(content="hi")]})

    # 找到所有 ToolMessage，检查断点
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) >= 2, f"期望 ≥ 2 个 ToolMessage，得到 {len(tool_msgs)}"

    # 第一个 ToolMessage（正常执行）
    assert tool_msgs[0].content == "纯文本结果"
    # 第二个 ToolMessage（截断提示）
    assert isinstance(tool_msgs[1].content, str)
    assert "工具调用上限" in tool_msgs[1].content


# ============================================================================
# 2.3.5: 未知工具
# ============================================================================


def test_call_tools_handles_unknown_tool():
    """tool_calls 含不存在的工具名 → ToolMessage.content 是 str 错误提示。"""
    llm = MagicMock()
    _stub_llm_with_tool_call(llm, "unknown_tool_xyz")

    agent = create_exact_agent(llm, [_StrTool()])  # 工具列表不含 unknown_tool_xyz
    result = agent.invoke({"messages": [HumanMessage(content="hi")]})

    tm = _last_tool_message(result["messages"])
    assert isinstance(tm.content, str)
    assert "未知工具" in tm.content
    assert "unknown_tool_xyz" in tm.content


# ============================================================================
# 2.3.6: DEFAULT_SYSTEM_PROMPT 文本契约
# ============================================================================


def test_default_system_prompt_text_contract():
    """DEFAULT_SYSTEM_PROMPT 文本契约：

    - 包含 'observe 工具会返回'（明确告知会返回图像）
    - 包含「本次执行看到了 N 个视角的图像」的肯定前缀（强制智能体声明是否看到图）
    - 包含「本次执行未能获取图像」的兜底前缀
    - 包含「失败归因：」标记（失败时输出归因行）
    - 不包含旧版 '如果当前系统暂未提供图像能力'（已替换）
    """
    assert "observe 工具会返回" in DEFAULT_SYSTEM_PROMPT
    # iter11-reset-multicam:措辞升级为"N 个视角",前缀变为"✓ 本次执行看到了"
    assert "✓ 本次执行看到了" in DEFAULT_SYSTEM_PROMPT
    assert "✗ 本次执行" in DEFAULT_SYSTEM_PROMPT
    assert "失败归因：" in DEFAULT_SYSTEM_PROMPT
    assert "如果当前系统暂未提供图像能力" not in DEFAULT_SYSTEM_PROMPT
