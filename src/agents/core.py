"""agents/core.py：Agent 相关 dataclass 定义。

历史遗留：本文件曾经包含 ExActAgent 与 SYSTEM_PROMPT（手写 ReAct Agent）。
Iteration 1 重构后，ExActAgent 已被 LangGraph 实现（agents.agent）取代。
本迭代（iter1-pipeline-refactor-config）删除 ExActAgent，仅保留 ToolCallRecord /
AgentResult 两个 dataclass，供 LangGraph 版本共享返回结构。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToolCallRecord:
    """单次工具调用记录。

    Attributes:
        tool_name: 被调用的工具名。
        args: 调用参数字典。
        result_text: 工具返回的字符串结果。
        step_index: 第几次工具调用（从 1 开始）。
    """
    tool_name: str
    args: dict
    result_text: str
    step_index: int


@dataclass
class AgentResult:
    """Agent.run 的返回值。

    Attributes:
        success: 任务是否完成（LLM 给出最终文本回答即视为成功）。
        trajectory: 工具调用轨迹列表。
        final_answer: LLM 最终文本回答。
        total_tool_calls: 实际工具调用次数。
    """
    success: bool
    trajectory: list[ToolCallRecord]
    final_answer: str
    total_tool_calls: int