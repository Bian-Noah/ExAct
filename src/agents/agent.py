"""LangGraph 版 Agent：用 StateGraph + 自写 tool_node（主线程同步）。

不用 create_agent（其内部 ToolNode 用 ThreadPoolExecutor 执行工具，
与 PyBullet GUI 的线程约束冲突会段错误）。
改为 StateGraph 拼装 + 普通函数 tool_node，工具在主线程同步执行。
官方 Quickstart 推荐写法：https://docs.langchain.com/oss/python/langgraph/quickstart

iter1-pipeline-refactor-config 后：
- MAX_REACT_ROUNDS / MAX_TOOL_CALLS 作为模块级常量保留作默认值
- create_exact_agent 接收 max_react_rounds / max_tool_calls 参数（默认 5 / 3）
- run_agent 接受可选的 max_react_rounds 参数透传给 agent.invoke 的 recursion_limit
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from typing_extensions import TypedDict

from agents.core import AgentResult, ToolCallRecord
from agents.prompt import DEFAULT_SYSTEM_PROMPT

# ReAct 循环最大轮数（默认值，可被 create_exact_agent 参数覆盖）
# 每轮：LLM 决策 + 工具执行。
# 设为 5 以容纳 MAX_TOOL_CALLS=3 时的截断场景（最坏 4 次工具调用 + final answer = 9 次节点访问）。
MAX_REACT_ROUNDS = 5

# 工具调用次数硬上限默认值（防止 LLM 不收敛）
# 达到上限后强制返回提示消息，让 LLM 基于现有信息回答
MAX_TOOL_CALLS = 3

_log = logging.getLogger(__name__)


# 扩展 MessagesState：增加 tool_call_count 字段
class _AgentState(MessagesState):
    """Agent 状态：messages + 工具调用计数。"""
    tool_call_count: int


def create_exact_agent(
    llm: Any,
    tools: list,
    system_prompt: str = None,
    max_react_rounds: int = MAX_REACT_ROUNDS,
    max_tool_calls: int = MAX_TOOL_CALLS,
):
    """用 StateGraph 拼装 agent（主线程同步执行工具）。

    Args:
        llm: ChatOpenAI 实例（或其他 BaseChatModel）。
        tools: BaseTool 实例列表。
        system_prompt: 可选系统提示词。
        max_react_rounds: 超过此节点访问次数后中断（默认 5）。
        max_tool_calls: 工具调用累计上限（默认 3）。

    Returns:
        CompiledStateGraph，可通过 .invoke({"{"..."） 调用。
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: MessagesState) -> dict:
        """model node：调用 LLM 决策。主线程同步。"""
        messages = state["messages"]
        # 若首条不是 SystemMessage，插入 system_prompt
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def call_tools(state: _AgentState) -> dict:
        """tool node：同步执行工具。主线程，无线程池。

        官方 Quickstart 写法：普通函数循环调用 tool.invoke()。
        超过 max_tool_calls 后强制返回截断提示，让 LLM 给出 final answer。
        """
        messages = state["messages"]
        last_message: AIMessage = messages[-1]
        tool_calls = getattr(last_message, "tool_calls", None) or []

        # 工具调用计数（跨轮累计）
        count = state.get("tool_call_count", 0)

        results = []
        for tc in tool_calls:
            count += 1
            tc_id = tc.get("id", "")

            # 硬截断：超过上限后不再执行工具，返回提示
            if count > max_tool_calls:
                content = (
                    f"已达到工具调用上限({max_tool_calls})，"
                    "请基于现有观察和动作结果，用自然语言回答任务结果。"
                )
                results.append(ToolMessage(content=content, tool_call_id=tc_id))
                continue

            name = tc.get("name", "")
            args = tc.get("args", {})

            tool = tools_by_name.get(name)
            if tool is None:
                content = f"错误：未知工具 '{name}'"
            else:
                try:
                    # Iteration 5：透传 tool.invoke 返回值（str 或 list[dict]），不再 str(...) 强转
                    # 工具返回 list[dict] 时 ToolMessage.content 保持原 list 形态（多模态）
                    content = tool.invoke(args)
                except Exception as e:
                    # 工具异常：完整 traceback 走 logger.exception 落日志（含类型/堆栈），
                    # f-string 同时拼接 type(e).__name__ + str(e) 写入 ToolMessage.content，
                    # 保留原行为兼容性（上游测试断言 "boom" in content），且当 str(e) 为空
                    # 时仍能暴露类型名（如 RuntimeError / ValueError 等），便于 LLM 反馈。
                    _log.exception("工具执行出错：tool=%s args=%s", name, args)
                    content = f"工具执行出错: {type(e).__name__}: {e}"

            results.append(ToolMessage(content=content, tool_call_id=tc_id))

        return {"messages": results, "tool_call_count": count}

    def should_continue(state: MessagesState) -> Literal["tools", END]:
        """条件边：LLM 返回有 tool_calls 则路由到 tools，否则结束。"""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    # 拼装 graph（用扩展的 _AgentState，含 tool_call_count）
    builder = StateGraph(_AgentState)
    builder.add_node("model", call_model)
    builder.add_node("tools", call_tools)
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", should_continue, ["tools", END])
    builder.add_edge("tools", "model")  # 工具执行后回到 model

    return builder.compile()


def run_agent(
    agent: Any,
    user_goal: str,
    system_prompt: str = None,
    max_react_rounds: int = MAX_REACT_ROUNDS,
) -> AgentResult:
    """调用 LangGraph agent 并转换为 AgentResult。

    Args:
        agent: create_exact_agent 返回的 CompiledStateGraph。
        user_goal: 用户目标字符串。
        system_prompt: 未使用（system_prompt 已在 create_exact_agent 中设置）。
        max_react_rounds: agent.invoke 的 recursion_limit 上限（默认 MAX_REACT_ROUNDS）。

    Returns:
        AgentResult: 含 trajectory/final_answer/total_tool_calls。
    """
    _log = logging.getLogger("agent")

    messages: list[BaseMessage] = [HumanMessage(content=user_goal)]

    _log.info(f"agent.invoke 开始 (max_react_rounds={max_react_rounds})")
    try:
        result = agent.invoke(
            {"messages": messages},
            config={"recursion_limit": max_react_rounds * 2 + 1},
        )
    except Exception as e:
        _log.error(f"agent.invoke 失败: {e}")
        return AgentResult(
            success=False,
            trajectory=[],
            final_answer=f"Agent 执行出错: {e}",
            total_tool_calls=0,
        )

    return _parse_agent_result(result)


def _parse_agent_result(result: dict) -> AgentResult:
    """将 LangGraph agent.invoke 返回的 messages 解析为 AgentResult。

    Args:
        result: agent.invoke 返回的 dict，含 "messages" 键。

    Returns:
        AgentResult。
    """
    messages: list[BaseMessage] = result.get("messages", [])

    if not messages:
        return AgentResult(
            success=False,
            trajectory=[],
            final_answer="",
            total_tool_calls=0,
        )

    trajectory: list[ToolCallRecord] = []
    # tool_call_id → trajectory 索引，用于匹配 ToolMessage 到对应的 ToolCallRecord
    id_to_index: dict[str, int] = {}
    final_answer = ""
    tool_call_index = 0

    for msg in messages:
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                for tc in tool_calls:
                    tool_call_index += 1
                    tc_id = tc.get("id", "")
                    id_to_index[tc_id] = len(trajectory)
                    trajectory.append(ToolCallRecord(
                        tool_name=tc.get("name", "unknown"),
                        args=tc.get("args", {}),
                        result_text="",  # 结果在后续 ToolMessage 中
                        step_index=tool_call_index,
                    ))
            else:
                content = msg.content
                if isinstance(content, str) and content.strip():
                    final_answer = content

        elif isinstance(msg, ToolMessage):
            # 用 tool_call_id 精确匹配，避免一条消息多 tool_calls 时被覆盖
            tc_id = getattr(msg, "tool_call_id", "")
            idx = id_to_index.get(tc_id)
            if idx is not None:
                trajectory[idx].result_text = str(msg.content)

    # 无最终回答时取最后一条 AIMessage content
    if not final_answer:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = msg.content
                if isinstance(content, str) and content.strip():
                    final_answer = content
                break

    return AgentResult(
        success=bool(final_answer),
        trajectory=trajectory,
        final_answer=final_answer,
        total_tool_calls=len(trajectory),
    )