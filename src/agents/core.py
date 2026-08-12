"""ExActAgent：手写简化 ReAct 主循环。

不依赖 LangChain，完全掌控 LLM ↔ Tool 交互流程。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agents.llm_client import MiniMaxClient
from tools.base import BaseTool


SYSTEM_PROMPT = (
    "你是 ExActAgent，一个具身智能助手。你可以调用以下工具来感知和操作环境：\n"
    "- observe(target?: str): 观察当前场景，返回物体列表和末端执行器位置。\n"
    "- action(instruction: str): 对场景执行自然语言动作指令。\n"
    "请根据用户目标，先观察场景，再执行动作，最后用自然语言回答任务结果。\n"
    "如果任务已完成或无法继续，请直接用文本回复（不调用工具）。"
)


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


class ExActAgent:
    """手写简化 ReAct Agent。

    通过 MiniMaxClient 调用 LLM，根据 LLM 返回的 tool_calls 调度对应工具执行，
    将工具结果追加到对话，循环直到 LLM 给出最终文本回答或达到工具调用上限。
    """

    def __init__(self, llm: MiniMaxClient, tools: list[BaseTool]):
        """初始化 Agent。

        Args:
            llm: MiniMaxClient 实例（或 duck typing 对象，含 chat 方法）。
            tools: BaseTool 实例列表。
        """
        self.llm = llm
        self.tools = tools
        self.tool_map: dict[str, BaseTool] = {t.name: t for t in tools}
        self.tool_schemas: list[dict] = [
            self._build_tool_schema(t) for t in tools
        ]

    def _build_tool_schema(self, tool: BaseTool) -> dict:
        """根据工具名构造 OpenAI function tool schema。

        Args:
            tool: BaseTool 实例。

        Returns:
            OpenAI tool schema dict，如
            {"type": "function", "function": {"name": ..., "description": ...,
             "parameters": {"type": "object", "properties": {...}, "required": [...]}}}
        """
        if tool.name == "observe":
            parameters = {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "可选，过滤特定物体名（大小写不敏感）。",
                    },
                },
                "required": [],
            }
        elif tool.name == "action":
            parameters = {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "自然语言动作指令，如『移动到红色方块上方』。",
                    },
                },
                "required": ["instruction"],
            }
        else:
            # 通用兜底：空 properties
            parameters = {
                "type": "object",
                "properties": {},
                "required": [],
            }

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            },
        }

    def run(self, user_goal: str, max_tool_calls: int = 30) -> AgentResult:
        """运行 ReAct 主循环。

        Args:
            user_goal: 用户目标字符串。
            max_tool_calls: 最大工具调用次数，防止无限循环。

        Returns:
            AgentResult dataclass。
        """
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_goal},
        ]
        trajectory: list[ToolCallRecord] = []
        tool_call_count = 0
        success = False
        final_answer = ""

        while tool_call_count < max_tool_calls:
            resp = self.llm.chat(messages, tools=self.tool_schemas)

            # 构造 assistant 消息并追加到对话
            assistant_msg: dict[str, Any] = {"role": resp["role"]}
            if resp["content"]:
                assistant_msg["content"] = resp["content"]
            if resp.get("tool_calls"):
                assistant_msg["tool_calls"] = resp["tool_calls"]
            messages.append(assistant_msg)

            tool_calls = resp.get("tool_calls") or []

            if tool_calls:
                # 有工具调用，逐个执行
                for tc in tool_calls:
                    tc_id = tc.get("id", "")
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    args_str = func.get("arguments", "{}")

                    # 解析 arguments JSON
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except (json.JSONDecodeError, TypeError):
                        args = {"raw": args_str}
                    if not isinstance(args, dict):
                        args = {"raw": args_str}

                    # 调度工具
                    tool = self.tool_map.get(name)
                    if tool is None:
                        result = f"错误：未知工具 '{name}'"
                    else:
                        result = tool.run(**args) if isinstance(args, dict) else f"错误：工具参数非法 {args}"

                    tool_call_count += 1
                    trajectory.append(ToolCallRecord(
                        tool_name=name,
                        args=args if isinstance(args, dict) else {},
                        result_text=result,
                        step_index=tool_call_count,
                    ))

                    # 追加 tool role 消息
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result,
                    })

                    if tool_call_count >= max_tool_calls:
                        break
                # 继续下一轮 LLM 调用
                continue
            else:
                # 无工具调用，LLM 给出最终文本回答
                final_answer = resp["content"] or ""
                success = bool(final_answer)
                break

        if tool_call_count >= max_tool_calls and not final_answer:
            final_answer = f"已达工具调用上限 {max_tool_calls} 次，任务未完成"
            success = False

        return AgentResult(
            success=success,
            trajectory=trajectory,
            final_answer=final_answer,
            total_tool_calls=tool_call_count,
        )
