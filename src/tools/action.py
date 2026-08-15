"""ActionTool + target_pos 正则解析。

提供 parse_target_pos 函数和 ActionTool（langchain_core.tools.BaseTool 子类）。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


def parse_target_pos(instruction: str) -> Optional[tuple[float, float, float]]:
    """从自然语言指令中解析目标坐标。

    支持两种格式：
    1. "x=0.5, y=0.0, z=0.4" 或 "x: 0.5" （需至少 3 个坐标）
    2. "坐标 (0.5, 0.0, 0.3)" 或 "(0.5, 0, 0.4)" 或中文括号

    Args:
        instruction: 自然语言动作指令。

    Returns:
        (x, y, z) tuple 或 None（无法解析时）。
    """
    if not instruction:
        return None

    # 模式 1: x=0.5 / x: 0.5 / x =0.5 等
    pattern1 = re.compile(r'[xyz]\s*[=:]\s*([\-0-9.]+)', re.IGNORECASE)
    matches1 = pattern1.findall(instruction)
    if len(matches1) >= 3:
        try:
            coords = [float(m) for m in matches1[:3]]
            return tuple(coords)
        except ValueError:
            pass

    # 模式 2: (0.5, 0.0, 0.3) 或（0.5, 0.0, 0.3）
    pattern2 = re.compile(
        r'[\(（]([\-0-9.]+)\s*,\s*([\-0-9.]+)\s*,\s*([\-0-9.]+)[\)）]'
    )
    match2 = pattern2.search(instruction)
    if match2:
        try:
            return tuple(float(g) for g in match2.groups())
        except ValueError:
            pass

    return None


class ActionInput(BaseModel):
    """ActionTool 输入参数。"""

    instruction: str = Field(description="自然语言动作指令，例如 '移动到红色方块上方'")


class ActionTool(BaseTool):
    """对场景执行自然语言动作指令。

    内部调用 Executor.run_action，返回 ExecResult.message。
    会尝试从 instruction 中解析目标坐标传给 executor。
    """

    name: str = "action"
    description: str = (
        "对场景执行自然语言动作指令，"
        "例如『移动到红色方块上方』『夹取红色方块』。"
    )
    args_schema: Type[BaseModel] = ActionInput
    env: Any = None
    executor: Any = None
    adapter: Any = None  # 注入：get_adapter(...) 返回的转换函数；None 时 _ensure_adapter 兜底

    def _ensure_adapter(self, vla, env) -> Any:
        """确保 self.adapter 非 None。

        已注入（runner 正常接线）→ 直接返回；
        未注入 → 按 spec 自动构造并 print 警告（漏接兜底，但让问题可见）。
        """
        if self.adapter is not None:
            return self.adapter
        from utils.adapter import get_adapter
        self.adapter = get_adapter(vla.output_spec, env.input_spec)
        print(
            "[ActionTool] ⚠️ 未注入 adapter，已自动按 spec 构造并临时使用: "
            f"{vla.output_spec.space} → {env.input_spec.space}。"
            "建议在 runner 组装时显式接线。"
        )
        return self.adapter

    def _run(self, instruction: str) -> str:
        """执行动作指令。

        Args:
            instruction: 自然语言动作指令字符串。

        Returns:
            ExecResult.message 字符串（成功/超时/失败描述）。
        """
        _log = logging.getLogger("action")
        _log.info(f"action 调用开始 instruction={instruction}")
        if not instruction or not instruction.strip():
            return "错误：动作指令不能为空"

        # 尝试从指令中解析目标坐标
        target_pos = parse_target_pos(instruction)

        # ★ fallback 入口：确保 adapter 非 None（正常由 runner 注入，漏接自动构造）
        adapter = self._ensure_adapter(self.executor.vla, self.env)

        result = self.executor.run_action(
            self.env, instruction,
            done_criteria="reached",
            target_pos=target_pos,
            adapter=adapter,
        )
        _log.info(f"action 调用完成 success={result.success}")
        return result.message