"""ActionTool：统一动作执行工具。

供 Agent 调用以通过 Executor 驱动 VLA 在 env 中执行自然语言动作指令。
"""

from __future__ import annotations

from typing import Any

from tools.base import BaseTool


class ActionTool(BaseTool):
    """对场景执行自然语言动作指令。

    例如『移动到红色方块上方』『夹取红色方块』。
    内部调用 Executor.run_action，返回 ExecResult.message。
    """

    name = "action"
    description = (
        "对场景执行自然语言动作指令，"
        "例如『移动到红色方块上方』『夹取红色方块』。"
    )

    def __init__(self, env: Any, executor: Any):
        """初始化。

        Args:
            env: 环境对象（duck typing，含 step/get_obs 方法）。
            executor: Executor 实例（含 run_action 方法）。
        """
        self.env = env
        self.executor = executor

    def _run(self, instruction: str) -> str:
        """执行动作指令。

        Args:
            instruction: 自然语言动作指令字符串。

        Returns:
            ExecResult.message 字符串（成功/超时/失败描述）。
        """
        if not instruction or not instruction.strip():
            return "错误：动作指令不能为空"

        result = self.executor.run_action(
            self.env, instruction, done_criteria="reached"
        )
        return result.message
