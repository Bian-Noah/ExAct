"""工具抽象基类。

定义所有 Agent 可调用工具的统一契约：
- 子类必须实现 _run(**kwargs) -> str
- 公共入口 run(**kwargs) 捕获异常并转为 ToolError 字符串，保证 Agent 主循环不崩溃
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """工具抽象基类。

    子类必须设置类属性 name 和 description，并实现 _run 方法。
    公共入口 run 内部调用 _run，捕获所有异常转为 ToolError 字符串。

    Attributes:
        name: 工具名（与 LLM tool schema 中的 function.name 对应）。
        description: 工具描述（供 LLM 理解工具用途）。
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def _run(self, **kwargs: Any) -> str:
        """子类实现具体工具逻辑。

        Args:
            **kwargs: 工具参数（由 LLM tool_calls 的 arguments JSON 解析而来）。

        Returns:
            工具执行结果字符串（将作为 tool role message 的 content 追加到对话）。
        """

    def run(self, **kwargs: Any) -> str:
        """公共入口：调用 _run，异常转 ToolError 字符串。

        保证无论子类实现如何，都不会抛异常给 Agent 主循环。

        Returns:
            成功时返回 _run 的结果字符串；
            异常时返回 f"ToolError[{name}]: {exception}"。
        """
        try:
            return self._run(**kwargs)
        except Exception as e:  # noqa: BLE001 — 工具层必须兜底
            tool_name = self.name or "unknown"
            return f"ToolError[{tool_name}]: {e}"
