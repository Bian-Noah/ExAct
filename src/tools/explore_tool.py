"""ExploreTool：探索笔记工具。

iter9-verify-explore-design 引入：单工具 + sub_action 参数派发，
避免挤占 max_tool_calls=3 的预算。

继承 langchain_core.tools.BaseTool，注入 Explore 实例。
"""

from __future__ import annotations

from typing import Any, Literal, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from explore import Explore


# 允许的子动作枚举（pydantic Literal 校验 + 运行时再校验兜底）
_SUB_ACTIONS = ("read_notes", "write_note")


class ExploreInput(BaseModel):
    """ExploreTool 输入参数。"""

    sub_action: Literal["read_notes", "write_note"] = Field(
        description="read_notes：读取已有笔记；write_note：写入新笔记"
    )
    note: str = Field(
        default="",
        description="write_note 时的笔记内容（单行，不含换行）",
    )


class ExploreTool(BaseTool):
    """探索笔记工具。先调用 read_notes 查阅既有经验，再调用 write_note 记录新发现。"""

    name: str = "explore"
    description: str = (
        "探索笔记工具：查阅与记录 LLM 在任务执行过程中的经验。"
        "建议工作流是先调用 read_notes 查阅既有笔记，再调用 write_note 记录新发现，"
        "便于跨实验复盘。"
    )
    args_schema: Type[BaseModel] = ExploreInput
    explore: Explore = Field(default=None)  # 实际注入由 runner 完成

    def __init__(self, explore: Explore, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # pydantic 实例化后会保留 explore 字段；这里再确保一下
        self.explore = explore

    def _run(self, sub_action: str, note: str = "") -> str:
        """分派到 explore.read() / explore.write(note)。

        Args:
            sub_action: "read_notes" | "write_note"
            note: write_note 时的笔记内容；read_notes 时忽略。

        Returns:
            结果文本（操作结果或错误消息）。
        """
        if sub_action == "read_notes":
            return self.explore.read()

        if sub_action == "write_note":
            if not note or not note.strip():
                return "错误：write_note 的 note 不能为空"
            if "\n" in note:
                return "错误：note 不能含换行（请使用 write_note 多次写入）"
            self.explore.write(note)
            return f"笔记已记录（当前共 {len(self.explore._notes)} 条）"

        return f"错误：未知 sub_action {sub_action!r}，期望 {list(_SUB_ACTIONS)}"