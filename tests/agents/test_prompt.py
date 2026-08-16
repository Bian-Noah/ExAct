"""agents.prompt 单元测试：DEFAULT_SYSTEM_PROMPT 与 parse_attribution。"""

from __future__ import annotations

from agents.prompt import (
    ATTRIBUTION_VALUES,
    DEFAULT_SYSTEM_PROMPT,
    parse_attribution,
)


class TestDefaultSystemPrompt:
    """DEFAULT_SYSTEM_PROMPT 文本契约。"""

    def test_has_observe_tool_description(self):
        """提示词应包含 observe 工具说明（多模态视觉）。"""
        assert "observe 工具会返回" in DEFAULT_SYSTEM_PROMPT

    def test_has_image_declaration_requirements(self):
        """最终回答须声明是否看到图像（肯定/兜底前缀）。"""
        assert "✓ 本次执行看到了图像" in DEFAULT_SYSTEM_PROMPT
        assert "✗ 本次执行未能获取图像" in DEFAULT_SYSTEM_PROMPT

    def test_has_attribution_requirement(self):
        """失败时应输出归因行。"""
        assert "失败归因：" in DEFAULT_SYSTEM_PROMPT

    def test_has_all_attribution_values(self):
        """归因枚举全部在提示词中给出。"""
        for v in ATTRIBUTION_VALUES:
            assert v in DEFAULT_SYSTEM_PROMPT


class TestParseAttribution:
    """parse_attribution 归因提取。"""

    def test_empty_answer(self):
        """空回答 → 空列表。"""
        assert parse_attribution("") == []

    def test_no_attribution_marker(self):
        """成功回答（无失败归因行）→ 空列表。"""
        answer = "✓ 本次执行看到了图像。任务已完成。"
        assert parse_attribution(answer) == []

    def test_single_attribution(self):
        """单个归因 → 只返回该项。"""
        answer = "✓ 本次执行看到了图像。任务失败。\n失败归因：规划问题"
        assert parse_attribution(answer) == ["规划问题"]

    def test_multiple_attributions(self):
        """多归因逗号分隔 → 按枚举顺序返回全部命中项。"""
        answer = "✗ 本次执行未能获取图像。任务失败。\n失败归因：执行器问题, 工具问题"
        assert parse_attribution(answer) == ["工具问题", "执行器问题"]

    def test_partial_match_only(self):
        """只命中部分枚举值 → 只返回命中项（不误报）。"""
        answer = "任务失败。\n失败归因：工具问题"
        assert parse_attribution(answer) == ["工具问题"]
