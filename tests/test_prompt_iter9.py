"""iter9 系统提示词契约测试。

验证 DEFAULT_SYSTEM_PROMPT 包含：
- smolVLA 指令规范（动词开头/30 字符）
- 探索流程（read_notes / write_note）
- 既有 ATTRIBUTION_VALUES 与 parse_attribution 行为不破坏
"""

from __future__ import annotations

from agents.prompt import (
    ATTRIBUTION_VALUES,
    DEFAULT_SYSTEM_PROMPT,
    parse_attribution,
)


class TestIter9Contract:
    def test_contains_verb_prefix_rule(self):
        # iter9-extend：英文动词原形开头
        assert "动作动词原形开头" in DEFAULT_SYSTEM_PROMPT

    def test_contains_30_char_rule(self):
        assert "30 字符" in DEFAULT_SYSTEM_PROMPT

    def test_contains_english_only_rule(self):
        # iter9-extend：必须全英文
        assert "指令必须为英文" in DEFAULT_SYSTEM_PROMPT

    def test_contains_read_notes(self):
        assert "read_notes" in DEFAULT_SYSTEM_PROMPT

    def test_contains_write_note(self):
        assert "write_note" in DEFAULT_SYSTEM_PROMPT

    def test_contains_explore_flow_marker(self):
        assert "探索流程" in DEFAULT_SYSTEM_PROMPT

    def test_action_rule_explains_rejection(self):
        assert "拒绝" in DEFAULT_SYSTEM_PROMPT

    def test_contains_english_verb_examples(self):
        # 动词表英文示例
        for verb in ["pick", "grab", "place", "move", "turn"]:
            assert verb in DEFAULT_SYSTEM_PROMPT


class TestAttributionUnchanged:
    """既有失败归因契约不破坏。"""

    def test_attribution_values_intact(self):
        assert "自身指令问题" in ATTRIBUTION_VALUES
        assert "规划问题" in ATTRIBUTION_VALUES
        assert "工具问题" in ATTRIBUTION_VALUES
        assert "执行器问题" in ATTRIBUTION_VALUES

    def test_parse_attribution_basic(self):
        text = "任务失败。\n失败归因：自身指令问题, 工具问题"
        result = parse_attribution(text)
        assert "自身指令问题" in result
        assert "工具问题" in result
        assert "规划问题" not in result
        assert "执行器问题" not in result

    def test_parse_attribution_no_marker(self):
        assert parse_attribution("一切正常") == []

    def test_parse_attribution_empty(self):
        assert parse_attribution("") == []


class TestPromptMentionsExploreTool:
    def test_explore_tool_described(self):
        # explore 工具应在提示词中描述
        assert "explore" in DEFAULT_SYSTEM_PROMPT


# ========== iter9-extend：VLA 能力边界 + 探索习惯 ==========


class TestVLACapabilityBoundary:
    """提示词应明确告诉 LLM：VLA 不擅长坐标，应多用相对位置 + 探索。"""

    def test_mentions_coordinate_limitation(self):
        # 明确告诉 LLM VLA 不擅长精确坐标
        assert "坐标" in DEFAULT_SYSTEM_PROMPT

    def test_mentions_relative_position_recommendation(self):
        # 推荐用相对位置而不是精确坐标
        assert "相对位置" in DEFAULT_SYSTEM_PROMPT

    def test_mentions_vla_capability_scope(self):
        # 告诉 LLM VLA 能做什么（简单动作）
        assert "VLA 能力边界" in DEFAULT_SYSTEM_PROMPT or "能力边界" in DEFAULT_SYSTEM_PROMPT


class TestExplorationEncouragement:
    """不确定时应鼓励 LLM 用 explore.write_note 记录。"""

    def test_mentions_uncertainty_should_explore(self):
        # 遇到拿不准要先 write_note
        assert "不确定" in DEFAULT_SYSTEM_PROMPT

    def test_write_note_for_uncertainty(self):
        # 在不确定时显式提到 write_note
        # （提示词中出现多次 write_note，至少包含"对...先 write_note"模式）
        assert DEFAULT_SYSTEM_PROMPT.count("write_note") >= 2