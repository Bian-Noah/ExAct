"""agents.prompt 单元测试:config-extra-prompt 额外提示词拼接。"""

from __future__ import annotations

from agents.prompt import (
    COMMON_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    build_system_prompt,
)


class TestBuildSystemPrompt:
    """build_system_prompt 拼接语义。"""

    def test_none_returns_default_full_text(self):
        """extra_prompt=None → 历史完整提示词(行为不变)。"""
        assert build_system_prompt(None) == DEFAULT_SYSTEM_PROMPT

    def test_empty_returns_default_full_text(self):
        """extra_prompt 为空/纯空白 → 历史完整提示词。"""
        assert build_system_prompt("") == DEFAULT_SYSTEM_PROMPT
        assert build_system_prompt("   \n  ") == DEFAULT_SYSTEM_PROMPT

    def test_non_empty_returns_common_plus_extra(self):
        """非空 extra_prompt → COMMON + 配置文案,文案替换默认的模型/策略说明。"""
        extra = "一次 action 只移动约 1~2 厘米,位移小是正常的。\n不要轻易调用 reset。"
        expected = COMMON_SYSTEM_PROMPT.rstrip("\n") + "\n\n" + extra.strip() + "\n"
        assert build_system_prompt(extra) == expected

    def test_extra_does_not_override_common_contract(self):
        """配置 extra_prompt 后,COMMON 中的报告契约仍保留(归因/复位/视角声明)。"""
        result = build_system_prompt("自定义规则")
        assert "失败归因：" in result
        assert "复位工具" in result
        assert "个视角" in result


class TestCommonSystemPrompt:
    """COMMON_SYSTEM_PROMPT:通用段净化范围。"""

    def test_common_keeps_contract_requirements(self):
        """COMMON 必须保留报告契约(parse_attribution 依赖)。"""
        assert "失败归因：" in COMMON_SYSTEM_PROMPT
        for value in ("自身指令问题", "规划问题", "工具问题", "执行器问题"):
            assert value in COMMON_SYSTEM_PROMPT
        assert "复位工具" in COMMON_SYSTEM_PROMPT
        assert "个视角" in COMMON_SYSTEM_PROMPT

    def test_common_removes_backend_model_specific_rules(self):
        """COMMON 不含应交给额外提示词段的模型/策略说明(不与配置文案冲突)。"""
        assert "关节极限" not in COMMON_SYSTEM_PROMPT   # reset 时机策略不残留
        assert "smolVLA 规范" not in COMMON_SYSTEM_PROMPT


class TestDefaultSystemPromptPreserved:
    """DEFAULT_SYSTEM_PROMPT 未被拆分改动(默认路径行为不变)。"""

    def test_default_still_contains_backend_model_specific_rules(self):
        """未配置 extra_prompt 时,历史完整提示词仍含模型/策略说明。"""
        assert "smolVLA" in DEFAULT_SYSTEM_PROMPT
        assert "关节极限" in DEFAULT_SYSTEM_PROMPT
        assert "operation='reset'" in DEFAULT_SYSTEM_PROMPT
