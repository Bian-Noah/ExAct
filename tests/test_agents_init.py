"""agents 包公共导出验证测试。"""

from __future__ import annotations

import src.agents as agents_module
from src.agents import (
    MiniMaxClient,
    ExActAgent,
    ToolCallRecord,
    AgentResult,
    create_llm,
    create_exact_agent,
    run_agent,
)


def test_agents_all_exports_present():
    expected = {
        "MiniMaxClient",
        "ExActAgent",
        "ToolCallRecord",
        "AgentResult",
        "create_llm",
        "create_exact_agent",
        "run_agent",
    }
    assert set(agents_module.__all__) == expected
    for name in expected:
        assert hasattr(agents_module, name), f"src.agents 缺少导出：{name}"


def test_imported_classes_are_correct_types():
    from config.loader import LLMConfig

    # MiniMaxClient 可实例化
    client = MiniMaxClient(LLMConfig(api_key="sk-test"))
    assert client.config.api_key == "sk-test"

    # ToolCallRecord / AgentResult dataclass 可实例化
    record = ToolCallRecord(tool_name="x", args={}, result_text="r", step_index=1)
    assert record.tool_name == "x"

    result = AgentResult(success=True, trajectory=[], final_answer="", total_tool_calls=0)
    assert result.success is True

    # ExActAgent 需要 llm 和 tools，验证类可引用即可
    assert ExActAgent.__name__ == "ExActAgent"
