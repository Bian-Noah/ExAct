"""agents 包公共导出验证测试。

iter1-pipeline-refactor-config 后：
- MiniMaxClient / ExActAgent / SYSTEM_PROMPT 已删除
- 仅保留 AgentResult / ToolCallRecord / create_llm / create_exact_agent / run_agent
"""

from __future__ import annotations

import src.agents as agents_module
from src.agents import (
    AgentResult,
    ToolCallRecord,
    create_exact_agent,
    create_llm,
    run_agent,
)


def test_agents_all_exports_present():
    expected = {
        "AgentResult",
        "ToolCallRecord",
        "create_llm",
        "create_exact_agent",
        "run_agent",
    }
    assert set(agents_module.__all__) == expected
    for name in expected:
        assert hasattr(agents_module, name), f"src.agents 缺少导出：{name}"


def test_dead_exports_removed():
    """MiniMaxClient / ExActAgent / SYSTEM_PROMPT 死代码已清理。"""
    for name in ("MiniMaxClient", "ExActAgent", "SYSTEM_PROMPT"):
        assert not hasattr(agents_module, name), f"src.agents 不应再导出：{name}"


def test_dataclasses_preserved():
    """ToolCallRecord / AgentResult 保留。"""
    record = ToolCallRecord(tool_name="x", args={}, result_text="r", step_index=1)
    assert record.tool_name == "x"

    result = AgentResult(success=True, trajectory=[], final_answer="", total_tool_calls=0)
    assert result.success is True
    assert result.final_answer == ""


def test_create_llm_is_callable():
    assert callable(create_llm)


def test_create_exact_agent_is_callable():
    assert callable(create_exact_agent)


def test_run_agent_is_callable():
    assert callable(run_agent)