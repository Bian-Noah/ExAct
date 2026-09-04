"""AgentConfig.extra_prompt 配置解析测试(config-extra-prompt)。"""

from __future__ import annotations

import os
import tempfile

from src.config.loader import AgentConfig, load_config


def _write_yaml(content: str) -> str:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name


def test_agent_config_extra_prompt_default_empty():
    """AgentConfig 默认 extra_prompt=""。"""
    cfg = AgentConfig()
    assert cfg.extra_prompt == ""


def test_load_config_extra_prompt_absent_uses_default():
    """yaml 不含 agent.extra_prompt → 默认空串。"""
    path = _write_yaml("agent:\n  max_react_rounds: 8\n  max_tool_calls: 6\n")
    try:
        cfg = load_config(path)
        assert cfg.agent.extra_prompt == ""
        assert cfg.agent.max_react_rounds == 8
        assert cfg.agent.max_tool_calls == 6
    finally:
        os.unlink(path)


def test_load_config_extra_prompt_multiline_block():
    """yaml 多行 | 块解析为带换行的 extra_prompt 字符串。"""
    path = _write_yaml(
        "agent:\n"
        "  extra_prompt: |\n"
        "    第一行规则\n"
        "    第二行规则\n"
    )
    try:
        cfg = load_config(path)
        assert cfg.agent.extra_prompt == "第一行规则\n第二行规则\n"
    finally:
        os.unlink(path)
