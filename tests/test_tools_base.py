"""BaseTool 抽象基类单元测试。"""

from __future__ import annotations

import pytest

from tools.base import BaseTool


def test_base_tool_cannot_be_instantiated():
    """BaseTool 是 ABC，直接实例化抛 TypeError。"""
    with pytest.raises(TypeError, match="abstract class"):
        BaseTool()


class _EchoTool(BaseTool):
    name = "echo"
    description = "echo back kwargs"

    def _run(self, **kwargs):
        return f"echo: {kwargs}"


def test_run_forwards_kwargs_and_returns_string():
    tool = _EchoTool()
    result = tool.run(x=1, y="a")
    assert result == "echo: {'x': 1, 'y': 'a'}"
    # 直接调用 _run 也能工作
    assert tool._run(a=2) == "echo: {'a': 2}"


class _RaiseTool(BaseTool):
    name = "raise"
    description = "always raises"

    def _run(self, **kwargs):
        raise RuntimeError("boom")


def test_run_catches_exception_and_returns_tool_error_string():
    tool = _RaiseTool()
    result = tool.run()
    assert isinstance(result, str)
    assert "ToolError[raise]" in result
    assert "boom" in result


class _NoNameTool(BaseTool):
    description = "no name set"

    def _run(self, **kwargs):
        raise ValueError("oops")


def test_run_uses_unknown_when_name_empty():
    tool = _NoNameTool()
    result = tool.run()
    assert "ToolError[unknown]" in result
    assert "oops" in result
