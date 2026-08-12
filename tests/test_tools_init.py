"""tools 包公共导出验证测试。"""

from __future__ import annotations

import src.tools as tools_module
from src.tools import (
    BaseTool,
    ObserveTool,
    ActionTool,
    LCObserveTool,
    LCActionTool,
    parse_target_pos,
)


def test_tools_all_exports_present():
    expected = {
        "BaseTool",
        "ObserveTool",
        "ActionTool",
        "LCObserveTool",
        "LCActionTool",
        "parse_target_pos",
    }
    assert set(tools_module.__all__) == expected
    for name in expected:
        assert hasattr(tools_module, name), f"src.tools 缺少导出：{name}"


def test_imported_classes_are_correct_types():
    # BaseTool 是 ABC，无法直接实例化
    import pytest
    with pytest.raises(TypeError):
        BaseTool()

    # 旧版 ObserveTool / ActionTool 是普通类，name 是类属性
    assert ObserveTool.name == "observe"
    assert ActionTool.name == "action"

    # LangChain 版 LCObserveTool / LCActionTool 是 pydantic BaseTool 子类，
    # name 是实例字段，类访问会 AttributeError → 用 model_fields 取默认值
    assert LCObserveTool.model_fields["name"].default == "observe"
    assert LCActionTool.model_fields["name"].default == "action"

    # parse_target_pos 是可调用函数
    assert callable(parse_target_pos)
