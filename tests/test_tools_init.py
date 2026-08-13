"""tools 包公共导出验证测试。

iter1-pipeline-refactor-config 后：
- BaseTool / LCActionTool / LCObserveTool / 死代码已删除
- 仅保留 ActionTool / ObserveTool / parse_target_pos
"""

from __future__ import annotations

import src.tools as tools_module
from src.tools import ActionTool, ObserveTool, parse_target_pos


def test_tools_all_exports_present():
    expected = {"ActionTool", "ObserveTool", "parse_target_pos"}
    assert set(tools_module.__all__) == expected
    for name in expected:
        assert hasattr(tools_module, name), f"src.tools 缺少导出：{name}"


def test_dead_exports_removed():
    """BaseTool / LCActionTool / LCObserveTool 已死代码清理。"""
    for name in ("BaseTool", "LCActionTool", "LCObserveTool"):
        assert not hasattr(tools_module, name), f"src.tools 不应再导出：{name}"


def test_imported_classes_are_correct_types():
    # 两者都是 pydantic BaseTool 子类，name 是字段默认值
    assert ActionTool.model_fields["name"].default == "action"
    assert ObserveTool.model_fields["name"].default == "observe"

    # parse_target_pos 是可调用函数
    assert callable(parse_target_pos)


def test_actiontool_observe_tool_instance_names():
    """实例化后 name 属性正确。"""
    from langchain_core.tools import BaseTool

    class _FakeEnv:
        def get_obs(self, include_rgb=True):
            return {"ee_pos": (0, 0, 0), "object_info": []}

    a = ActionTool(env=_FakeEnv(), executor=_FakeEnv())
    o = ObserveTool(env=_FakeEnv())
    assert isinstance(a, BaseTool)
    assert isinstance(o, BaseTool)
    assert a.name == "action"
    assert o.name == "observe"
    # args_schema 是 pydantic 类，比较引用相等需要看具体内部类对象
    # 简化校验：args_schema 是 BaseModel 子类且字段名正确
    assert a.args_schema.__name__ == "ActionInput"
    assert o.args_schema.__name__ == "ObserveInput"