"""ExploreTool 单元测试。

覆盖：sub_action 分派、note 校验、工具元数据。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from explore import Explore
from tools.explore_tool import ExploreInput, ExploreTool


# ========== 子动作派发 ==========


class TestSubActionDispatch:
    def test_write_note(self, tmp_path: Path):
        e = Explore(tmp_path)
        tool = ExploreTool(explore=e)
        result = tool._run("write_note", "夹取红方块 → 合规")
        assert "笔记已记录" in result
        assert "共 1 条" in result
        assert e._notes == ["夹取红方块 → 合规"]

    def test_read_notes_empty(self, tmp_path: Path):
        e = Explore(tmp_path)
        tool = ExploreTool(explore=e)
        result = tool._run("read_notes")
        assert result == Explore.EMPTY_NOTICE

    def test_read_notes_with_content(self, tmp_path: Path):
        e = Explore(tmp_path)
        e.write("a")
        e.write("b")
        tool = ExploreTool(explore=e)
        result = tool._run("read_notes")
        assert result == "a\nb"

    def test_unknown_sub_action(self, tmp_path: Path):
        e = Explore(tmp_path)
        tool = ExploreTool(explore=e)
        result = tool._run("unknown_action")
        assert "未知 sub_action" in result


# ========== write_note 校验 ==========


class TestWriteNoteValidation:
    def test_empty_note(self, tmp_path: Path):
        e = Explore(tmp_path)
        tool = ExploreTool(explore=e)
        result = tool._run("write_note", "")
        assert "不能为空" in result
        assert e._notes == []

    def test_whitespace_note(self, tmp_path: Path):
        e = Explore(tmp_path)
        tool = ExploreTool(explore=e)
        result = tool._run("write_note", "   ")
        assert "不能为空" in result
        assert e._notes == []

    def test_note_with_newline(self, tmp_path: Path):
        e = Explore(tmp_path)
        tool = ExploreTool(explore=e)
        result = tool._run("write_note", "line1\nline2")
        assert "不能含换行" in result
        assert e._notes == []


# ========== 工具元数据 ==========


class TestToolMetadata:
    def test_name(self, tmp_path: Path):
        tool = ExploreTool(explore=Explore(tmp_path))
        assert tool.name == "explore"

    def test_description_mentions_read_write(self, tmp_path: Path):
        tool = ExploreTool(explore=Explore(tmp_path))
        assert "先调用 read_notes" in tool.description or "先" in tool.description

    def test_args_schema_is_input(self, tmp_path: Path):
        tool = ExploreTool(explore=Explore(tmp_path))
        assert tool.args_schema is ExploreInput


# ========== 输入 schema 校验 ==========


class TestInputSchema:
    def test_valid_input(self):
        # pydantic Literal 校验
        inp = ExploreInput(sub_action="read_notes")
        assert inp.sub_action == "read_notes"

    def test_invalid_sub_action(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            ExploreInput(sub_action="unknown")