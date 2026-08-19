"""Explore 类单元测试。

覆盖：enabled=False no-op、write/read 拼接、flush 落盘、flush 后清空、
重复 flush 防重入、父目录自动创建、日期格式、跨次 flush 累加。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from explore import Explore


# ========== enabled=False 路径 ==========


class TestDisabled:
    def test_disabled_write_noop(self, tmp_path: Path):
        e = Explore(tmp_path, enabled=False)
        e.write("x")
        assert e._notes == []

    def test_disabled_read_returns_notice(self, tmp_path: Path):
        e = Explore(tmp_path, enabled=False)
        assert e.read() == Explore.EMPTY_NOTICE

    def test_disabled_flush_creates_nothing(self, tmp_path: Path):
        e = Explore(tmp_path, enabled=False)
        e.write("x")
        e.flush()
        assert not list(tmp_path.iterdir())


# ========== write / read 基础 ==========


class TestWriteRead:
    def test_multiple_write_concat(self, tmp_path: Path):
        e = Explore(tmp_path)
        e.write("a")
        e.write("b")
        assert e.read() == "a\nb"

    def test_empty_read_returns_notice(self, tmp_path: Path):
        e = Explore(tmp_path)
        assert e.read() == Explore.EMPTY_NOTICE

    def test_single_write(self, tmp_path: Path):
        e = Explore(tmp_path)
        e.write("hello")
        assert e.read() == "hello"


# ========== flush 落盘 ==========


class TestFlush:
    def test_basic_flush(self, tmp_path: Path):
        e = Explore(tmp_path)
        e.write("x")
        e.flush()
        log_path = tmp_path / f"{date.today().isoformat()}.log"
        assert log_path.exists()
        assert log_path.read_text(encoding="utf-8") == "x\n"

    def test_flush_clears_memory(self, tmp_path: Path):
        e = Explore(tmp_path)
        e.write("x")
        e.flush()
        assert e.read() == Explore.EMPTY_NOTICE
        assert e._notes == []

    def test_flush_idempotent(self, tmp_path: Path):
        """重复 flush 不会重复落盘。"""
        e = Explore(tmp_path)
        e.write("x")
        e.flush()
        e.flush()
        log_path = tmp_path / f"{date.today().isoformat()}.log"
        assert log_path.read_text(encoding="utf-8") == "x\n"

    def test_flush_creates_parent_dirs(self, tmp_path: Path):
        root = tmp_path / "a" / "b" / "c"
        e = Explore(root)
        e.write("x")
        e.flush()
        assert root.is_dir()

    def test_flush_empty_noop(self, tmp_path: Path):
        """内存为空时 flush 不创建文件。"""
        e = Explore(tmp_path)
        e.flush()
        assert not list(tmp_path.iterdir())

    def test_flush_appends_to_existing_file(self, tmp_path: Path):
        """两次 flush 累加（同日期追加）。"""
        e1 = Explore(tmp_path)
        e1.write("a")
        e1.write("b")
        e1.flush()
        e2 = Explore(tmp_path)
        e2.write("c")
        e2.flush()
        log_path = tmp_path / f"{date.today().isoformat()}.log"
        assert log_path.read_text(encoding="utf-8") == "a\nb\nc\n"

    def test_flush_filename_uses_date_today(self, tmp_path: Path, monkeypatch):
        """monkeypatch date.today 后文件名为该日期。"""
        fake_date = date(2026, 1, 15)
        monkeypatch.setattr("explore.explore.date", type(date))
        # 替换 module 内的 date.today
        from explore import explore as explore_mod
        monkeypatch.setattr(explore_mod, "date", type("D", (), {"today": staticmethod(lambda: fake_date)}))

        e = Explore(tmp_path)
        e.write("x")
        e.flush()
        assert (tmp_path / "2026-01-15.log").exists()


# ========== 路径解析 ==========


class TestPathResolve:
    def test_absolute_path_kept(self, tmp_path: Path):
        e = Explore(tmp_path)
        assert e.root == tmp_path

    def test_relative_path_resolved(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        e = Explore("myexplore")
        assert e.root.is_absolute()
        assert e.root.name == "myexplore"


# ========== 多实例隔离 ==========


class TestMultipleInstances:
    def test_separate_notes(self, tmp_path: Path):
        e1 = Explore(tmp_path)
        e2 = Explore(tmp_path)
        e1.write("from-1")
        e2.write("from-2")
        assert e1.read() == "from-1"
        assert e2.read() == "from-2"