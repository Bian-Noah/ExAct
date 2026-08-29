"""experiment/utils.py 通用工具单元测试（iter12-video-recording 任务 1）。

覆盖：
- make_timestamp 格式
- ensure_dir 多层目录创建 + 已存在幂等
- validate_even_resolution 偶数通过 / 奇数抛 ValueError / 非正数抛 ValueError
- resolve_path 绝对路径原样 / 相对路径按 base 解析
- next_available_dir 同秒冲突 `_001` `_002` 递增
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from experiment.utils import (
    ensure_dir,
    make_timestamp,
    next_available_dir,
    resolve_path,
    validate_even_resolution,
)


# ============================================================================
# make_timestamp
# ============================================================================


def test_make_timestamp_format():
    """make_timestamp 应返回 `%Y%m%d_%H%M%S` 格式字符串。"""
    ts = make_timestamp()
    assert re.match(r"\d{8}_\d{6}$", ts), f"时间戳格式不对: {ts!r}"


# ============================================================================
# ensure_dir
# ============================================================================


def test_ensure_dir_creates_parents(tmp_path: Path):
    """ensure_dir 多层不存在目录自动创建，返回 Path。"""
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert result == target
    assert result.is_dir()
    assert (tmp_path / "a" / "b").is_dir()


def test_ensure_dir_existing_idempotent(tmp_path: Path):
    """ensure_dir 已存在目录幂等不报错。"""
    existing = tmp_path / "x"
    existing.mkdir()
    result = ensure_dir(existing)
    assert result == existing
    assert result.is_dir()


def test_ensure_dir_accepts_str(tmp_path: Path):
    """ensure_dir 接受字符串路径。"""
    result = ensure_dir(str(tmp_path / "str_dir"))
    assert result.is_dir()


# ============================================================================
# validate_even_resolution
# ============================================================================


def test_validate_even_resolution_ok():
    """偶数分辨率通过，不抛异常。"""
    validate_even_resolution(640, 480)  # 不应抛


def test_validate_even_resolution_odd():
    """奇数宽度抛 ValueError。"""
    with pytest.raises(ValueError):
        validate_even_resolution(641, 480)


def test_validate_even_resolution_odd_height():
    """奇数高度抛 ValueError。"""
    with pytest.raises(ValueError):
        validate_even_resolution(640, 481)


def test_validate_even_resolution_non_positive():
    """非正数抛 ValueError。"""
    with pytest.raises(ValueError):
        validate_even_resolution(0, 480)
    with pytest.raises(ValueError):
        validate_even_resolution(-2, 480)


# ============================================================================
# resolve_path
# ============================================================================


def test_resolve_path_absolute():
    """绝对路径原样返回。"""
    abs_path = "/tmp/abc"
    assert resolve_path(abs_path) == Path("/tmp/abc")


def test_resolve_path_relative_uses_cwd(tmp_path: Path, monkeypatch):
    """相对路径默认按 CWD 解析。"""
    monkeypatch.chdir(tmp_path)
    assert resolve_path("data/exp") == tmp_path / "data/exp"


def test_resolve_path_relative_with_base(tmp_path: Path):
    """相对路径按传入 base 解析。"""
    base = tmp_path / "proj"
    assert resolve_path("data/exp", base=base) == base / "data/exp"


# ============================================================================
# next_available_dir
# ============================================================================


def test_next_available_dir_first(tmp_path: Path):
    """目录不存在时直接返回 timestamp 目录。"""
    result = next_available_dir(tmp_path, "20260829_103000")
    assert result == tmp_path / "20260829_103000"


def test_next_available_dir_suffix(tmp_path: Path):
    """同名目录已存在时追加 `_001`。"""
    (tmp_path / "20260829_103000").mkdir()
    result = next_available_dir(tmp_path, "20260829_103000")
    assert result == tmp_path / "20260829_103000_001"


def test_next_available_dir_increments(tmp_path: Path):
    """连续冲突依次递增 `_001` `_002`。"""
    (tmp_path / "20260829_103000").mkdir()
    (tmp_path / "20260829_103000_001").mkdir()
    result = next_available_dir(tmp_path, "20260829_103000")
    assert result == tmp_path / "20260829_103000_002"


def test_next_available_dir_returns_uncreated(tmp_path: Path):
    """返回的目录路径不应已存在（由调用方 mkdir）。"""
    (tmp_path / "20260829_103000").mkdir()
    result = next_available_dir(tmp_path, "20260829_103000")
    assert not result.exists()
