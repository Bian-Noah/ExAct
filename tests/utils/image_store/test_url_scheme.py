"""Unit tests for src.utils.image_store.url_scheme."""

from __future__ import annotations

import pytest

from src.utils.image_store.url_scheme import (
    build_url,
    is_valid_url,
    parse_url,
)


# ---------- build_url 正常路径 ----------

class TestBuildUrlNormal:
    def test_simple_observation(self):
        assert build_url("observations", "001.png") == "img://observations/001.png"

    def test_action_category(self):
        assert build_url("actions", "frame_42.png") == "img://actions/frame_42.png"

    def test_dash_in_filename(self):
        assert build_url("obs", "2026-08-13_001.png") == "img://obs/2026-08-13_001.png"

    def test_dotted_filename(self):
        # filename pattern allows dots between alphanumeric/dash/underscore
        assert build_url("obs", "snapshot.v2.png") == "img://obs/snapshot.v2.png"

    def test_single_char_category_and_filename(self):
        assert build_url("a", "b.png") == "img://a/b.png"


# ---------- build_url 异常路径 ----------

class TestBuildUrlInvalid:
    def test_empty_category(self):
        with pytest.raises(ValueError, match="[Cc]ategory"):
            build_url("", "001.png")

    def test_empty_filename(self):
        with pytest.raises(ValueError, match="[Ff]ilename"):
            build_url("obs", "")

    def test_category_with_space(self):
        with pytest.raises(ValueError, match="[Cc]ategory"):
            build_url("obs category", "001.png")

    def test_category_with_slash(self):
        with pytest.raises(ValueError, match="[Cc]ategory"):
            build_url("obs/cat", "001.png")

    def test_category_with_colon(self):
        with pytest.raises(ValueError, match="[Cc]ategory"):
            build_url("obs:cat", "001.png")

    def test_chinese_category(self):
        with pytest.raises(ValueError, match="[Cc]ategory"):
            build_url("观测", "001.png")

    def test_filename_without_png_suffix(self):
        with pytest.raises(ValueError, match="[Ff]ilename"):
            build_url("obs", "001.jpg")

    def test_filename_with_space(self):
        with pytest.raises(ValueError, match="[Ff]ilename"):
            build_url("obs", "001 copy.png")


# ---------- parse_url 正常路径 ----------

class TestParseUrlNormal:
    def test_roundtrip_simple(self):
        assert parse_url("img://observations/001.png") == ("observations", "001.png")

    def test_roundtrip_with_dashes(self):
        url = "img://actions/2026-08-13_frame.png"
        assert parse_url(url) == ("actions", "2026-08-13_frame.png")

    def test_roundtrip_with_dots(self):
        url = "img://obs/snapshot.v2.png"
        assert parse_url(url) == ("obs", "snapshot.v2.png")


# ---------- parse_url 异常路径 ----------

class TestParseUrlInvalid:
    def test_empty_string(self):
        with pytest.raises(ValueError):
            parse_url("")

    def test_non_string(self):
        with pytest.raises(ValueError):
            parse_url(None)  # type: ignore[arg-type]

    def test_wrong_scheme(self):
        with pytest.raises(ValueError, match="scheme|start"):
            parse_url("http://obs/001.png")

    def test_no_scheme(self):
        with pytest.raises(ValueError):
            parse_url("obs/001.png")

    def test_missing_filename(self):
        with pytest.raises(ValueError):
            parse_url("img://observations/")

    def test_missing_category(self):
        with pytest.raises(ValueError):
            parse_url("img:///001.png")

    def test_invalid_category(self):
        with pytest.raises(ValueError):
            parse_url("img://bad category/001.png")

    def test_invalid_filename(self):
        with pytest.raises(ValueError):
            parse_url("img://obs/001.jpg")


# ---------- is_valid_url ----------

class TestIsValidUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "img://observations/001.png",
            "img://actions/frame_42.png",
            "img://obs/2026-08-13_001.png",
            "img://a/b.png",
        ],
    )
    def test_valid_urls(self, url):
        assert is_valid_url(url) is True
        # consistency: every URL is_valid_url says is valid can also be parsed
        cat, fn = parse_url(url)
        assert isinstance(cat, str)
        assert isinstance(fn, str)

    @pytest.mark.parametrize(
        "url",
        [
            "",
            "img://obs/",
            "img:///001.png",
            "img://obs/001.jpg",
            "http://obs/001.png",
            "obs/001.png",
            "img://bad category/001.png",
            None,
        ],
    )
    def test_invalid_urls(self, url):
        assert is_valid_url(url) is False  # type: ignore[arg-type]


# ---------- build_url ⇄ parse_url 一致性 ----------

class TestBuildParseConsistency:
    @pytest.mark.parametrize(
        "category,filename",
        [
            ("observations", "001.png"),
            ("actions", "frame_42.png"),
            ("explorations", "2026-08-13_001.png"),
            ("a", "b.png"),
            ("cat_with_underscore", "fn-with-dash.png"),
        ],
    )
    def test_build_then_parse(self, category, filename):
        url = build_url(category, filename)
        assert parse_url(url) == (category, filename)

    def test_is_valid_url_after_build(self):
        url = build_url("obs", "001.png")
        assert is_valid_url(url) is True
