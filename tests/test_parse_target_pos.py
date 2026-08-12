"""parse_target_pos 正则解析函数的单元测试。"""

import pytest

from tools.lc_action import parse_target_pos


class TestParseTargetPos:
    """parse_target_pos 各种格式测试。"""

    def test_xyz_equals_format(self):
        """x=0.5, y=0.0, z=0.4 格式。"""
        result = parse_target_pos("x=0.5, y=0.0, z=0.4")
        assert result == (0.5, 0.0, 0.4)

    def test_xyz_colon_format(self):
        """x: 0.5, y: 0, z: 0.4 格式。"""
        result = parse_target_pos("x: 0.5, y: 0, z: 0.4")
        assert result == (0.5, 0.0, 0.4)

    def test_parentheses_format(self):
        """括号格式 (0.5, 0.0, 0.3)。"""
        result = parse_target_pos("坐标 (0.5, 0.0, 0.3)")
        assert result == (0.5, 0.0, 0.3)

    def test_chinese_parentheses(self):
        """中文括号格式。"""
        result = parse_target_pos("移动到（0.5, 0.0, 0.3）")
        assert result == (0.5, 0.0, 0.3)

    def test_no_coordinates_returns_none(self):
        """无坐标返回 None。"""
        result = parse_target_pos("移动到红色方块上方")
        assert result is None

    def test_only_two_coords_returns_none(self):
        """只有 2 个坐标返回 None。"""
        result = parse_target_pos("x=0.5, y=0.0")
        assert result is None

    def test_negative_numbers(self):
        """负数坐标。"""
        result = parse_target_pos("x=-0.1, y=0.2, z=-0.3")
        assert result == (-0.1, 0.2, -0.3)

    def test_empty_string_returns_none(self):
        """空字符串返回 None。"""
        assert parse_target_pos("") is None

    def test_none_returns_none(self):
        """None 返回 None。"""
        assert parse_target_pos(None) is None
