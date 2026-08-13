"""EnvConfig 字段校验单元测试（iter2-renderer-env-mode）。

覆盖：
- 默认值（mode="direct"、renderer="auto"、use_gui=False）
- 合法值（mode="direct"|"gui"、renderer="auto"|"cpu"|"gpu"）
- 非法值（mode="headless"、renderer="vulkan"）抛 ValueError
- 字段缺失时回退默认值
- 旧字段 use_gui 向后兼容
"""

from __future__ import annotations

import pytest

from config.loader import EnvConfig, _from_dict


# ============================================================
# 默认值
# ============================================================

def test_envconfig_defaults():
    """EnvConfig() 默认值符合 iter2 规范。"""
    cfg = EnvConfig()
    assert cfg.mode == "direct"
    assert cfg.renderer == "auto"
    assert cfg.use_gui is False
    assert cfg.camera_resolution == (640, 480)


# ============================================================
# 合法值
# ============================================================

@pytest.mark.parametrize("mode", ["direct", "gui"])
def test_envconfig_mode_valid(mode: str):
    cfg = EnvConfig(mode=mode)
    assert cfg.mode == mode


@pytest.mark.parametrize("renderer", ["auto", "cpu", "gpu"])
def test_envconfig_renderer_valid(renderer: str):
    cfg = EnvConfig(renderer=renderer)
    assert cfg.renderer == renderer


def test_envconfig_all_legal_values():
    cfg = EnvConfig(mode="gui", renderer="cpu", camera_resolution=(320, 240))
    assert cfg.mode == "gui"
    assert cfg.renderer == "cpu"
    assert cfg.camera_resolution == (320, 240)


# ============================================================
# 非法值（仅 _from_dict 层校验，直接构造 EnvConfig 不校验非法值）
# ============================================================

@pytest.mark.parametrize(
    "bad_mode",
    [
        "headless",
        # 注：Literal 类型不运行时校验，直接构造 EnvConfig(mode=...) 接受任意字符串
    ],
)
def test_from_dict_mode_invalid_raises(bad_mode: str):
    """_from_dict 对非法 mode 抛 ValueError，错误信息包含字段名。"""
    with pytest.raises(ValueError) as exc_info:
        _from_dict({"mode": bad_mode}, EnvConfig)
    assert "EnvConfig.mode" in str(exc_info.value)
    assert bad_mode in str(exc_info.value)


@pytest.mark.parametrize(
    "bad_renderer",
    [
        "vulkan",
        "fast",
        "null",
    ],
)
def test_from_dict_renderer_invalid_raises(bad_renderer: str):
    """_from_dict 对非法 renderer 抛 ValueError，错误信息包含字段名。"""
    with pytest.raises(ValueError) as exc_info:
        _from_dict({"renderer": bad_renderer}, EnvConfig)
    assert "EnvConfig.renderer" in str(exc_info.value)
    assert bad_renderer in str(exc_info.value)


def test_from_dict_mode_non_string_raises():
    """mode 非字符串时抛 ValueError。"""
    with pytest.raises((ValueError, TypeError)):
        _from_dict({"mode": 123}, EnvConfig)


def test_from_dict_renderer_non_string_raises():
    """renderer 非字符串时抛 ValueError。"""
    with pytest.raises((ValueError, TypeError)):
        _from_dict({"renderer": None}, EnvConfig)


# ============================================================
# 字段缺失回退默认值
# ============================================================

def test_from_dict_empty_dict_uses_defaults():
    cfg = _from_dict({}, EnvConfig)
    assert cfg.mode == "direct"
    assert cfg.renderer == "auto"
    assert cfg.use_gui is False
    assert cfg.camera_resolution == (640, 480)


def test_from_dict_partial_dict_fills_remaining():
    """仅提供部分字段时，其余字段用默认值。"""
    cfg = _from_dict({"mode": "gui"}, EnvConfig)
    assert cfg.mode == "gui"
    assert cfg.renderer == "auto"
    assert cfg.use_gui is False


# ============================================================
# use_gui 字段向后兼容
# ============================================================

def test_envconfig_use_gui_field_exists():
    """use_gui 字段保留，构造时不报错。"""
    cfg = EnvConfig(use_gui=True)
    assert cfg.use_gui is True
    # mode/renderer 仍使用默认值
    assert cfg.mode == "direct"
    assert cfg.renderer == "auto"


def test_from_dict_accepts_legacy_use_gui():
    """旧 yaml 仍可使用 use_gui 字段。"""
    cfg = _from_dict({"use_gui": True}, EnvConfig)
    assert cfg.use_gui is True
