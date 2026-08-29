"""VideoConfig 数据类单元测试（iter12-video-recording 任务 2）。

覆盖：
- 默认值全字段
- from_dict 全字段解析
- from_dict 缺失字段用默认
- resolution list → tuple 转换
- 奇数 resolution 抛 ValueError
- 空 dict 全默认
"""

from __future__ import annotations

import pytest

from experiment.video.config import VideoConfig


# ============================================================================
# 默认值
# ============================================================================


def test_video_config_defaults():
    """VideoConfig() 默认值全字段正确。"""
    cfg = VideoConfig()
    assert cfg.enabled is True
    assert cfg.filename == "demo.mp4"
    assert cfg.fps == 15
    assert cfg.resolution == (640, 480)
    assert cfg.camera is None
    assert cfg.capture_every_n_steps == 1


# ============================================================================
# from_dict
# ============================================================================


def test_from_dict_full():
    """全字段 dict 正确解析。"""
    d = {
        "enabled": False,
        "filename": "x.mp4",
        "fps": 30,
        "resolution": [320, 240],
        "camera": "cam1",
        "capture_every_n_steps": 2,
    }
    cfg = VideoConfig.from_dict(d)
    assert cfg.enabled is False
    assert cfg.filename == "x.mp4"
    assert cfg.fps == 30
    assert cfg.resolution == (320, 240)
    assert cfg.camera == "cam1"
    assert cfg.capture_every_n_steps == 2


def test_from_dict_missing_fields_use_defaults():
    """缺失字段用默认。"""
    cfg = VideoConfig.from_dict({"enabled": False})
    assert cfg.enabled is False
    assert cfg.filename == "demo.mp4"
    assert cfg.fps == 15
    assert cfg.resolution == (640, 480)
    assert cfg.camera is None
    assert cfg.capture_every_n_steps == 1


def test_from_dict_resolution_list_to_tuple():
    """resolution list → tuple 转换。"""
    cfg = VideoConfig.from_dict({"resolution": [1280, 720]})
    assert cfg.resolution == (1280, 720)
    assert isinstance(cfg.resolution, tuple)


def test_from_dict_resolution_tuple_ok():
    """resolution 已是 tuple 时直接接受。"""
    cfg = VideoConfig.from_dict({"resolution": (800, 600)})
    assert cfg.resolution == (800, 600)


def test_from_dict_invalid_resolution_odd():
    """奇数 resolution 抛 ValueError。"""
    with pytest.raises(ValueError):
        VideoConfig.from_dict({"resolution": [641, 480]})
    with pytest.raises(ValueError):
        VideoConfig.from_dict({"resolution": [640, 481]})


def test_from_dict_invalid_resolution_non_positive():
    """非正数 resolution 抛 ValueError。"""
    with pytest.raises(ValueError):
        VideoConfig.from_dict({"resolution": [0, 480]})


def test_from_dict_invalid_resolution_length():
    """长度不为 2 的 resolution 抛 ValueError。"""
    with pytest.raises(ValueError):
        VideoConfig.from_dict({"resolution": [640]})
    with pytest.raises(ValueError):
        VideoConfig.from_dict({"resolution": [640, 480, 3]})


def test_from_dict_empty_dict():
    """空 dict 全默认。"""
    cfg = VideoConfig.from_dict({})
    assert cfg == VideoConfig()


def test_from_dict_not_dict_raises():
    """非 dict 输入抛 ValueError。"""
    with pytest.raises(ValueError):
        VideoConfig.from_dict("not-a-dict")  # type: ignore[arg-type]
