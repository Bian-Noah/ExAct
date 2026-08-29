"""ExperimentConfig.video 配置解析单元测试（iter12-video-recording 任务 7）。

覆盖：
- 旧 yaml（无 video 段）video is None
- yaml 含 video 段时正确解析
- 非法 resolution 回退 None + warning
- default.yaml 含 experiment.video 段且字段齐全
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from config import load_config
from config.loader import ExperimentConfig
from experiment.video.config import VideoConfig


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


# ============================================================================
# 缺省行为
# ============================================================================


def test_experiment_config_video_default_none():
    """ExperimentConfig() 默认 video is None（旧调用兼容）。"""
    cfg = ExperimentConfig()
    assert cfg.video is None


def test_old_yaml_no_video_section(tmp_path: Path):
    """旧 yaml（无 video 段）video is None。"""
    cfg_path = _write_yaml(
        tmp_path,
        {
            "env": {"mode": "direct"},
            "vla": {"backend": "mock"},
            "llm": {"api_key": "x"},
            "robot": {"type": "panda"},
            "task": {"default_user_goal": "x"},
            "agent": {"max_react_rounds": 1},
            "experiment": {"enabled": True, "root": "data/experiment"},
        },
    )
    cfg = load_config(str(cfg_path))
    assert cfg.experiment.video is None


def test_experiment_section_absent_video_none(tmp_path: Path):
    """整个 experiment 段缺失时 video 用默认 None。"""
    cfg_path = _write_yaml(
        tmp_path,
        {
            "env": {"mode": "direct"},
            "vla": {"backend": "mock"},
            "llm": {"api_key": "x"},
            "robot": {"type": "panda"},
        },
    )
    cfg = load_config(str(cfg_path))
    assert cfg.experiment.video is None


# ============================================================================
# 解析
# ============================================================================


def test_yaml_video_parsed(tmp_path: Path):
    """yaml 含 video 段时正确解析。"""
    cfg_path = _write_yaml(
        tmp_path,
        {
            "env": {"mode": "direct"},
            "vla": {"backend": "mock"},
            "llm": {"api_key": "x"},
            "robot": {"type": "panda"},
            "experiment": {
                "enabled": True,
                "video": {
                    "enabled": True,
                    "filename": "exp.mp4",
                    "fps": 30,
                    "resolution": [1280, 720],
                    "camera": "observation.images.top",
                    "capture_every_n_steps": 2,
                },
            },
        },
    )
    cfg = load_config(str(cfg_path))
    video = cfg.experiment.video
    assert isinstance(video, VideoConfig)
    assert video.enabled is True
    assert video.filename == "exp.mp4"
    assert video.fps == 30
    assert video.resolution == (1280, 720)
    assert video.camera == "observation.images.top"
    assert video.capture_every_n_steps == 2


def test_yaml_video_enabled_false(tmp_path: Path):
    """video.enabled=False 正确解析。"""
    cfg_path = _write_yaml(
        tmp_path,
        {
            "env": {"mode": "direct"},
            "vla": {"backend": "mock"},
            "llm": {"api_key": "x"},
            "robot": {"type": "panda"},
            "experiment": {"video": {"enabled": False}},
        },
    )
    cfg = load_config(str(cfg_path))
    assert cfg.experiment.video is not None
    assert cfg.experiment.video.enabled is False


# ============================================================================
# 非法配置回退
# ============================================================================


def test_yaml_video_invalid_resolution_fallback(tmp_path: Path):
    """非法 resolution（奇数）回退 None + warning。"""
    cfg_path = _write_yaml(
        tmp_path,
        {
            "env": {"mode": "direct"},
            "vla": {"backend": "mock"},
            "llm": {"api_key": "x"},
            "robot": {"type": "panda"},
            "experiment": {"video": {"resolution": [641, 480]}},
        },
    )
    with pytest.warns(UserWarning):
        cfg = load_config(str(cfg_path))
    assert cfg.experiment.video is None


# ============================================================================
# default.yaml 校验
# ============================================================================


def test_default_yaml_has_video_section():
    """default.yaml 含 experiment.video 段且字段齐全。"""
    cfg = load_config("configs/default.yaml")
    video = cfg.experiment.video
    assert video is not None, "default.yaml 应含 experiment.video 段"
    assert video.enabled is True
    assert video.filename == "demo.mp4"
    assert video.fps == 15
    assert video.resolution == (640, 480)
    assert video.camera is None
    assert video.capture_every_n_steps == 1


def test_default_yaml_video_matches_design():
    """default.yaml video 字段与设计文档 4.3 一致。"""
    cfg = load_config("configs/default.yaml")
    video = cfg.experiment.video
    assert video.resolution[0] % 2 == 0
    assert video.resolution[1] % 2 == 0
