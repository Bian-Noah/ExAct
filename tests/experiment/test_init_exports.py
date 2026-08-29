"""experiment 子包导出单元测试（iter12-video-recording 任务 6）。

覆盖：`from experiment import VideoConfig` 可用。
"""

from __future__ import annotations

from experiment import (
    ExperimentRecorder,
    VideoConfig,
    get_recorder,
    set_recorder,
)


def test_experiment_exports_video_config():
    """experiment 子包导出 VideoConfig。"""
    assert VideoConfig is not None
    assert VideoConfig().enabled is True


def test_experiment_exports_recorder_api():
    """原有导出不受影响（回归）。"""
    assert ExperimentRecorder is not None
    assert callable(get_recorder)
    assert callable(set_recorder)


def test_video_config_usable_from_experiment():
    """通过 experiment 导出的 VideoConfig 可正常构造。"""
    cfg = VideoConfig.from_dict({"fps": 30, "resolution": [320, 240]})
    assert cfg.fps == 30
    assert cfg.resolution == (320, 240)
