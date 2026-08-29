"""experiment 子包导出。

Iteration 3 引入：ExperimentRecorder（实验数据落盘）+ 进程级单例 API。
Iteration 12 扩展（iter12-video-recording）：导出 VideoConfig（视频录制配置）。
"""

from .recorder import (
    ExperimentRecorder,
    _SafeRecorder,
    get_recorder,
    set_recorder,
)
from .video.config import VideoConfig

__all__ = [
    "ExperimentRecorder",
    "_SafeRecorder",
    "get_recorder",
    "set_recorder",
    "VideoConfig",
]