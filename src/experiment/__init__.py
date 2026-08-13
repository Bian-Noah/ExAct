"""experiment 子包导出。

Iteration 3 引入：ExperimentRecorder（实验数据落盘）+ 进程级单例 API。
"""

from .recorder import (
    ExperimentRecorder,
    _SafeRecorder,
    get_recorder,
    set_recorder,
)

__all__ = [
    "ExperimentRecorder",
    "_SafeRecorder",
    "get_recorder",
    "set_recorder",
]