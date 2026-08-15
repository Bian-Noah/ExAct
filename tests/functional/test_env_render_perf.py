"""CPU 渲染单帧性能基线（iter2-renderer-env-mode）。

记录 TINY_RENDERER 单帧 render() 平均耗时作为基线。
不设硬阈值，仅打印日志。
"""

from __future__ import annotations

import logging
import time

import pytest

from config.loader import EnvConfig
from env.pybullet_env import PyBulletEnv


@pytest.mark.slow
def test_render_cpu_baseline_timing(caplog):
    """CPU 模式下 5 次 render() 耗时基线（warmup 1 次 + 4 次平均）。"""
    env = PyBulletEnv(env_config=EnvConfig(mode="direct", renderer="cpu"))
    try:
        env.reset(task_spec={"objects": []})
        # warmup
        env.render()
        # 测量 4 次
        durations = []
        for _ in range(4):
            t0 = time.perf_counter()
            env.render()
            durations.append(time.perf_counter() - t0)
        avg_ms = sum(durations) / len(durations) * 1000
        with caplog.at_level(logging.INFO):
            logging.getLogger("baseline").info(
                f"CPU render 平均耗时: {avg_ms:.2f} ms (min={min(durations)*1000:.2f}, max={max(durations)*1000:.2f})"
            )
        # 软断言：不应超 1 秒（M4 Air CPU 软渲染典型值 < 500ms）
        assert avg_ms < 1000, f"CPU render 平均耗时 {avg_ms:.2f}ms 超过 1s"
    finally:
        env.close()
