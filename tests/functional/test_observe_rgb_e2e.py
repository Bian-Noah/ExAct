"""observe 端到端视觉链路（iter2-renderer-env-mode）。

真实启动 env + ObserveTool，验证能拿到 RGB 图像，shape 正确。
"""

from __future__ import annotations

import numpy as np
import pytest

from config.loader import EnvConfig
from env.pybullet_env import PyBulletPandaEnv
from tools.observe import ObserveTool


@pytest.mark.slow
def test_observe_gets_rgb_end_to_end():
    """env.reset + ObserveTool._run() 真实链路 → obs['rgb'] shape 正确。"""
    env = PyBulletPandaEnv(env_config=EnvConfig(mode="direct", renderer="cpu"))
    try:
        env.reset(task_spec={"objects": []})
        tool = ObserveTool(env=env)
        # Iteration 5：_run() 返回 list[dict]
        result = tool._run()
        assert isinstance(result, list)
        text = result[0]["text"]
        # 文本格式
        assert "末端执行器位置" in text
        # 验证 env 拿到了 RGB
        obs = env.get_obs(include_rgb=True)
        assert obs["rgb"] is not None
        assert obs["rgb"].shape == (480, 640, 3)
        assert obs["rgb"].dtype == np.uint8
    finally:
        env.close()
