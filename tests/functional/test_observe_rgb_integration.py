"""ObserveTool 端到端集成（iter2-renderer-env-mode）。

mock env.get_obs 返回含 RGB 的 obs，验证 ObserveTool 调用契约。
"""

from __future__ import annotations

import numpy as np
from unittest.mock import MagicMock

from tools.observe import ObserveTool


def test_observe_calls_get_obs_with_rgb_true():
    env = MagicMock()
    env.get_obs.return_value = {
        "ee_pos": (0.5, 0.0, 0.4),
        "object_info": [],
        "state_desc": "...",
        "rgb": {"cam1": np.zeros((480, 640, 3), dtype=np.uint8)},
    }
    tool = ObserveTool(env=env)
    tool._run()
    env.get_obs.assert_called_with(include_rgb=True)


def test_observe_executes_with_rgb_in_obs():
    env = MagicMock()
    env.get_obs.return_value = {
        "ee_pos": (0.5, 0.0, 0.4),
        "object_info": [
            {"id": 1, "name": "cube", "pos": [0.5, 0.0, 0.1], "quat": [0, 0, 0, 1]},
        ],
        "state_desc": "...",
        "rgb": {"cam1": np.zeros((480, 640, 3), dtype=np.uint8)},
    }
    tool = ObserveTool(env=env)
    result = tool._run()
    # Iteration 5：_run() 返回 list[dict]
    assert isinstance(result, list)
    text = result[0]["text"]
    assert "末端执行器位置" in text
    assert "cube" in text
