"""ObserveTool 多相机 image block 单元测试（iter11-reset-multicam）。

验证遍历 obs["rgb"] dict 每相机,ImageStore 按相机名分目录,生成多 image block。
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from tools.observe import ObserveTool


@pytest.fixture
def mock_env_single():
    """env 配 1 相机。"""
    env = MagicMock()
    env.get_obs = MagicMock(return_value={
        "ee_pos": (0.1, 0.2, 0.3),
        "object_info": [],
        "state_desc": "test",
        "rgb": {"cam1": np.zeros((480, 640, 3), dtype=np.uint8)},
    })
    return env


@pytest.fixture
def mock_env_multi():
    """env 配 3 相机。"""
    env = MagicMock()
    env.get_obs = MagicMock(return_value={
        "ee_pos": (0.1, 0.2, 0.3),
        "object_info": [],
        "state_desc": "test",
        "rgb": {
            "cam1": np.zeros((480, 640, 3), dtype=np.uint8),
            "cam2": np.ones((480, 640, 3), dtype=np.uint8) * 50,
            "cam3": np.ones((480, 640, 3), dtype=np.uint8) * 100,
        },
    })
    return env


@pytest.fixture
def mock_image_store():
    """Mock ImageStore,save 返回 img:// URL,upload_to_minimax 返回 mm_file://."""
    store = MagicMock()
    counter = [0]

    def fake_save(image, category):
        counter[0] += 1
        return f"img://{category}/{counter[0]:03d}.png"

    store.save = MagicMock(side_effect=fake_save)
    store.upload_to_minimax = MagicMock(side_effect=lambda url: f"mm_file://fake_{url}")
    return store


def test_observe_single_camera_returns_text_plus_one_image(mock_env_single, mock_image_store):
    """1 camera 时返回 list 长度 2(text + 1 image)。"""
    tool = ObserveTool(env=mock_env_single, image_store=mock_image_store)
    content = tool._run()
    assert len(content) == 2
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image"


def test_observe_multi_camera_returns_text_plus_n_images(mock_env_multi, mock_image_store):
    """3 cameras 时返回 list 长度 4(text + 3 images)。"""
    tool = ObserveTool(env=mock_env_multi, image_store=mock_image_store)
    content = tool._run()
    assert len(content) == 4
    assert content[0]["type"] == "text"
    assert all(c["type"] == "image" for c in content[1:])


def test_observe_image_store_per_camera_category(mock_env_multi, mock_image_store):
    """ImageStore 中 3 个 camera 图分别存到不同 category。"""
    tool = ObserveTool(env=mock_env_multi, image_store=mock_image_store)
    tool._run()
    categories_used = [call.kwargs.get("category") for call in mock_image_store.save.call_args_list]
    assert categories_used == ["observations_cam1", "observations_cam2", "observations_cam3"]


def test_observe_no_image_store_returns_text_only(mock_env_single):
    """image_store=None 时只返回 text 块。"""
    tool = ObserveTool(env=mock_env_single, image_store=None)
    content = tool._run()
    assert len(content) == 1
    assert content[0]["type"] == "text"


def test_observe_rgb_none_returns_text_only():
    """obs["rgb"]=None 时只返回 text 块。"""
    env = MagicMock()
    env.get_obs = MagicMock(return_value={
        "ee_pos": (0.1, 0.2, 0.3),
        "object_info": [],
        "state_desc": "test",
        "rgb": None,
    })
    tool = ObserveTool(env=env, image_store=MagicMock())
    content = tool._run()
    assert len(content) == 1


def test_observe_recorder_emit_per_camera(mock_env_multi):
    """recorder.emit("observe_image") 每个相机调一次。"""
    from experiment.recorder import ExperimentRecorder, _global_recorder
    import experiment.recorder as rec_module

    # 用 MagicMock 替换全局 recorder
    mock_recorder = MagicMock()
    rec_module._global_recorder = mock_recorder

    tool = ObserveTool(env=mock_env_multi, image_store=None)
    tool._run()

    # 3 相机 × 1 emit("observe_image") = 3 次
    observe_image_calls = [
        c for c in mock_recorder.emit.call_args_list
        if c.args[0] == "observe_image"
    ]
    assert len(observe_image_calls) == 3

    # 还原(避免污染其他测试)
    rec_module._global_recorder = None