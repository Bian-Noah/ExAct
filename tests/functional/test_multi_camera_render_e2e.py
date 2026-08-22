"""多相机渲染集成功能测试（iter11-reset-multicam）。

在真实 PyBullet env 上验证:
  1. env 配 3 相机,render() 返回 3 张独立图
  2. ObserveTool 返回 1 text + 3 image block
  （LeRobotVLA 端验证见 test_vla_reset_recovery.py,本文件不 import lerobot 避免 torch 依赖）
"""
from __future__ import annotations

import numpy as np

from config.loader import CameraSpec, EnvConfig, RobotConfig
from env.pybullet_env import PyBulletEnv
from tools.observe import ObserveTool
from utils.image_store import ImageStore, MemoryBackend


def _make_env(camera_count: int = 3):
    """构造 3 相机 env(overhead + 2 side)。"""
    cameras = (
        CameraSpec(name="observation.images.top", target=(0.5, 0.0, 0.5),
                   distance=1.5, yaw=50, pitch=-35, roll=0),
        CameraSpec(name="observation.images.side", target=(0.0, 0.5, 0.3),
                   distance=1.0, yaw=0, pitch=-30, roll=0),
        CameraSpec(name="observation.images.wrist", target=(0.3, 0.0, 0.2),
                   distance=0.3, yaw=180, pitch=0, roll=0,
                   fov=90, resolution=(256, 256)),
    )
    env = PyBulletEnv(
        env_config=EnvConfig(mode="direct", renderer="cpu", cameras=cameras),
        robot_config=RobotConfig(type="so101"),
    )
    env.reset(task_spec={"objects": [{"type": "cube", "pos": [0.5, 0, 0.1], "color": "yellow"}]})
    return env


def test_multi_camera_render_e2e():
    """3 相机 env → render() 返回 3 张独立图,形状正确。"""
    env = _make_env()
    images = env.render()
    assert len(images) == 3
    assert set(images.keys()) == {
        "observation.images.top", "observation.images.side", "observation.images.wrist"
    }
    # overhead 640x480,wrist 256x256
    assert images["observation.images.top"].shape == (480, 640, 3)
    assert images["observation.images.wrist"].shape == (256, 256, 3)
    # 不同视角图像不同
    assert not np.array_equal(images["observation.images.top"], images["observation.images.side"])
    env.close()


def test_observe_tool_multi_camera_e2e(tmp_path):
    """ObserveTool 多相机 → 返回 1 text + 3 image block。"""
    env = _make_env()
    image_store = ImageStore(MemoryBackend(max_memory_items=10, spill_dir=tmp_path))
    image_store.upload_to_minimax = lambda url: f"mm_file://fake_{url}"
    tool = ObserveTool(env=env, image_store=image_store)
    content = tool._run()
    assert len(content) == 4  # 1 text + 3 image
    assert content[0]["type"] == "text"
    assert all(c["type"] == "image" for c in content[1:])
    # 3 张图存到不同 category(相机名含点已 sanitize 为 _)
    categories = {
        c["url"].split("img://")[1].split("/")[0]
        for c in content[1:]
    }
    assert len(categories) == 3
    assert categories == {
        "observations_observation_images_top",
        "observations_observation_images_side",
        "observations_observation_images_wrist",
    }
    env.close()