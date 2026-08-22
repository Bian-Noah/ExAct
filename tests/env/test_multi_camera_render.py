"""env.render() 多相机单元测试（iter11-reset-multicam）。

测试 env.render() 按 cameras 列表返回 dict[str, np.ndarray],
验证 dict 长度、shape、dtype、key 一致性、多相机视图差异、
每相机独立 resolution。
"""
from __future__ import annotations

import numpy as np
import pytest

from config.loader import CameraSpec, EnvConfig
from env.pybullet_env import PyBulletEnv


# ----- U3 单元测试 -----

@pytest.fixture
def env_single_camera():
    """单相机(env 默认配置)PyBulletEnv。"""
    env = PyBulletEnv(env_config=EnvConfig())
    env.reset(task_spec={"objects": []})
    yield env
    env.close()


@pytest.fixture
def env_multi_camera():
    """3 相机配置 PyBulletEnv(top / side / wrist)。"""
    env = PyBulletEnv(
        env_config=EnvConfig(
            cameras=(
                CameraSpec(
                    name="observation.images.top",
                    target=(0.5, 0.0, 0.5),
                    distance=1.5,
                    yaw=50,
                    pitch=-35,
                    roll=0,
                ),
                CameraSpec(
                    name="observation.images.side",
                    target=(0.0, 0.5, 0.3),
                    distance=1.0,
                    yaw=0,
                    pitch=-30,
                    roll=0,
                ),
                CameraSpec(
                    name="observation.images.wrist",
                    target=(0.3, 0.0, 0.2),
                    distance=0.3,
                    yaw=180,
                    pitch=0,
                    roll=0,
                    fov=90,
                    resolution=(256, 256),
                ),
            ),
        ),
    )
    env.reset(task_spec={"objects": []})
    yield env
    env.close()


def test_render_single_camera_returns_dict_length_1(env_single_camera):
    """单相机配置时 render() 返回 dict 长度 1。"""
    images = env_single_camera.render()
    assert isinstance(images, dict)
    assert len(images) == 1


def test_render_multi_camera_returns_dict_length_n(env_multi_camera):
    """3 相机配置时 render() 返回 dict 长度 3。"""
    images = env_multi_camera.render()
    assert isinstance(images, dict)
    assert len(images) == 3


def test_render_image_shape_and_dtype(env_single_camera):
    """返回 ndarray shape (H, W, 3) dtype uint8。"""
    images = env_single_camera.render()
    for name, rgb in images.items():
        assert rgb.dtype == np.uint8
        assert rgb.ndim == 3
        assert rgb.shape[2] == 3
        assert rgb.shape[0] > 0 and rgb.shape[1] > 0


def test_render_dict_key_matches_camera_spec_name(env_multi_camera):
    """dict key 与 CameraSpec.name 一致。"""
    images = env_multi_camera.render()
    assert set(images.keys()) == {
        "observation.images.top",
        "observation.images.side",
        "observation.images.wrist",
    }


def test_render_multi_camera_different_views(env_multi_camera):
    """多相机视图矩阵不同,返回的 RGB 数组不完全相同。"""
    images = env_multi_camera.render()
    rgb_top = images["observation.images.top"]
    rgb_side = images["observation.images.side"]
    # 两个视角应该看到不同的场景分布,绝大多数像素不相等
    assert not np.array_equal(rgb_top, rgb_side)


def test_render_per_camera_resolution(env_multi_camera):
    """wrist 相机独立配 256x256,与 overhead 的 640x480 不同。"""
    images = env_multi_camera.render()
    assert images["observation.images.top"].shape[:2] == (480, 640)
    assert images["observation.images.wrist"].shape[:2] == (256, 256)


def test_render_obs_rgb_is_dict(env_multi_camera):
    """env.get_obs(include_rgb=True) 中 obs["rgb"] 是 dict,与 render() 一致。"""
    obs = env_multi_camera.get_obs(include_rgb=True)
    assert isinstance(obs["rgb"], dict)
    assert len(obs["rgb"]) == 3


def test_render_obs_rgb_none_when_include_rgb_false(env_multi_camera):
    """env.get_obs(include_rgb=False) 中 obs["rgb"] 为 None。"""
    obs = env_multi_camera.get_obs(include_rgb=False)
    assert obs["rgb"] is None