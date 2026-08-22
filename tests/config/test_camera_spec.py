"""CameraSpec + EnvConfig.cameras 单元测试（iter11-reset-multicam）。

测试 CameraSpec dataclass 序列化、EnvConfig.cameras 字段加载、
yaml 中 cameras list → tuple[CameraSpec, ...] 转换,name 重复 warn。
"""
from __future__ import annotations

import warnings
from textwrap import dedent

import pytest
import yaml

from config.loader import CameraSpec, EnvConfig, _from_dict, load_config


# ----- U1 单元测试 -----

def test_camera_spec_construction():
    """CameraSpec 构造后字段值正确。"""
    cam = CameraSpec(
        name="cam1",
        target=(0.5, 0.0, 0.5),
        distance=1.5,
        yaw=50,
        pitch=-35,
        roll=0,
    )
    assert cam.name == "cam1"
    assert cam.target == (0.5, 0.0, 0.5)
    assert cam.distance == 1.5
    assert cam.yaw == 50
    assert cam.pitch == -35
    assert cam.roll == 0
    assert cam.fov == 60.0
    assert cam.resolution == (640, 480)


def test_camera_spec_frozen():
    """CameraSpec 是 frozen dataclass,修改字段报错。"""
    cam = CameraSpec(
        name="cam1",
        target=(0.5, 0.0, 0.5),
        distance=1.5,
        yaw=50,
        pitch=-35,
        roll=0,
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        cam.name = "cam2"  # type: ignore[misc]


def test_camera_spec_custom_resolution():
    """CameraSpec 可独立配 fov / resolution。"""
    cam = CameraSpec(
        name="wrist",
        target=(0.3, 0.0, 0.2),
        distance=0.5,
        yaw=180,
        pitch=0,
        roll=0,
        fov=90.0,
        resolution=(256, 256),
    )
    assert cam.fov == 90.0
    assert cam.resolution == (256, 256)


def test_env_config_default_camera():
    """EnvConfig 默认 cameras 含 1 个 overhead 相机。"""
    env_cfg = EnvConfig()
    assert len(env_cfg.cameras) == 1
    cam = env_cfg.cameras[0]
    assert cam.name == "observation.images.top"
    assert cam.distance == 1.5


def test_env_config_load_yaml_cameras(tmp_path):
    """yaml 中配 3 个 camera,加载后 cameras 长度 3,name 正确。"""
    cfg_file = tmp_path / "env.yaml"
    cfg_file.write_text(dedent("""
        env:
          cameras:
            - name: observation.images.top
              target: [0.5, 0.0, 0.5]
              distance: 1.5
              yaw: 50
              pitch: -35
              roll: 0
            - name: observation.images.side
              target: [0.0, 0.5, 0.3]
              distance: 1.0
              yaw: 0
              pitch: -30
              roll: 0
            - name: observation.images.wrist
              target: [0.3, 0.0, 0.2]
              distance: 0.3
              yaw: 180
              pitch: 0
              roll: 0
              fov: 90
              resolution: [256, 256]
    """).strip())

    cfg = load_config(str(cfg_file))
    assert len(cfg.env.cameras) == 3
    assert cfg.env.cameras[0].name == "observation.images.top"
    assert cfg.env.cameras[1].name == "observation.images.side"
    assert cfg.env.cameras[2].name == "observation.images.wrist"
    assert cfg.env.cameras[2].resolution == (256, 256)


def test_env_config_missing_cameras_field(tmp_path):
    """yaml 不写 cameras 字段,使用默认 1 个 overhead。"""
    cfg_file = tmp_path / "env.yaml"
    cfg_file.write_text("env:\n  mode: direct\n")
    cfg = load_config(str(cfg_file))
    assert len(cfg.env.cameras) == 1
    assert cfg.env.cameras[0].name == "observation.images.top"


def test_camera_spec_duplicate_name_warning():
    """CameraSpec 同 name 重复加载时 warn（但不阻断）。"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # 直接构造 EnvConfig 时 cameras 字段重复 name
        result = EnvConfig(
            cameras=(
                CameraSpec(name="cam1", target=(0.5, 0.0, 0.5),
                           distance=1.5, yaw=50, pitch=-35, roll=0),
                CameraSpec(name="cam1", target=(0.0, 0.5, 0.3),
                           distance=1.0, yaw=0, pitch=-30, roll=0),
            )
        )
    assert len(result.cameras) == 2  # 重复也保留两个,dict key 冲突由 consumer 处理
    # 构造阶段不 warn(因为 dataclass 构造时无校验);只有 _from_dict 路径才 warn
    assert not any("重复 name" in str(warning.message) for warning in w)


def test_env_config_cameras_duplicate_name_warning_via_yaml(tmp_path):
    """yaml cameras 中同 name 重复 → _from_dict warn。"""
    cfg_file = tmp_path / "env.yaml"
    cfg_file.write_text(dedent("""
        env:
          cameras:
            - name: cam1
              target: [0.5, 0.0, 0.5]
              distance: 1.5
              yaw: 50
              pitch: -35
              roll: 0
            - name: cam1
              target: [0.0, 0.5, 0.3]
              distance: 1.0
              yaw: 0
              pitch: -30
              roll: 0
    """).strip())

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_config(str(cfg_file))
    assert len(cfg.env.cameras) == 2
    assert any("重复 name" in str(warning.message) for warning in w)


def test_env_config_cameras_not_list():
    """EnvConfig.cameras 字段不是 list 时报错。"""
    with pytest.raises(ValueError, match="cameras.*期望 list"):
        _from_dict({"cameras": {"name": "cam1"}}, EnvConfig)


def test_env_config_cameras_item_not_dict():
    """EnvConfig.cameras 元素不是 dict 时报错。"""
    with pytest.raises(ValueError, match=r"cameras\[0\].*期望 dict"):
        _from_dict({"cameras": ["not_a_dict"]}, EnvConfig)