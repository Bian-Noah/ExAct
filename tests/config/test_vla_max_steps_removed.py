"""config 清理 vla.max_steps + runner 接线 单元测试（iter11-reset-multicam）。

验证 VLAConfig 无 max_steps 字段,yaml 加载报错时 warn 忽略,
Executor 构造不再含 max_steps。
"""
from __future__ import annotations

import warnings
from textwrap import dedent

import pytest

from config.loader import VLAConfig, load_config


def test_vla_config_no_max_steps_field():
    """VLAConfig 实例无 max_steps 属性。"""
    vla_cfg = VLAConfig()
    assert not hasattr(vla_cfg, "max_steps")


def test_load_yaml_no_max_steps_succeeds(tmp_path):
    """yaml 不写 vla.max_steps 加载成功。"""
    cfg_file = tmp_path / "env.yaml"
    cfg_file.write_text("vla:\n  backend: mock\n")
    cfg = load_config(str(cfg_file))
    assert cfg.vla.backend == "mock"


def test_load_yaml_with_max_steps_warns_and_ignores(tmp_path):
    """yaml 写 vla.max_steps 时 warn 忽略(视为未知字段)。"""
    cfg_file = tmp_path / "env.yaml"
    cfg_file.write_text("vla:\n  backend: mock\n  max_steps: 99\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_config(str(cfg_file))
    assert cfg.vla.backend == "mock"
    # max_steps 字段已被忽略,cfg.vla 无该属性
    assert not hasattr(cfg.vla, "max_steps")
    # 应该有警告
    assert any("max_steps" in str(warning.message) for warning in w)


def test_load_yaml_with_cameras_succeeds(tmp_path):
    """yaml 写 env.cameras 加载成功(1 overhead)。"""
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
    """).strip())
    cfg = load_config(str(cfg_file))
    assert len(cfg.env.cameras) == 1
    assert cfg.env.cameras[0].name == "cam1"


def test_load_default_yaml_loads_successfully():
    """configs/default.yaml 加载成功(已被清理 max_steps,新增 cameras)。"""
    import os
    project_root = "/Users/noah/项目/保研练习项目/ExAct"
    cfg = load_config(os.path.join(project_root, "configs/default.yaml"))
    assert cfg.vla.backend == "mock"
    assert len(cfg.env.cameras) >= 1