"""ExperimentConfig 字段扩展单元测试。

覆盖（按 tasks.md 3.3 任务列表）：
- ExperimentConfig() 默认值（enabled=True、root='data/experiment'、log_to_stdout=True）
- ExperimentConfig 自定义字段
- load_config("configs/default.yaml").experiment 字段匹配
- yaml 缺 experiment 节时 fallback 默认实例
- yaml experiment.enabled 类型错误时抛 TypeError
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config import ExperimentConfig, load_config


# ============================================================================
# ExperimentConfig 默认值与字段
# ============================================================================


def test_experiment_config_defaults():
    """ExperimentConfig() 默认值应匹配设计文档。"""
    cfg = ExperimentConfig()
    assert cfg.enabled is True
    assert cfg.root == "data/experiment"
    assert cfg.log_to_stdout is True


def test_experiment_config_custom_fields():
    """ExperimentConfig 自定义字段。"""
    cfg = ExperimentConfig(enabled=False, root="/tmp/exp", log_to_stdout=False)
    assert cfg.enabled is False
    assert cfg.root == "/tmp/exp"
    assert cfg.log_to_stdout is False


# ============================================================================
# load_config 加载 configs/default.yaml
# ============================================================================


def test_load_default_yaml_experiment_section():
    """configs/default.yaml 应包含完整的 experiment 节。"""
    config = load_config("configs/default.yaml")
    assert hasattr(config, "experiment")
    assert config.experiment.enabled is True
    assert config.experiment.root == "data/experiment"
    assert config.experiment.log_to_stdout is True


# ============================================================================
# yaml 缺 experiment 节时 fallback
# ============================================================================


def test_load_yaml_without_experiment_fallback(tmp_path: Path):
    """yaml 缺 experiment 节时 fallback 到 ExperimentConfig() 默认实例。"""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """\
env:
  mode: direct
  renderer: auto
  camera_resolution: [64, 48]
vla:
  backend: mock
  max_steps: 50
llm:
  api_key: "test-key"
  model: "MiniMax-M3"
  base_url: "https://api.minimax.chat/v1"
  max_tokens: 2048
explore:
  enabled: false
robot:
  urdf_path: "franka_panda/panda.urdf"
  base_position: [0.0, 0.0, 0.0]
  arm_joint_indices: [0, 1, 2, 3, 4, 5, 6]
  ee_link_index: 11
  finger_joint_indices: [9, 10]
task:
  default_user_goal: "goal"
  objects:
    - type: cube
      pos: [0.5, 0, 0.1]
      color: red
agent:
  max_react_rounds: 5
  max_tool_calls: 3
"""
    )
    config = load_config(str(yaml_path))
    # 缺 experiment 节时，应 fallback 到默认实例（Iteration 3 设计）
    assert isinstance(config.experiment, ExperimentConfig)
    assert config.experiment.enabled is True
    assert config.experiment.root == "data/experiment"
    assert config.experiment.log_to_stdout is True


# ============================================================================
# yaml 字段类型错误
# ============================================================================


def test_load_yaml_experiment_enabled_type_error(tmp_path: Path):
    """yaml experiment.enabled 不是 bool 时应抛 TypeError。"""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """\
env:
  mode: direct
  renderer: auto
  camera_resolution: [64, 48]
vla:
  backend: mock
  max_steps: 50
llm:
  api_key: "test-key"
  model: "MiniMax-M3"
  base_url: "https://api.minimax.chat/v1"
  max_tokens: 2048
explore:
  enabled: false
robot:
  urdf_path: "franka_panda/panda.urdf"
  base_position: [0.0, 0.0, 0.0]
  arm_joint_indices: [0, 1, 2, 3, 4, 5, 6]
  ee_link_index: 11
  finger_joint_indices: [9, 10]
task:
  default_user_goal: "goal"
  objects:
    - type: cube
      pos: [0.5, 0, 0.1]
      color: red
agent:
  max_react_rounds: 5
  max_tool_calls: 3
experiment:
  enabled: "not_a_bool"
  root: "data/experiment"
  log_to_stdout: true
"""
    )
    with pytest.raises(TypeError, match="enabled"):
        load_config(str(yaml_path))


def test_load_yaml_experiment_log_to_stdout_type_error(tmp_path: Path):
    """yaml experiment.log_to_stdout 不是 bool 时应抛 TypeError。"""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """\
env:
  mode: direct
  renderer: auto
  camera_resolution: [64, 48]
vla:
  backend: mock
  max_steps: 50
llm:
  api_key: "test-key"
  model: "MiniMax-M3"
  base_url: "https://api.minimax.chat/v1"
  max_tokens: 2048
explore:
  enabled: false
robot:
  urdf_path: "franka_panda/panda.urdf"
  base_position: [0.0, 0.0, 0.0]
  arm_joint_indices: [0, 1, 2, 3, 4, 5, 6]
  ee_link_index: 11
  finger_joint_indices: [9, 10]
task:
  default_user_goal: "goal"
  objects:
    - type: cube
      pos: [0.5, 0, 0.1]
      color: red
agent:
  max_react_rounds: 5
  max_tool_calls: 3
experiment:
  enabled: true
  root: "data/experiment"
  log_to_stdout: "not_a_bool"
"""
    )
    with pytest.raises(TypeError, match="log_to_stdout"):
        load_config(str(yaml_path))


def test_load_yaml_experiment_root_type_error(tmp_path: Path):
    """yaml experiment.root 不是 str 时应抛 TypeError。"""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """\
env:
  mode: direct
  renderer: auto
  camera_resolution: [64, 48]
vla:
  backend: mock
  max_steps: 50
llm:
  api_key: "test-key"
  model: "MiniMax-M3"
  base_url: "https://api.minimax.chat/v1"
  max_tokens: 2048
explore:
  enabled: false
robot:
  urdf_path: "franka_panda/panda.urdf"
  base_position: [0.0, 0.0, 0.0]
  arm_joint_indices: [0, 1, 2, 3, 4, 5, 6]
  ee_link_index: 11
  finger_joint_indices: [9, 10]
task:
  default_user_goal: "goal"
  objects:
    - type: cube
      pos: [0.5, 0, 0.1]
      color: red
agent:
  max_react_rounds: 5
  max_tool_calls: 3
experiment:
  enabled: true
  root: 12345
  log_to_stdout: true
"""
    )
    with pytest.raises(TypeError, match="root"):
        load_config(str(yaml_path))


# ============================================================================
# 部分字段存在 + 部分 fallback
# ============================================================================


def test_load_yaml_experiment_partial_fields_fallback(tmp_path: Path):
    """yaml experiment 仅提供部分字段时，其他字段 fallback 到默认值。"""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """\
env:
  mode: direct
  renderer: auto
  camera_resolution: [64, 48]
vla:
  backend: mock
  max_steps: 50
llm:
  api_key: "test-key"
  model: "MiniMax-M3"
  base_url: "https://api.minimax.chat/v1"
  max_tokens: 2048
explore:
  enabled: false
robot:
  urdf_path: "franka_panda/panda.urdf"
  base_position: [0.0, 0.0, 0.0]
  arm_joint_indices: [0, 1, 2, 3, 4, 5, 6]
  ee_link_index: 11
  finger_joint_indices: [9, 10]
task:
  default_user_goal: "goal"
  objects:
    - type: cube
      pos: [0.5, 0, 0.1]
      color: red
agent:
  max_react_rounds: 5
  max_tool_calls: 3
experiment:
  root: "data/custom_exp"
"""
    )
    config = load_config(str(yaml_path))
    # 提供的字段
    assert config.experiment.root == "data/custom_exp"
    # 未提供的字段 fallback 默认值
    assert config.experiment.enabled is True
    assert config.experiment.log_to_stdout is True