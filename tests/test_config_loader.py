from __future__ import annotations

import os
import tempfile
import warnings

import pytest
import yaml

from src.config.loader import (
    AgentConfig,
    AppConfig,
    EnvConfig,
    ExperimentConfig,
    ExploreConfig,
    LerobotConfig,
    LLMConfig,
    RobotConfig,
    TaskConfig,
    VLAConfig,
    _from_dict,
    load_config,
)


# ---------- 1. dataclass 默认值 ----------

def test_env_config_defaults():
    cfg = EnvConfig()
    # iter2-renderer-env-mode: use_gui deprecated, 默认 mode=direct 推导为 False
    assert cfg.use_gui is False
    assert cfg.mode == "direct"
    assert cfg.renderer == "auto"
    assert cfg.camera_resolution == (640, 480)


def test_vla_config_defaults():
    cfg = VLAConfig()
    assert cfg.backend == "mock"
    assert cfg.model_path is None
    # iter11-reset-multicam:max_steps 已移除
    assert not hasattr(cfg, "max_steps")


def test_lerobot_config_defaults():
    cfg = LerobotConfig()
    assert cfg.policy_type == "act"
    assert cfg.device == "cuda:0"
    assert cfg.quantization == "none"
    assert cfg.image_key == "observation.images.top"
    assert cfg.action_dim == 14


def test_vla_config_lerobot_nested_default():
    # VLAConfig 默认的 lerobot 嵌套字段使用默认 LerobotConfig
    cfg = VLAConfig()
    assert isinstance(cfg.lerobot, LerobotConfig)
    assert cfg.lerobot.policy_type == "act"
    assert cfg.lerobot.quantization == "none"


def test_from_dict_vla_lerobot_nested():
    data = {
        "backend": "lerobot",
        "model_path": "lerobot/pi0_libero_finetuned",
        "lerobot": {
            "policy_type": "pi0",
            "device": "cuda:0",
            "quantization": "4bit",
        },
    }
    vla = _from_dict(data, VLAConfig)
    assert vla.backend == "lerobot"
    assert vla.model_path == "lerobot/pi0_libero_finetuned"
    assert vla.lerobot.policy_type == "pi0"
    assert vla.lerobot.quantization == "4bit"
    # 未提供的字段走默认
    assert vla.lerobot.image_key == "observation.images.top"
    assert vla.lerobot.action_dim == 14


def test_llm_config_defaults():
    cfg = LLMConfig()
    assert cfg.api_key == ""
    assert cfg.model == "MiniMax-M3"
    assert cfg.base_url == "https://api.minimax.chat/v1"
    assert cfg.max_tokens == 2048


def test_explore_config_defaults():
    cfg = ExploreConfig()
    assert cfg.enabled is False


# ---------- 1b. 新增 dataclass 默认值 ----------

def test_robot_config_defaults():
    cfg = RobotConfig()
    assert cfg.type == "panda"
    assert cfg.urdf_path == "franka_panda/panda.urdf"
    assert cfg.base_position == (0.0, 0.0, 0.0)
    assert cfg.panda.arm_joint_indices == (0, 1, 2, 3, 4, 5, 6)
    assert cfg.panda.ee_link_index == 11
    assert cfg.panda.finger_joint_indices == (9, 10)


def test_task_config_defaults():
    cfg = TaskConfig()
    assert cfg.default_user_goal == "把机械臂移到红色方块上方"
    assert len(cfg.objects) == 1
    assert cfg.objects[0]["type"] == "cube"
    assert cfg.objects[0]["color"] == "red"


def test_agent_config_defaults():
    cfg = AgentConfig()
    assert cfg.max_react_rounds == 5
    assert cfg.max_tool_calls == 3


def test_experiment_config_defaults():
    cfg = ExperimentConfig()
    # Iteration 3：默认开启录制
    assert cfg.enabled is True
    assert cfg.root == "data/experiment"
    assert cfg.log_to_stdout is True


# ---------- 2. _from_dict 正常场景 ----------

def test_from_dict_env_camera_list_to_tuple():
    cfg = _from_dict({"camera_resolution": [320, 240]}, EnvConfig)
    assert cfg.camera_resolution == (320, 240)
    assert isinstance(cfg.camera_resolution, tuple)
    # iter2: use_gui 默认 False（由 mode=direct 推导）
    assert cfg.use_gui is False


def test_from_dict_vla_model_path_null():
    cfg = _from_dict({"model_path": None, "max_steps": 100}, VLAConfig)
    assert cfg.model_path is None
    # iter11-reset-multicam:max_steps 已移除,传入被 warn 忽略
    assert not hasattr(cfg, "max_steps")
    assert cfg.backend == "mock"


def test_from_dict_llm_full():
    cfg = _from_dict(
        {
            "api_key": "sk-xxx",
            "model": "CustomModel",
            "base_url": "https://example.com/v1",
            "max_tokens": 1024,
        },
        LLMConfig,
    )
    assert cfg.api_key == "sk-xxx"
    assert cfg.model == "CustomModel"
    assert cfg.base_url == "https://example.com/v1"
    assert cfg.max_tokens == 1024


def test_from_dict_app_config_nested():
    raw = {
        "env": {"use_gui": False, "camera_resolution": [100, 200]},
        "vla": {"backend": "openvla", "model_path": "/tmp/model", "max_steps": 30},
        "llm": {"api_key": "key"},
        "explore": {"enabled": True},
    }
    app = _from_dict(raw, AppConfig)
    assert isinstance(app.env, EnvConfig)
    assert app.env.use_gui is False
    assert app.env.camera_resolution == (100, 200)
    assert isinstance(app.vla, VLAConfig)
    assert app.vla.backend == "openvla"
    assert app.vla.model_path == "/tmp/model"
    assert not hasattr(app.vla, "max_steps")
    assert isinstance(app.llm, LLMConfig)
    assert app.llm.api_key == "key"
    assert isinstance(app.explore, ExploreConfig)
    assert app.explore.enabled is True


def test_from_dict_unknown_field_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = _from_dict({"use_gui": True, "unknown_key": 123}, EnvConfig)
        assert cfg.use_gui is True
        assert len(w) == 1
        assert "unknown_key" in str(w[0].message)


# ---------- 3. _from_dict 异常校验 ----------

def test_from_dict_missing_required_field():
    @dataclass  # type: ignore
    class NoDefault:
        required: str

    # 临时加字段，用 _check_type 级别的 path 验证
    with pytest.raises(ValueError, match="缺少必填字段"):
        _from_dict({}, NoDefault)


def test_from_dict_env_camera_wrong_length():
    with pytest.raises(ValueError, match="camera_resolution.*长度"):
        _from_dict({"camera_resolution": [640]}, EnvConfig)


def test_from_dict_env_camera_non_int():
    with pytest.raises(ValueError, match="camera_resolution.*整数"):
        _from_dict({"camera_resolution": ["wide", 480]}, EnvConfig)


def test_from_dict_not_mapping():
    with pytest.raises(ValueError, match="不是 mapping"):
        _from_dict([1, 2, 3], EnvConfig)  # type: ignore[arg-type]


def test_check_type_bool_not_int():
    # iter11-reset-multicam:max_steps 字段已移除,改用 use_gui(bool 字段)验证类型拒绝
    with pytest.raises(TypeError, match="期望 bool"):
        _from_dict({"use_gui": "yes"}, EnvConfig)


def test_check_type_int_not_bool():
    with pytest.raises(TypeError, match="期望 bool"):
        _from_dict({"use_gui": 1}, EnvConfig)


def test_check_type_str_not_number():
    with pytest.raises(TypeError, match="期望 str"):
        _from_dict({"model": 1234}, LLMConfig)


# ---------- 4. load_config 文件场景 ----------

def _write_yaml(content: str) -> str:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name


def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("/tmp/definitely_does_not_exist_12345.yaml")


def test_load_config_invalid_yaml_syntax():
    path = _write_yaml("env:\n  use_gui: [invalid\n")
    try:
        with pytest.raises(ValueError, match="YAML 解析失败"):
            load_config(path)
    finally:
        os.unlink(path)


def test_load_config_top_level_not_mapping():
    path = _write_yaml("- 1\n- 2\n")
    try:
        with pytest.raises(ValueError, match="YAML 顶层必须是 mapping"):
            load_config(path)
    finally:
        os.unlink(path)


def test_load_config_section_not_mapping():
    path = _write_yaml("env: [1, 2]\n")
    try:
        with pytest.raises(ValueError, match="配置节 'env' 必须是 mapping"):
            load_config(path)
    finally:
        os.unlink(path)


def test_load_config_normal_full_yaml():
    path = _write_yaml(
        """\
env:
  use_gui: false
  camera_resolution: [320, 240]
vla:
  backend: mock
  model_path: null
  max_steps: 50
llm:
  api_key: "sk-test-key"
  model: "MiniMax-M3"
  base_url: "https://api.minimax.chat/v1"
  max_tokens: 2048
explore:
  enabled: false
"""
    )
    try:
        cfg = load_config(path)
        assert isinstance(cfg, AppConfig)
        assert cfg.env.use_gui is False
        assert cfg.env.camera_resolution == (320, 240)
        assert cfg.vla.backend == "mock"
        assert cfg.vla.model_path is None
        assert cfg.llm.api_key == "sk-test-key"
        assert cfg.llm.model == "MiniMax-M3"
        assert cfg.llm.base_url == "https://api.minimax.chat/v1"
        assert cfg.llm.max_tokens == 2048
        assert cfg.explore.enabled is False
    finally:
        os.unlink(path)


def test_load_config_empty_yaml_uses_defaults():
    path = _write_yaml("")
    try:
        cfg = load_config(path)
        # iter2: use_gui 默认 False（由 mode=direct 推导）
        assert cfg.env.use_gui is False
        assert cfg.env.mode == "direct"
        assert cfg.env.renderer == "auto"
        assert cfg.env.camera_resolution == (640, 480)
        assert cfg.vla.backend == "mock"
        assert cfg.vla.model_path is None
        # iter11-reset-multicam:max_steps 已移除
        assert not hasattr(cfg.vla, "max_steps")
        assert cfg.llm.api_key == ""
        assert cfg.llm.model == "MiniMax-M3"
        assert cfg.explore.enabled is False
    finally:
        os.unlink(path)


# ---------- 5. 新增节的 YAML 解析 ----------

def test_robot_config_yaml_parse():
    cfg = _from_dict(
        {
            "type": "panda",
            "urdf_path": "custom/robot.urdf",
            "panda": {
                "arm_joint_indices": [1, 2, 3],
            },
        },
        RobotConfig,
    )
    assert cfg.type == "panda"
    assert cfg.urdf_path == "custom/robot.urdf"
    assert cfg.panda.arm_joint_indices == (1, 2, 3)
    assert isinstance(cfg.panda.arm_joint_indices, tuple)
    # 未指定字段使用默认值
    assert cfg.panda.ee_link_index == 11


def test_robot_config_missing_uses_defaults():
    cfg = _from_dict({}, RobotConfig)
    assert cfg.type == "panda"
    assert cfg.urdf_path == "franka_panda/panda.urdf"
    assert cfg.panda.arm_joint_indices == (0, 1, 2, 3, 4, 5, 6)


def test_robot_config_legacy_flat_format_merged_into_panda():
    """旧平铺格式（arm_joint_indices 在顶层）→ 合并进 panda 特化块，不丢配置。"""
    cfg = _from_dict(
        {
            "urdf_path": "custom/robot.urdf",
            "arm_joint_indices": [1, 2, 3],
            "ee_link_index": 5,
        },
        RobotConfig,
    )
    assert cfg.type == "panda"
    assert cfg.urdf_path == "custom/robot.urdf"
    assert cfg.panda.arm_joint_indices == (1, 2, 3)
    assert cfg.panda.ee_link_index == 5
    # 未提供的夹爪索引用默认值
    assert cfg.panda.finger_joint_indices == (9, 10)


def test_task_config_yaml_parse():
    cfg = _from_dict(
        {
            "default_user_goal": "拾起红方块",
            "objects": [{"type": "cube", "pos": [0.6, 0.2, 0.1], "color": "red"}],
        },
        TaskConfig,
    )
    assert cfg.default_user_goal == "拾起红方块"
    assert len(cfg.objects) == 1
    assert cfg.objects[0]["type"] == "cube"


def test_agent_config_override():
    cfg = _from_dict({"max_react_rounds": 10, "max_tool_calls": 5}, AgentConfig)
    assert cfg.max_react_rounds == 10
    assert cfg.max_tool_calls == 5


def test_experiment_config_in_appconfig():
    raw = {
        "env": {"use_gui": False},
        "vla": {"backend": "mock"},
        "llm": {"api_key": "k"},
        "explore": {"enabled": False},
    }
    app = _from_dict(raw, AppConfig)
    assert isinstance(app.experiment, ExperimentConfig)
    # Iteration 3：默认开启录制
    assert app.experiment.enabled is True
    assert app.experiment.root == "data/experiment"
    assert app.experiment.log_to_stdout is True


def test_appconfig_full_yaml_load():
    path = _write_yaml(
        """\
env:
  use_gui: false
  camera_resolution: [320, 240]
vla:
  backend: mock
  max_steps: 50
llm:
  api_key: "sk-test"
  model: "MiniMax-M3"
  base_url: "https://api.minimax.chat/v1"
  max_tokens: 2048
explore:
  enabled: false
robot:
  urdf_path: "custom/panda.urdf"
  arm_joint_indices: [0, 1, 2, 3, 4, 5, 6]
  ee_link_index: 11
task:
  default_user_goal: "把机械臂移到红色方块上方"
agent:
  max_react_rounds: 5
  max_tool_calls: 3
experiment:
  enabled: false
"""
    )
    try:
        cfg = load_config(path)
        assert isinstance(cfg.robot, RobotConfig)
        assert cfg.robot.urdf_path == "custom/panda.urdf"
        assert isinstance(cfg.task, TaskConfig)
        assert cfg.task.default_user_goal == "把机械臂移到红色方块上方"
        assert isinstance(cfg.agent, AgentConfig)
        assert cfg.agent.max_react_rounds == 5
        assert isinstance(cfg.experiment, ExperimentConfig)
        assert cfg.experiment.enabled is False
    finally:
        os.unlink(path)


def test_appconfig_minimal_yaml_load():
    """仅有老四节时，新增四节使用默认值。"""
    path = _write_yaml(
        """\
env:
  use_gui: false
vla:
  backend: mock
llm:
  api_key: "k"
explore:
  enabled: false
"""
    )
    try:
        cfg = load_config(path)
        assert isinstance(cfg.robot, RobotConfig)
        assert cfg.robot.urdf_path == "franka_panda/panda.urdf"
        assert isinstance(cfg.task, TaskConfig)
        assert cfg.task.default_user_goal == "把机械臂移到红色方块上方"
        assert isinstance(cfg.agent, AgentConfig)
        assert cfg.agent.max_react_rounds == 5
        assert isinstance(cfg.experiment, ExperimentConfig)
        # Iteration 3：默认开启录制
        assert cfg.experiment.enabled is True
        assert cfg.experiment.root == "data/experiment"
        assert cfg.experiment.log_to_stdout is True
    finally:
        os.unlink(path)


from dataclasses import dataclass  # noqa: E402 (用于 NoDefault 测试)
