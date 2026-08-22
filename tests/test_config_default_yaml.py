from __future__ import annotations

import os

from src.config.loader import (
    AgentConfig,
    AppConfig,
    ExperimentConfig,
    RobotConfig,
    TaskConfig,
    load_config,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_YAML_PATH = os.path.join(PROJECT_ROOT, "configs", "default.yaml")


def test_configs_default_yaml_exists():
    assert os.path.isfile(DEFAULT_YAML_PATH), (
        f"默认配置文件不存在：{DEFAULT_YAML_PATH}"
    )


def test_functional_scenario_a_load_default_yaml():
    """功能场景 A：加载磁盘上真实的 configs/default.yaml 并校验核心字段。"""
    cfg = load_config(DEFAULT_YAML_PATH)

    assert isinstance(cfg, AppConfig)
    # env (iter2-renderer-env-mode: use_gui 由 mode 替代，默认 mode=direct / renderer=auto)
    assert cfg.env.use_gui is False
    assert cfg.env.mode == "direct"
    assert cfg.env.renderer == "auto"
    assert cfg.env.camera_resolution == (640, 480)
    # vla
    assert cfg.vla.backend == "mock"
    assert cfg.vla.model_path is None
    # iter11-reset-multicam:max_steps 已移除
    assert not hasattr(cfg.vla, "max_steps")
    # llm（api_key 只要是占位符或空均合法）
    assert cfg.llm.api_key in {"YOUR_API_KEY_HERE", ""} or len(cfg.llm.api_key) > 0
    assert cfg.llm.model == "MiniMax-M3"
    assert cfg.llm.base_url == "https://api.minimax.chat/v1"
    assert cfg.llm.max_tokens == 2048
    # explore
    assert cfg.explore.enabled is False
    # robot
    assert isinstance(cfg.robot, RobotConfig)
    assert cfg.robot.type == "panda"
    assert cfg.robot.urdf_path == "franka_panda/panda.urdf"
    assert cfg.robot.panda.arm_joint_indices == (0, 1, 2, 3, 4, 5, 6)
    assert cfg.robot.panda.ee_link_index == 11
    # task
    assert isinstance(cfg.task, TaskConfig)
    assert cfg.task.default_user_goal == "把机械臂移到红色方块上方"
    assert len(cfg.task.objects) == 1
    # agent
    assert isinstance(cfg.agent, AgentConfig)
    assert cfg.agent.max_react_rounds == 5
    assert cfg.agent.max_tool_calls == 3
    # experiment（Iteration 3：默认开启录制）
    assert isinstance(cfg.experiment, ExperimentConfig)
    assert cfg.experiment.enabled is True
    assert cfg.experiment.root == "data/experiment"
    assert cfg.experiment.log_to_stdout is True
