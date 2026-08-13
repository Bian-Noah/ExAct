from __future__ import annotations

import os

import src.config as config_module
from src.config import (
    AgentConfig,
    AppConfig,
    EnvConfig,
    ExperimentConfig,
    ExploreConfig,
    LLMConfig,
    RobotConfig,
    TaskConfig,
    VLAConfig,
    load_config,
)


def test_config_all_exports_present():
    expected = {
        "AgentConfig",
        "AppConfig",
        "EnvConfig",
        "ExperimentConfig",
        "ExploreConfig",
        "LLMConfig",
        "RobotConfig",
        "TaskConfig",
        "VLAConfig",
        "load_config",
    }
    assert set(config_module.__all__) == expected
    for name in expected:
        assert hasattr(config_module, name), f"src.config 缺少导出：{name}"


def test_imported_dataclasses_are_correct_types():
    # 确保导入的 dataclass 可以直接实例化
    assert EnvConfig().use_gui is False  # iter2: use_gui deprecated, 默认由 mode=direct 推导为 False
    assert EnvConfig().mode == "direct"
    assert EnvConfig().renderer == "auto"
    assert VLAConfig().backend == "mock"
    assert LLMConfig().model == "MiniMax-M3"
    assert ExploreConfig().enabled is False
    assert RobotConfig().urdf_path == "franka_panda/panda.urdf"
    assert TaskConfig().default_user_goal == "把机械臂移到红色方块上方"
    assert AgentConfig().max_react_rounds == 5
    # Iteration 3：ExperimentConfig 默认开启录制
    assert ExperimentConfig().enabled is True
    assert ExperimentConfig().root == "data/experiment"
    assert ExperimentConfig().log_to_stdout is True
    # AppConfig 需要显式传入子配置实例
    app = AppConfig(
        env=EnvConfig(),
        vla=VLAConfig(),
        llm=LLMConfig(),
        explore=ExploreConfig(),
        robot=RobotConfig(),
        task=TaskConfig(),
        agent=AgentConfig(),
        experiment=ExperimentConfig(),
    )
    assert isinstance(app.robot, RobotConfig)
    assert isinstance(app.task, TaskConfig)
    assert isinstance(app.agent, AgentConfig)
    assert isinstance(app.experiment, ExperimentConfig)


def test_load_config_via_public_import():
    """通过 src.config.load_config 公共接口调用，确保其与 loader.load_config 行为一致。"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_path = os.path.join(project_root, "configs", "default.yaml")
    cfg = load_config(default_path)
    assert isinstance(cfg, AppConfig)
    assert cfg.vla.backend == "mock"
    assert cfg.env.camera_resolution == (640, 480)
    # 新增节也都被加载
    assert isinstance(cfg.robot, RobotConfig)
    assert isinstance(cfg.task, TaskConfig)
    assert isinstance(cfg.agent, AgentConfig)
    assert isinstance(cfg.experiment, ExperimentConfig)
