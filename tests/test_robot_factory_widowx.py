"""WidowX 机器人配置与 build_robot 工厂分派单元测试（add-widowx-robot）。

覆盖两类：
1. 配置层：WidowxRobotConfig 默认值 / RobotConfig 嵌套 widowx 字段 /
   _from_dict 解析（含/不含 widowx 特化块）；
2. 工厂层：build_robot(type="widowx") 分派到 WidowxRobot、未知类型错误
   消息列出 widowx、默认 type=panda 回归，以及 WidowxRobot 行为。

WidowxRobot.__init__ 会调用 ensure_urdf()（本地缺失时可能触发网络下载），
因此构造机器人时统一 monkeypatch 下载器（env.robot.widowx.widowx_robot
模块命名空间中的 ensure_urdf_downloaded），或让 urdf_local_path 指向已存在
文件，保证测试零网络请求。
运行方式：cd 项目根目录 && PYTHONPATH=src pytest tests/test_robot_factory_widowx.py
"""

import pytest

import env.robot.widowx.urdf_downloader as urdf_downloader
import env.robot.widowx.widowx_robot as wr

from config.loader import RobotConfig, WidowxRobotConfig, _from_dict
from env.robot import build_robot
from env.robot.widowx import WidowxRobot


# ---------- 1. WidowxRobotConfig / RobotConfig 默认值 ----------

def test_widowx_config_defaults():
    """WidowxRobotConfig() 默认值：5 臂 (0-4) + ee=11 + 2 夹爪 (9,10) + URDF 字段。"""
    cfg = WidowxRobotConfig()
    assert cfg.arm_joint_indices == (0, 1, 2, 3, 4)
    assert cfg.ee_link_index == 11
    assert cfg.gripper_joint_indices == (9, 10)
    assert "wx250.urdf" in cfg.urdf_url
    assert cfg.urdf_local_path == "robot/widowx/wx250.urdf"


def test_robot_config_has_widowx_field():
    """RobotConfig() 嵌套 widowx 字段为 WidowxRobotConfig 实例且默认值正确。"""
    cfg = RobotConfig()
    assert isinstance(cfg.widowx, WidowxRobotConfig)
    assert cfg.widowx.arm_joint_indices == (0, 1, 2, 3, 4)
    assert cfg.widowx.ee_link_index == 11
    assert cfg.widowx.gripper_joint_indices == (9, 10)
    assert cfg.widowx.urdf_url.endswith("wx250.urdf")
    assert cfg.widowx.urdf_local_path == "robot/widowx/wx250.urdf"


def test_robot_config_default_type_is_panda_unaffected():
    """回归：新增 widowx 嵌套字段不改变 RobotConfig 默认 type（仍为 panda）。"""
    cfg = RobotConfig()
    assert cfg.type == "panda"
    assert cfg.urdf_path == "franka_panda/panda.urdf"


# ---------- 2. _from_dict 解析 widowx 特化块 ----------

def test_from_dict_widowx_block():
    """含 widowx 特化块的 dict → 解析进 RobotConfig，list 转 tuple。"""
    cfg = _from_dict(
        {
            "type": "widowx",
            "widowx": {
                "arm_joint_indices": [0, 1, 2, 3, 4],
                "ee_link_index": 11,
                "gripper_joint_indices": [9, 10],
                "urdf_url": "http://example.com/wx250.urdf",
                "urdf_local_path": "robot/widowx/wx250.urdf",
            },
        },
        RobotConfig,
    )
    assert cfg.type == "widowx"
    assert cfg.widowx.arm_joint_indices == (0, 1, 2, 3, 4)
    assert isinstance(cfg.widowx.arm_joint_indices, tuple)
    assert cfg.widowx.ee_link_index == 11
    assert cfg.widowx.gripper_joint_indices == (9, 10)
    assert cfg.widowx.urdf_url == "http://example.com/wx250.urdf"
    assert cfg.widowx.urdf_local_path == "robot/widowx/wx250.urdf"


def test_from_dict_without_widowx_block():
    """不含 widowx 块的 dict → widowx 走默认值兜底，不抛错。"""
    cfg = _from_dict({"type": "widowx"}, RobotConfig)
    assert isinstance(cfg.widowx, WidowxRobotConfig)
    assert cfg.widowx.arm_joint_indices == (0, 1, 2, 3, 4)
    assert cfg.widowx.ee_link_index == 11
    assert cfg.widowx.gripper_joint_indices == (9, 10)
    assert cfg.widowx.urdf_local_path == "robot/widowx/wx250.urdf"


# ---------- 3. build_robot 工厂分派 ----------

def test_build_robot_widowx(monkeypatch):
    """type="widowx" → WidowxRobot 实例，且下载器按配置路径被调用。"""
    calls = []

    def _fake_ensure_urdf(*args, **kwargs):
        calls.append((args, kwargs))
        return True

    monkeypatch.setattr(wr, "ensure_urdf_downloaded", _fake_ensure_urdf)
    monkeypatch.setattr(wr, "ensure_assets_downloaded", lambda *a, **k: 0)
    robot = build_robot(RobotConfig(type="widowx"))
    assert isinstance(robot, WidowxRobot)
    assert robot.arm_joint_indices == (0, 1, 2, 3, 4)
    assert robot.ee_link_index == 11
    assert robot.gripper_joint_indices == (9, 10)
    # ensure_urdf() 以 (local_path, url) 位置参数调用下载器
    assert calls
    assert calls[0][0][0] == "robot/widowx/wx250.urdf"


def test_build_robot_widowx_uses_custom_config(monkeypatch):
    """自定义 widowx 特化配置 → 机器人属性随配置。"""
    monkeypatch.setattr(wr, "ensure_urdf_downloaded", lambda *a, **k: True)
    monkeypatch.setattr(wr, "ensure_assets_downloaded", lambda *a, **k: 0)
    cfg = RobotConfig(
        type="widowx",
        widowx=WidowxRobotConfig(
            arm_joint_indices=(1, 2, 3, 4, 5),
            ee_link_index=8,
            gripper_joint_indices=(12, 13),
            urdf_url="http://example.com/wx250.urdf",
            urdf_local_path="robot/widowx/wx250.urdf",
        ),
    )
    robot = build_robot(cfg)
    assert isinstance(robot, WidowxRobot)
    assert robot.arm_joint_indices == (1, 2, 3, 4, 5)
    assert robot.ee_link_index == 8
    assert robot.gripper_joint_indices == (12, 13)
    assert robot.urdf_url == "http://example.com/wx250.urdf"
    assert robot.urdf_local_path == "robot/widowx/wx250.urdf"


def test_build_robot_unknown_type_lists_widowx():
    """type="ghost" → ValueError 且消息列出 widowx 可选类型。"""
    with pytest.raises(ValueError) as exc_info:
        build_robot(RobotConfig(type="ghost"))
    assert "widowx" in str(exc_info.value)
    assert "panda" in str(exc_info.value)


# ---------- 4. WidowxRobot 行为 ----------

def test_widowx_robot_input_spec(monkeypatch):
    """input_spec → task 空间 7 维 / gripper_index=6。"""
    monkeypatch.setattr(wr, "ensure_urdf_downloaded", lambda *a, **k: True)
    monkeypatch.setattr(wr, "ensure_assets_downloaded", lambda *a, **k: 0)
    robot = WidowxRobot(RobotConfig(type="widowx"))
    spec = robot.input_spec
    assert spec.space == "task"
    assert spec.dim == 7
    assert spec.gripper_index == 6


def test_widowx_robot_home_joint_positions(monkeypatch):
    """home_joint_positions → 5 维全零。"""
    monkeypatch.setattr(wr, "ensure_urdf_downloaded", lambda *a, **k: True)
    monkeypatch.setattr(wr, "ensure_assets_downloaded", lambda *a, **k: 0)
    robot = WidowxRobot(RobotConfig(type="widowx"))
    assert robot.home_joint_positions() == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_widowx_robot_skips_download_when_local_exists(tmp_path, monkeypatch):
    """urdf_local_path 指向已存在文件 → 走真实下载器且零网络请求。"""
    local_path = tmp_path / "wx250.urdf"
    local_path.write_bytes(b"<urdf>")

    def _should_not_call(*args, **kwargs):
        raise AssertionError("requests.get 不应被调用（本地 URDF 已存在）")

    monkeypatch.setattr(urdf_downloader.requests, "get", _should_not_call)
    cfg = RobotConfig(
        type="widowx",
        widowx=WidowxRobotConfig(urdf_local_path=str(local_path)),
    )
    robot = WidowxRobot(cfg)
    assert robot.urdf_local_path == str(local_path)
    assert local_path.read_bytes() == b"<urdf>"
