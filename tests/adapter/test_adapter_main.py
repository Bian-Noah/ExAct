"""get_adapter 注册表分派单元测试（robot-vla-adapter 任务 3）。"""

import pytest

from env.base import ActionSpec
from utils.adapter import get_adapter
from utils.adapter.main import AdapterNotFoundError
from utils.adapter.adapters.identity import identity_transform
from utils.adapter.adapters.joint_to_joint import joint_to_joint_transform
from utils.adapter.adapters.task_to_joint import task_to_joint_transform


def _task_7d():
    return ActionSpec("task", ("dx", "dy", "dz", "drx", "dry", "drz", "gripper"))


def _joint_6d():
    return ActionSpec("joint", ("joint",) * 6)


def test_get_adapter_task_task():
    """(task, task) → identity_transform。"""
    adapter = get_adapter(_task_7d(), _task_7d())
    assert adapter is identity_transform


def test_get_adapter_joint_joint():
    """(joint, joint) → joint_to_joint_transform。"""
    adapter = get_adapter(_joint_6d(), _joint_6d())
    assert adapter is joint_to_joint_transform


def test_get_adapter_task_joint():
    """(task, joint) → task_to_joint_transform。"""
    adapter = get_adapter(_task_7d(), _joint_6d())
    assert adapter is task_to_joint_transform


def test_get_adapter_unknown_raises():
    """(joint, task) 未注册组合 → AdapterNotFoundError 含 spec 描述。"""
    with pytest.raises(AdapterNotFoundError) as exc_info:
        get_adapter(_joint_6d(), _task_7d())
    msg = str(exc_info.value)
    assert "joint" in msg and "task" in msg


def test_adapter_not_found_error_attributes():
    """AdapterNotFoundError 保留 vla_spec / env_spec 供排查。"""
    vla = _joint_6d()
    env = _task_7d()
    with pytest.raises(AdapterNotFoundError) as exc_info:
        get_adapter(vla, env)
    assert exc_info.value.vla_spec is vla
    assert exc_info.value.env_spec is env
