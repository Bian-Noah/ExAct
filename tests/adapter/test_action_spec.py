"""ActionSpec + BaseEnv 能力接口单元测试（robot-vla-adapter 任务 1）。"""

import dataclasses

import pytest

from env.base import ActionSpec, BaseEnv


# ========== ActionSpec 派生字段 ==========


def test_action_spec_task_7d_derives():
    """task 空间 7 维（含 gripper）→ dim=7, gripper_index=6。"""
    spec = ActionSpec("task", ("dx", "dy", "dz", "drx", "dry", "drz", "gripper"))
    assert spec.space == "task"
    assert spec.dim == 7
    assert spec.gripper_index == 6


def test_action_spec_joint_6d_no_gripper():
    """joint 空间 6 维无 gripper → dim=6, gripper_index=None。"""
    spec = ActionSpec("joint", ("joint",) * 6)
    assert spec.space == "joint"
    assert spec.dim == 6
    assert spec.gripper_index is None


def test_action_spec_gripper_first_dim():
    """gripper 出现在非末尾维时索引正确。"""
    spec = ActionSpec("joint", ("gripper", "j1", "j2", "j3"))
    assert spec.dim == 4
    assert spec.gripper_index == 0


# ========== ActionSpec 校验 ==========


def test_action_spec_invalid_space():
    """非法 space 抛 ValueError。"""
    with pytest.raises(ValueError):
        ActionSpec("spatial", ("dx",))


def test_action_spec_duplicate_component():
    """components 含重复元素抛 ValueError。"""
    with pytest.raises(ValueError):
        ActionSpec("joint", ("j", "j", "j"))


def test_action_spec_empty_component():
    """components 含空串抛 ValueError。"""
    with pytest.raises(ValueError):
        ActionSpec("task", ("dx", ""))


def test_action_spec_frozen():
    """frozen dataclass：不可修改字段。"""
    spec = ActionSpec("task", ("dx", "dy", "dz", "drx", "dry", "drz", "gripper"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.space = "joint"


def test_action_spec_hashable():
    """frozen + tuple → 可哈希，可用于注册表 dict key。"""
    spec = ActionSpec("joint", ("joint",) * 6)
    assert hash(spec) == hash(ActionSpec("joint", ("joint",) * 6))


# ========== BaseEnv 能力接口（ik/fk 默认 NotImplementedError） ==========


def _make_minimal_env():
    """构造实现所有抽象方法的最小 BaseEnv 子类（含 input_spec）。"""
    class MinimalEnv(BaseEnv):
        def reset(self, task_spec, seed=0):
            return {}

        def step(self, action):
            return {}, 0.0, False, {}

        def render(self):
            return None

        def get_obs(self, include_rgb=True):
            return {}

        def close(self):
            pass

        @property
        def input_spec(self):
            return ActionSpec("task", ("dx", "dy", "dz", "drx", "dry", "drz", "gripper"))

    return MinimalEnv()


def test_baseenv_ik_default_not_implemented():
    """未实现 ik() 时调用抛 NotImplementedError。"""
    env = _make_minimal_env()
    with pytest.raises(NotImplementedError):
        env.ik((0.5, 0.0, 0.4))


def test_baseenv_fk_default_not_implemented():
    """未实现 fk() 时调用抛 NotImplementedError。"""
    env = _make_minimal_env()
    with pytest.raises(NotImplementedError):
        env.fk((0.0,) * 7)
