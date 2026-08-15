"""env.base 单元测试（Action7D + BaseEnv ABC）。"""

import inspect

import numpy as np
import pytest

from env.base import BaseEnv, Action7D, ActionSpec


# ============================================================
# Action7D 测试
# ============================================================

def test_action7d_fields_correct_order():
    """7 个字段值按顺序正确。"""
    a = Action7D(0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.8)
    assert a.dx == 0.1
    assert a.dy == 0.2
    assert a.dz == 0.3
    assert a.drx == 0.01
    assert a.dry == 0.02
    assert a.drz == 0.03
    assert a.gripper == 0.8


def test_action7d_unpack():
    """解构 7 个变量。"""
    a = Action7D(1, 2, 3, 4, 5, 6, 7)
    dx, dy, dz, drx, dry, drz, g = a
    assert dx == 1
    assert dy == 2
    assert dz == 3
    assert drx == 4
    assert dry == 5
    assert drz == 6
    assert g == 7


def test_action7d_len_and_iter():
    """len=7，list 解构正确。"""
    a = Action7D(1, 2, 3, 4, 5, 6, 7)
    assert len(a) == 7
    assert list(a) == [1, 2, 3, 4, 5, 6, 7]


def test_action7d_is_namedtuple_immutable():
    """NamedTuple 不可变，赋值抛 AttributeError。"""
    a = Action7D(1, 2, 3, 4, 5, 6, 7)
    with pytest.raises(AttributeError):
        a.dx = 99


# ============================================================
# BaseEnv ABC 测试
# ============================================================

def test_baseenv_is_abstract():
    """BaseEnv 是抽象类。"""
    assert inspect.isabstract(BaseEnv)


def test_baseenv_cannot_instantiate():
    """BaseEnv 不能直接实例化。"""
    with pytest.raises(TypeError):
        BaseEnv()


def test_baseenv_subclass_missing_methods_cannot_instantiate():
    """空子类不能实例化（5 个抽象方法未实现）。"""
    class Bad(BaseEnv):
        pass

    with pytest.raises(TypeError):
        Bad()


def test_baseenv_subclass_partial_methods_cannot_instantiate():
    """只实现 2 个方法的子类不能实例化。"""
    class Partial(BaseEnv):
        def reset(self, task_spec, seed=0):
            return {}

        def step(self, action):
            return {}, 0.0, False, {}

    with pytest.raises(TypeError):
        Partial()


def test_baseenv_full_subclass_can_instantiate_and_call():
    """全实现子类可实例化，5 个方法返回类型符合契约。"""
    class Good(BaseEnv):
        def reset(self, task_spec, seed=0):
            return {"rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                    "object_info": [], "ee_pos": (0.0, 0.0, 0.0),
                    "state_desc": "test"}

        def step(self, action):
            return self.get_obs(), 0.0, False, {}

        def render(self):
            return np.zeros((4, 4, 3), dtype=np.uint8)

        def get_obs(self, include_rgb: bool = True):
            return {"rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                    "object_info": [], "ee_pos": (0.0, 0.0, 0.0),
                    "state_desc": "test"}

        def close(self):
            pass

        @property
        def input_spec(self):
            return ActionSpec("task", ("dx", "dy", "dz", "drx", "dry", "drz", "gripper"))

    env = Good()

    # reset 返回 dict 含 4 个键
    obs = env.reset({})
    assert isinstance(obs, dict)
    assert set(obs.keys()) == {"rgb", "object_info", "ee_pos", "state_desc"}

    # step 返回 4-tuple
    result = env.step(Action7D(1, 2, 3, 4, 5, 6, 7))
    assert isinstance(result, tuple)
    assert len(result) == 4

    # render 返回 ndarray (H,W,3) uint8
    img = env.render()
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.dtype == np.uint8

    # get_obs 返回 dict
    obs2 = env.get_obs()
    assert isinstance(obs2, dict)

    # close 返回 None
    assert env.close() is None
