"""executor/vla.py 单元测试。

覆盖 BaseVLA 抽象契约 + MockVLA 行为（字段范围、可复现性、忽略 image）。
"""

import inspect

import numpy as np
import pytest

from env.base import Action7D
from executor.vla import BaseVLA, MockVLA


# ========== BaseVLA 抽象契约测试 ==========


def test_basevla_is_abstract():
    """BaseVLA 应为抽象类。"""
    assert inspect.isabstract(BaseVLA) is True


def test_basevla_cannot_instantiate():
    """BaseVLA 不可直接实例化。"""
    with pytest.raises(TypeError):
        BaseVLA()


def test_basevla_subclass_missing_predict_cannot_instantiate():
    """子类不实现 predict 不可实例化。"""
    class BadVLA(BaseVLA):
        pass

    with pytest.raises(TypeError):
        BadVLA()


def test_basevla_full_subclass_can_instantiate():
    """子类实现 predict 后可实例化且 predict 返回 Action7D。"""
    class GoodVLA(BaseVLA):
        def predict(self, image, instruction):
            return Action7D(0, 0, 0, 0, 0, 0, 0.5)

    v = GoodVLA()
    result = v.predict(None, "test")
    assert isinstance(result, Action7D)


# ========== MockVLA 行为测试 ==========


def test_mockvla_can_instantiate():
    """MockVLA 可实例化。"""
    MockVLA(seed=0)


def test_mockvla_predict_returns_action7d():
    """predict 返回 Action7D 实例。"""
    v = MockVLA(seed=0)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    result = v.predict(image, "move")
    assert isinstance(result, Action7D)


def test_mockvla_field_ranges():
    """位移在 [-0.02, 0.02]，旋转在 [-0.05, 0.05]，gripper=0.5。"""
    v = MockVLA(seed=0)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    for i in range(10):
        result = v.predict(image, f"instruction_{i}")
        assert -0.02 <= result.dx <= 0.02
        assert -0.02 <= result.dy <= 0.02
        assert -0.02 <= result.dz <= 0.02
        assert -0.05 <= result.drx <= 0.05
        assert -0.05 <= result.dry <= 0.05
        assert -0.05 <= result.drz <= 0.05
        assert result.gripper == 0.5


def test_mockvla_reproducible_same_seed_and_instruction():
    """相同 seed + 相同 instruction 输出完全一致。"""
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    v1 = MockVLA(seed=42)
    v2 = MockVLA(seed=42)
    r1 = v1.predict(image, "move to red")
    r2 = v2.predict(image, "move to red")
    assert r1 == r2
    # 7 个字段逐一断言
    assert r1.dx == r2.dx
    assert r1.dy == r2.dy
    assert r1.dz == r2.dz
    assert r1.drx == r2.drx
    assert r1.dry == r2.dry
    assert r1.drz == r2.drz
    assert r1.gripper == r2.gripper


def test_mockvla_different_instruction_different_output():
    """不同 instruction 至少有一组输出不同。"""
    v = MockVLA(seed=0)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    instructions = ["move_a", "move_b", "move_c", "move_d", "move_e"]
    outputs = [v.predict(image, ins) for ins in instructions]
    # 至少存在一组不同
    found_diff = False
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            if outputs[i] != outputs[j]:
                found_diff = True
                break
        if found_diff:
            break
    assert found_diff, "不同 instruction 应产生不同输出"


def test_mockvla_ignores_image():
    """相同 seed + instruction 下，image 变化不影响输出。"""
    v = MockVLA(seed=0)
    img_none_result = v.predict(None, "move")
    img_random = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
    img_random_result = v.predict(img_random, "move")
    img_empty = np.zeros((0, 0, 3), dtype=np.uint8)
    img_empty_result = v.predict(img_empty, "move")
    assert img_none_result == img_random_result == img_empty_result
