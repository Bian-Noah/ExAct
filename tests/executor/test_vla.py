"""executor/model 单元测试。

覆盖 BaseVLA 抽象契约 + MockVLA 行为（字段范围、可复现性、忽略 image）。

注：BaseVLA / MockVLA 已迁移到 executor.model 子包下，本文件从
`executor.model.base` 与 `executor.model.mock.mock_vla` 导入，并保留从
`executor` 的 re-export 入口，确保兼容性。
"""

import inspect

import numpy as np
import pytest

from env.base import Action7D, ActionSpec
from executor import BaseVLA as BaseVLA_re
from executor import MockVLA as MockVLA_re
from executor.model.base import BaseVLA, VLAOutput
from executor.model.mock.mock_vla import JointMockVLA, MockVLA


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
    """子类实现 predict + output_spec 后可实例化且 predict 返回 VLAOutput。"""
    class GoodVLA(BaseVLA):
        @property
        def output_spec(self):
            return ActionSpec("task", ("dx", "dy", "dz", "drx", "dry", "drz", "gripper"))

        def predict(self, image, instruction):
            return VLAOutput(
                values=Action7D(0, 0, 0, 0, 0, 0, 0.5),
                spec=self.output_spec,
            )

    v = GoodVLA()
    result = v.predict(None, "test")
    assert isinstance(result, VLAOutput)
    assert isinstance(result.values, Action7D)


# ========== MockVLA 行为测试 ==========


def test_mockvla_can_instantiate():
    """MockVLA 可实例化。"""
    MockVLA(seed=0)


def test_mockvla_predict_returns_vlaoutput():
    """predict 返回 VLAOutput 且 values 是 Action7D。"""
    v = MockVLA(seed=0)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    result = v.predict(image, "move")
    assert isinstance(result, VLAOutput)
    assert isinstance(result.values, Action7D)


def test_mockvla_field_ranges():
    """位移在 [-0.02, 0.02]，旋转在 [-0.05, 0.05]，gripper=0.5。"""
    v = MockVLA(seed=0)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    for i in range(10):
        result = v.predict(image, f"instruction_{i}").values
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
    r1 = v1.predict(image, "move to red").values
    r2 = v2.predict(image, "move to red").values
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
    outputs = [v.predict(image, ins).values for ins in instructions]
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
    assert (
        img_none_result.values
        == img_random_result.values
        == img_empty_result.values
    )


# ========== re-export 兼容性测试 ==========


def test_mockvla_output_spec():
    """MockVLA.output_spec → task 7 维 / gripper_index=6。"""
    v = MockVLA(seed=0)
    spec = v.output_spec
    assert spec.space == "task"
    assert spec.dim == 7
    assert spec.gripper_index == 6


def test_lerobot_vla_output_spec():
    """LeRobotVLA.output_spec → joint 空间，dim 由 action_dim 决定（构造不加载）。

    lerobot_vla 顶层 import torch，M4 无 torch 环境跳过（阶段二 RTX4060 验证）。
    """
    pytest.importorskip("torch")
    from executor.model.lerobot.lerobot_vla import LeRobotVLA

    v = LeRobotVLA(model_path="x", action_dim=14)
    spec = v.output_spec
    assert spec.space == "joint"
    assert spec.dim == 14


def test_executor_reexports_basetypes():
    """executor.__init__ 应 re-export BaseVLA / MockVLA / JointMockVLA，便于老代码兼容。"""
    assert BaseVLA_re is BaseVLA
    assert MockVLA_re is MockVLA
    assert inspect.isabstract(BaseVLA_re) is True
    assert issubclass(MockVLA_re, BaseVLA_re)
    # JointMockVLA 也从 executor 直接 re-export（与 MockVLA 对称）
    from executor import JointMockVLA as JointMockVLA_re
    assert JointMockVLA_re is JointMockVLA
    assert issubclass(JointMockVLA_re, BaseVLA_re)


# ========== JointMockVLA 行为测试 ==========


def test_jointmockvla_can_instantiate():
    """JointMockVLA 可实例化。"""
    JointMockVLA(seed=0)


def test_jointmockvla_predict_returns_vlaoutput():
    """predict 返回 VLAOutput 且 values 是 np.ndarray 长度 6。"""
    v = JointMockVLA(seed=0)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    result = v.predict(image, "move")
    assert isinstance(result, VLAOutput)
    assert isinstance(result.values, np.ndarray)
    assert result.values.shape == (6,)


def test_jointmockvla_output_spec():
    """JointMockVLA.output_spec → joint 6 维（与 so101 input_spec 等维）。"""
    v = JointMockVLA(seed=0)
    spec = v.output_spec
    assert spec.space == "joint"
    assert spec.dim == 6
    # so101 的 input_spec 用 6 个 "joint" 组件名（无 gripper 维）
    assert spec.gripper_index is None


def test_jointmockvla_field_ranges():
    """关节增量在 [-0.1, 0.1] rad，最后一维 gripper=0.5。"""
    v = JointMockVLA(seed=0)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    for i in range(10):
        result = v.predict(image, f"instruction_{i}").values
        for j in range(5):
            assert -0.1 <= result[j] <= 0.1, f"joint[{j}] out of range"
        assert result[5] == 0.5


def test_jointmockvla_reproducible_same_seed_and_instruction():
    """相同 seed + 相同 instruction 输出完全一致。"""
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    v1 = JointMockVLA(seed=42)
    v2 = JointMockVLA(seed=42)
    r1 = v1.predict(image, "move to red").values
    r2 = v2.predict(image, "move to red").values
    assert np.array_equal(r1, r2)
    for j in range(6):
        assert r1[j] == r2[j]


def test_jointmockvla_different_instruction_different_output():
    """不同 instruction 至少有一组输出不同。"""
    v = JointMockVLA(seed=0)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    instructions = ["move_a", "move_b", "move_c", "move_d", "move_e"]
    outputs = [v.predict(image, ins).values for ins in instructions]
    found_diff = False
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            if not np.array_equal(outputs[i], outputs[j]):
                found_diff = True
                break
        if found_diff:
            break
    assert found_diff


def test_jointmockvla_ignores_image():
    """相同 seed + instruction 下，image 变化不影响输出。"""
    v = JointMockVLA(seed=0)
    img_none = v.predict(None, "move").values
    img_random = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
    img_random_r = v.predict(img_random, "move").values
    assert np.array_equal(img_none, img_random_r)


def test_jointmockvla_passes_joint_to_joint_adapter():
    """JointMockVLA(joint,6) + so101(joint,6) → joint_to_joint_transform 等维直通。"""
    from utils.adapter import get_adapter

    v = JointMockVLA(seed=0)
    env_spec = ActionSpec("joint", ("joint",) * 6)
    adapter = get_adapter(v.output_spec, env_spec)
    assert adapter.__name__ == "joint_to_joint_transform"

    class _FakeEnv:
        @property
        def input_spec(self):
            return env_spec

    out = v.predict(None, "move forward").values
    mapped = adapter(out, _FakeEnv())
    assert mapped.shape == (6,)
    assert np.allclose(mapped, out)


# ========== 新结构验证 ==========


def test_subclass_must_implement_predict():
    """未实现 predict 的子类无法实例化。"""

    class BadVLA(BaseVLA):
        pass

    with pytest.raises(TypeError):
        BadVLA()


# ========== _build_raw_obs 顶层 task 键回归测试 ==========

def test_lerobot_vla_build_raw_obs_has_top_level_task_key():
    """_build_raw_obs 必须把 instruction 同时放到顶层 "task" 键。

    背景：SmolVLA 的 TokenizerProcessorStep 通过 complementary_data["task"]
    取任务说明；batch_to_transition 只抽 batch 顶层名为 "task" 的键。
    若缺失，predict 时会抛 KeyError("task")，action 工具会报
    "工具执行出错: 'task'"。本测试防止再次回归。
    """
    pytest.importorskip("torch")
    from executor.model.lerobot.lerobot_vla import LeRobotVLA

    v = LeRobotVLA(model_path="x", action_dim=6, device="cpu")
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    raw_obs = v._build_raw_obs(image, "pick up the yellow cube")
    assert "task" in raw_obs, "顶层缺少 'task' 键，SmolVLA tokenizer 会 KeyError"
    assert raw_obs["task"] == "pick up the yellow cube"
    # observation.language_instruction 也保留（其他后端如 OpenVLA 需要）
    assert raw_obs["observation.language_instruction"] == "pick up the yellow cube"


def test_lerobot_vla_build_raw_obs_task_survives_batch_to_transition():
    """raw_obs 经 lerobot batch_to_transition 后，task 进入 complementary_data。

    端到端验证：直接走 lerobot 官方的 batch_to_transition，确保 complementary_data
    含 task（即 TokenizerProcessorStep.get_task 不会再抛 KeyError）。
    """
    pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    from executor.model.lerobot.lerobot_vla import LeRobotVLA
    from lerobot.processor.converters import batch_to_transition
    from lerobot.processor.pipeline import TransitionKey

    v = LeRobotVLA(model_path="x", action_dim=6, device="cpu")
    raw_obs = v._build_raw_obs(np.zeros((8, 8, 3), dtype=np.uint8), "grasp cube")
    transition = batch_to_transition(raw_obs)
    complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA)
    assert complementary is not None, "complementary_data 不应为 None"
    assert complementary.get("task") == "grasp cube", (
        f"complementary_data 缺 'task' 键，实际={complementary!r}"
    )
