"""executor/__init__.py 主循环单元测试（Iteration 10 chunk 契约）。

使用 FakeEnv（内存 mock）替代 PyBulletEnv，验证 Executor 按 VLA 输出的
整 chunk 执行：循环边界 = len(actions)，中间不调 check_done。
"""

import numpy as np
import pytest

from env.base import ActionSpec, BaseEnv
from executor import Executor, ExecResult, MockVLA
from executor.model.base import BaseVLA, VLAOutput


def _task_spec() -> ActionSpec:
    return ActionSpec("task", ("dx", "dy", "dz", "drx", "dry", "drz", "gripper"))


# ========== FakeEnv 测试辅助类 ==========


class FakeEnv(BaseEnv):
    """内存中的 BaseEnv 子类，统计 env.step 调用次数。"""

    def __init__(self):
        self._current_obs = {
            "rgb": {"cam1": np.zeros((10, 10, 3), dtype=np.uint8)},
            "ee_pos": (0.0, 0.0, 0.0),
            "object_info": [],
            "state_desc": "",
        }
        self.step_count = 0

    def reset(self, task_spec=None, seed=0):
        self.step_count = 0
        return self._current_obs

    def step(self, action):
        self.step_count += 1
        return self._current_obs, 0.0, False, {}

    def render(self):
        return self._current_obs["rgb"]

    def get_obs(self, include_rgb: bool = True):
        return self._current_obs

    def close(self):
        pass

    @property
    def input_spec(self):
        return _task_spec()


class FakeEnvWithEePosControl(BaseEnv):
    """可控制 ee_pos 随步数推进的 FakeEnv。"""

    def __init__(self, satisfy_at_step=3, move_step=0.05):
        self._step_count = 0
        self._satisfy_at_step = satisfy_at_step
        self._move_step = move_step
        self._ee_pos = (0.0, 0.0, 0.0)
        self.step_count = 0

    def reset(self, task_spec=None, seed=0):
        self._step_count = 0
        self._ee_pos = (0.0, 0.0, 0.0)
        return self._make_obs()

    def step(self, action):
        self._step_count += 1
        self.step_count += 1
        if self._step_count < self._satisfy_at_step:
            self._ee_pos = (
                self._ee_pos[0] + self._move_step,
                self._ee_pos[1],
                self._ee_pos[2],
            )
        return self._make_obs(), 0.0, False, {}

    def render(self):
        return {"cam1": np.zeros((10, 10, 3), dtype=np.uint8)}

    def get_obs(self, include_rgb: bool = True):
        return self._make_obs()

    def close(self):
        pass

    @property
    def input_spec(self):
        return _task_spec()

    def _make_obs(self):
        return {
            "rgb": {"cam1": np.zeros((10, 10, 3), dtype=np.uint8)},
            "ee_pos": self._ee_pos,
            "object_info": [],
            "state_desc": "",
        }


class FakeEnvRaisingStep(BaseEnv):
    """step 时抛异常的 FakeEnv。"""

    def reset(self, task_spec=None, seed=0):
        return {
            "rgb": {"cam1": np.zeros((10, 10, 3), dtype=np.uint8)},
            "ee_pos": (0.0, 0.0, 0.0),
        }

    def step(self, action):
        raise RuntimeError("FakeEnv step error for testing")

    def render(self):
        return {"cam1": np.zeros((10, 10, 3), dtype=np.uint8)}

    def get_obs(self, include_rgb: bool = True):
        return {
            "rgb": {"cam1": np.zeros((10, 10, 3), dtype=np.uint8)},
            "ee_pos": (0.0, 0.0, 0.0),
        }

    def close(self):
        pass

    @property
    def input_spec(self):
        return _task_spec()


# ========== Iteration 10 chunk 契约：基本执行 ==========


def test_mock_vla_runs_chunk_of_1():
    """MockVLA 返回 (1, 7) → executor 跑 1 步（不是 max_steps 步）。"""
    env = FakeEnv()
    vla = MockVLA(seed=0)
    executor = Executor(vla)  # max_steps 兼容保留，不用于循环
    result = executor.run_action(env, "move", done_criteria="unknown")
    assert result.success is False
    assert result.steps == 1
    assert env.step_count == 1  # 真的只 step 了 1 次


def test_executor_runs_full_chunk_no_truncation():
    """MockVLA chunk=1 → executor 跑 1 步即使 max_steps=50 也不截断。"""
    env = FakeEnv()
    vla = MockVLA(seed=0)
    executor = Executor(vla)
    result = executor.run_action(env, "move", done_criteria="unknown")
    assert result.steps == 1
    assert env.step_count == 1


def test_check_done_called_once_after_chunk():
    """executor 跑完 chunk 后 check_done 只被调 1 次（事后报告）。"""
    import executor as executor_pkg
    real_check_done = executor_pkg.check_done
    call_count = [0]

    def spy_check_done(*args, **kwargs):
        call_count[0] += 1
        return real_check_done(*args, **kwargs)

    executor_pkg.check_done = spy_check_done
    try:
        env = FakeEnv()
        vla = MockVLA(seed=0)
        executor = Executor(vla)
        executor.run_action(env, "move", done_criteria="unknown")
        assert call_count[0] == 1
    finally:
        executor_pkg.check_done = real_check_done


def test_no_check_done_truncation():
    """Iteration 10：target_pos=None + 单步 0 位移不会被反向兜底截断。

    FakeEnv ee_pos 不变 → chunk=1 完整跑完（不被"位移 ≤ 0.01 判 done"截断）。
    """
    env = FakeEnv()
    vla = MockVLA(seed=0)
    executor = Executor(vla)
    result = executor.run_action(env, "stay", done_criteria="reached")  # 无 target_pos
    assert result.steps == 1  # chunk 跑完，不是被反向兜底截断成 0
    assert result.success is False  # target_pos 未提供 → False
    assert "target_pos 未提供" in result.message


def test_adapter_called_once_per_run_action():
    """adapter 只被调 1 次（一次性转换整 chunk）。"""
    env = FakeEnv()
    vla = MockVLA(seed=0)
    executor = Executor(vla)

    call_count = [0]
    received_chunks = []

    def spy_adapter(raw, env):
        call_count[0] += 1
        received_chunks.append(np.asarray(raw).copy())
        return raw  # 直通

    executor.run_action(env, "move", done_criteria="unknown", adapter=spy_adapter)
    assert call_count[0] == 1
    # adapter 收到的是整 chunk (1, 7)
    assert received_chunks[0].shape == (1, 7)


# ========== ExecResult 结构 ==========


def test_executor_result_field_types():
    """ExecResult 字段类型正确。"""
    env = FakeEnv()
    vla = MockVLA(seed=0)
    executor = Executor(vla)
    result = executor.run_action(env, "move", done_criteria="unknown")
    assert isinstance(result, ExecResult)
    assert isinstance(result.success, bool)
    assert isinstance(result.steps, int)
    assert isinstance(result.final_obs, dict)
    assert isinstance(result.message, str)


def test_executor_does_not_hold_env():
    """同一 Executor 两次 run_action 传入不同 FakeEnv → 互不影响。"""
    env1 = FakeEnv()
    env2 = FakeEnv()
    env2._current_obs["ee_pos"] = (1.0, 1.0, 1.0)
    vla = MockVLA(seed=0)
    executor = Executor(vla)
    result1 = executor.run_action(env1, "move", done_criteria="unknown")
    result2 = executor.run_action(env2, "move", done_criteria="unknown")
    assert result1.final_obs["ee_pos"] == (0.0, 0.0, 0.0)
    assert result2.final_obs["ee_pos"] == (1.0, 1.0, 1.0)


def test_executor_exception_passthrough():
    """FakeEnv.step 抛 RuntimeError → run_action 不捕获，异常向上抛。"""
    env = FakeEnvRaisingStep()
    vla = MockVLA(seed=0)
    executor = Executor(vla)
    with pytest.raises(RuntimeError, match="FakeEnv step error"):
        executor.run_action(env, "move", done_criteria="unknown")


def test_executor_message_includes_step_count():
    """message 含 "执行 VLA 规划的 N 步"。"""
    env = FakeEnv()
    vla = MockVLA(seed=0)
    executor = Executor(vla)
    result = executor.run_action(env, "move", done_criteria="unknown")
    assert "执行 VLA 规划的 1 步" in result.message


# ========== target_pos 场景 ==========


class FakeEnvWithTargetApproach(BaseEnv):
    """可控制 ee_pos 逐步接近 target_pos 的 FakeEnv。"""

    def __init__(self, target_pos, satisfy_at_step=3):
        self._target_pos = target_pos
        self._satisfy_at_step = satisfy_at_step
        self._step_count = 0
        self._ee_pos = (
            target_pos[0] + 0.5,
            target_pos[1] + 0.5,
            target_pos[2] + 0.5,
        )
        self.step_count = 0

    def reset(self, task_spec=None, seed=0):
        self._step_count = 0
        self._ee_pos = (
            self._target_pos[0] + 0.5,
            self._target_pos[1] + 0.5,
            self._target_pos[2] + 0.5,
        )
        return self._make_obs()

    def step(self, action):
        self._step_count += 1
        self.step_count += 1
        if self._step_count >= self._satisfy_at_step:
            self._ee_pos = self._target_pos
        else:
            self._ee_pos = (
                self._target_pos[0] + 0.1,
                self._target_pos[1] + 0.1,
                self._target_pos[2] + 0.1,
            )
        return self._make_obs(), 0.0, False, {}

    def render(self):
        return {"cam1": np.zeros((10, 10, 3), dtype=np.uint8)}

    def get_obs(self, include_rgb: bool = True):
        return self._make_obs()

    def close(self):
        pass

    @property
    def input_spec(self):
        return _task_spec()

    def _make_obs(self):
        return {
            "rgb": {"cam1": np.zeros((10, 10, 3), dtype=np.uint8)},
            "ee_pos": self._ee_pos,
            "object_info": [],
            "state_desc": "",
        }


def test_executor_with_target_pos_chunk_1_reached():
    """target_pos 提供 + chunk=1 + 第 1 步 ee_pos 到达 → success=True, steps=1。"""
    target_pos = (0.5, 0.0, 0.5)
    # satisfy_at_step=1：第 1 步就到达
    env = FakeEnvWithTargetApproach(target_pos, satisfy_at_step=1)
    vla = MockVLA(seed=0)
    executor = Executor(vla)
    result = executor.run_action(
        env, "move to target", done_criteria="reached", target_pos=target_pos
    )
    assert result.success is True
    assert result.steps == 1
    assert "距目标" in result.message


def test_executor_with_target_pos_chunk_1_not_reached():
    """target_pos 远离 ee_pos → chunk=1 跑完 success=False。"""
    target_pos = (0.5, 0.0, 0.5)
    # satisfy_at_step=100 远超 chunk=1
    env = FakeEnvWithTargetApproach(target_pos, satisfy_at_step=100)
    vla = MockVLA(seed=0)
    executor = Executor(vla)
    result = executor.run_action(
        env, "move to target", done_criteria="reached", target_pos=target_pos
    )
    assert result.success is False
    assert result.steps == 1  # chunk 跑完
    assert "未到达" in result.message


# ========== robot-vla-adapter：adapter 参数 ==========

_ADAPTER_TEST_OBS = {
    "rgb": {"cam1": np.zeros((10, 10, 3), dtype=np.uint8)},
    "ee_pos": (0.0, 0.0, 0.0),
    "object_info": [],
    "state_desc": "adapter test",
}


def test_run_action_with_adapter_transforms_chunk():
    """adapter 收到的是 VLAOutput 解包后的裸 chunk (1, 7)，转换后 env.step 1 次。"""
    observed = {}

    def capturing_env(action):
        observed.setdefault("actions", []).append(np.asarray(action).copy())
        return _ADAPTER_TEST_OBS, 0.0, False, {}

    env = FakeEnv()
    env.step = capturing_env

    vla = MockVLA(seed=0)
    executor = Executor(vla)

    def fake_adapter(vla_output, env):
        observed["adapter_received"] = np.asarray(vla_output).copy()
        # 把 chunk 转成 list[ndarray] 让 env.step 逐个处理（这里 chunk=1 故 1 个）
        chunk = np.asarray(vla_output)
        return chunk  # 直接返回 chunk（executor 按第一维迭代 step）

    executor.run_action(env, "move", done_criteria="unknown", adapter=fake_adapter)
    # adapter 收到 shape (1, 7) 的整 chunk
    assert observed["adapter_received"].shape == (1, 7)


def test_run_action_adapter_none_direct():
    """adapter=None → VLAOutput.values 裸 chunk 原样 step（按第一维迭代）。"""
    observed = {}

    def capturing_env(action):
        observed.setdefault("actions", []).append(np.asarray(action).copy())
        return _ADAPTER_TEST_OBS, 0.0, False, {}

    env = FakeEnv()
    env.step = capturing_env

    vla = MockVLA(seed=0)
    executor = Executor(vla)
    executor.run_action(env, "move", done_criteria="unknown", adapter=None)
    # env.step 被调 1 次（chunk=1），收到 shape (7,) 单步动作
    assert len(observed["actions"]) == 1
    assert observed["actions"][0].shape == (7,)


def test_run_action_joint_to_joint_chunked_link():
    """VLA 输出 chunk (1, 8) joint → joint_to_joint → SO101 6 维 chunk (1, 6) → env.step 1 次。"""
    observed = {}

    def capturing_env(action):
        observed["step_action"] = np.asarray(action).copy()
        return _ADAPTER_TEST_OBS, 0.0, False, {}

    class JointVLA(BaseVLA):
        """输出 joint 8 维 chunk (1, 8)。"""

        @property
        def output_spec(self):
            return ActionSpec("joint", ("joint",) * 8)

        def predict(self, image, instruction):
            return VLAOutput(
                values=np.array([[1.0, 2, 3, 4, 5, 6, 7, 8]]),  # (1, 8)
                spec=self.output_spec,
            )

    class So101FakeEnv(BaseEnv):
        @property
        def input_spec(self):
            return ActionSpec("joint", ("joint",) * 6)

        def reset(self, task_spec=None, seed=0):
            return _ADAPTER_TEST_OBS

        def step(self, action):
            return _ADAPTER_TEST_OBS, 0.0, False, {}

        def render(self):
            return _ADAPTER_TEST_OBS["rgb"]

        def get_obs(self, include_rgb=True):
            return _ADAPTER_TEST_OBS

        def close(self):
            pass

    env = So101FakeEnv()
    env.step = capturing_env

    from utils.adapter.adapters.joint_to_joint import joint_to_joint_transform

    vla = JointVLA()
    executor = Executor(vla)
    executor.run_action(env, "move", done_criteria="unknown", adapter=joint_to_joint_transform)

    # executor 按第一维迭代 step（chunk=1 故 step 1 次）；env.step 收到单步 (6,) 动作
    step_action = observed["step_action"]
    assert isinstance(step_action, np.ndarray)
    assert step_action.shape == (6,)
    np.testing.assert_allclose(step_action, [1.0, 2, 3, 4, 5, 6])