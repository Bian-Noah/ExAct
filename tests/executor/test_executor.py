"""executor/__init__.py 主循环单元测试。

使用 FakeEnv（内存 mock）替代 PyBulletPandaEnv，验证 Executor 主循环的
两种退出分支、ExecResult 字段正确性、异常透传。
"""

import numpy as np
import pytest

from env.base import BaseEnv
from executor import Executor, ExecResult, MockVLA


# ========== FakeEnv 测试辅助类 ==========


class FakeEnv(BaseEnv):
    """内存中的 BaseEnv 子类，可编程控制 obs 序列。

    用于验证 Executor 主循环逻辑，不连 PyBullet。
    """

    def __init__(self, obs_sequence=None):
        self._obs_sequence = obs_sequence or []
        self._idx = 0
        self._current_obs = (
            self._obs_sequence[0] if self._obs_sequence else {}
        )

    def reset(self, task_spec=None, seed=0):
        self._idx = 0
        self._current_obs = (
            self._obs_sequence[0] if self._obs_sequence else {}
        )
        return self._current_obs

    def step(self, action):
        # 推进到下一个 obs
        self._idx += 1
        if self._idx < len(self._obs_sequence):
            self._current_obs = self._obs_sequence[self._idx]
        # 返回 (obs, reward, done, info)
        return self._current_obs, 0.0, False, {}

    def render(self):
        return self._current_obs.get("rgb", np.zeros((10, 10, 3), dtype=np.uint8))

    def get_obs(self):
        return self._current_obs

    def close(self):
        pass


class FakeEnvWithEePosControl(BaseEnv):
    """可控制 ee_pos 随步数推进的 FakeEnv。

    用于测试 reached 满足/不满足分支：
    - 前 satisfy_at_step 步：ee_pos 位移 > 阈值（reached 不满足）
    - 第 satisfy_at_step 步后：ee_pos 不再变化（reached 满足）
    """

    def __init__(self, satisfy_at_step=3, move_step=0.05):
        self._step_count = 0
        self._satisfy_at_step = satisfy_at_step
        self._move_step = move_step  # 每步位移量（> REACHED_THRESHOLD）
        self._ee_pos = (0.0, 0.0, 0.0)

    def reset(self, task_spec=None, seed=0):
        self._step_count = 0
        self._ee_pos = (0.0, 0.0, 0.0)
        return self._make_obs()

    def step(self, action):
        self._step_count += 1
        if self._step_count < self._satisfy_at_step:
            # 位移大于阈值
            self._ee_pos = (
                self._ee_pos[0] + self._move_step,
                self._ee_pos[1],
                self._ee_pos[2],
            )
        # 第 satisfy_at_step 步后 ee_pos 不再变化（位移=0 < 阈值）
        return self._make_obs(), 0.0, False, {}

    def render(self):
        return np.zeros((10, 10, 3), dtype=np.uint8)

    def get_obs(self):
        return self._make_obs()

    def close(self):
        pass

    def _make_obs(self):
        return {
            "rgb": np.zeros((10, 10, 3), dtype=np.uint8),
            "ee_pos": self._ee_pos,
            "object_info": [],
            "state_desc": "",
        }


class FakeEnvRaisingStep(BaseEnv):
    """step 时抛异常的 FakeEnv，用于测试异常透传。"""

    def reset(self, task_spec=None, seed=0):
        return {
            "rgb": np.zeros((10, 10, 3), dtype=np.uint8),
            "ee_pos": (0.0, 0.0, 0.0),
        }

    def step(self, action):
        raise RuntimeError("FakeEnv step error for testing")

    def render(self):
        return np.zeros((10, 10, 3), dtype=np.uint8)

    def get_obs(self):
        return {
            "rgb": np.zeros((10, 10, 3), dtype=np.uint8),
            "ee_pos": (0.0, 0.0, 0.0),
        }

    def close(self):
        pass


# ========== Executor 主循环测试 ==========


def test_executor_max_steps_exit():
    """max_steps=5 + done_criteria='unknown'（强制 done=False）→ 5 步后退出。"""
    obs = {
        "rgb": np.zeros((10, 10, 3), dtype=np.uint8),
        "ee_pos": (0.0, 0.0, 0.0),
    }
    env = FakeEnv([obs])  # 永远返回同一 obs
    vla = MockVLA(seed=0)
    executor = Executor(vla, max_steps=5)
    result = executor.run_action(env, "move", done_criteria="unknown")
    assert result.success is False
    assert result.steps == 5
    assert "达到最大步数" in result.message


def test_executor_done_criteria_satisfied():
    """FakeEnvWithEePosControl 第 3 步后位移=0 < 阈值 → reached 满足。"""
    env = FakeEnvWithEePosControl(satisfy_at_step=3, move_step=0.05)
    vla = MockVLA(seed=0)
    executor = Executor(vla, max_steps=10)
    result = executor.run_action(env, "move", done_criteria="reached")
    assert result.success is True
    assert result.steps == 3
    assert "位移" in result.message


def test_executor_result_field_types():
    """ExecResult 字段类型正确。"""
    obs = {
        "rgb": np.zeros((10, 10, 3), dtype=np.uint8),
        "ee_pos": (0.0, 0.0, 0.0),
    }
    env = FakeEnv([obs])
    vla = MockVLA(seed=0)
    executor = Executor(vla, max_steps=3)
    result = executor.run_action(env, "move", done_criteria="unknown")
    assert isinstance(result, ExecResult)
    assert isinstance(result.success, bool)
    assert isinstance(result.steps, int)
    assert isinstance(result.final_obs, dict)
    assert isinstance(result.message, str)


def test_executor_does_not_hold_env():
    """同一 Executor 两次 run_action 传入不同 FakeEnv → 互不影响。"""
    obs1 = {
        "rgb": np.zeros((10, 10, 3), dtype=np.uint8),
        "ee_pos": (0.0, 0.0, 0.0),
    }
    obs2 = {
        "rgb": np.zeros((10, 10, 3), dtype=np.uint8),
        "ee_pos": (1.0, 1.0, 1.0),
    }
    env1 = FakeEnv([obs1])
    env2 = FakeEnv([obs2])
    vla = MockVLA(seed=0)
    executor = Executor(vla, max_steps=3)
    result1 = executor.run_action(env1, "move", done_criteria="unknown")
    result2 = executor.run_action(env2, "move", done_criteria="unknown")
    # 两次结果独立，且 final_obs 各自对应自己的 env
    assert result1.final_obs["ee_pos"] == (0.0, 0.0, 0.0)
    assert result2.final_obs["ee_pos"] == (1.0, 1.0, 1.0)


def test_executor_exception_passthrough():
    """FakeEnv.step 抛 RuntimeError → run_action 不捕获，异常向上抛。"""
    env = FakeEnvRaisingStep()
    vla = MockVLA(seed=0)
    executor = Executor(vla, max_steps=5)
    with pytest.raises(RuntimeError, match="FakeEnv step error"):
        executor.run_action(env, "move", done_criteria="unknown")


def test_executor_max_steps_zero():
    """max_steps=0 → 立即返回，steps=0, success=False, final_obs={}。"""
    env = FakeEnv([])
    vla = MockVLA(seed=0)
    executor = Executor(vla, max_steps=0)
    result = executor.run_action(env, "move", done_criteria="unknown")
    assert result.steps == 0
    assert result.success is False
    assert result.final_obs == {}


# ========== target_pos 场景测试（修复 reached 语义 Bug） ==========


class FakeEnvWithTargetApproach(BaseEnv):
    """可控制 ee_pos 逐步接近 target_pos 的 FakeEnv。

    用于验证 Executor + target_pos 的 reached 判断：
    - 前 satisfy_at_step 步：ee_pos 远离 target_pos（> 阈值）
    - 第 satisfy_at_step 步：ee_pos 等于 target_pos（到达）
    """

    def __init__(self, target_pos, satisfy_at_step=3):
        self._target_pos = target_pos
        self._satisfy_at_step = satisfy_at_step
        self._step_count = 0
        # 初始 ee_pos 远离 target
        self._ee_pos = (
            target_pos[0] + 0.5,
            target_pos[1] + 0.5,
            target_pos[2] + 0.5,
        )

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
        if self._step_count >= self._satisfy_at_step:
            # 到达目标
            self._ee_pos = self._target_pos
        else:
            # 仍远离目标
            self._ee_pos = (
                self._target_pos[0] + 0.1,
                self._target_pos[1] + 0.1,
                self._target_pos[2] + 0.1,
            )
        return self._make_obs(), 0.0, False, {}

    def render(self):
        return np.zeros((10, 10, 3), dtype=np.uint8)

    def get_obs(self):
        return self._make_obs()

    def close(self):
        pass

    def _make_obs(self):
        return {
            "rgb": np.zeros((10, 10, 3), dtype=np.uint8),
            "ee_pos": self._ee_pos,
            "object_info": [],
            "state_desc": "",
        }


def test_executor_with_target_pos_reached():
    """target_pos 提供 + 第 3 步 ee_pos 到达 target → success=True, steps=3。"""
    target_pos = (0.5, 0.0, 0.5)
    env = FakeEnvWithTargetApproach(target_pos, satisfy_at_step=3)
    vla = MockVLA(seed=0)
    executor = Executor(vla, max_steps=10)
    result = executor.run_action(
        env, "move to target", done_criteria="reached", target_pos=target_pos
    )
    assert result.success is True
    assert result.steps == 3
    assert "距目标" in result.message


def test_executor_with_target_pos_not_reached_within_max_steps():
    """target_pos 远离 ee_pos 且永远无法到达 → 循环到 max_steps 退出。"""
    target_pos = (0.5, 0.0, 0.5)
    # satisfy_at_step=100 远超 max_steps=5，循环内永远到达不了
    env = FakeEnvWithTargetApproach(target_pos, satisfy_at_step=100)
    vla = MockVLA(seed=0)
    executor = Executor(vla, max_steps=5)
    result = executor.run_action(
        env, "move to target", done_criteria="reached", target_pos=target_pos
    )
    assert result.success is False
    assert result.steps == 5
    assert "达到最大步数" in result.message


def test_executor_without_target_pos_uses_fallback():
    """未提供 target_pos → 使用前后位移兜底逻辑（向后兼容）。"""
    obs = {
        "rgb": np.zeros((10, 10, 3), dtype=np.uint8),
        "ee_pos": (0.0, 0.0, 0.0),
    }
    env = FakeEnv([obs])  # ee_pos 永远不变，前后位移=0
    vla = MockVLA(seed=0)
    executor = Executor(vla, max_steps=5)
    result = executor.run_action(env, "move", done_criteria="reached")
    # 兜底逻辑：前后位移=0 < 阈值 → 第 1 步就退出
    assert result.steps == 1
    assert result.success is True
    assert "兜底" in result.message
