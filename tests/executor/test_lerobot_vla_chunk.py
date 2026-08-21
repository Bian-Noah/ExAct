"""LeRobotVLA chunk 契约单元测试（Iteration 10）。

依赖 torch/lerobot，本机 M4 Mac 不安装这些 stage2 依赖，用 pytest.importorskip 跳过。
在 RTX 4060 + stage2 依赖装好时跑。
"""

import numpy as np
import pytest

pytest.importorskip("torch")


def _make_mock_LeRobotVLA():
    """构造 LeRobotVLA 但 mock 掉 _ensure_loaded 与内部组件。"""
    from executor.model.lerobot.lerobot_vla import LeRobotVLA

    v = LeRobotVLA(model_path="/tmp/fake", policy_type="act", device="cpu", action_dim=6)
    # Mock 内部组件，跳过真实加载
    v._policy = _MockPolicy()
    v._preprocessor = lambda raw_obs: raw_obs  # 直通
    v._postprocessor = lambda x: x  # 直通
    return v


class _MockPolicy:
    """Mock lerobot policy：跟踪 select_action / predict_action_chunk 调用。"""

    def __init__(self):
        self.select_action_called = 0
        self.predict_action_chunk_called = 0
        # 默认返回 chunk (1, 50, 6) → executor 处理为 (50, 6)
        self._next_return = None

    def set_next_return(self, value):
        self._next_return = value

    def select_action(self, preprocessed):
        self.select_action_called += 1
        if self._next_return is not None:
            return self._next_return
        return np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])  # (1, 6)

    def predict_action_chunk(self, preprocessed):
        self.predict_action_chunk_called += 1
        if self._next_return is not None:
            return self._next_return
        # 默认 (1, 50, 6) — smolVLA chunk_size=50
        return np.zeros((1, 50, 6), dtype=float)


def test_predict_calls_predict_action_chunk_not_select_action():
    """Iteration 10：predict 走 predict_action_chunk，**不**调 select_action。

    背景：旧实现调 select_action 走 deque 队列，会跨 action 调用污染新 chunk。
    """
    v = _make_mock_LeRobotVLA()
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    v.predict(image, "move")

    assert v._policy.predict_action_chunk_called == 1
    assert v._policy.select_action_called == 0


def test_chunk_postprocess_drops_batch_dim():
    """postprocessor 输出 (1, 50, 6) → squeeze batch → (50, 6)。"""
    v = _make_mock_LeRobotVLA()
    v._policy.set_next_return(np.zeros((1, 50, 6), dtype=float))

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    result = v.predict(image, "move")

    assert isinstance(result.values, np.ndarray)
    assert result.values.shape == (50, 6)


def test_chunk_postprocess_keeps_2d_input():
    """postprocessor 输出 (50, 6) → 保持 (50, 6)。"""
    v = _make_mock_LeRobotVLA()
    v._policy.set_next_return(np.zeros((50, 6), dtype=float))

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    result = v.predict(image, "move")

    assert result.values.shape == (50, 6)


def test_chunk_postprocess_ndim_1_reshapes_to_1_n():
    """postprocessor 输出 (6,) → reshape (1, 6)（兼容单步 postprocessor）。"""
    v = _make_mock_LeRobotVLA()
    v._policy.set_next_return(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    result = v.predict(image, "move")

    assert result.values.shape == (1, 6)


def test_chunk_postprocess_ndim_4_raises():
    """postprocessor 输出 (1, 50, 6, 1) → ValueError（ndim > 3 不支持）。"""
    v = _make_mock_LeRobotVLA()
    v._policy.set_next_return(np.zeros((1, 50, 6, 1), dtype=float))

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="ndim"):
        v.predict(image, "move")


def test_repeated_predict_no_deque_state_leak():
    """Iteration 10：连续 2 次 predict 都触发 predict_action_chunk（无 deque 残留）。"""
    v = _make_mock_LeRobotVLA()
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    v.predict(image, "move_1")
    v.predict(image, "move_2")

    assert v._policy.predict_action_chunk_called == 2
    assert v._policy.select_action_called == 0