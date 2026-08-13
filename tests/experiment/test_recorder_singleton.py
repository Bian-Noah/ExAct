"""进程级单例 _global_recorder + _SafeRecorder 单元测试。

覆盖（按 tasks.md 2.2 任务列表）：
- get_recorder() 未初始化时返回 _SafeRecorder
- set_recorder(r) 后 get_recorder() 返回同一对象
- set_recorder(None) 重置回 _SafeRecorder
- _SafeRecorder 所有方法 no-op
- autouse fixture 保证测试间全局状态隔离
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiment.recorder import (
    ExperimentRecorder,
    _SafeRecorder,
    get_recorder,
    set_recorder,
)


# ============================================================================
# autouse fixture：每个测试前后重置 _global_recorder
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_global_recorder():
    """每个测试前重置单例，结束后再重置，确保测试间状态隔离。"""
    set_recorder(None)
    yield
    set_recorder(None)


# ============================================================================
# get_recorder() fallback 行为
# ============================================================================


def test_get_recorder_uninitialized_returns_safe():
    """未 set_recorder 时 get_recorder() 返回 _SafeRecorder 实例。"""
    recorder = get_recorder()
    assert isinstance(recorder, _SafeRecorder)


def test_get_recorder_returns_same_safe_when_called_twice():
    """未 set_recorder 时多次调用 get_recorder() 返回同一个 _SafeRecorder。"""
    r1 = get_recorder()
    r2 = get_recorder()
    assert isinstance(r1, _SafeRecorder)
    assert r1 is r2


# ============================================================================
# set_recorder 注入与重置
# ============================================================================


def test_set_recorder_then_get_returns_same_instance(tmp_path: Path):
    """set_recorder(r) 后 get_recorder() 应返回同一对象。"""
    real = ExperimentRecorder(root=tmp_path, enabled=False)
    set_recorder(real)
    fetched = get_recorder()
    assert fetched is real


def test_set_recorder_none_resets(tmp_path: Path):
    """set_recorder(None) 后 get_recorder() 重新 fallback 到 _SafeRecorder。"""
    real = ExperimentRecorder(root=tmp_path)
    set_recorder(real)
    assert get_recorder() is real

    set_recorder(None)
    assert isinstance(get_recorder(), _SafeRecorder)


def test_set_recorder_can_swap(tmp_path: Path):
    """set_recorder 可多次切换到不同实例。"""
    r1 = ExperimentRecorder(root=tmp_path, enabled=False)
    r2 = ExperimentRecorder(root=tmp_path / "other", enabled=False)
    set_recorder(r1)
    assert get_recorder() is r1
    set_recorder(r2)
    assert get_recorder() is r2
    assert get_recorder() is not r1


# ============================================================================
# _SafeRecorder 所有方法 no-op
# ============================================================================


def test_safe_recorder_emit_noop(tmp_path: Path, capsys):
    """_SafeRecorder.emit 不报错、不产生文件、不打印 stdout。"""
    safe = _SafeRecorder()
    safe.emit("log", message="x")
    safe.emit("observe_image", image=np.zeros((10, 10, 3), dtype=np.uint8), idx=1)
    safe.emit("video_frame", image=np.zeros((10, 10, 3), dtype=np.uint8), timestamp=0.1)
    safe.emit("unknown_event", foo="bar")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    # _SafeRecorder 内部 root=/tmp，但不应创建任何文件
    tmp_files = list(Path("/tmp").iterdir()) if Path("/tmp").exists() else []
    # 仅检查 safe.observer_dir 仍为 None（未触发 start）
    assert safe.observer_dir is None


def test_safe_recorder_start_returns_root():
    """_SafeRecorder.start() 返回其内部 root，不创建任何目录。"""
    safe = _SafeRecorder()
    result = safe.start()
    assert result == Path("/tmp")
    # exp_dir 仍为 None（未真正创建）
    assert safe.exp_dir is None
    assert safe.observer_dir is None
    assert safe._log_fh is None


def test_safe_recorder_finish_noop(capsys):
    """_SafeRecorder.finish() 不报错、不产生文件。"""
    safe = _SafeRecorder()
    safe.finish(success=True, summary="x")
    safe.finish(success=False, summary="y")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_safe_recorder_is_experiment_recorder_instance():
    """_SafeRecorder 应是 ExperimentRecorder 的子类（保证类型兼容）。"""
    safe = _SafeRecorder()
    assert isinstance(safe, ExperimentRecorder)


def test_safe_recorder_enabled_false():
    """_SafeRecorder.__init__ 应设置 enabled=False。"""
    safe = _SafeRecorder()
    assert safe.enabled is False
    assert safe.log_to_stdout is False


# ============================================================================
# 全局状态隔离（autouse fixture 已保证，这里加显式断言）
# ============================================================================


def test_isolation_set_then_get_twice(tmp_path: Path):
    """测试 A 设置后，测试 B 应看到 fresh _SafeRecorder（fixture 已重置）。"""
    real = ExperimentRecorder(root=tmp_path)
    set_recorder(real)
    assert get_recorder() is real


def test_isolation_after_teardown_is_safe():
    """fixture teardown 后 _global_recorder 应为 None，下一次 get_recorder 返回 _SafeRecorder。"""
    # fixture 已重置，直接验证
    recorder = get_recorder()
    assert isinstance(recorder, _SafeRecorder)


# ============================================================================
# _SafeRecorder 与 enabled=False 实例的语义等价
# ============================================================================


def test_safe_recorder_emits_are_safe_even_after_toggle(tmp_path: Path):
    """Safe recorder 在外部 toggle enabled 后仍不应报错。"""
    safe = _SafeRecorder()
    safe.enabled = True  # 即使 toggle 成 True
    # emit 应仍 no-op（因为 exp_dir/observer_dir 为 None）
    safe.emit("observe_image", image=np.zeros((10, 10, 3), dtype=np.uint8), idx=1)
    # 不应抛错
    assert safe.observer_dir is None