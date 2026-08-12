"""executor/check_done.py 单元测试。"""

import pytest

from executor.check_done import REACHED_THRESHOLD, check_done


# ========== reached 规则 - target_pos 场景（正确语义） ==========


def test_check_done_reached_with_target_satisfied():
    """target_pos 距 ee_pos < 阈值 → (True, reason 含 '距目标')。"""
    obs_before = {"ee_pos": (0.5, 0.0, 0.5)}
    obs_after = {"ee_pos": (0.503, 0.004, 0.5)}  # 距目标 0.005m
    target_pos = (0.5, 0.0, 0.5)
    done, reason = check_done(obs_before, obs_after, "reached", target_pos=target_pos)
    assert done is True
    assert "距目标" in reason
    assert "0.0050" in reason


def test_check_done_reached_with_target_not_satisfied():
    """target_pos 距 ee_pos > 阈值 → (False, reason 含 '未到达')。"""
    obs_before = {"ee_pos": (0.0, 0.0, 0.0)}
    obs_after = {"ee_pos": (0.02, 0.0, 0.0)}  # 距目标 0.02m
    target_pos = (0.5, 0.0, 0.5)
    done, reason = check_done(obs_before, obs_after, "reached", target_pos=target_pos)
    assert done is False
    assert "未到达" in reason


def test_check_done_reached_with_target_boundary():
    """target_pos 距 ee_pos 恰好 = 阈值 → (True, ...)（小于等于阈值算到达）。"""
    obs_before = {"ee_pos": (0.0, 0.0, 0.0)}
    obs_after = {"ee_pos": (0.01, 0.0, 0.0)}  # 距目标 0.01m
    target_pos = (0.0, 0.0, 0.0)
    done, _ = check_done(obs_before, obs_after, "reached", target_pos=target_pos)
    assert done is True


def test_check_done_reached_with_target_case_insensitive():
    """criteria='Reached'/'REACHED' + target_pos → 按 reached 规则判断。"""
    obs_before = {"ee_pos": (0.0, 0.0, 0.0)}
    obs_after = {"ee_pos": (0.003, 0.004, 0.0)}  # 距目标 0.005m
    target_pos = (0.0, 0.0, 0.0)
    for criteria in ["Reached", "REACHED", "ReAcHeD"]:
        done, _ = check_done(obs_before, obs_after, criteria, target_pos=target_pos)
        assert done is True, f"criteria={criteria} 应匹配 reached 规则"


def test_check_done_reached_with_target_chinese():
    """criteria='到达' + target_pos → 按 reached 规则判断。"""
    obs_before = {"ee_pos": (0.0, 0.0, 0.0)}
    obs_after = {"ee_pos": (0.003, 0.004, 0.0)}  # 距目标 0.005m
    target_pos = (0.0, 0.0, 0.0)
    done, _ = check_done(obs_before, obs_after, "到达", target_pos=target_pos)
    assert done is True


def test_check_done_reached_target_ignores_before_pos():
    """target_pos 提供时，只看 obs_after 与 target 的距离，obs_before 不影响结果。"""
    target_pos = (0.5, 0.0, 0.5)
    obs_after = {"ee_pos": (0.503, 0.0, 0.5)}  # 距目标 0.005m
    # obs_before 任意（即使变化很大），只要 obs_after 接近 target 即到达
    obs_before_far = {"ee_pos": (0.0, 0.0, 0.0)}
    done, _ = check_done(obs_before_far, obs_after, "reached", target_pos=target_pos)
    assert done is True


# ========== reached 规则 - 兜底场景（无 target_pos，向后兼容） ==========


def test_check_done_reached_fallback_satisfied():
    """无 target_pos + 前后位移 0.005m < 0.01m → (True, reason 含 '兜底')。"""
    obs_before = {"ee_pos": (0, 0, 0)}
    obs_after = {"ee_pos": (0.003, 0.004, 0)}  # 位移 0.005m
    done, reason = check_done(obs_before, obs_after, "reached")
    assert done is True
    assert "兜底" in reason


def test_check_done_reached_fallback_not_satisfied():
    """无 target_pos + 前后位移 0.02m > 0.01m → (False, reason 含 '未到达')。"""
    obs_before = {"ee_pos": (0, 0, 0)}
    obs_after = {"ee_pos": (0.02, 0, 0)}  # 位移 0.02m
    done, reason = check_done(obs_before, obs_after, "reached")
    assert done is False
    assert "未到达" in reason


def test_check_done_reached_fallback_boundary():
    """无 target_pos + 前后位移恰好 0.01m → (True, ...)。"""
    obs_before = {"ee_pos": (0, 0, 0)}
    obs_after = {"ee_pos": (0.01, 0, 0)}  # 位移恰好 0.01m
    done, _ = check_done(obs_before, obs_after, "reached")
    assert done is True


def test_check_done_reached_fallback_case_insensitive():
    """无 target_pos + criteria='Reached'/'REACHED' → 按 reached 规则判断。"""
    obs_before = {"ee_pos": (0, 0, 0)}
    obs_after = {"ee_pos": (0.003, 0.004, 0)}  # 位移 0.005m
    for criteria in ["Reached", "REACHED", "ReAcHeD"]:
        done, _ = check_done(obs_before, obs_after, criteria)
        assert done is True, f"criteria={criteria} 应匹配 reached 规则"


def test_check_done_reached_fallback_chinese():
    """无 target_pos + criteria='到达' → 按 reached 规则判断。"""
    obs_before = {"ee_pos": (0, 0, 0)}
    obs_after = {"ee_pos": (0.003, 0.004, 0)}  # 位移 0.005m
    done, _ = check_done(obs_before, obs_after, "到达")
    assert done is True


# ========== grasped 规则 ==========


def test_check_done_grasped_placeholder():
    """criteria='grasped' → (False, reason 含 '未实现' 或 '需感知')。"""
    done, reason = check_done({}, {}, "grasped")
    assert done is False
    assert "未实现" in reason or "需感知" in reason


def test_check_done_grasped_chinese():
    """criteria='夹取' / '抓取' → 同 grasped 占位。"""
    for criteria in ["夹取", "抓取"]:
        done, reason = check_done({}, {}, criteria)
        assert done is False
        assert "未实现" in reason or "需感知" in reason


# ========== 未知 criteria ==========


def test_check_done_unknown_criteria():
    """criteria='unknown' → (False, reason 含 'unknown_criteria')。"""
    done, reason = check_done({}, {}, "unknown")
    assert done is False
    assert "unknown_criteria" in reason


# ========== 异常场景 ==========


def test_check_done_reached_missing_ee_pos_before():
    """reached 规则下 obs_before 缺 ee_pos → ValueError。"""
    obs_after = {"ee_pos": (0, 0, 0)}
    with pytest.raises(ValueError, match="obs_before"):
        check_done({}, obs_after, "reached")


def test_check_done_reached_missing_ee_pos_after():
    """reached 规则下 obs_after 缺 ee_pos → ValueError。"""
    obs_before = {"ee_pos": (0, 0, 0)}
    with pytest.raises(ValueError, match="obs_after"):
        check_done(obs_before, {}, "reached")


def test_check_done_grasped_no_ee_pos_needed():
    """grasped/未知 criteria 下 obs 缺 ee_pos → 不报错。"""
    # grasped
    done, reason = check_done({}, {}, "grasped")
    assert done is False
    # 未知 criteria
    done, reason = check_done({}, {}, "unknown")
    assert done is False


def test_reached_threshold_constant():
    """REACHED_THRESHOLD 应为 0.01。"""
    assert REACHED_THRESHOLD == 0.01
