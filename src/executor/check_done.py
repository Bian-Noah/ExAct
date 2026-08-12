"""check_done 纯函数：动作前后状态对比，判断完成。

基于 criteria 字符串匹配分支判断规则：
- "reached" / "到达" → 比较 ee_pos 与目标位置的距离（需 target_pos）
                       target_pos 缺失时回退到前后位移逻辑（兜底）
- "grasped" / "夹取" / "抓取" → 占位实现
- 默认 → unknown_criteria
"""

import math
from typing import Optional, Tuple

REACHED_THRESHOLD: float = 0.01  # 模块常量，米


def check_done(
    obs_before: dict,
    obs_after: dict,
    criteria: str,
    target_pos: Optional[Tuple[float, float, float]] = None,
) -> tuple[bool, str]:
    """判断动作是否完成。

    Args:
        obs_before: env.step 前的 obs dict。
        obs_after: env.step 后的 obs dict。
        criteria: 完成标准字符串，支持 "reached"/"到达"、
            "grasped"/"夹取"/"抓取"。大小写不敏感。
        target_pos: 目标位置 (x, y, z)。reached 规则下优先使用：
            - 提供 → 比较 obs_after["ee_pos"] 与 target_pos 的距离
            - 未提供 → 回退到比较前后位移（兜底，不推荐用于真实到达判断）

    Returns:
        (is_done: bool, reason: str)。

    Raises:
        ValueError: reached 规则下 obs_before 或 obs_after 缺 ee_pos 字段。
    """
    criteria_lower = criteria.lower()

    # 分支 reached：优先用 target_pos 判断，缺失时回退到前后位移
    if "reached" in criteria_lower or "到达" in criteria:
        if "ee_pos" not in obs_before:
            raise ValueError("reached 规则下 obs_before 缺少 'ee_pos' 字段")
        if "ee_pos" not in obs_after:
            raise ValueError("reached 规则下 obs_after 缺少 'ee_pos' 字段")

        if target_pos is not None:
            # 正确语义：ee_pos 距目标位置的距离 < 阈值
            dist_to_target = math.dist(obs_after["ee_pos"], target_pos)
            if dist_to_target <= REACHED_THRESHOLD:
                return (
                    True,
                    f"ee_pos 距目标 {dist_to_target:.4f} m，小于阈值 {REACHED_THRESHOLD}",
                )
            return (
                False,
                f"ee_pos 距目标 {dist_to_target:.4f} m，大于阈值 {REACHED_THRESHOLD}，未到达",
            )

        # 兜底：target_pos 缺失，用前后位移判断（用于无目标的测试场景）
        dist = math.dist(obs_before["ee_pos"], obs_after["ee_pos"])
        if dist <= REACHED_THRESHOLD:
            return (
                True,
                f"ee_pos 前后位移 {dist:.4f} m，小于阈值 {REACHED_THRESHOLD}（未提供 target_pos，使用前后位移兜底）",
            )
        return (
            False,
            f"ee_pos 前后位移 {dist:.4f} m，大于阈值 {REACHED_THRESHOLD}，未到达（未提供 target_pos，使用前后位移兜底）",
        )

    # 分支 grasped：占位实现
    if "grasped" in criteria_lower or "夹取" in criteria or "抓取" in criteria:
        return (False, "grasped 判断未实现，需感知支持")

    # 默认：未知 criteria
    return (False, f"unknown_criteria: {criteria}")
