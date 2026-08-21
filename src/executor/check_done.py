"""check_done 纯函数：动作前后状态对比，判断完成。

基于 criteria 字符串匹配分支判断规则：
- "reached" / "到达" → 比较 ee_pos 与目标位置的距离（需 target_pos）
                       target_pos 缺失时**不**做反向兜底，由调用方（LLM）
                       通过 final_obs 自己判断到位与否。
- "grasped" / "夹取" / "抓取" → 占位实现
- 默认 → unknown_criteria

Iteration 10 重要变更：
  - 删除 `target_pos=None` 时的反向兜底分支（"单步位移 ≤ 0.01 m 判 done"）。
    原因：chunk 契约下 executor 在循环中**不再调** check_done；check_done 仅
    用于 chunk 跑完后的事后报告。target_pos 缺失时返回 (False, ...)，让 LLM
    通过 final_obs["ee_pos"] 自主判断语义层面的"到位"。
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

    Iteration 10：target_pos 缺失时**不**触发反向兜底——避免"单步位移小
    → 判 done → 截断 chunk"的 bug 语义复发（详见
    `docs/develop/2026-08-20-vla-chunk-execution-issues.md` 第 3 节）。

    Args:
        obs_before: env.step 前的 obs dict。
        obs_after: env.step 后的 obs dict。
        criteria: 完成标准字符串，支持 "reached"/"到达"、
            "grasped"/"夹取"/"抓取"。大小写不敏感。
        target_pos: 目标位置 (x, y, z)。reached 规则下使用：
            - 提供 → 比较 obs_after["ee_pos"] 与 target_pos 的距离
            - 未提供 → 返回 (False, "...")，由调用方（ActionTool / LLM）
                       通过 final_obs 自己判断到位与否

    Returns:
        (is_done: bool, reason: str)。

    Raises:
        ValueError: reached 规则下 obs_before 或 obs_after 缺 ee_pos 字段。
    """
    criteria_lower = criteria.lower()

    # 分支 reached：仅在 target_pos 提供时严格判定
    if "reached" in criteria_lower or "到达" in criteria:
        if "ee_pos" not in obs_before:
            raise ValueError("reached 规则下 obs_before 缺少 'ee_pos' 字段")
        if "ee_pos" not in obs_after:
            raise ValueError("reached 规则下 obs_after 缺少 'ee_pos' 字段")

        if target_pos is None:
            # Iteration 10：删除反向兜底。LLM 通过 final_obs["ee_pos"] 自行判断。
            return (
                False,
                "target_pos 未提供，LLM 通过 final_obs 判断到位与否",
            )

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

    # 分支 grasped：占位实现
    if "grasped" in criteria_lower or "夹取" in criteria or "抓取" in criteria:
        return (False, "grasped 判断未实现，需感知支持")

    # 默认：未知 criteria
    return (False, f"unknown_criteria: {criteria}")
