"""ObserveTool：观察当前场景工具。

供 Agent 调用以获取场景物体列表和末端执行器位置。
"""

from __future__ import annotations

from typing import Any, Optional

from tools.base import BaseTool


def _get_pos(obj: dict) -> tuple:
    """从物体 dict 中取位置，兼容 pos / position 两种字段名。"""
    pos = obj.get("position", obj.get("pos"))
    if pos is None:
        return (None, None, None)
    if isinstance(pos, (list, tuple)) and len(pos) >= 3:
        return (pos[0], pos[1], pos[2])
    return (None, None, None)


class ObserveTool(BaseTool):
    """观察当前场景，返回物体列表和末端执行器位置。

    可指定 target 参数过滤特定物体（大小写不敏感，先精确后包含匹配）。
    """

    name = "observe"
    description = (
        "观察当前场景，返回物体列表和末端执行器位置。"
        "可指定 target 参数（可选）过滤特定物体。"
    )

    def __init__(self, env: Any):
        """初始化。

        Args:
            env: 环境对象（duck typing，含 get_obs() 方法即可，不强制继承 BaseEnv）。
        """
        self.env = env

    def _run(self, target: Optional[str] = None) -> str:
        """执行观察。

        Args:
            target: 可选物体名过滤，大小写不敏感。None 时返回全部物体。

        Returns:
            多行字符串：第一行末端执行器位置，其后物体列表。
        """
        # observe 工具只需要 ee_pos 和 object_info，跳过 RGB 渲染
        obs = self.env.get_obs(include_rgb=False)
        lines: list[str] = []

        # 末端执行器位置
        ee_pos = obs.get("ee_pos")
        if (
            ee_pos is not None
            and isinstance(ee_pos, (list, tuple))
            and len(ee_pos) >= 3
        ):
            lines.append(
                f"末端执行器位置: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})"
            )
        else:
            lines.append("末端执行器位置: unknown")

        # 物体列表
        lines.append("场景物体列表:")
        object_info = obs.get("object_info", []) or []

        # 过滤物体
        if target is None:
            filtered = object_info
        else:
            target_lower = target.lower()
            # 先精确匹配（大小写不敏感）
            filtered = [
                obj for obj in object_info
                if obj.get("name", "").lower() == target_lower
            ]
            # 没有再做包含匹配
            if not filtered:
                filtered = [
                    obj for obj in object_info
                    if target_lower in obj.get("name", "").lower()
                ]

        for obj in filtered:
            name = obj.get("name", "unknown")
            p0, p1, p2 = _get_pos(obj)
            lines.append(f"  - {name}: pos=({p0}, {p1}, {p2})")

        if target and not filtered:
            lines.append(f"  (未找到目标物体 '{target}')")

        return "\n".join(lines)
