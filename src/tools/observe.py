"""ObserveTool：观察当前场景工具。

继承 langchain_core.tools.BaseTool。

Iteration 3：在 _run() 拿到 obs['rgb'] ndarray 后埋点
recorder.emit('observe_image', ...) + emit('log', ...)，
用于 ExperimentRecorder 落盘 observer/{idx:03d}.png 与 experiment.log。
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from experiment.recorder import get_recorder


class ObserveInput(BaseModel):
    """ObserveTool 输入参数。"""

    target: Optional[str] = Field(
        default=None, description="可选，过滤特定物体名（大小写不敏感）"
    )


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

    Iteration 3 扩展：每次 _run() 拿到 obs['rgb'] ndarray 后，调用
    recorder.emit('observe_image', ...) 与 emit('log', ...) 埋点。
    """

    name: str = "observe"
    description: str = (
        "观察当前场景，返回物体列表和末端执行器位置。"
        "可指定 target 参数（可选）过滤特定物体。"
    )
    args_schema: Type[BaseModel] = ObserveInput
    env: Any = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Iteration 3：埋点计数器 + 全局 recorder 引用
        self._call_count: int = 0
        self._recorder = get_recorder()

    def _run(self, target: Optional[str] = None) -> str:
        """执行观察。

        Args:
            target: 可选物体名过滤，大小写不敏感。None 时返回全部物体。

        Returns:
            多行字符串：第一行末端执行器位置，其后物体列表。
        """
        _log = logging.getLogger("observe")
        _log.info(f"observe 调用开始 target={target}")
        # iter2-renderer-env-mode：恢复 RGB 图像获取（之前为绕开段错误硬编码 False）
        obs = self.env.get_obs(include_rgb=True)
        _log.info("observe 调用完成")

        # Iteration 3：埋点保存 RGB 图像（异常隔离由 recorder 内部 try/except 处理）
        rgb = obs.get("rgb")
        if rgb is not None:
            self._recorder.emit("observe_image", image=rgb, idx=self._call_count)
            self._recorder.emit(
                "log",
                message=f"[observe] saved observer/{self._call_count:03d}.png",
            )
            self._call_count += 1

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