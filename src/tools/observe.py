"""ObserveTool：观察当前场景工具。

继承 langchain_core.tools.BaseTool。

Iteration 3：在 _run() 拿到 obs['rgb'] ndarray 后埋点
recorder.emit('observe_image', ...) + emit('log', ...)，
用于 ExperimentRecorder 落盘 observer/{idx:03d}.png 与 experiment.log。

Iteration 5：_run() 返回结构化 list[dict]（LangChain 标准 content blocks），
    - 始终含 1 个 text 块
    - 若 image_store 已注入且 obs['rgb'] 可用，追加 1 个 image 块
    - image 块字段：{"type": "image", "url": "img://observations/..."}
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional, Type

import numpy as np
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


def _format_text_lines(obs: dict, target: Optional[str]) -> list[str]:
    """从 obs 构造格式化文本行（ee_pos + 场景物体列表）。

    复用位：函数返回值作为 LangChain text content block 的 "text" 字段。

    Args:
        obs: env.get_obs() 返回的 dict（含 ee_pos / object_info）。
        target: 可选物体名过滤，大小写不敏感。

    Returns:
        多行字符串列表。
    """
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

    return lines


class ObserveTool(BaseTool):
    """观察当前场景，返回物体列表和末端执行器位置。

    可指定 target 参数过滤特定物体（大小写不敏感，先精确后包含匹配）。

    Iteration 3 扩展：每次 _run() 拿到 obs['rgb'] ndarray 后，调用
    recorder.emit('observe_image', ...) 与 emit('log', ...) 埋点。

    Iteration 5 扩展：_run() 返回结构化 list[dict]（LangChain 标准 content blocks）：
        - 始终含 1 个 text 块（原有格式化文本）
        - 若 image_store 已注入且 obs['rgb'] 可用，追加 1 个 image 块
        - image 块字段：{"type": "image", "url": "img://observations/..."}
    """

    name: str = "observe"
    description: str = (
        "观察当前场景，返回物体列表和末端执行器位置。"
        "可指定 target 参数（可选）过滤特定物体。"
    )
    args_schema: Type[BaseModel] = ObserveInput
    env: Any = None
    image_store: Any = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Iteration 3：埋点计数器 + 全局 recorder 引用
        self._call_count: int = 0
        self._recorder = get_recorder()

    def _run(self, target: Optional[str] = None) -> list[dict]:
        """执行观察,返回结构化 list[dict]。

        返回形态(LangChain 标准 content blocks):
            [
                {"type": "text", "text": "..."},
                {"type": "image", "url": "img://observations/{cam_name}/..."},  # 每相机 1 个
                ...
            ]

        iter11-reset-multicam:遍历 obs["rgb"] dict 每个相机,每相机生成 1 个
        image block。ImageStore category 改为 f"observations/{cam_name}"。

        Args:
            target: 可选物体名过滤,大小写不敏感。None 时返回全部物体。

        Returns:
            list[dict]:至少 1 个 text 块;每相机 1 个 image 块(若 image_store 与 rgb 可用)。
        """
        _log = logging.getLogger("observe")
        _log.info(f"observe 调用开始 target={target}")
        # iter2-renderer-env-mode：恢复 RGB 图像获取（之前为绕开段错误硬编码 False）
        obs = self.env.get_obs(include_rgb=True)
        _log.info("observe 调用完成")

        # Iteration 3：埋点保存 RGB 图像（异常隔离由 recorder 内部 try/except 处理）
        rgb_dict = obs.get("rgb")
        image_blocks: list[dict] = []
        if rgb_dict is not None:
            # iter11-reset-multicam:遍历每个相机,分别存 ImageStore + 生成 image block
            for cam_name, rgb in rgb_dict.items():
                if not isinstance(rgb, np.ndarray):
                    continue
                self._recorder.emit("observe_image", image=rgb, idx=self._call_count)
                self._recorder.emit(
                    "log",
                    message=f"[observe] saved observer/{self._call_count:03d}.png "
                    f"(camera={cam_name})",
                )

                # Iteration 5：若 image_store 已注入,保存到 ImageStore 并追加 image content block
                # iter11-reset-multicam:category 用 f"observations_{cam_name}" 替代 f"observations/{cam_name}"
                # ImageStore url_scheme 只允许 [a-zA-Z0-9_],相机名中的 . 和 / 需替换为 _
                if self.image_store is not None:
                    safe_cam = re.sub(r"[^a-zA-Z0-9_]", "_", cam_name)
                    image_url = self.image_store.save(rgb, category=f"observations_{safe_cam}")
                    # Iteration 5 fix：把 img:// 翻译成 provider 可消费的 mm_file://{file_id}
                    image_url = self.image_store.upload_to_minimax(image_url)
                    image_blocks.append({"type": "image", "url": image_url})

            self._call_count += 1

        text_lines = _format_text_lines(obs, target)
        content: list[dict] = [{"type": "text", "text": "\n".join(text_lines)}]
        content.extend(image_blocks)

        return content
