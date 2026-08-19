"""Explore 类：内存笔记 + 按日期落盘。

iter9-verify-explore-design 引入：单类实现，构造时接 root + enabled，
write/read/flush 三件套在 enabled=False 时全部 no-op。
flush 按当天日期追加到 root/{YYYY-MM-DD}.log，flush 后清空内存。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Union


def _resolve_path(root: Union[str, Path]) -> Path:
    """相对路径相对 CWD 解析，与 project 内 _resolve_experiment_root 风格一致。"""
    p = Path(root)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


class Explore:
    """探索笔记收集器（内存 + 按日落盘）。

    Attributes:
        root: 落盘根目录（解析后为绝对 Path）。
        enabled: 是否启用。False 时所有方法 no-op。
    """

    EMPTY_NOTICE = "(暂无探索笔记)"

    def __init__(self, root: Union[str, Path] = "data/explore", enabled: bool = True) -> None:
        self.root: Path = _resolve_path(root)
        self.enabled: bool = enabled
        self._notes: list[str] = []

    def write(self, note: str) -> None:
        """追加一条笔记到内存列表。enabled=False 时 no-op。"""
        if not self.enabled:
            return
        self._notes.append(note)

    def read(self) -> str:
        """返回全部笔记的换行拼接。空时返回提示语。

        enabled=False 时同样返回提示语（不暴露任何信息）。
        """
        if not self.enabled or not self._notes:
            return self.EMPTY_NOTICE
        return "\n".join(self._notes)

    def flush(self) -> None:
        """按当天日期把内存笔记追加到 root/{YYYY-MM-DD}.log，然后清空内存。

        enabled=False 时 no-op。重复调用不会重复落盘（flush 后已清空）。
        """
        if not self.enabled:
            return
        if not self._notes:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        log_path = self.root / f"{date.today().isoformat()}.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(self._notes) + "\n")
        self._notes.clear()