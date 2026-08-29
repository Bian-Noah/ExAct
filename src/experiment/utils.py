"""experiment 通用工具函数（iter12-video-recording 引入）。

recorder 与 video/ 子包共用的小工具，消除重复：
- 时间戳生成（实验目录命名）
- 目录创建（含父级）
- 偶数分辨率校验（mp4v 编码器要求）
- 相对路径解析（相对 base，默认 CWD）
- 同秒并发目录名去冲突（`_001` `_002` 递增）
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def make_timestamp() -> str:
    """生成 `%Y%m%d_%H%M%S` 格式时间戳（实验目录命名）。

    Returns:
        str，如 `20260829_103000`。
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path | str) -> Path:
    """创建目录（含父级），返回 Path。

    Args:
        path: 目录路径（Path 或 str）。

    Returns:
        Path: 创建后的目录路径。
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def validate_even_resolution(w: int, h: int) -> None:
    """校验宽高均为偶数（mp4v 编码器要求）。

    Args:
        w: 宽度。
        h: 高度。

    Raises:
        ValueError: w 或 h 非正数 / 非偶数。
    """
    if w <= 0 or h <= 0:
        raise ValueError(f"分辨率宽高必须为正数，得到 w={w}, h={h}")
    if w % 2 != 0 or h % 2 != 0:
        raise ValueError(f"分辨率宽高必须为偶数（mp4v 编码器要求），得到 w={w}, h={h}")


def resolve_path(root_value: str, base: Path | None = None) -> Path:
    """相对路径相对 base（默认 CWD）解析为绝对路径。

    Args:
        root_value: 路径值（相对或绝对）。
        base: 解析相对路径的基准目录；None 时用当前工作目录。

    Returns:
        Path: 绝对路径。
    """
    path = Path(root_value)
    if path.is_absolute():
        return path
    base_dir = base if base is not None else Path.cwd()
    return base_dir / path


def next_available_dir(root: Path, timestamp: str) -> Path:
    """同秒并发场景下自动追加 `_001` `_002` 后缀，返回不冲突的目录路径。

    Args:
        root: 实验根目录。
        timestamp: make_timestamp() 生成的值。

    Returns:
        Path: 不冲突的目录路径（未创建，由调用方 mkdir）。
    """
    candidate = root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{timestamp}_{suffix:03d}"
        suffix += 1
    return candidate
