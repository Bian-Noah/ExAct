"""图像工具模块。

提供 uint8 RGB ndarray 与 base64 PNG 字符串、文件之间的转换。
"""

import base64
import io
import os

import numpy as np
from PIL import Image


def _validate_image(image: np.ndarray) -> None:
    """校验图像合法性：dtype=uint8、shape=(H,W,3)。"""
    if image.dtype != np.uint8:
        raise ValueError(f"image dtype 必须是 uint8，收到 {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"image shape 必须是 (H,W,3)，收到 {image.shape}"
        )


def encode_b64(image: np.ndarray) -> str:
    """将 uint8 RGB 图像编码为带前缀的 base64 PNG 字符串。

    Args:
        image: shape=(H,W,3), dtype=uint8。

    Returns:
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."

    Raises:
        ValueError: dtype 非 uint8 或 shape 非法。
    """
    _validate_image(image)
    pil_img = Image.fromarray(image, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64_str}"


def save_image(image: np.ndarray, path: str) -> None:
    """保存图像到文件。格式按路径后缀推断（.png/.jpg）。

    Raises:
        ValueError: dtype 非 uint8 或 shape 非法。
    """
    _validate_image(image)
    pil_img = Image.fromarray(image, mode="RGB")
    pil_img.save(path)


def load_image(path: str) -> np.ndarray:
    """从文件加载图像为 uint8 RGB ndarray。

    Returns:
        shape=(H,W,3), dtype=uint8。

    Raises:
        FileNotFoundError: 路径不存在。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"图像文件不存在: {path}")
    pil_img = Image.open(path).convert("RGB")
    return np.array(pil_img, dtype=np.uint8)
