"""utils.image 单元测试。"""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from utils.image import encode_b64, save_image, load_image


def test_encode_b64_returns_prefixed_string():
    """返回值以 data:image/png;base64, 开头。"""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    result = encode_b64(img)
    assert isinstance(result, str)
    assert result.startswith("data:image/png;base64,")


def test_encode_b64_roundtrip():
    """编码 + 手动解码，像素一致。"""
    img = np.array([[[255, 0, 0]] * 10] * 10, dtype=np.uint8)  # 10x10 全红
    s = encode_b64(img)
    # 去前缀解码
    b64_data = s.split(",", 1)[1]
    raw = base64.b64decode(b64_data)
    pil_img = Image.open(io.BytesIO(raw))
    restored = np.array(pil_img, dtype=np.uint8)
    assert np.array_equal(img, restored)


def test_save_load_roundtrip(tmp_path):
    """save_image + load_image 往返一致。"""
    img = np.random.randint(0, 256, (20, 30, 3), dtype=np.uint8)
    path = str(tmp_path / "a.png")
    save_image(img, path)
    loaded = load_image(path)
    assert loaded.shape == (20, 30, 3)
    assert loaded.dtype == np.uint8
    assert np.array_equal(img, loaded)


def test_encode_b64_rejects_float_dtype():
    """float32 → ValueError。"""
    img = np.ones((5, 5, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        encode_b64(img)


def test_encode_b64_rejects_2d_shape():
    """(H,W) → ValueError。"""
    img = np.ones((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        encode_b64(img)


def test_encode_b64_rejects_rgba_shape():
    """(H,W,4) → ValueError。"""
    img = np.ones((5, 5, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        encode_b64(img)


def test_load_image_file_not_found():
    """不存在路径 → FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        load_image("/tmp/nonexistent_12345.png")
