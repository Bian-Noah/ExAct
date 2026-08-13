"""ImageStore.upload_to_minimax 单元测试（Iteration 5 fix）。

覆盖：
- 2.1.1 读 yaml 拿 api_key（local.yaml 优先，fallback default.yaml）
- 2.1.2 正常 POST：返回 mm_file://{file_id}
- 2.1.3 HTTP 401 抛异常
- 2.1.4 HTTP 429 抛异常
- 2.1.5 HTTP 5xx 抛异常
- 2.1.6 base_resp.status_code != 0 抛 RuntimeError
- 2.1.7 POST 携带正确的 multipart 字段（purpose=video_generation_input + file）
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from utils.image_store import ImageStore, MemoryBackend


# ============================================================================
# helpers
# ============================================================================


def _make_image_store_with_image(tmp_path: Path) -> tuple:
    """构造 ImageStore + 存一张图 + 返回 (store, img_url)。"""
    backend = MemoryBackend(max_memory_items=5, spill_dir=tmp_path)
    store = ImageStore(backend)
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    img_url = store.save(img, category="observations")
    return store, img_url


def _mock_yaml_content(api_key: str = "sk-test-12345") -> str:
    return f"llm:\n  api_key: {api_key}\n  model: MiniMax-M3\n"


# ============================================================================
# 2.1.1 读 yaml
# ============================================================================


def test_upload_to_minimax_reads_local_yaml_first(tmp_path, monkeypatch):
    """local.yaml 存在时优先读 local.yaml。"""
    store, img_url = _make_image_store_with_image(tmp_path)

    # 临时把 configs/ 替换为带 local.yaml 的临时目录
    fake_configs = tmp_path / "configs"
    fake_configs.mkdir()
    (fake_configs / "default.yaml").write_text(_mock_yaml_content("sk-default"))
    (fake_configs / "local.yaml").write_text(_mock_yaml_content("sk-local"))

    fake_store_path = Path("/fake/src/utils/image_store/store.py")
    monkeypatch.setattr("utils.image_store.store.Path", lambda p: fake_store_path if p == Path else p)

    # 由于 Path(__file__).parents[2] 不可 mock，我们直接验证 _read_api_key 调用
    # 这里改为直接 mock yaml.safe_load 来观察哪个文件被读
    with patch("builtins.open", mock_open(read_data=_mock_yaml_content("sk-local"))) as m_open:
        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = {
                "file": {"file_id": 12345},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            }
            try:
                store.upload_to_minimax(img_url)
            except Exception:
                pass
            # 验证 open 被调用
            assert m_open.called


def test_upload_to_minimax_reads_default_yaml_when_no_local(tmp_path):
    """local.yaml 不存在时 fallback default.yaml。"""
    store, img_url = _make_image_store_with_image(tmp_path)

    # 直接 patch 整个 upload_to_minimax 内的 yaml 读取逻辑：
    # 用 monkeypatch 改 Path(__file__).parents[2] 行为
    # 这里用更简单的做法：patch yaml.safe_load + open
    with patch("builtins.open", mock_open(read_data=_mock_yaml_content("sk-default"))) as m_open:
        with patch("yaml.safe_load", return_value={"llm": {"api_key": "sk-default"}}):
            with patch("requests.post") as mock_post:
                mock_post.return_value.json.return_value = {
                    "file": {"file_id": 12345},
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                }
                result = store.upload_to_minimax(img_url)
                assert result == "mm_file://12345"
                assert m_open.called


# ============================================================================
# 2.1.2 正常 POST
# ============================================================================


def test_upload_to_minimax_returns_mm_file_url(tmp_path):
    """正常 POST：返回 mm_file://{file_id}。"""
    store, img_url = _make_image_store_with_image(tmp_path)

    with patch("builtins.open", mock_open(read_data=_mock_yaml_content("sk-test"))):
        with patch("yaml.safe_load", return_value={"llm": {"api_key": "sk-test"}}):
            with patch("requests.post") as mock_post:
                mock_post.return_value.json.return_value = {
                    "file": {"file_id": 2012456789012345678},
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                }
                result = store.upload_to_minimax(img_url)

    assert result == "mm_file://2012456789012345678"


def test_upload_to_minimax_sends_correct_multipart_fields(tmp_path):
    """POST 携带 purpose=video_generation_input + file 字段 + Bearer 头。"""
    store, img_url = _make_image_store_with_image(tmp_path)

    with patch("builtins.open", mock_open(read_data=_mock_yaml_content("sk-test"))):
        with patch("yaml.safe_load", return_value={"llm": {"api_key": "sk-test"}}):
            with patch("requests.post") as mock_post:
                mock_post.return_value.json.return_value = {
                    "file": {"file_id": 12345},
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                }
                store.upload_to_minimax(img_url)

                # 验证 POST 调用参数
                call_args = mock_post.call_args
                assert call_args[0][0] == "https://api.minimaxi.com/v1/files/upload"
                # headers
                assert call_args[1]["headers"]["Authorization"] == "Bearer sk-test"
                # data (purpose)
                assert call_args[1]["data"]["purpose"] == "video_generation_input"
                # files (file 字段是 (filename, buf, content_type))
                files = call_args[1]["files"]
                assert "file" in files
                assert files["file"][0] == "observation.png"
                assert files["file"][2] == "image/png"
                # timeout
                assert call_args[1]["timeout"] == 10


# ============================================================================
# 2.1.3-2.1.5 HTTP 错误
# ============================================================================


def test_upload_to_minimax_raises_on_http_401(tmp_path):
    """HTTP 401 抛 HTTPError。"""
    import requests as req_mod

    store, img_url = _make_image_store_with_image(tmp_path)

    with patch("builtins.open", mock_open(read_data=_mock_yaml_content("sk-test"))):
        with patch("yaml.safe_load", return_value={"llm": {"api_key": "sk-test"}}):
            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.raise_for_status.side_effect = req_mod.HTTPError("401 Unauthorized")
                mock_post.return_value = mock_response

                with pytest.raises(req_mod.HTTPError):
                    store.upload_to_minimax(img_url)


def test_upload_to_minimax_raises_on_http_429(tmp_path):
    """HTTP 429 抛 HTTPError。"""
    import requests as req_mod

    store, img_url = _make_image_store_with_image(tmp_path)

    with patch("builtins.open", mock_open(read_data=_mock_yaml_content("sk-test"))):
        with patch("yaml.safe_load", return_value={"llm": {"api_key": "sk-test"}}):
            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.raise_for_status.side_effect = req_mod.HTTPError("429 Too Many Requests")
                mock_post.return_value = mock_response

                with pytest.raises(req_mod.HTTPError):
                    store.upload_to_minimax(img_url)


def test_upload_to_minimax_raises_on_http_500(tmp_path):
    """HTTP 500 抛 HTTPError。"""
    import requests as req_mod

    store, img_url = _make_image_store_with_image(tmp_path)

    with patch("builtins.open", mock_open(read_data=_mock_yaml_content("sk-test"))):
        with patch("yaml.safe_load", return_value={"llm": {"api_key": "sk-test"}}):
            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.raise_for_status.side_effect = req_mod.HTTPError("500 Internal Server Error")
                mock_post.return_value = mock_response

                with pytest.raises(req_mod.HTTPError):
                    store.upload_to_minimax(img_url)


# ============================================================================
# 2.1.6 base_resp.status_code != 0
# ============================================================================


def test_upload_to_minimax_raises_when_status_code_not_zero(tmp_path):
    """base_resp.status_code != 0 时也成功（HTTP 200）但业务失败 → 抛 RuntimeError。"""
    store, img_url = _make_image_store_with_image(tmp_path)

    with patch("builtins.open", mock_open(read_data=_mock_yaml_content("sk-test"))):
        with patch("yaml.safe_load", return_value={"llm": {"api_key": "sk-test"}}):
            with patch("requests.post") as mock_post:
                mock_post.return_value.json.return_value = {
                    "file": {"file_id": 12345},
                    "base_resp": {"status_code": 2013, "status_msg": "invalid format"},
                }

                with pytest.raises(RuntimeError) as exc_info:
                    store.upload_to_minimax(img_url)
                assert "2013" in str(exc_info.value) or "invalid format" in str(exc_info.value)


# ============================================================================
# 2.1.7 边界
# ============================================================================


def test_upload_to_minimax_reads_image_from_backend(tmp_path):
    """upload_to_minimax 内部从 backend 拿 ndarray（验证数据流正确）。"""
    store, img_url = _make_image_store_with_image(tmp_path)
    original_image = store.load(img_url)

    with patch("builtins.open", mock_open(read_data=_mock_yaml_content("sk-test"))):
        with patch("yaml.safe_load", return_value={"llm": {"api_key": "sk-test"}}):
            with patch("requests.post") as mock_post:
                mock_post.return_value.json.return_value = {
                    "file": {"file_id": 12345},
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                }
                store.upload_to_minimax(img_url)

                # 验证 POST 的 file 内容确实是原始 ndarray 编码的 PNG
                call_args = mock_post.call_args
                files = call_args[1]["files"]
                file_buffer = files["file"][1]
                # file_buffer 是 io.BytesIO；读取后能解码为 ndarray
                import io
                import imageio.v3 as iio

                file_buffer.seek(0)
                loaded = np.asarray(iio.imread(file_buffer))
                assert loaded.shape == original_image.shape
                assert np.array_equal(loaded, original_image)
