"""WidowX URDF 下载器 ensure_urdf_downloaded 单元测试。

覆盖四类行为：
1. 本地已存在 → 跳过（零网络请求，返回 False）；
2. 正常下载 → 原子落盘 + 自动建父目录（返回 True）；
3. HTTP 非 200 → 抛 RuntimeError（含 url / local_path 指引）且清理 .tmp；
4. 网络异常 → 抛 RuntimeError 且清理残留 .tmp。

仅依赖 pytest 内置的 monkeypatch / tmp_path，不引入额外 mock 库。
运行方式：cd 项目根目录 && PYTHONPATH=src pytest tests/test_urdf_downloader.py
"""

import os

import requests

import pytest

from env.robot.widowx import urdf_downloader
from env.robot.widowx.urdf_downloader import ensure_urdf_downloaded


class FakeResponse:
    """极简 requests.Response 替身：仅提供被测代码用到的 status_code / content / raise_for_status。"""

    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP error {self.status_code} for url")


def test_skip_when_exists(tmp_path, monkeypatch):
    """本地文件已存在 → 返回 False，且不发起任何网络请求。"""
    local_path = tmp_path / "wx250.urdf"
    local_path.write_bytes(b"existing")

    def _should_not_call(*args, **kwargs):
        raise AssertionError("requests.get 不应被调用（文件已存在）")

    monkeypatch.setattr(urdf_downloader.requests, "get", _should_not_call)

    assert ensure_urdf_downloaded(str(local_path), "http://example.com/wx250.urdf") is False
    assert local_path.read_bytes() == b"existing"


def test_download_success(tmp_path, monkeypatch):
    """本地不存在 → 下载成功，返回 True，内容正确落盘且父目录自动创建。"""
    local_path = tmp_path / "nested" / "dir" / "wx250.urdf"

    calls = {}

    def _fake_get(url, timeout):
        calls["url"] = url
        calls["timeout"] = timeout
        return FakeResponse(status_code=200, content=b"<urdf>")

    monkeypatch.setattr(urdf_downloader.requests, "get", _fake_get)

    assert ensure_urdf_downloaded(str(local_path), "http://example.com/wx250.urdf") is True
    assert local_path.read_bytes() == b"<urdf>"
    assert local_path.parent.is_dir()
    assert not os.path.exists(str(local_path) + ".tmp")
    assert calls["timeout"] == 30


def test_download_http_error(tmp_path, monkeypatch):
    """HTTP 404 → 抛 RuntimeError，message 含 url 与 local_path 指引，且 .tmp 被清理。"""
    local_path = tmp_path / "wx250.urdf"

    def _fake_get(url, timeout):
        return FakeResponse(status_code=404, content=b"not found")

    monkeypatch.setattr(urdf_downloader.requests, "get", _fake_get)

    with pytest.raises(RuntimeError) as exc_info:
        ensure_urdf_downloaded(str(local_path), "http://example.com/wx250.urdf")

    message = str(exc_info.value)
    assert "http://example.com/wx250.urdf" in message
    assert str(local_path) in message
    assert not os.path.exists(str(local_path) + ".tmp")
    assert not local_path.exists()


def test_network_error_cleans_tmp(tmp_path, monkeypatch):
    """网络异常 → 抛 RuntimeError，且预置的残留 .tmp 被清理、无目标文件落盘。"""
    local_path = tmp_path / "wx250.urdf"
    tmp_path_file = str(local_path) + ".tmp"
    # 预置上次失败遗留的半成品临时文件，验证异常清理路径
    with open(tmp_path_file, "wb") as f:
        f.write(b"partial")

    def _fake_get(url, timeout):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(urdf_downloader.requests, "get", _fake_get)

    with pytest.raises(RuntimeError) as exc_info:
        ensure_urdf_downloaded(str(local_path), "http://example.com/wx250.urdf")

    assert "connection refused" in str(exc_info.value)
    assert not os.path.exists(tmp_path_file)
    assert not local_path.exists()
