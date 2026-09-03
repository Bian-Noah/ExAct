"""WidowX URDF 下载器 ensure_urdf_downloaded 单元测试。

覆盖四类行为：
1. 本地已存在 → 跳过（零网络请求，返回 False）；
2. 正常下载 → 原子落盘 + 自动建父目录（返回 True）；
3. HTTP 非 200 → 抛 RuntimeError（含 url / local_path 指引）且清理 .tmp；
4. 网络异常 → 抛 RuntimeError 且清理残留 .tmp。

以及 ensure_assets_downloaded（资产补齐）：
5. URDF 引用资产缺失 → 批量下载到 URDF 同目录，返回补齐数；
6. URDF 缺失 / 无引用 → 零网络请求，返回 0；
7. 无法从 urdf_url 推导资产根 → 抛 RuntimeError。

仅依赖 pytest 内置的 monkeypatch / tmp_path，不引入额外 mock 库。
运行方式：cd 项目根目录 && PYTHONPATH=src pytest tests/test_urdf_downloader.py
"""

import os

import requests

import pytest

from env.robot.widowx import urdf_downloader
from env.robot.widowx.urdf_downloader import ensure_assets_downloaded, ensure_urdf_downloaded


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


# ============================================================================
# ensure_assets_downloaded：URDF 引用资产补齐
# ============================================================================

_URDF_WITH_MESH_REF = (
    '<robot name="widowx">'
    '<link name="base_link"><visual><geometry>'
    '<mesh filename="package://widowx/meshes/meshes_wx250/WXA-250-M-1-Base.stl"/>'
    "</geometry></visual></link>"
    "</robot>"
).encode("utf-8")


def test_assets_downloads_missing_meshes(tmp_path, monkeypatch):
    """URDF 引用资产缺失 → 下载到 URDF 同目录，返回补齐数；再次调用跳过返回 0。"""
    urdf_path = tmp_path / "wx250.urdf"
    urdf_path.write_bytes(_URDF_WITH_MESH_REF)

    fetched = []

    def _fake_get(url, timeout):
        fetched.append((url, timeout))
        return FakeResponse(status_code=200, content=b"stl-bytes")

    monkeypatch.setattr(urdf_downloader.requests, "get", _fake_get)

    # urdf_url 形如 <root>/urdf/wx250.urdf → 资产根可推导
    count = ensure_assets_downloaded(
        str(urdf_path),
        "https://github.com/example/repo/raw/main/pkg/widowx/urdf/wx250.urdf",
    )
    assert count == 1
    target = tmp_path / "meshes" / "meshes_wx250" / "WXA-250-M-1-Base.stl"
    assert target.read_bytes() == b"stl-bytes"
    # URL 由资产根 + package:// rest 拼出
    assert fetched[0][0].endswith(
        "/pkg/widowx/meshes/meshes_wx250/WXA-250-M-1-Base.stl"
    )
    assert not os.path.exists(str(target) + ".tmp")

    # 幂等：资产已存在 → 零网络请求，返回 0
    fetched.clear()
    assert ensure_assets_downloaded(
        str(urdf_path),
        "https://github.com/example/repo/raw/main/pkg/widowx/urdf/wx250.urdf",
    ) == 0
    assert fetched == []


def test_assets_skip_when_urdf_missing_or_no_refs(tmp_path, monkeypatch):
    """URDF 缺失或无可解析引用 → 返回 0，且不发任何网络请求。"""
    missing_urdf = tmp_path / "not_exist.urdf"

    def _should_not_call(*args, **kwargs):
        raise AssertionError("requests.get 不应被调用")

    monkeypatch.setattr(urdf_downloader.requests, "get", _should_not_call)

    assert ensure_assets_downloaded(
        str(missing_urdf), "https://example.com/pkg/widowx/urdf/wx250.urdf"
    ) == 0

    # 无 package:// 引用的 URDF → 解析为空，同样零请求
    plain_urdf = tmp_path / "plain.urdf"
    plain_urdf.write_bytes(b"<urdf/>")
    assert ensure_assets_downloaded(
        str(plain_urdf), "https://example.com/pkg/widowx/urdf/wx250.urdf"
    ) == 0


def test_assets_derive_fail_raises(tmp_path, monkeypatch):
    """urdf_url 非 <root>/urdf/<file> 布局（无法推导资产根）→ RuntimeError。"""
    urdf_path = tmp_path / "wx250.urdf"
    urdf_path.write_bytes(_URDF_WITH_MESH_REF)

    def _should_not_call(*args, **kwargs):
        raise AssertionError("requests.get 不应被调用（推导失败应先报错）")

    monkeypatch.setattr(urdf_downloader.requests, "get", _should_not_call)

    with pytest.raises(RuntimeError) as exc_info:
        ensure_assets_downloaded(str(urdf_path), "http://example.com/wx250.urdf")
    assert "资产根" in str(exc_info.value) or "urdf_url" in str(exc_info.value)
