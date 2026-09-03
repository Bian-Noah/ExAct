"""WidowX URDF 资产下载器（add-widowx-robot 任务 1）。

提供幂等的 URDF 资产下载：
- `ensure_urdf_downloaded`: 确保 URDF 单文件存在（缺失则下载）。
- `ensure_assets_downloaded`: 确保 URDF 引用的外部资产存在（缺失则批量下载）。

背景：wx250.urdf 非自包含——其 `<visual>` 元素通过
`package://widowx/meshes/...` 引用 10 个 .stl 与 1 个 .png（位于源仓库
`<pkg_root>/meshes/` 下）。只下载 URDF 不下载引用资产，pybullet
`loadURDF` 会因找不到 mesh 而失败（Cannot load URDF file）。
引用资产统一落盘到 URDF 同目录下，即 `package://widowx/<rest>` →
`<urdf_dir>/<rest>`，与本地资产布局（robot/widowx/meshes/...）一致。
"""

import os
import posixpath
import re

import requests

from utils.logging import setup_logging

logger = setup_logging("robot.widowx.urdf_downloader")

# 下载超时（秒）
_DOWNLOAD_TIMEOUT = 30

# 匹配 URDF 内 package:// 资源引用，捕获 <rest>（pkg 名后、不含引号/空白的部分）
_PACKAGE_URI_RE = re.compile(r'package://[A-Za-z0-9_.-]+/([^"\'\s]+)')


def ensure_urdf_downloaded(local_path: str, url: str) -> bool:
    """确保 URDF 文件存在于本地，缺失时从 url 下载（幂等）。

    Args:
        local_path: 落盘路径（如 "robot/widowx/wx250.urdf"）。
        url: 下载地址。

    Returns:
        True: 本次实际下载；False: 本地已存在，跳过。

    Raises:
        RuntimeError: 网络失败 / HTTP 非 200 / 写盘失败（含手动放置指引）。
    """
    if os.path.isfile(local_path):
        logger.debug("URDF 已存在，跳过下载: %s", local_path)
        return False

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    tmp_path = local_path + ".tmp"
    try:
        logger.info("下载 WidowX URDF: %s", url)
        resp = requests.get(url, timeout=_DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            f.write(resp.content)
        os.replace(tmp_path, local_path)
        logger.info("URDF 已落盘: %s (%d bytes)", local_path, len(resp.content))
        return True
    except Exception as e:
        # 清理半成品临时文件（存在才删，忽略清理失败）
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise RuntimeError(
            f"WidowX URDF 下载失败: {e}。"
            f"请手动下载 {url} 到 {local_path} 后重试。"
        ) from e


def _derive_asset_root_url(urdf_url: str) -> str | None:
    """从 urdf_url 推导 package 资产根 URL。

    urdf_url 约定形如 `<asset_root>/urdf/<file>.urdf`（如
    `.../assets/widowx/urdf/wx250.urdf`），则 asset_root 为去掉
    `urdf/<file>` 两段的 URL（`.../assets/widowx`），URDF 内
    `package://<pkg>/<rest>` 对应的源文件 URL 为 `asset_root/<rest>`。

    Args:
        urdf_url: URDF 下载地址。

    Returns:
        资产根 URL；层级不足（URDF 不在 `<root>/urdf/` 布局）时返回 None。
    """
    parent = posixpath.dirname(urdf_url)      # .../urdf
    asset_root = posixpath.dirname(parent)    # ...（package 根）
    if not (asset_root.startswith(("http://", "https://")) and asset_root.count("/") >= 2):
        return None
    return asset_root


def _parse_package_refs(urdf_path: str) -> list[str]:
    """解析 URDF 文本中全部 package:// 引用的 rest 路径（去重保序）。

    Args:
        urdf_path: 本地 URDF 文件路径。

    Returns:
        rest 路径列表（如 ["meshes/meshes_wx250/WXA-250-M-1-Base.stl"]）；
        URDF 不存在或无可解析引用时返回空列表。
    """
    if not os.path.isfile(urdf_path):
        return []
    try:
        with open(urdf_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        logger.warning("读取 URDF 失败，跳过资产补齐: %s (%s)", urdf_path, e)
        return []
    return list(dict.fromkeys(_PACKAGE_URI_RE.findall(text)))


def _ensure_within_dir(urdf_dir: str, target: str) -> bool:
    """target 绝对路径必须位于 urdf_dir 之内（防 package:// 路径穿越）。"""
    abs_target = os.path.abspath(target)
    return abs_target == urdf_dir or abs_target.startswith(urdf_dir + os.sep)


def ensure_assets_downloaded(urdf_path: str, urdf_url: str) -> int:
    """确保 URDF 引用的 package:// 资产本地齐全，缺失则从源仓库下载（幂等）。

    URDF 可能已存在而引用资产缺失（如仅下载过 URDF 单文件的服务器），
    因此本函数在 urdf 已存在时同样执行解析与补齐。单个资产下载失败会
    抛 RuntimeError，但已补齐的部分保留（下次重试跳过）。

    Args:
        urdf_path: 本地 URDF 文件路径（须已存在；不存在时直接返回 0）。
        urdf_url: URDF 下载地址，用于推导资产根 URL（`<root>/urdf/<file>` 布局）。

    Returns:
        本次实际下载补齐的资产文件数。

    Raises:
        RuntimeError: 资产根 URL 无法推导 / 单个资产下载失败（含手动放置指引）。
    """
    refs = _parse_package_refs(urdf_path)
    if not refs:
        return 0

    asset_root = _derive_asset_root_url(urdf_url)
    if asset_root is None:
        raise RuntimeError(
            f"无法从 urdf_url 推导 WidowX 资产根地址: {urdf_url!r}。"
            f"（约定 urdf_url 形如 <资产根>/urdf/<file>.urdf）"
            f"请手动将完整资产目录（含 meshes/）放置到 {os.path.dirname(urdf_path)} 后重试。"
        )

    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    downloaded = 0
    for rest in refs:
        target = os.path.join(urdf_dir, *rest.split("/"))
        if not _ensure_within_dir(urdf_dir, target):
            logger.warning("跳过越界资产引用（非 URDF 目录内）: %s", rest)
            continue
        if os.path.isfile(target):
            logger.debug("资产已存在，跳过: %s", target)
            continue

        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        url = asset_root + "/" + rest
        tmp_path = target + ".tmp"
        try:
            logger.info("下载 WidowX 资产: %s -> %s", url, target)
            resp = requests.get(url, timeout=_DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                f.write(resp.content)
            os.replace(tmp_path, target)
            logger.info("资产已落盘: %s (%d bytes)", target, len(resp.content))
            downloaded += 1
        except Exception as e:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            raise RuntimeError(
                f"WidowX 资产下载失败: {e}。"
                f"资产 {rest} 应位于本地 {target}。"
                f"请手动下载 {url} 到 {target} 后重试。"
            ) from e
    return downloaded
