"""WidowX URDF 资产下载器（add-widowx-robot 任务 1）。

提供幂等的 URDF 下载函数：本地已存在则跳过（零网络请求），
否则从配置 URL 下载并原子落盘。失败时清理临时文件并抛 RuntimeError。
"""

import os

import requests

from utils.logging import setup_logging

logger = setup_logging("robot.widowx.urdf_downloader")

# 下载超时（秒）
_DOWNLOAD_TIMEOUT = 30


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
