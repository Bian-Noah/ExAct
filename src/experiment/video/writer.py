"""mp4 视频写入封装（iter12-video-recording 引入）。

封装 `cv2.VideoWriter`，为 `ExperimentRecorder` 提供 open/write/close 生命周期：
- 按 `VideoConfig` 打开 `process/{filename}`
- RGB → BGR 颜色转换（mp4v 编码按 BGR 期望）
- 异常隔离：open/write/close 内部全捕获，不向上抛（录制失败不影响业务主线）

设计注：`cv2` 通过延迟 import 引入，未安装时 `open()` 返回 False，
后续 write/close 全部 no-op（设计文档 4.3 异常流程）。
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

from experiment.utils import validate_even_resolution
from experiment.video.config import VideoConfig


class VideoWriter:
    """cv2.VideoWriter 封装：open/write/close/is_open。

    Args:
        process_dir: process/ 目录（已由 recorder.start() 创建）。
        config: 视频配置。
    """

    def __init__(self, process_dir: Path, config: VideoConfig) -> None:
        self.process_dir = Path(process_dir)
        self.config = config
        self._writer = None
        self._cv2 = None

    @property
    def is_open(self) -> bool:
        """是否已成功打开底层 writer。"""
        return self._writer is not None

    def open(self) -> bool:
        """按 VideoConfig 打开 `process/{filename}`。

        Returns:
            True=打开成功；False=cv2 缺失 / 参数非法 / 路径不可写 / 打开异常。
        """
        try:
            import cv2

            self._cv2 = cv2
        except ImportError:
            warnings.warn(
                "[VideoWriter] opencv (cv2) 未安装，视频录制不可用。"
                "请安装 opencv-python-headless 以启用视频录制。",
                stacklevel=2,
            )
            return False

        try:
            validate_even_resolution(*self.config.resolution)
        except ValueError as e:
            warnings.warn(f"[VideoWriter] {e}，视频录制降级 no-op。", stacklevel=2)
            return False

        path = self.process_dir / self.config.filename
        try:
            fourcc = self._cv2.VideoWriter_fourcc(*"mp4v")
            writer = self._cv2.VideoWriter(
                str(path),
                fourcc,
                float(self.config.fps),
                self.config.resolution,
            )
            if writer is None or not writer.isOpened():
                # mp4v 打开失败时回退 avc1
                fourcc = self._cv2.VideoWriter_fourcc(*"avc1")
                writer = self._cv2.VideoWriter(
                    str(path),
                    fourcc,
                    float(self.config.fps),
                    self.config.resolution,
                )
            if writer is None or not writer.isOpened():
                warnings.warn(
                    f"[VideoWriter] 打开 {path} 失败（编码器不可用），视频录制降级 no-op。",
                    stacklevel=2,
                )
                return False
            self._writer = writer
            return True
        except Exception as e:
            warnings.warn(
                f"[VideoWriter] 打开 {path} 异常：{e}，视频录制降级 no-op。",
                stacklevel=2,
            )
            return False

    def write(self, image: np.ndarray) -> bool:
        """写入一帧 RGB 图像。

        Args:
            image: RGB ndarray，shape=(H, W, 3)，uint8。

        Returns:
            True=已写入；False=未打开 / 写入失败被吞。
        """
        if self._writer is None or self._cv2 is None:
            return False
        try:
            if image.ndim != 3 or image.shape[2] != 3:
                return False
            bgr = self._cv2.cvtColor(image, self._cv2.COLOR_RGB2BGR)
            self._writer.write(bgr)
            return True
        except Exception:
            return False

    def close(self) -> None:
        """release writer；重复调用安全；未打开时 no-op。"""
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
            finally:
                self._writer = None
