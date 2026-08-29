"""视频录制配置数据类（iter12-video-recording 引入）。

定义 `VideoConfig`（enabled/filename/fps/resolution/camera/capture_every_n_steps），
并提供 `from_dict` 从 yaml 配置 dict 构造（缺省容错 + 偶数分辨率校验）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from experiment.utils import validate_even_resolution


@dataclass
class VideoConfig:
    """实验过程视频录制配置。

    Attributes:
        enabled: 视频录制总开关。False 时 recorder 不 open writer、
            FrameCapturer 不抓帧，全部 no-op。
        filename: process/ 下输出文件名。
        fps: 视频文件帧率标签（播放速度，与物理步进 240Hz 解耦）。
        resolution: 抓帧分辨率 (W, H)，必须为偶数宽高（mp4v 编码器要求）。
        camera: 使用哪个相机名（env.cameras 中的 name）；None = 第一个相机。
        capture_every_n_steps: 每 N 个物理子步抓 1 帧（默认 1 = 每子步）。
    """

    enabled: bool = True
    filename: str = "demo.mp4"
    fps: int = 15
    resolution: tuple[int, int] = (640, 480)
    camera: str | None = None
    capture_every_n_steps: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "VideoConfig":
        """从配置 dict 构造，缺失字段用默认。

        Args:
            d: 配置 dict（yaml `experiment.video` 段）。

        Returns:
            VideoConfig 实例。

        Raises:
            ValueError: resolution 非偶数宽高（由上层 catch 回退默认 + warning）。
        """
        if not isinstance(d, dict):
            raise ValueError(f"VideoConfig 配置必须是 mapping，得到 {type(d).__name__}")

        resolution_raw = d.get("resolution", (640, 480))
        if isinstance(resolution_raw, (list, tuple)):
            if len(resolution_raw) != 2:
                raise ValueError(
                    f"VideoConfig.resolution 格式错误：期望长度为 2 的 [W, H]，得到 {resolution_raw!r}"
                )
            resolution = (int(resolution_raw[0]), int(resolution_raw[1]))
        else:
            raise ValueError(
                f"VideoConfig.resolution 格式错误：期望 list/tuple，得到 {type(resolution_raw).__name__}"
            )

        validate_even_resolution(*resolution)

        return cls(
            enabled=bool(d.get("enabled", True)),
            filename=str(d.get("filename", "demo.mp4")),
            fps=int(d.get("fps", 15)),
            resolution=resolution,
            camera=d.get("camera"),
            capture_every_n_steps=int(d.get("capture_every_n_steps", 1)),
        )
