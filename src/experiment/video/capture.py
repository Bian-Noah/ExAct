"""抓帧策略组件（iter12-video-recording 引入）。

`FrameCapturer` 由 `PyBulletEnv` 持有，在物理子步回调中驱动：
- 按 `capture_every_n_steps` 节流（每 N 子步抓 1 帧）
- 相机选择（config.camera 指定名；None 用 env.cameras 第一个）
- view/proj 矩阵缓存（reset 后首次计算，避免每帧重复 compute）
- RGBA → RGB，emit `video_frame` 事件给 recorder
- 异常隔离：try/except 吞错 + 节流日志（首次 + 每 10 次）
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from experiment.video.config import VideoConfig


class FrameCapturer:
    """抓帧策略：每 N 子步抓 1 帧、相机选择、RGBA→RGB、emit video_frame。

    Args:
        env: PyBulletEnv（读取 env_config.cameras / client_id / _renderer）。
        config: 视频配置。
        recorder: ExperimentRecorder（用于 emit video_frame）。
    """

    def __init__(self, env, config: VideoConfig, recorder) -> None:
        self.env = env
        self.config = config
        self.recorder = recorder
        self.enabled = config.enabled
        self._substep_counter = 0
        self._cam_matrices: tuple | None = None
        self._camera_index: int = 0
        self._error_count = 0

    def reset_timer(self) -> None:
        """重置子步计数与相机矩阵缓存（env.reset 时调用）。"""
        self._substep_counter = 0
        self._cam_matrices = None

    def _resolve_camera_index(self) -> int:
        """按 config.camera 匹配 env.cameras 的索引；None 用第一个。"""
        cameras = self.env.env_config.cameras
        if self.config.camera is not None:
            for i, cam in enumerate(cameras):
                if cam.name == self.config.camera:
                    return i
            warnings.warn(
                f"[FrameCapturer] 相机 '{self.config.camera}' 未在 env.cameras 中找到，"
                "使用第一个相机。",
                stacklevel=2,
            )
        return 0

    def _get_camera_matrices(self) -> tuple:
        """获取 (view_matrix, proj_matrix)，带缓存。"""
        if self._cam_matrices is not None:
            return self._cam_matrices
        cameras = self.env.env_config.cameras
        cam = cameras[self._camera_index]
        width, height = self.config.resolution
        import pybullet as p

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=list(cam.target),
            distance=cam.distance,
            yaw=cam.yaw,
            pitch=cam.pitch,
            roll=cam.roll,
            upAxisIndex=2,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=cam.fov,
            aspect=width / height,
            nearVal=0.1,
            farVal=100.0,
        )
        self._cam_matrices = (view_matrix, proj_matrix)
        return self._cam_matrices

    def capture(self) -> None:
        """抓当前帧并 emit('video_frame', image=rgb, timestamp=time.time())。

        内部 try/except 吞错 + 节流日志（首次 + 每 10 次），不阻断物理步进。
        """
        if not self.enabled:
            return
        # 节流：每 capture_every_n_steps 次调用实际抓 1 帧
        if self._substep_counter % self.config.capture_every_n_steps != 0:
            self._substep_counter += 1
            return
        self._substep_counter += 1
        try:
            self._camera_index = self._resolve_camera_index()
            view_matrix, proj_matrix = self._get_camera_matrices()
            import pybullet as p

            width, height = self.config.resolution
            (_, _, px, _, _) = p.getCameraImage(
                width,
                height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                physicsClientId=self.env._client_id,
                renderer=self.env._renderer,
            )
            rgba = np.array(px, dtype=np.uint8).reshape((height, width, 4))
            rgb = rgba[:, :, :3]
            self.recorder.emit(
                "video_frame", image=rgb, timestamp=time.time()
            )
        except Exception as e:
            self._error_count += 1
            if self._error_count == 1 or self._error_count % 10 == 0:
                warnings.warn(
                    f"[FrameCapturer] capture video frame failed: {e} "
                    f"(已失败 {self._error_count} 次，抓帧降级 no-op)",
                    stacklevel=2,
                )
