"""executor 模块：连接 Agent（LLM 大脑）与 env（物理世界）的执行器。

核心组件：
- ExecResult: 执行结果 dataclass
- Executor: 主循环类，循环调 VLA 驱动 env，直到完成或超时
- BaseVLA / MockVLA / create_vla: VLA 模型抽象与工厂（详见 executor.model 子包）
"""

from dataclasses import dataclass

import numpy as np

from env.base import BaseEnv
from executor.build_input import build_vla_input
from executor.check_done import check_done
from executor.model.base import BaseVLA
from executor.model.factory import create_vla
from executor.model.mock.mock_vla import MockVLA
from utils.logging import setup_logging


@dataclass
class ExecResult:
    """executor.run_action 的返回值。

    Attributes:
        success: 是否成功完成（达到 done_criteria）。
        steps: 实际执行的步数。
        final_obs: 最后一步的 obs dict。
        message: 完成或失败的原因描述。
    """

    success: bool
    steps: int
    final_obs: dict
    message: str


class Executor:
    """动作执行器主循环。

    聚合一个 BaseVLA 实例和 max_steps 参数，循环调用 VLA 驱动 env 推进物理，
    直到满足 done_criteria 或达到 max_steps。

    不持有 env 引用（env 作为 run_action 参数传入，降低耦合）。
    不捕获 env.step 异常（异常透传给上层）。
    """

    def __init__(self, vla: BaseVLA, max_steps: int = 50):
        """初始化 Executor。

        Args:
            vla: BaseVLA 实例（MockVLA / OpenVLA 等）。
            max_steps: 最大循环步数，默认 50。
        """
        self.vla = vla
        self.max_steps = max_steps
        self.logger = setup_logging(__name__)

    def run_action(
        self,
        env: BaseEnv,
        instruction: str,
        done_criteria: str,
        target_pos: tuple | None = None,
        adapter=None,
    ) -> ExecResult:
        """循环调 VLA 驱动 env，直到完成或超时。

        VLA 通过 env.get_obs()["rgb"] 获取图像——VLA 直接接触环境，
        不依赖 LLM 传入图片 URL。VLA 是否使用 image 参数由各后端自行决定
        （MockVLA 故意忽略，LLMVLA 用语义信息，SmallVLA/OpenVLA 必须使用）。

        Iteration 6 扩展：循环第一步 vla.predict 调用之后打印 image 链路验证
        信息（shape / dtype / 与 obs rgb 的一致性），用于确认 VLA 已能从 env
        拿到图——链路通。

        robot-vla-adapter：新增可选 adapter 参数。adapter 在 vla.predict 之后、
        env.step 之前执行，把 VLA 输出（VLAOutput）转换为 env 原生动作。
        adapter=None 时直通（VLA 输出原样喂给 env）——executor 保持纯净，
        不负责 adapter 的构造/兜底（那归 ActionTool._ensure_adapter）。

        Args:
            env: BaseEnv 实例（PyBulletEnv / FakeEnv 等）。
            instruction: 自然语言指令字符串。
            done_criteria: 完成标准字符串（如 "reached" / "grasped"）。
            target_pos: 目标位置 (x, y, z)。reached 规则下用于判断 ee_pos
                是否到达目标位置；未提供时 check_done 回退到前后位移兜底逻辑。
            adapter: 转换函数 `adapter(vla_output, env) -> env 原生动作`；
                None 时直通（不构造、不提示）。

        Returns:
            ExecResult dataclass。

        Raises:
            env.step 抛出的异常透传，不捕获。
        """
        obs_after: dict = {}
        done = False
        reason = ""
        step = 0

        for step in range(self.max_steps):
            obs_before = env.get_obs()
            vla_input = build_vla_input(obs_before, instruction)
            vla_output = self.vla.predict(vla_input["image"], instruction)
            # ★ adapter 插入点：VLA 输出 → env 原生动作
            action = adapter(vla_output, env) if adapter is not None else vla_output

            # Iteration 6：链路验证 print（仅第一步）
            if step == 0:
                image = vla_input["image"]
                rgb = obs_before.get("rgb")
                if image is not None:
                    print(
                        f"[executor] VLA.predict 收到 image: "
                        f"shape={image.shape}, dtype={image.dtype}"
                    )
                    if rgb is not None:
                        is_match = bool(np.array_equal(image, rgb))
                        print(
                            f"[executor] image 与 obs rgb array_equal: {is_match}"
                        )
                        if not is_match:
                            self.logger.warning(
                                "⚠️ VLA image 与 obs rgb 不一致"
                            )
                    else:
                        print("[executor] ⚠️ obs_before['rgb'] 为 None")
                else:
                    print("[executor] ⚠️ vla_input['image'] 为 None")

            obs_after, _, _, _ = env.step(action)
            done, reason = check_done(
                obs_before, obs_after, done_criteria, target_pos=target_pos
            )
            if done:
                break

        # 构造返回 message
        if done:
            message = reason
        else:
            message = f"达到最大步数 {self.max_steps}，未完成：{reason}"

        # max_steps=0 时 step 仍为 0（range(0) 不进入循环）
        steps = step + 1 if self.max_steps > 0 else 0

        return ExecResult(
            success=done,
            steps=steps,
            final_obs=obs_after,
            message=message,
        )


__all__ = [
    "BaseVLA",
    "ExecResult",
    "Executor",
    "MockVLA",
    "build_vla_input",
    "check_done",
    "create_vla",
]
