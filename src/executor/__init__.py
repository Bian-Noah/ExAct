"""executor 模块：连接 Agent（LLM 大脑）与 env（物理世界）的执行器。

核心组件：
- ExecResult: 执行结果 dataclass
- Executor: 主循环类，循环调 VLA 驱动 env，直到完成或超时
- BaseVLA / MockVLA / create_vla: VLA 模型抽象与工厂（详见 executor.model 子包）

Iteration 10 重大变更：
  - `Executor.run_action` 不再按 `range(max_steps)` 硬编码循环。
    VLA 的 chunk 是它对"接下来要做的事"的承诺，executor 照单全收：
    1) 调 `vla.predict(...)` 一次拿整 chunk VLAOutput
    2) 解包 VLAOutput.values，shape `(N, action_dim)`
    3) 调 `adapter(values, env)` 一次拿转换后的 actions，shape `(N, env_dim)`
    4) `for action in actions: env.step(action)`，中间**不再调** `check_done`
    5) 循环结束后调 `check_done(...)` 做**事后报告**（不控制循环）
  - 删除硬编码 `max_steps` 字段对循环的控制（构造时仍兼容，DeprecationWarning）
  - VLA OOD 行为的完整轨迹能跑到 LLM 眼前，不再被中途截断
"""

import inspect
import warnings
from dataclasses import dataclass

import numpy as np

from env.base import BaseEnv
from executor.build_input import build_vla_input
from executor.check_done import check_done
from executor.model.base import BaseVLA, VLAOutput
from executor.model.factory import create_vla
from executor.model.mock.mock_vla import JointMockVLA, MockVLA
from utils.logging import setup_logging


@dataclass
class ExecResult:
    """executor.run_action 的返回值。

    Attributes:
        success: 是否成功完成（达到 done_criteria 的事后判定）。
        steps: 实际执行的步数（= len(VLA 输出的 chunk)）。
        final_obs: 最后一步的 obs dict。
        message: 完成或失败的原因描述。
    """

    success: bool
    steps: int
    final_obs: dict
    message: str


class Executor:
    """动作执行器：按 VLA 输出的整 chunk 驱动 env，**不**在中间干预。

    聚合一个 BaseVLA 实例，按 chunk 契约执行：VLA 的 predict 返回 N 步轨迹，
    executor 循环 N 次 `env.step`，循环中**不再调** `check_done`。
    """

    def __init__(self, vla: BaseVLA, max_steps: int = 50):
        """初始化 Executor。

        Args:
            vla: BaseVLA 实例（MockVLA / OpenVLA 等）。
            max_steps: **已废弃**（Iteration 10）。构造时打 DeprecationWarning。
                字段仍存入 self.max_steps 但**不再用于循环控制**。
                循环边界 = VLA 输出的 N，由 `vla.predict` 自决。
                计划在 Iteration 11 完全移除该参数。
        """
        if max_steps != 50:
            # 默认 50 是基线值；如果调用方显式传了非默认 max_steps 则报警告
            warnings.warn(
                "Executor(max_steps=...) 已废弃（Iteration 10 chunk 契约）。"
                "循环边界 = len(VLA 输出的 chunk)，由 vla.predict 自决。"
                "Iteration 11 将删除该参数。",
                DeprecationWarning,
                stacklevel=2,
            )
        self.vla = vla
        self.max_steps = max_steps  # 保留字段以兼容旧测试 / 检查（但不再用于循环）
        self.logger = setup_logging(__name__)

    def run_action(
        self,
        env: BaseEnv,
        instruction: str,
        done_criteria: str,
        target_pos: tuple | None = None,
        adapter=None,
    ) -> ExecResult:
        """按 VLA 输出的整 chunk 执行动作（Iteration 10 chunk 契约）。

        流程：
          1) `vla.predict(image, instruction, state?)` 一次拿整 chunk VLAOutput
          2) 解包 `VLAOutput.values` → ndarray shape `(N, action_dim)`
          3) `adapter(values, env)` 一次拿转换后 actions shape `(N, env_dim)`
          4) `for action in actions: env.step(action)`，**不调** check_done
          5) 循环结束后 `check_done(...)` 做**事后报告**
          6) 返回 `ExecResult(success, steps=len(actions), ...)`

        VLA 通过 `env.get_obs()["rgb"]` 获取图像——VLA 直接接触环境，
        不依赖 LLM 传入图片 URL。VLA 是否使用 image 参数由各后端自行决定
        （MockVLA 故意忽略，LLMVLA 用语义信息，SmallVLA/OpenVLA 必须使用）。

        Args:
            env: BaseEnv 实例（PyBulletEnv / FakeEnv 等）。
            instruction: 自然语言指令字符串。
            done_criteria: 完成标准字符串（如 "reached" / "grasped"）。
            target_pos: 目标位置 (x, y, z)。reached 规则下用于判断 ee_pos
                是否到达目标位置；未提供时 check_done 返回 (False, ...)，
                由调用方（ActionTool / LLM）通过 final_obs 自行判断。
            adapter: 转换函数 `adapter(vla_output, env) -> env 原生动作`；
                shape `(N, env_dim)`，**只调用一次**。None 时直通。

        Returns:
            ExecResult dataclass。

        Raises:
            env.step 抛出的异常透传。
        """
        # 缓存 VLA.predict 是否接受 state 参数（兼容老 VLA 子类）
        _predict_accepts_state = getattr(self, "_predict_accepts_state", None)
        if _predict_accepts_state is None:
            try:
                _predict_accepts_state = (
                    "state" in inspect.signature(self.vla.predict).parameters
                )
            except (TypeError, ValueError):
                _predict_accepts_state = False
            self._predict_accepts_state = _predict_accepts_state

        # ★ 1. VLA 自决 N，predict 一次拿整 chunk
        obs_before = env.get_obs()
        vla_input = build_vla_input(obs_before, instruction)
        state_vec = None
        try:
            state_vec = env.get_joint_state()
        except (NotImplementedError, AttributeError):
            state_vec = None

        if _predict_accepts_state:
            vla_output = self.vla.predict(
                vla_input["image"], instruction, state=state_vec
            )
        else:
            vla_output = self.vla.predict(vla_input["image"], instruction)

        # ★ 2. 解包 VLAOutput，拿到 ndarray shape (N, action_dim)
        if isinstance(vla_output, VLAOutput):
            raw = vla_output.values
        else:
            raw = vla_output  # 兼容裸值输入

        # ★ 3. adapter 一次转换整 chunk → shape (N, env_dim)
        actions = adapter(raw, env) if adapter is not None else raw

        # ★ Iteration 6：链路验证 print（仅第一步）
        image = vla_input["image"]
        rgb = obs_before.get("rgb")
        if image is not None:
            print(
                f"[executor] VLA.predict 收到 image: "
                f"shape={image.shape}, dtype={image.dtype}"
            )
            if rgb is not None:
                is_match = bool(np.array_equal(image, rgb))
                print(f"[executor] image 与 obs rgb array_equal: {is_match}")
                if not is_match:
                    self.logger.warning("⚠️ VLA image 与 obs rgb 不一致")
            else:
                print("[executor] ⚠️ obs_before['rgb'] 为 None")
        else:
            print("[executor] ⚠️ vla_input['image'] 为 None")

        # ★ 4. 严格按 VLA 输出的 N 步执行，**不调** check_done
        obs_after = obs_before
        for action in actions:
            obs_after, _, _, _ = env.step(action)

        # ★ 5. 事后报告：check_done 不再控制循环，仅给 LLM 一个判定结果
        done, reason = check_done(
            obs_before, obs_after, done_criteria, target_pos=target_pos
        )

        steps = len(actions) if hasattr(actions, "__len__") else 0

        return ExecResult(
            success=done,
            steps=steps,
            final_obs=obs_after,
            message=f"执行 VLA 规划的 {steps} 步: {reason}",
        )


__all__ = [
    "BaseVLA",
    "ExecResult",
    "Executor",
    "JointMockVLA",
    "MockVLA",
    "VLAOutput",
    "build_vla_input",
    "check_done",
    "create_vla",
]