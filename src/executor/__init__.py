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

Iteration 11 重大变更（iter11-reset-multicam）：
  - 彻底删除 `Executor(max_steps=...)` 参数和 DeprecationWarning（plan 决策 D9）
  - `Executor.run_action` 新增 `operation: Literal["vla", "reset"]` 参数：
    - `"vla"`（默认）：走 VLA 推理 + env.step chunk 循环（iter 10 行为）
    - `"reset"`：直接调 `env.reset_arm_to_home()`，不走 VLA
  - image 参数类型：env → VLA 全链路改为 dict[str, np.ndarray]
"""

import inspect
from dataclasses import dataclass
from typing import Literal

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

    iter11-reset-multicam:`run_action` 新增 `operation` 参数,`"reset"` 走
    env.reset_arm_to_home() 不走 VLA,`"vla"` 走原 chunk 路径。
    """

    def __init__(self, vla: BaseVLA):
        """初始化 Executor。

        iter11-reset-multicam:删除 max_steps 参数(iter 10 已废弃,iter 11 完全移除)。

        Args:
            vla: BaseVLA 实例（MockVLA / LeRobotVLA / OpenVLA 等）。
        """
        self.vla = vla
        self.logger = setup_logging(__name__)

    def run_action(
        self,
        env: BaseEnv,
        instruction: str,
        done_criteria: str,
        target_pos: tuple | None = None,
        adapter=None,
        operation: Literal["vla", "reset"] = "vla",
    ) -> ExecResult:
        """按 VLA 输出的整 chunk 执行动作（Iteration 10 chunk 契约）。

        iter11-reset-multicam:operation 参数透传,`"reset"` 走 env 复位路径,
        `"vla"` 走原 VLA 推理路径。

        流程（vla 路径）：
          1) `vla.predict(image, instruction, state?)` 一次拿整 chunk VLAOutput
          2) 解包 `VLAOutput.values` → ndarray shape `(N, action_dim)`
          3) `adapter(values, env)` 一次拿转换后 actions shape `(N, env_dim)`
          4) `for action in actions: env.step(action)`，**不调** check_done
          5) 循环结束后 `check_done(...)` 做**事后报告**
          6) 返回 `ExecResult(success, steps=len(actions), ...)`

        流程（reset 路径）：
          1) `env.reset_arm_to_home()` 瞬时复位关节
          2) `env.get_obs()` 拿新 obs
          3) 返回 `ExecResult(success=None, steps=0, ...)`，**不走 VLA**

        VLA 通过 `env.get_obs()["rgb"]` 获取图像——VLA 直接接触环境，
        不依赖 LLM 传入图片 URL。VLA 是否使用 image 参数由各后端自行决定
        （MockVLA 故意忽略，LLMVLA 用语义信息，SmallVLA/OpenVLA 必须使用）。
        iter11 起 image 是 dict[str, np.ndarray](多相机)。

        Args:
            env: BaseEnv 实例（PyBulletEnv / FakeEnv 等）。
            instruction: 自然语言指令字符串(reset 路径可空)。
            done_criteria: 完成标准字符串（如 "reached" / "grasped"）。
            target_pos: 目标位置 (x, y, z)。reached 规则下用于判断 ee_pos
                是否到达目标位置；未提供时 check_done 返回 (False, ...)，
                由调用方（ActionTool / LLM）通过 final_obs 自行判断。
            adapter: 转换函数 `adapter(vla_output, env) -> env 原生动作`；
                shape `(N, env_dim)`，**只调用一次**。None 时直通。
            operation: `"vla"`（默认，走 VLA）或 `"reset"`（走 env.reset_arm_to_home）。

        Returns:
            ExecResult dataclass。

        Raises:
            env.step 抛出的异常透传。
        """
        # ★ reset 路径:直接调 env 复位,不走 VLA
        if operation == "reset":
            env.reset_arm_to_home()
            obs_after = env.get_obs(include_rgb=False)
            return ExecResult(
                success=None,
                steps=0,
                final_obs=obs_after,
                message="机械臂已复位到 home（0 个 VLA 步，不走 VLA 推理）",
            )

        # ★ vla 路径:iter 10 chunk 契约
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

        # iter11-reset-multicam:env.get_obs() 返回的 obs["rgb"] 已是 dict
        # vla_input["image"] 也是 dict,直接传给 vla.predict
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
        # 诊断(场景对齐 Stage1):打印 VLA 原始动作,区分"模型输出偏"与"执行映射错";
        # 仅进日志,不改 ToolMessage 内容(避免影响 LLM 决策与对照实验纯度)。
        if isinstance(raw, np.ndarray) and raw.size:
            first_row = np.asarray(raw[0]).reshape(-1)[:7]
            spec_components = getattr(
                getattr(vla_output, "spec", None), "components", None
            )
            print(
                "[executor] VLA raw action(7D): "
                f"{np.round(first_row, 4).tolist()}"
                + (f" spec={spec_components}" if spec_components else "")
            )
        actions = adapter(raw, env) if adapter is not None else raw

        # ★ Iteration 6：链路验证 print（多相机取首张）
        image_dict = vla_input["image"]
        rgb_dict = obs_before.get("rgb")
        if image_dict is not None:
            first_key = next(iter(image_dict))
            first_image = image_dict[first_key]
            print(
                f"[executor] VLA.predict 收到 image[{first_key}]: "
                f"shape={first_image.shape}, dtype={first_image.dtype}"
            )
            if rgb_dict is not None and first_key in rgb_dict:
                is_match = bool(np.array_equal(first_image, rgb_dict[first_key]))
                print(f"[executor] image[{first_key}] 与 obs rgb array_equal: {is_match}")
                if not is_match:
                    self.logger.warning(f"⚠️ VLA image[{first_key}] 与 obs rgb 不一致")
            else:
                print("[executor] ⚠️ obs_before['rgb'] 为 None 或不含该相机键")
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