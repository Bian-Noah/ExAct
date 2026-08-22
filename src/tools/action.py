"""ActionTool + target_pos 正则解析。

提供 parse_target_pos 函数和 ActionTool（langchain_core.tools.BaseTool 子类）。

iter9-verify-explore-design：在 _run 入口接入 validate_instruction，
拦截不合规指令并返回含失败原因的拒绝消息（不执行动作）。

iter9-extend（English-only）：指令必须全英文，描述与示例同步改为英文。

iter11-reset-multicam：ActionInput 加 operation: Literal["vla", "reset"] = "vla" 字段。
operation="reset" 时走 env.reset_arm_to_home() 路径不走 VLA、不走 validate_instruction。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from utils.verify import validate_instruction


def parse_target_pos(instruction: str) -> Optional[tuple[float, float, float]]:
    """从英文指令中解析目标坐标。

    支持两种格式：
    1. "x=0.5, y=0.0, z=0.4" 或 "x: 0.5" （需至少 3 个坐标）
    2. "position (0.5, 0.0, 0.3)" 或 "(0.5, 0, 0.4)"

    Args:
        instruction: 英文动作指令。

    Returns:
        (x, y, z) tuple 或 None（无法解析时）。
    """
    if not instruction:
        return None

    # 模式 1: x=0.5 / x: 0.5 / x =0.5 等
    pattern1 = re.compile(r'[xyz]\s*[=:]\s*([\-0-9.]+)', re.IGNORECASE)
    matches1 = pattern1.findall(instruction)
    if len(matches1) >= 3:
        try:
            coords = [float(m) for m in matches1[:3]]
            return tuple(coords)
        except ValueError:
            pass

    # 模式 2: (0.5, 0.0, 0.3) 或（0.5, 0.0, 0.3）
    pattern2 = re.compile(
        r'[\(（]([\-0-9.]+)\s*,\s*([\-0-9.]+)\s*,\s*([\-0-9.]+)[\)）]'
    )
    match2 = pattern2.search(instruction)
    if match2:
        try:
            return tuple(float(g) for g in match2.groups())
        except ValueError:
            pass

    return None


class ActionInput(BaseModel):
    """ActionTool 输入参数。

    iter11-reset-multicam:加 operation 字段。
      - "vla"(默认):执行 VLA 推理,instruction 必填且需符合 smolVLA 规范
      - "reset":把机械臂关节瞬时复位到 home,不走 VLA,instruction 可省略
    """

    instruction: str = Field(
        default="",
        description="English action instruction, e.g. 'pick red cube' or 'move to (0.5, 0, 0.3)'. "
        "Required for operation='vla', can be empty for operation='reset'.",
    )
    operation: Literal["vla", "reset"] = Field(
        default="vla",
        description="Operation type: 'vla' (default, run VLA inference) or "
        "'reset' (instantly reset arm joints to home, bypass VLA). "
        "Use 'reset' when VLA is OOD and joints are saturated.",
    )


class ActionTool(BaseTool):
    """执行英文自然语言动作指令。

    内部调用 Executor.run_action，返回 ExecResult.message。
    会尝试从 instruction 中解析目标坐标传给 executor。

    iter11-reset-multicam:支持 operation="reset" 触发 env.reset_arm_to_home(),
    绕过 VLA 推理,适用于 VLA OOD 后把关节从极限位姿拉回 home。
    """

    name: str = "action"
    description: str = (
        "Execute an action on the scene. Two operation modes:\n"
        "  - operation='vla' (default): run VLA inference with the given instruction.\n"
        "    Instruction must be a single verb-led English sentence (≤30 chars).\n"
        "  - operation='reset': instantly reset arm joints to home pose (bypass VLA).\n"
        "    Use this when VLA is OOD and joints are saturated to a limit pose.\n"
        "Examples:\n"
        "  action(operation='vla', instruction='pick red cube')\n"
        "  action(operation='reset')"
    )
    args_schema: Type[BaseModel] = ActionInput
    env: Any = None
    executor: Any = None  # 注入：get_adapter(...) 返回的转换函数；None 时 _ensure_adapter 兜底
    adapter: Any = None  # 注入：get_adapter(...) 返回的转换函数；None 时 _ensure_adapter 兜底

    def _ensure_adapter(self, vla, env) -> Any:
        """确保 self.adapter 非 None。

        已注入（runner 正常接线）→ 直接返回；
        未注入 → 按 spec 自动构造并 print 警告（漏接兜底，但让问题可见）。
        """
        if self.adapter is not None:
            return self.adapter
        from utils.adapter import get_adapter
        self.adapter = get_adapter(vla.output_spec, env.input_spec)
        print(
            "[ActionTool] ⚠️ 未注入 adapter，已自动按 spec 构造并临时使用: "
            f"{vla.output_spec.space} → {env.input_spec.space}。"
            "建议在 runner 组装时显式接线。"
        )
        return self.adapter

    def _run(self, instruction: str = "", operation: str = "vla") -> str:
        """执行动作指令。

        iter11-reset-multicam:operation="reset" 时短路到 env 复位路径,
        不走 validate_instruction、不走 vla.predict。

        Args:
            instruction: 自然语言动作指令字符串(reset 路径可空)。
            operation: "vla"(默认)走 VLA 推理,"reset" 走 env.reset_arm_to_home()。

        Returns:
            ExecResult.message 字符串(vla 路径),或
            "机械臂已复位到 home..." 字符串(reset 路径)。
        """
        _log = logging.getLogger("action")
        _log.info(f"action 调用开始 operation={operation} instruction={instruction}")

        # iter11-reset-multicam:reset 路径不走 validate_instruction,不消耗 VLA 规范
        if operation == "reset":
            return self._execute_reset()

        # vla 路径:校验 instruction 非空 + smolVLA 规范
        if not instruction or not instruction.strip():
            return "错误：动作指令不能为空"

        # iter9：smolVLA 指令校验拦截（只拒绝不修，返回原因让 LLM 自我纠正）
        reason = validate_instruction(instruction)
        if reason is not None:
            _log.info(f"action 被指令校验拒绝 reason={reason}")
            return f"指令不符合 smolVLA 规范：{reason}"

        # 尝试从指令中解析目标坐标
        target_pos = parse_target_pos(instruction)

        # ★ fallback 入口：确保 adapter 非 None（正常由 runner 注入，漏接自动构造）
        adapter = self._ensure_adapter(self.executor.vla, self.env)

        result = self.executor.run_action(
            self.env, instruction,
            done_criteria="reached",
            target_pos=target_pos,
            adapter=adapter,
            operation="vla",
        )
        _log.info(f"action 调用完成 success={result.success}")

        # ★ Iteration 10 chunk 契约：在 message 中附带 final_obs["ee_pos"]
        # 和"LLM 观察判断"字样，让 LLM 能看到末端位置，自主判断到位与否。
        ee_pos = result.final_obs.get("ee_pos") if result.final_obs else None
        if ee_pos is not None:
            ee_str = f"({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})"
            return (
                f"{result.message}\n"
                f"最终末端位置: {ee_str}\n"
                f"LLM 观察当前画面判断任务是否完成。"
            )
        return result.message

    def _execute_reset(self) -> str:
        """iter11-reset-multicam:走 env.reset_arm_to_home() 路径,不走 VLA。

        Returns:
            "机械臂已复位到 home..." 字符串,含末端位置反馈。
        """
        _log = logging.getLogger("action")
        _log.info("action reset 路径：调用 env.reset_arm_to_home()")
        self.env.reset_arm_to_home()
        obs_after = self.env.get_obs(include_rgb=False)
        ee_pos = obs_after.get("ee_pos")
        if ee_pos is not None:
            ee_str = f"({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})"
            return f"机械臂已复位到 home,末端位置:{ee_str}"
        return "机械臂已复位到 home"