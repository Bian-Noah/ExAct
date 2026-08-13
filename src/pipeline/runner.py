"""Pipeline 编排层：组装 env + vla + executor + tools + agent 并运行。

iter1-pipeline-refactor-config 引入：把原本 app.py 中的手拼装配代码
内聚到 `run_pipeline(config, user_goal=None, task_spec=None)` 函数中。
本迭代的 PipelineResult 是轻量版本（不含实验持久化），Iteration 3 会
扩展 storage 写入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.core import AgentResult
from agents.agent import create_exact_agent, run_agent
from agents.llm_factory import create_llm
from config.loader import AppConfig
from env.pybullet_env import PyBulletPandaEnv
from executor import Executor
from executor.model.factory import create_vla
from tools import ActionTool, ObserveTool
from utils.logging import setup_logging

logger = setup_logging("pipeline")


@dataclass
class PipelineResult:
    """Pipeline 运行结果（轻量版本，Iteration 3 会扩展 storage）。

    Attributes:
        agent_result: AgentResult（来自 agents.core，含 trajectory/final_answer/total_tool_calls）。
        env_closed: env.close() 是否被调用（异常路径下也应为 True）。
    """
    agent_result: AgentResult
    env_closed: bool = False


def _to_task_spec_dict(task_config_objects: tuple[dict, ...]) -> dict:
    """把 `config.task.objects` 的 tuple[dict] 转成 env.reset 期望的 dict 格式。"""
    return {"objects": [dict(obj) for obj in task_config_objects]}


def run_pipeline(
    config: AppConfig,
    user_goal: str | None = None,
    task_spec: dict | None = None,
) -> PipelineResult:
    """组装 env + vla + executor + tools + agent，运行任务，关闭 env。

    Args:
        config: 全局配置（env/vla/llm/agent/task/robot/...）。
        user_goal: 用户目标字符串。为 None 时使用 config.task.default_user_goal。
        task_spec: 任务定义（如 {"objects": [...]}）。为 None 时使用
            config.task.objects 转为 dict。

    Returns:
        PipelineResult，含 agent_result 与 env_closed 标志。

    Raises:
        NotImplementedError: vla backend 未实现。
        RuntimeError: env 创建失败 / agent 执行异常。
    """
    # 1. 解析 user_goal（缺省用 config.task.default_user_goal）
    if user_goal is None:
        user_goal = config.task.default_user_goal

    # 2. 解析 task_spec（缺省用 config.task.objects 转 dict）
    if task_spec is None:
        task_spec = _to_task_spec_dict(config.task.objects)

    # 3. 组装 env
    env = PyBulletPandaEnv(
        env_config=config.env,
        robot_config=config.robot,
    )

    result: PipelineResult | None = None
    try:
        # 3a. env.reset（不区分 GUI/DIRECT，按 config.env.use_gui 由 env 自行决策）
        env.reset(task_spec=task_spec, seed=0)

        # 4. 组装 vla（工厂）
        vla = create_vla(config.vla, config.llm)

        # 5. 组装 executor
        executor = Executor(vla, max_steps=config.vla.max_steps)

        # 6. 组装 llm
        llm = create_llm(config.llm)

        # 7. 组装 tools
        tools = [ObserveTool(env=env), ActionTool(env=env, executor=executor)]

        # 8. 组装 agent（max_react_rounds / max_tool_calls 从 AgentConfig 读取）
        agent = create_exact_agent(
            llm,
            tools,
            max_react_rounds=config.agent.max_react_rounds,
            max_tool_calls=config.agent.max_tool_calls,
        )

        # 9. 执行 agent
        agent_result = run_agent(agent, user_goal)

        # env_closed 暂记 False，finally 中 mutate 为 True
        result = PipelineResult(agent_result=agent_result, env_closed=False)

    finally:
        # 10. env.close() 始终执行
        try:
            env.close()
        except Exception as e:
            logger.warning(f"env.close() 失败（已忽略）: {e}")
        # 异常路径下 result 仍为 None，调用方拿不到 result，会收到原始异常
        # 正常路径下 mutate env_closed = True
        if result is not None:
            result.env_closed = True
        logger.info("pipeline 完成，env 已关闭")

    return result