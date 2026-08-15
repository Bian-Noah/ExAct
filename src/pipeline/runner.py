"""Pipeline 编排层：组装 env + vla + executor + tools + agent 并运行。

iter1-pipeline-refactor-config 引入：把原本 app.py 中的手拼装配代码
内聚到 `run_pipeline(config, user_goal=None, task_spec=None)` 函数中。

Iteration 3 扩展：在 run_pipeline 启动时构造 ExperimentRecorder 并
通过 set_recorder() 注入全局单例；结束时调 finish() 写收尾日志，
set_recorder(None) 重置全局单例。PipelineResult 字段保持纯净。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.core import AgentResult
from agents.agent import create_exact_agent, run_agent
from agents.llm_factory import create_llm
from config.loader import AppConfig
from env.pybullet_env import PyBulletEnv
from executor import Executor
from executor.model.factory import create_vla
from experiment.recorder import (
    ExperimentRecorder,
    get_recorder,
    set_recorder,
)
from tools import ActionTool, ObserveTool
from utils.adapter import get_adapter
from utils.image_store import create_image_store
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


def _resolve_experiment_root(root_value: str) -> Path:
    """解析 experiment.root 为绝对路径。

    相对路径时，相对当前工作目录（CWD）解析，与项目内其他相对路径一致。
    """
    path = Path(root_value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _build_recorder(config: AppConfig) -> ExperimentRecorder:
    """根据 config.experiment 构造 ExperimentRecorder。"""
    exp_cfg = config.experiment
    root = _resolve_experiment_root(exp_cfg.root)
    return ExperimentRecorder(
        root=root,
        enabled=exp_cfg.enabled,
        log_to_stdout=exp_cfg.log_to_stdout,
    )


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

    # Iteration 3：构造 ExperimentRecorder + 注入全局单例 + start 创建目录
    recorder = _build_recorder(config)
    set_recorder(recorder)
    exp_dir = recorder.start()
    recorder.emit(
        "log",
        message=f"Pipeline started, user_goal={user_goal!r}, "
                f"vla_backend={config.vla.backend}, "
                f"experiment_enabled={config.experiment.enabled}",
    )

    # 3. 组装 env
    env = PyBulletEnv(
        env_config=config.env,
        robot_config=config.robot,
    )

    result: PipelineResult | None = None
    success = False
    summary = "未执行"
    try:
        # 3a. env.reset（不区分 GUI/DIRECT，按 config.env.use_gui 由 env 自行决策）
        env.reset(task_spec=task_spec, seed=0)

        # 4. 组装 vla（工厂）
        vla = create_vla(config.vla, config.llm)

        # 5. 组装 executor
        executor = Executor(vla, max_steps=config.vla.max_steps)

        # 6. 组装 llm
        llm = create_llm(config.llm)

        # 7. 构造 ImageStore（迭代 5：observe 工具需要它来回传 image content block）
        image_store = create_image_store(config.image_store)

        # 8. 组装 tools（ObserveTool 注入 image_store；ActionTool 注入 adapter）
        adapter = get_adapter(vla.output_spec, env.input_spec)
        tools = [
            ObserveTool(env=env, image_store=image_store),
            ActionTool(env=env, executor=executor, adapter=adapter),
        ]

        # 8. 组装 agent（max_react_rounds / max_tool_calls 从 AgentConfig 读取）
        agent = create_exact_agent(
            llm,
            tools,
            max_react_rounds=config.agent.max_react_rounds,
            max_tool_calls=config.agent.max_tool_calls,
        )

        # 9. 执行 agent（max_react_rounds 透传，决定 recursion_limit）
        agent_result = run_agent(
            agent,
            user_goal,
            max_react_rounds=config.agent.max_react_rounds,
        )

        # env_closed 暂记 False，finally 中 mutate 为 True
        result = PipelineResult(agent_result=agent_result, env_closed=False)
        success = bool(agent_result.success)
        summary = f"steps={agent_result.total_tool_calls}, success={success}"

    except Exception as e:
        success = False
        summary = f"exception={type(e).__name__}: {e}"
        recorder.emit("log", level="ERROR", message=f"Pipeline 异常: {summary}")
        raise
    finally:
        # Iteration 3：finish 写收尾日志 + set_recorder(None) 重置全局单例
        try:
            recorder.finish(success=success, summary=summary)
        except Exception as e:
            logger.warning(f"recorder.finish 失败（已忽略）: {e}")
        finally:
            set_recorder(None)

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