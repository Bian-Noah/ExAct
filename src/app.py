"""ExAct 第四步迭代验证脚本。

使用 LangChain create_agent 替代手写 ReAct 循环：
1. 加载 configs/local.yaml（含真实 API Key）
2. 创建 PyBulletPandaEnv（GUI 模式）+ MockVLA + Executor
3. ChatOpenAI + LCObserveTool + LCActionTool + create_agent
4. Agent 自主决策：先 observe 观察场景，再 action 执行动作，最后文本回答
5. 打印工具调用轨迹和最终结果
6. close env

运行方式：
    conda activate exact
    PYTHONPATH=src python src/app.py
"""

import os
import sys

# PYTHONPATH fallback：确保 src 目录在搜索路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import load_config
from env.pybullet_env import PyBulletPandaEnv
from executor import Executor, MockVLA
from agents import create_llm, create_exact_agent, run_agent, AgentResult
from tools import LCObserveTool, LCActionTool
from utils.logging import setup_logging

logger = setup_logging("app")


def main():
    # 1. 加载配置（优先 local.yaml，回退 default.yaml）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_path = os.path.join(project_root, "configs", "local.yaml")
    default_path = os.path.join(project_root, "configs", "default.yaml")
    config_path = local_path if os.path.isfile(local_path) else default_path

    logger.info(f"加载配置文件: {config_path}")
    cfg = load_config(config_path)
    logger.info(
        f"配置加载完成: backend={cfg.vla.backend}, "
        f"model={cfg.llm.model}, gui={cfg.env.use_gui}"
    )

    # 2. 创建 env + executor
    env = PyBulletPandaEnv(
        use_gui=cfg.env.use_gui,
        camera_resolution=cfg.env.camera_resolution,
    )
    task_spec = {
        "objects": [
            {"type": "cube", "pos": [0.5, 0, 0.1], "color": "red"},
        ]
    }

    try:
        logger.info("重置环境...")
        env.reset(task_spec=task_spec, seed=0)

        vla = MockVLA(seed=0)
        executor = Executor(vla, max_steps=cfg.vla.max_steps)

        # 3. 初始化 LLM + tools + agent（LangChain create_agent）
        llm = create_llm(cfg.llm)
        # LangChain BaseTool 是 pydantic 模型，必须用关键字参数传字段
        tools = [LCObserveTool(env=env), LCActionTool(env=env, executor=executor)]
        agent = create_exact_agent(llm, tools)

        # 4. 运行 Agent
        user_goal = "把机械臂移到红色方块上方"
        logger.info(f"开始执行任务: {user_goal}")
        result: AgentResult = run_agent(agent, user_goal)

        # 5. 打印结果
        logger.info(f"任务结果 success={result.success}")
        logger.info(f"最终回答: {result.final_answer}")
        logger.info(f"总工具调用次数: {result.total_tool_calls}")
        for t in result.trajectory:
            preview = t.result_text[:100] if t.result_text else ""
            logger.info(
                f"  [{t.step_index}] {t.tool_name}(args={t.args}) → {preview}"
            )
    finally:
        logger.info("关闭环境...")
        env.close()
        logger.info("Done.")


if __name__ == "__main__":
    main()
