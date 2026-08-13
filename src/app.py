"""ExAct CLI 入口。"""
import os

from config import load_config
from pipeline import run_pipeline
from utils.logging import setup_logging

logger = setup_logging("app")

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local = os.path.join(root, "configs", "local.yaml")
    default = os.path.join(root, "configs", "default.yaml")
    config_path = local if os.path.isfile(local) else default

    logger.info(f"加载配置文件: {config_path}")
    cfg = load_config(config_path)
    logger.info(f"配置: backend={cfg.vla.backend}, model={cfg.llm.model}, gui={cfg.env.use_gui}")

    result = run_pipeline(cfg, user_goal=cfg.task.default_user_goal)
    logger.info(f"任务结果 success={result.agent_result.success}")
    logger.info(f"最终回答: {result.agent_result.final_answer}")
    logger.info(f"总工具调用次数: {result.agent_result.total_tool_calls}")
    for t in result.agent_result.trajectory:
        preview = t.result_text[:100] if t.result_text else ""
        logger.info(f"  [{t.step_index}] {t.tool_name}(args={t.args}) → {preview}")


if __name__ == "__main__":
    main()