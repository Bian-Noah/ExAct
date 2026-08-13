"""pipeline 包：顶层编排层。

暴露 `run_pipeline(config, user_goal=None, task_spec=None) -> PipelineResult`
与 `PipelineResult` dataclass。
"""

from .runner import PipelineResult, run_pipeline

__all__ = ["PipelineResult", "run_pipeline"]