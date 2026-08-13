"""Iteration 6 链路验证脚本：手动构造 env/vla/executor，跑一次 run_action，确认 print 输出。"""
import sys
sys.path.insert(0, "src")

from env.pybullet_env import PyBulletPandaEnv
from config.loader import load_config
from executor import Executor
from executor.model.mock.mock_vla import MockVLA

config = load_config("configs/default.yaml")
env = PyBulletPandaEnv(env_config=config.env, robot_config=config.robot)
env.reset(task_spec={"objects": [dict(obj) for obj in config.task.objects]}, seed=0)

vla = MockVLA()
executor = Executor(vla, max_steps=1)

result = executor.run_action(env, "move", done_criteria="reached", target_pos=(0.0, 0.0, 0.0))
print(f"[smoke] result.success={result.success}, steps={result.steps}, message={result.message!r}")
env.close()