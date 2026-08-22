"""ActionTool adapter 注入 / fallback 单元测试（robot-vla-adapter 任务 7）。"""

from env.base import Action7D, ActionSpec, BaseEnv
from executor import Executor, MockVLA
from tools.action import ActionTool
from utils.adapter import get_adapter
from utils.adapter.adapters.identity import identity_transform


class _FakeEnv(BaseEnv):
    """最小 FakeEnv：ee_pos 不移动，reached 永不满足（跑 max_steps）。"""

    def __init__(self):
        self._ee_pos = (0.0, 0.0, 0.0)

    def reset(self, task_spec=None, seed=0):
        return self._make_obs()

    def step(self, action):
        return self._make_obs(), 0.0, False, {}

    def render(self):
        import numpy as np
        return np.zeros((10, 10, 3), dtype=np.uint8)

    def get_obs(self, include_rgb=True):
        return self._make_obs()

    def close(self):
        pass

    @property
    def input_spec(self):
        return ActionSpec("task", ("dx", "dy", "dz", "drx", "dry", "drz", "gripper"))

    def _make_obs(self):
        import numpy as np
        # iter11-reset-multicam:rgb 改为 dict[str, ndarray](单相机 cam1)
        return {
            "rgb": {"cam1": np.zeros((10, 10, 3), dtype=np.uint8)},
            "object_info": [],
            "ee_pos": self._ee_pos,
            "state_desc": "",
        }


def _make_env_and_executor():
    env = _FakeEnv()
    env.reset()
    vla = MockVLA(seed=0)
    # iter11-reset-multicam:Executor 不再接收 max_steps 参数
    executor = Executor(vla)
    return env, executor


def test_action_tool_injected_adapter_no_warning(capsys):
    """注入 adapter → _run 透传，不 print 警告。"""
    env, executor = _make_env_and_executor()
    tool = ActionTool(env=env, executor=executor, adapter=identity_transform)
    result = tool._run("move")
    captured = capsys.readouterr()
    assert "未注入 adapter" not in captured.out
    assert isinstance(result, str)


def test_action_tool_fallback_adapter_auto_constructs(capsys):
    """未注入 adapter → _ensure_adapter 自动按 spec 构造 + print 警告。"""
    env, executor = _make_env_and_executor()
    tool = ActionTool(env=env, executor=executor)  # 不传 adapter
    result = tool._run("move")
    captured = capsys.readouterr()
    assert "未注入 adapter" in captured.out
    assert isinstance(result, str)


def test_action_tool_fallback_sets_internal_adapter():
    """fallback 后 tool.adapter 被填充为 get_adapter 结果。"""
    env, executor = _make_env_and_executor()
    tool = ActionTool(env=env, executor=executor)
    tool._run("move")
    expected = get_adapter(executor.vla.output_spec, env.input_spec)
    assert tool.adapter is expected


def test_action_tool_with_coords_still_works():
    """含坐标指令 + adapter 注入 → target_pos 传递不破坏。"""
    env, executor = _make_env_and_executor()
    original = executor.run_action
    captured = {}

    def capturing(env_arg, instr, done_criteria, **kw):
        captured["target_pos"] = kw.get("target_pos")
        captured["adapter"] = kw.get("adapter")
        return original(env_arg, instr, done_criteria, **kw)

    executor.run_action = capturing
    tool = ActionTool(env=env, executor=executor, adapter=identity_transform)
    tool._run("move to x=0.5, y=0.0, z=0.4")
    assert captured["target_pos"] == (0.5, 0.0, 0.4)
    assert captured["adapter"] is identity_transform


def test_action_tool_empty_instruction_short_circuits(capsys):
    """空指令不调 executor、不构造 adapter。"""
    env, executor = _make_env_and_executor()
    tool = ActionTool(env=env, executor=executor)  # 未注入
    result = tool._run("")
    captured = capsys.readouterr()
    assert "不能为空" in result
    assert "未注入 adapter" not in captured.out  # 短路，未触发 _ensure_adapter
