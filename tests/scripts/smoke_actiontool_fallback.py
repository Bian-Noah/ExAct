"""robot-vla-adapter 功能场景 C：ActionTool 未注入 adapter 的 fallback。

验证漏接路径：手动构造 ActionTool 不传 adapter，_run 时自动按 spec 构造并
print 警告，仍能跑通。

强制 EnvConfig(mode="direct")，不开 GUI 弹窗。PASS/FAIL 通过退出码表达。
运行：PYTHONPATH=src python tests/scripts/smoke_actiontool_fallback.py
"""

import sys

sys.path.insert(0, "src")

from env.base import Action7D, ActionSpec, BaseEnv
from executor import Executor, MockVLA
from tools.action import ActionTool


class _MiniEnv(BaseEnv):
    """最小 FakeEnv：ee_pos 不移动，跑 max_steps 后结束。"""

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
        return {
            "rgb": np.zeros((10, 10, 3), dtype=np.uint8),
            "object_info": [],
            "ee_pos": self._ee_pos,
            "state_desc": "",
        }


def main() -> int:
    import contextlib
    import io

    env = _MiniEnv()
    env.reset()
    vla = MockVLA(seed=0)
    executor = Executor(vla, max_steps=2)

    # 手动构造 ActionTool，不传 adapter（漏接路径）
    tool = ActionTool(env=env, executor=executor)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = tool._run("移动到红色方块上方")

    out = buf.getvalue()
    assert "[ActionTool] ⚠️ 未注入 adapter" in out, (
        f"应打印 fallback 警告，实际输出：{out!r}"
    )
    assert isinstance(result, str) and len(result) > 0, (
        f"应返回非空 message，实际 {result!r}"
    )
    assert tool.adapter is not None, "fallback 后 tool.adapter 应被填充"
    print("[smoke] ✓ fallback 警告已打印")
    print(f"[smoke] ✓ _run 返回 message 非空: {result!r}")
    print("[smoke] ✓ tool.adapter 已自动填充")

    print("\n=== PASS ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n=== FAIL: {type(e).__name__}: {e} ===")
        sys.exit(1)
