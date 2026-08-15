"""适配分派：按 (vla_spec, env_spec) 查注册表返回转换函数。

注册表键为 (vla_space, env_space)：
  - ("joint", "joint"): joint_to_joint_transform（同空间数值映射）
  - ("task",  "joint"): task_to_joint_hardcode_transform（硬编码前 N 维直通）
  - ("task",  "task"):  identity_transform（同空间直通，Mock→Panda）

输入约定：各 adapter 接收 executor 统一解包后的裸动作值（不含 VLAOutput 包装）。
"""

from env.base import ActionSpec
from utils.adapter.adapters.identity import identity_transform
from utils.adapter.adapters.joint_to_joint import joint_to_joint_transform
from utils.adapter.adapters.task_to_joint_hardcode import (
    task_to_joint_hardcode_transform,
)


class AdapterNotFoundError(Exception):
    """找不到覆盖 (vla_spec, env_spec) 组合的 adapter。"""

    def __init__(self, vla_spec: ActionSpec, env_spec: ActionSpec):
        self.vla_spec = vla_spec
        self.env_spec = env_spec
        super().__init__(
            f"无适配器覆盖 {vla_spec.space} → {env_spec.space} 组合"
            f"（vla_spec={vla_spec!r}, env_spec={env_spec!r}）。"
            f"请检查 spec 声明，或新增对应 adapter。"
        )


# (vla_space, env_space) → 转换函数
_ADAPTER_REGISTRY: dict[tuple[str, str], callable] = {
    ("joint", "joint"): joint_to_joint_transform,
    # task→joint：无真实 IK 时用硬编码占位（前 N 维直通），保证链路不崩
    ("task", "joint"): task_to_joint_hardcode_transform,
    ("task", "task"): identity_transform,        # 同空间直通（Mock→Panda）
    # ("joint", "task"): fk_based_transform,      # 预留，未实现
}


def get_adapter(vla_spec: ActionSpec, env_spec: ActionSpec):
    """按 (vla_space, env_space) 查注册表，返回转换函数。

    Args:
        vla_spec: VLA 输出 spec。
        env_spec: env 输入 spec。

    Returns:
        转换函数：`transform(vla_output, env) -> env 原生动作`。

    Raises:
        AdapterNotFoundError: 组合不在注册表。
    """
    key = (vla_spec.space, env_spec.space)
    try:
        return _ADAPTER_REGISTRY[key]
    except KeyError:
        raise AdapterNotFoundError(vla_spec, env_spec) from None
