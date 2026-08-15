"""adapter 适配层：把 VLA 输出转换为 env 原生动作。

对外入口：`get_adapter(vla_spec, env_spec)` 查注册表返回转换函数。
"""

from utils.adapter.main import (
    AdapterNotFoundError,
    get_adapter,
)

__all__ = [
    "AdapterNotFoundError",
    "get_adapter",
]
