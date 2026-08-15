from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field, fields, is_dataclass
from typing import TypeVar, Type, Optional, get_type_hints, Literal

import yaml

from .image_store_config import ImageStoreConfig

T = TypeVar("T")

# EnvConfig 字段值合法集合（iter2-renderer-env-mode 引入）
_ENV_MODE_VALUES: frozenset[str] = frozenset({"direct", "gui"})
_ENV_RENDERER_VALUES: frozenset[str] = frozenset({"auto", "cpu", "gpu"})


def _resolve_field_types(cls: type) -> dict[str, type]:
    """使用 get_type_hints 解析 dataclass 的真实字段类型（处理 PEP 563 字符串注解）。"""
    try:
        return get_type_hints(cls, include_extras=True)
    except Exception:
        # 某些特殊场景（如命名空间不完整）fallback：eval 每个字段 type 字符串
        resolved: dict[str, type] = {}
        module_ns = getattr(sys.modules.get(cls.__module__, None), "__dict__", {}) or {}
        local_ns = {cls.__name__: cls}
        for f in fields(cls):
            t = f.type
            if isinstance(t, str):
                try:
                    resolved[f.name] = eval(t, module_ns, local_ns)  # noqa: S307
                except Exception:
                    resolved[f.name] = object  # 无法解析就放过（后续 isinstance 总是 True）
            else:
                resolved[f.name] = t
        return resolved


@dataclass
class EnvConfig:
    """环境配置。

    iter2-renderer-env-mode 新增字段：
      - mode: "direct" | "gui"，是否开启 PyBullet GUI 窗口
      - renderer: "auto" | "cpu" | "gpu"，getCameraImage 使用的渲染器

    use_gui 字段保留为 deprecated（iter1 引入），由 mode 推导。
    """

    mode: Literal["direct", "gui"] = "direct"
    renderer: Literal["auto", "cpu", "gpu"] = "auto"
    camera_resolution: tuple[int, int] = (640, 480)
    use_gui: bool = False  # deprecated: 由 mode 替代，保留向后兼容


@dataclass
class LerobotConfig:
    """LeRobot 后端 policy 特化配置（仅 vla.backend == "lerobot" 时生效）。

    Attributes:
        policy_type: policy 类型，决定从 lerobot.policies 导入哪个类（0.5 及更早
            为 lerobot.common.policies）。
            常见值：act（轻量、默认）、diffusion、smolvla、pi0。
        device: 推理设备，默认 "cuda:0"。
        quantization: 量化等级，"none" | "8bit" | "4bit"。
            仅对 VLA 类 policy（smolvla / pi0 / pi0fast）生效；
            ACT / Diffusion 纯 PyTorch 实现无量化概念，保持 "none"。
        image_key: obs dict 中图像键名。默认 "observation.images.top"，
            仅作 fallback；加载后优先从 policy.config 自动探测。
        action_dim: policy 输出维度；超过 7 时截前 7 维映射为 Action7D。
    """
    policy_type: str = "act"
    device: str = "cuda:0"
    quantization: str = "none"
    image_key: str = "observation.images.top"
    action_dim: int = 14


@dataclass
class VLAConfig:
    backend: str = "mock"
    model_path: Optional[str] = None
    max_steps: int = 50
    lerobot: LerobotConfig = field(default_factory=LerobotConfig)


@dataclass
class LLMConfig:
    api_key: str = ""
    model: str = "MiniMax-M3"
    base_url: str = "https://api.minimax.chat/v1"
    max_tokens: int = 2048


@dataclass
class ExploreConfig:
    enabled: bool = False


@dataclass
class RobotConfig:
    """机械臂配置（URDF 路径 + 关节索引常量）。"""
    urdf_path: str = "franka_panda/panda.urdf"
    base_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    arm_joint_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    ee_link_index: int = 11
    finger_joint_indices: tuple[int, ...] = (9, 10)


@dataclass
class TaskConfig:
    """任务定义（默认 user_goal + 物体列表）。"""
    default_user_goal: str = "把机械臂移到红色方块上方"
    objects: tuple[dict, ...] = (
        {"type": "cube", "pos": [0.5, 0, 0.1], "color": "red"},
    )


@dataclass
class AgentConfig:
    """Agent 行为配置（ReAct 轮数 + 工具调用上限）。"""
    max_react_rounds: int = 5
    max_tool_calls: int = 3


@dataclass
class ExperimentConfig:
    """实验数据持久化配置（Iteration 3 启用）。

    Attributes:
        enabled: 是否开启实验录制。False 时 ExperimentRecorder 全 no-op。
        root: 实验产物根目录（相对项目根的字符串路径）。
        log_to_stdout: log 事件是否同步输出到终端。
    """
    enabled: bool = True
    root: str = "data/experiment"
    log_to_stdout: bool = True


@dataclass
class AppConfig:
    env: EnvConfig
    vla: VLAConfig
    llm: LLMConfig
    explore: ExploreConfig
    robot: RobotConfig
    task: TaskConfig
    agent: AgentConfig
    experiment: ExperimentConfig
    image_store: ImageStoreConfig


def _check_type(name: str, value: object, expected: type) -> None:
    """校验 value 的类型是否匹配 expected（处理 None、tuple/list、基本类型）。"""
    if expected is type(None):  # noqa: E721
        if value is not None:
            raise TypeError(f"字段 '{name}' 类型错误：期望 None，得到 {type(value).__name__}")
        return
    origin = getattr(expected, "__origin__", None)
    # Optional[X] → typing.Union[X, None]
    if origin is not None and hasattr(expected, "__args__"):
        args = expected.__args__
        # Optional[X] = Union[X, None]
        if type(None) in args:
            if value is None:
                return
            non_none = [a for a in args if a is not type(None)]  # noqa: E721
            if len(non_none) == 1:
                _check_type(name, value, non_none[0])
                return
        # tuple[int, int] / list[...] 等泛型
        if origin is tuple or origin is list:
            if not isinstance(value, (list, tuple)):
                raise TypeError(
                    f"字段 '{name}' 类型错误：期望 list/tuple，得到 {type(value).__name__}"
                )
            return
        raise TypeError(f"字段 '{name}' 包含不受支持的泛型：{expected}")
    # bool 是 int 的子类，先单独判断
    if expected is bool:
        if not isinstance(value, bool):
            raise TypeError(
                f"字段 '{name}' 类型错误：期望 bool，得到 {type(value).__name__}"
            )
        return
    if expected is int:
        # bool 不应被当作 int
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"字段 '{name}' 类型错误：期望 int，得到 {type(value).__name__}"
            )
        return
    if expected is str:
        if not isinstance(value, str):
            raise TypeError(
                f"字段 '{name}' 类型错误：期望 str，得到 {type(value).__name__}"
            )
        return
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"字段 '{name}' 类型错误：期望 float，得到 {type(value).__name__}"
            )
        return
    if not isinstance(value, expected):
        raise TypeError(
            f"字段 '{name}' 类型错误：期望 {expected.__name__}，得到 {type(value).__name__}"
        )


def _from_dict(data: dict, cls: Type[T]) -> T:
    """dict → dataclass 实例，带字段存在性 + 类型校验 + 未知字段警告。"""
    if not isinstance(data, dict):
        raise ValueError(f"加载 {cls.__name__} 失败：顶层内容不是 mapping，而是 {type(data).__name__}")
    if not is_dataclass(cls):
        raise TypeError(f"_from_dict 只接受 dataclass 类型，{cls.__name__} 不是")

    cls_fields = {f.name: f for f in fields(cls)}
    resolved_types = _resolve_field_types(cls)
    # 未知字段警告
    for k in data.keys():
        if k not in cls_fields:
            warnings.warn(f"配置文件中包含未知字段 '{cls.__name__}.{k}'，将被忽略")

    kwargs: dict = {}
    for fname, f in cls_fields.items():
        ftype = resolved_types.get(fname, object)
        default_value = f.default if f.default is not f.default_factory else f.default_factory() if callable(f.default_factory) else None
        has_default = f.default is not f.default_factory or f.default_factory is not f.default_factory
        present = fname in data

        if not present:
            # 嵌套 dataclass 字段：未提供时用空 dict，递归时由子 dataclass 默认值兜底
            if is_dataclass(ftype):
                kwargs[fname] = _from_dict({}, ftype)
                continue
            if has_default:
                kwargs[fname] = default_value
                continue
            raise ValueError(f"配置文件缺少必填字段 '{cls.__name__}.{fname}'")

        raw_value = data[fname]
        # camera_resolution 特殊处理：yaml list → tuple
        if cls is EnvConfig and fname == "camera_resolution":
            if not isinstance(raw_value, list) or len(raw_value) != 2:
                raise ValueError(
                    f"字段 'EnvConfig.camera_resolution' 格式错误：期望长度为 2 的 list [W, H]，得到 {raw_value!r}"
                )
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in raw_value):
                raise ValueError(
                    f"字段 'EnvConfig.camera_resolution' 格式错误：两个元素都必须是整数，得到 {raw_value!r}"
                )
            kwargs[fname] = (raw_value[0], raw_value[1])
            continue
        # EnvConfig.mode / renderer 字段值合法性校验（iter2-renderer-env-mode）
        if cls is EnvConfig and fname == "mode":
            if not isinstance(raw_value, str) or raw_value not in _ENV_MODE_VALUES:
                raise ValueError(
                    f"字段 'EnvConfig.mode' 取值错误：期望 {_ENV_MODE_VALUES} 之一，得到 {raw_value!r}"
                )
            kwargs[fname] = raw_value
            continue
        if cls is EnvConfig and fname == "renderer":
            if not isinstance(raw_value, str) or raw_value not in _ENV_RENDERER_VALUES:
                raise ValueError(
                    f"字段 'EnvConfig.renderer' 取值错误：期望 {_ENV_RENDERER_VALUES} 之一，得到 {raw_value!r}"
                )
            kwargs[fname] = raw_value
            continue
        # 通用 tuple 字段：list → tuple 转换 + 元素类型校验
        # 适用于 RobotConfig.arm_joint_indices / base_position 等
        origin = getattr(ftype, "__origin__", None)
        if origin is tuple and isinstance(raw_value, list):
            type_args = getattr(ftype, "__args__", ())
            if len(type_args) == 2 and type_args[1] is Ellipsis:
                # tuple[X, ...] 变长
                elem_t = type_args[0]
                for i, v in enumerate(raw_value):
                    _check_type(f"{cls.__name__}.{fname}[{i}]", v, elem_t)
                kwargs[fname] = tuple(raw_value)
                continue
            if len(type_args) > 0:
                # 固定长度 tuple[X, Y, ...]
                if len(raw_value) != len(type_args):
                    raise ValueError(
                        f"字段 '{cls.__name__}.{fname}' 长度错误：期望 {len(type_args)}，得到 {len(raw_value)}"
                    )
                for i, (v, t) in enumerate(zip(raw_value, type_args)):
                    _check_type(f"{cls.__name__}.{fname}[{i}]", v, t)
                kwargs[fname] = tuple(raw_value)
                continue
        # 子 dataclass 嵌套（目前没用，但保留可扩展性）
        if is_dataclass(ftype) and isinstance(raw_value, dict):
            kwargs[fname] = _from_dict(raw_value, ftype)
            continue
        # 普通类型校验
        _check_type(f"{cls.__name__}.{fname}", raw_value, ftype)
        kwargs[fname] = raw_value

    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ValueError(f"构造 {cls.__name__} 失败：{e}") from e


def load_config(path: str) -> AppConfig:
    """加载 yaml 配置文件，返回类型化 AppConfig。

    可能抛出：
      - FileNotFoundError: path 不存在
      - ValueError: YAML 语法错误 / 缺字段 / 格式不匹配
      - TypeError: 字段类型不匹配
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"配置文件不存在：{path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"YAML 解析失败（{path}）：{e}") from e

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML 顶层必须是 mapping，实际为 {type(raw).__name__}")

    env_raw = raw.get("env", {}) or {}
    vla_raw = raw.get("vla", {}) or {}
    llm_raw = raw.get("llm", {}) or {}
    explore_raw = raw.get("explore", {}) or {}
    robot_raw = raw.get("robot", {}) or {}
    task_raw = raw.get("task", {}) or {}
    agent_raw = raw.get("agent", {}) or {}
    experiment_raw = raw.get("experiment", {}) or {}
    image_store_raw = raw.get("image_store", {}) or {}

    for key, section in (
        ("env", env_raw),
        ("vla", vla_raw),
        ("llm", llm_raw),
        ("explore", explore_raw),
        ("robot", robot_raw),
        ("task", task_raw),
        ("agent", agent_raw),
        ("experiment", experiment_raw),
        ("image_store", image_store_raw),
    ):
        if not isinstance(section, dict):
            raise ValueError(f"配置节 '{key}' 必须是 mapping，实际为 {type(section).__name__}")

    return AppConfig(
        env=_from_dict(env_raw, EnvConfig),
        vla=_from_dict(vla_raw, VLAConfig),
        llm=_from_dict(llm_raw, LLMConfig),
        explore=_from_dict(explore_raw, ExploreConfig),
        robot=_from_dict(robot_raw, RobotConfig),
        task=_from_dict(task_raw, TaskConfig),
        agent=_from_dict(agent_raw, AgentConfig),
        experiment=_from_dict(experiment_raw, ExperimentConfig),
        image_store=_from_dict(image_store_raw, ImageStoreConfig),
    )
