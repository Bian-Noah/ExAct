from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, fields, is_dataclass
from typing import TypeVar, Type, Optional, get_type_hints

import yaml

T = TypeVar("T")


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
    use_gui: bool = True
    camera_resolution: tuple[int, int] = (640, 480)


@dataclass
class VLAConfig:
    backend: str = "mock"
    model_path: Optional[str] = None
    max_steps: int = 50


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
class AppConfig:
    env: EnvConfig
    vla: VLAConfig
    llm: LLMConfig
    explore: ExploreConfig


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

    for key, section in (("env", env_raw), ("vla", vla_raw), ("llm", llm_raw), ("explore", explore_raw)):
        if not isinstance(section, dict):
            raise ValueError(f"配置节 '{key}' 必须是 mapping，实际为 {type(section).__name__}")

    return AppConfig(
        env=_from_dict(env_raw, EnvConfig),
        vla=_from_dict(vla_raw, VLAConfig),
        llm=_from_dict(llm_raw, LLMConfig),
        explore=_from_dict(explore_raw, ExploreConfig),
    )
