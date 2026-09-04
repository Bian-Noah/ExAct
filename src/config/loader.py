from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field, fields, is_dataclass
from typing import TypeVar, Type, Optional, get_type_hints, Literal

import yaml

from .image_store_config import ImageStoreConfig
# iter12-video-recording:VideoConfig 定义在 experiment/video/config.py,
# loader 复用 import 避免双份定义漂移(设计文档 1.8 注)
from experiment.video.config import VideoConfig

T = TypeVar("T")

# EnvConfig 字段值合法集合（iter2-renderer-env-mode 引入）
_ENV_MODE_VALUES: frozenset[str] = frozenset({"direct", "gui"})
_ENV_RENDERER_VALUES: frozenset[str] = frozenset({"auto", "cpu", "gpu"})


# iter11-reset-multicam:多相机支持,env 端相机列表配置项
@dataclass(frozen=True)
class CameraSpec:
    """env 相机配置项（iter11-reset-multicam）。

    Attributes:
        name: 相机名,作为 obs["rgb"] 的 key。需与下游 VLA policy config
            的 VISUAL 键名一致（如 smolVLA: observation.images.camera1/2/3）。
        target: 注视点 (x, y, z),世界坐标。
        distance: 相机到 target 的距离（米）。
        yaw / pitch / roll: 相机姿态（度）。
        fov: 视角（度）,默认 60。
        resolution: (W, H) 像素,默认 (640, 480)。
            VLA 模型期望 256x256 时可在每相机独立配小分辨率,
            避免 render 后再 resize 的开销。
    """

    name: str
    target: tuple[float, float, float]
    distance: float
    yaw: float
    pitch: float
    roll: float
    fov: float = 60.0
    resolution: tuple[int, int] = (640, 480)


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
    # iter11-reset-multicam:多相机支持。默认 1 个 overhead 相机,与 iter 2 行为兼容。
    cameras: tuple[CameraSpec, ...] = (
        CameraSpec(
            name="observation.images.top",
            target=(0.5, 0.0, 0.5),
            distance=1.5,
            yaw=50,
            pitch=-35,
            roll=0,
        ),
    )
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
class MockConfig:
    """Mock VLA 后端特化配置（仅 vla.backend == "mock" 时生效）。

    与 LerobotConfig 同模式：顶层 backend 决定大类，嵌套特化字段决定细节。

    Attributes:
        variant: mock 实现变体。
            - "task"：默认，task 空间 7 维（MockVLA，用于 Panda 链路）
            - "joint"：joint 空间 6 维（JointMockVLA，用于 so101 链路）
    """
    variant: str = "task"


@dataclass
class OpenVLAConfig:
    """OpenVLA 后端特化配置（仅 vla.backend == "openvla" 时生效）。

    与 LerobotConfig / MockConfig 同模式：顶层 backend 决定大类，嵌套特化字段决定细节。

    Attributes:
        unnorm_key: 反归一化键，决定动作物理量纲与顺序。
            "bridge_orig"（BridgeData V2 训练版）顺序与 Action7D 一致；
            LIBERO 等微调版顺序可能不同，使用前需核对。
        attn_impl: attention 实现，"flash_attention_2"（需安装 flash-attn，
            缺包时后端自动回退 eager）| "eager"。
        device: 推理设备，默认 "cuda:0"（quantization="4bit" 时被 device_map 覆盖）。
        dtype: 推理精度字符串（构造期不解析为 torch.dtype，避免 import torch），
            取值 "bfloat16" | "float16" | "float32"。
        quantization: 量化等级，"4bit"（bnb nf4，4060 8GB 默认，~4GB 显存）
            | "none"（bf16 全精度，需 16GB+ 显存）。
    """
    unnorm_key: str = "bridge_orig"
    attn_impl: str = "flash_attention_2"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    quantization: str = "4bit"


@dataclass
class VLAConfig:
    backend: str = "mock"
    model_path: Optional[str] = None
    # iter11-reset-multicam:max_steps 字段已删除(iter 10 后无消费方)
    mock: MockConfig = field(default_factory=MockConfig)
    lerobot: LerobotConfig = field(default_factory=LerobotConfig)
    openvla: OpenVLAConfig = field(default_factory=OpenVLAConfig)


@dataclass
class LLMConfig:
    api_key: str = ""
    model: str = "MiniMax-M3"
    base_url: str = "https://api.minimax.chat/v1"
    max_tokens: int = 2048


@dataclass
class ExploreConfig:
    """探索机制配置。

    Attributes:
        enabled: 是否启用 Explore（False 时 ExploreTool 不注入 agent，
            Explore 实例也不构造，flush 不执行）。
        root: 探索笔记落盘根目录（相对项目根的字符串路径）。
    """
    enabled: bool = False
    root: str = "data/explore"


@dataclass
class PandaRobotConfig:
    """Panda 机器人特化配置（仅 robot.type == "panda" 时生效）。

    Attributes:
        arm_joint_indices: 7 个机械臂关节索引。
        ee_link_index: 末端执行器 link 索引（panda_hand）。
        finger_joint_indices: 夹爪关节索引。
    """
    arm_joint_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    ee_link_index: int = 11
    finger_joint_indices: tuple[int, ...] = (9, 10)


@dataclass
class SO101RobotConfig:
    """SO101 机器人特化配置（仅 robot.type == "so101" 时生效）。

    Attributes:
        arm_joint_indices: 5 个机械臂关节索引（joint[0-4]）。
        ee_link_index: 末端执行器 link 索引（wrist_roll 之后）。
        gripper_joint_index: 夹爪关节索引（joint[6]=gripper 铰链）。
    """
    arm_joint_indices: tuple[int, ...] = (0, 1, 2, 3, 4)
    ee_link_index: int = 4
    gripper_joint_index: int = 6


@dataclass
class WidowxRobotConfig:
    """WidowX 机器人特化配置（仅 robot.type == "widowx" 时生效）。

    add-widowx-robot 新增：wx250.urdf 关节布局 5 臂 (0-4) + 2 夹爪 (9,10)，
    ee_link = /ee_gripper_link (index 11)。URDF 首次运行时自动下载
    （urdf_url → urdf_local_path），本地已有则跳过（幂等）。

    Attributes:
        arm_joint_indices: 5 个机械臂关节索引。
        ee_link_index: 末端执行器 link 索引（/ee_gripper_link）。
        gripper_joint_indices: 夹爪 2 指关节索引。
        urdf_url: URDF 下载地址（github.com/raw 通道，本环境实测可用）。
        urdf_local_path: URDF 本地落盘路径（与 robot.urdf_path 顶层字段须一致）。
    """
    arm_joint_indices: tuple[int, ...] = (0, 1, 2, 3, 4)
    ee_link_index: int = 11
    gripper_joint_indices: tuple[int, ...] = (9, 10)
    urdf_url: str = (
        "https://github.com/ismarou/manipulator_gym/raw/main/"
        "manipulator_gym/utils/assets/widowx/urdf/wx250.urdf"
    )
    urdf_local_path: str = "robot/widowx/wx250.urdf"


@dataclass
class RobotConfig:
    """机械臂配置。

    robot-vla-adapter 重构：
      - type: 顶层类型分派字段（决定加载 env/robot/<type>/ 下哪个 Robot）。
      - urdf_path / base_position: 保留顶层公共字段（所有机器人通用）。
      - panda: Panda 特化嵌套配置（arm_joint_indices / ee_link_index /
        finger_joint_indices 迁入此处，带默认值 fallback）。
      - so101: SO101 特化嵌套配置（arm_joint_indices / ee_link_index）。
      - widowx: WidowX 特化嵌套配置（add-widowx-robot 新增）。
    """
    type: str = "panda"
    urdf_path: str = "franka_panda/panda.urdf"
    base_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    panda: PandaRobotConfig = field(default_factory=PandaRobotConfig)
    so101: SO101RobotConfig = field(default_factory=SO101RobotConfig)
    widowx: WidowxRobotConfig = field(default_factory=WidowxRobotConfig)


@dataclass
class TaskConfig:
    """任务定义（默认 user_goal + 物体列表）。"""
    default_user_goal: str = "把机械臂移到红色方块上方"
    objects: tuple[dict, ...] = (
        {"type": "cube", "pos": [0.5, 0, 0.1], "color": "red"},
    )


@dataclass
class AgentConfig:
    """Agent 行为配置（ReAct 轮数 + 工具调用上限 + 额外提示词段）。"""
    max_react_rounds: int = 5
    max_tool_calls: int = 3
    # 额外提示词：非空时替换系统提示词中的"后端模型/运行策略"说明段
    # （COMMON 通用段与报告契约固定保留）。空 = 用历史完整提示词，行为不变。
    # yaml 多行文本用 | 块；命名取"额外提示词"，不是"VLA 模型的提示词"。
    extra_prompt: str = ""


@dataclass
class ExperimentConfig:
    """实验数据持久化配置（Iteration 3 启用）。

    Attributes:
        enabled: 是否开启实验录制。False 时 ExperimentRecorder 全 no-op。
        root: 实验产物根目录（相对项目根的字符串路径）。
        log_to_stdout: log 事件是否同步输出到终端。
        video: 视频录制配置（iter12-video-recording 引入）；None = 不录视频（旧配置兼容）。
    """
    enabled: bool = True
    root: str = "data/experiment"
    log_to_stdout: bool = True
    video: VideoConfig | None = None


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

    # robot-vla-adapter 兼容：RobotConfig 旧平铺格式（arm_joint_indices 等在顶层）
    # 合并进 panda 特化嵌套块，并从顶层移除，避免未知字段警告 + 丢配置。
    if cls is RobotConfig:
        legacy_keys = ("arm_joint_indices", "ee_link_index", "finger_joint_indices")
        if any(k in data for k in legacy_keys):
            panda_data = dict(data.get("panda", {}) or {})
            new_data = dict(data)
            for k in legacy_keys:
                if k in data:
                    panda_data.setdefault(k, data[k])
                    new_data.pop(k, None)
            new_data["panda"] = panda_data
            data = new_data

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
        # iter11-reset-multicam:cameras 特殊处理——list of dict → tuple[CameraSpec, ...]
        if cls is EnvConfig and fname == "cameras":
            if not isinstance(raw_value, list):
                raise ValueError(
                    f"字段 'EnvConfig.cameras' 格式错误：期望 list，得到 {type(raw_value).__name__}"
                )
            cameras_list: list[CameraSpec] = []
            seen_names: set[str] = set()
            for i, cam_data in enumerate(raw_value):
                if not isinstance(cam_data, dict):
                    raise ValueError(
                        f"字段 'EnvConfig.cameras[{i}]' 格式错误：期望 dict，得到 {type(cam_data).__name__}"
                    )
                cam_spec = _from_dict(cam_data, CameraSpec)
                if cam_spec.name in seen_names:
                    warnings.warn(
                        f"配置 'EnvConfig.cameras' 中存在重复 name '{cam_spec.name}'，"
                        f"后者将覆盖前者的渲染结果（dict key 冲突）"
                    )
                seen_names.add(cam_spec.name)
                cameras_list.append(cam_spec)
            kwargs[fname] = tuple(cameras_list)
            continue
        # iter12-video-recording:ExperimentConfig.video 特殊处理——
        # dict → VideoConfig;解析失败回退 None + warning(设计文档 4.3 配置类错误策略)
        if cls is ExperimentConfig and fname == "video":
            if raw_value is None:
                kwargs[fname] = None
                continue
            try:
                kwargs[fname] = VideoConfig.from_dict(raw_value)
            except (ValueError, TypeError) as e:
                warnings.warn(
                    f"配置 'ExperimentConfig.video' 解析失败（{e}），"
                    "视频录制回退为禁用（video=None）。"
                )
                kwargs[fname] = None
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
