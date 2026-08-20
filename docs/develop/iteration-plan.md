# ExAct 迭代计划

## 项目愿景

研究**大模型（LLM）配合 VLA**，在**探索机制**下是否能更好地完成具身操控任务。

核心假设：让 LLM 大脑先探索环境、积累经验，再指导 VLA 脊髓执行任务，能弥补 VLA 泛化能力不足的问题。

---

## 迭代路线图总览

| 迭代  | 名称                                | 目标                                                            | VLA 后端           | 设备        | 状态          |
| ----- | ----------------------------------- | --------------------------------------------------------------- | ------------------ | ----------- | ------------- |
| 1     | 流水线重构与配置化                  | 清理技术债，建立可扩展结构                                       | Mock（不替换）     | M4 Air      | ✅ 已完成     |
| 2     | 渲染与环境模式配置化                | 解决 M4 Mac 段错误，渲染器(GPU/CPU)与环境模式均可配置             | Mock               | M4 Air      | ✅ 已完成     |
| 3     | 实验数据持久化（埋点解耦）          | 通过埋点机制收集实验数据，与业务管线解耦                          | Mock               | M4 Air      | ✅ 已完成     |
| 4     | 图片存储器基础                      | 图片存取与 URL 协议，仅图片存储与引用                             | Mock               | M4 Air      | ⏳ 待启动     |
| 5     | LLM 多模态视觉接入                  | LLM 通过 URL 看到图片（多模态输入），observe 工具返回图片 URL      | Mock               | M4 Air      | ✅ 已完成     |
| 6     | VLA 链路验证                      | 验证 VLA 已能从 env 拿到图（链路通了，无需 LLM 介入）           | Mock               | M4 Air      | ⏳ 待启动     |
| 7     | LLM-VLA 适配器                      | 用 MiniMax-M3 本身作为伪 VLA，机械臂真正响应指令                 | **LLMVLA**         | M4 Air      | ⏳ 待启动     |
| 9     | smolVLA 指令契约 + 基础探索         | action 指令拦截 + 系统提示词约束 + 笔记本式探索工具               | LLMVLA             | M4 Air      | ⏳ 待启动     |
| 10    | VLA chunk 接口契约重构              | 把 BaseVLA.predict 从单步改为多步轨迹，executor 按 VLA 输出执行   | smolVLA / Mock    | M4 Air      | ⏳ 待启动     |

> 拆分说明：原 Iteration 4「图片存储器与 LLM 视觉能力」任务过多（包含图片存储、observe 改造、action 改造、executor 链路、BaseVLA 契约、config 扩展、pipeline 集成 7 件事），现拆为 4 / 5 / 6 三个迭代。原 Iteration 5（LLM-VLA 适配器）顺延为新 Iteration 7。原 Iteration 3 数据保存方案由"扩展管线接口"改为"ExperimentRecorder 一体化"——业务代码 `recorder.emit(...)` 埋点，由 `src/experiment/recorder.py` 一个类同时负责收集与保存三类数据（`log` → `experiment.log`、`observe_image` → `observer/*.png`、`video_frame` 占位）。每次实验产物在 `ExAct/data/experiment/{时间戳}/` 下。Iteration 6 原计划"image_url 传递链路"反思后改为"VLA 链路验证"——VLA 已能从 env 拿到图，无需 LLM 介入传递，仅补一个 print 验证链路工作。详见下文。

---

## Iteration 1: 流水线重构与配置化

**状态**：✅ 已完成

**目标**：清理技术债，建立可扩展的项目结构。

### 背景：当前需要清理的问题

- `tools/` 下存在 `lc_action.py` / `lc_observe.py`（LangChain 版）与 `action.py` / `observe.py`（手写版）两套实现，lc_ 前缀的标签毫无必要
- `tools/base.py` 中的手写 `BaseTool` 抽象基类已废弃（被 LangChain BaseTool 取代）
- `agents/core.py` 中的 `ExActAgent` 手写 ReAct 循环已废弃（被 LangGraph lc_agent 取代）
- `agents/llm_client.py` 中的 `MiniMaxClient`（urllib 手写 HTTP）已废弃（被 ChatOpenAI 取代）
- 机械臂 URDF 路径、task_spec 硬编码在 `pybullet_env.py` 和 `app.py` 中
- `MAX_REACT_ROUNDS`、`MAX_TOOL_CALLS` 硬编码在 `lc_agent.py` 中
- `executor/` 下 VLA 模型实现混在扁平文件中，不便于多后端扩展

### 任务

1. **迁移主流程到 `src/pipeline/` 模块**

   - 创建 `pipeline/runner.py`，封装 env + executor + agent 的组装与运行
   - `app.py` 瘦身为入口，只负责加载配置和调用 pipeline
   - pipeline 从 config 提取所有参数，显式传入各模块
2. **清理 lc_ 前缀死代码**

   - 删除 `tools/lc_action.py`，将其逻辑（LangChain BaseTool + parse_target_pos）合并到 `tools/action.py`
   - 删除 `tools/lc_observe.py`，将其逻辑合并到 `tools/observe.py`
   - 删除 `tools/base.py` 中废弃的手写 BaseTool 抽象基类
   - 删除 `agents/core.py` 中的 `ExActAgent` 类（保留 `AgentResult`、`ToolCallRecord` dataclass，因 lc_agent 依赖）
   - 删除 `agents/llm_client.py` 中的 `MiniMaxClient`（已被 ChatOpenAI 替代）
3. **配置化机械臂与任务**

   - config 新增 `robot` 节：URDF 路径、初始位姿、关节索引常量
   - config 新增 `task` 节：物体列表 task_spec
   - config 新增 `agent` 节：`max_react_rounds`、`max_tool_calls`
   - 程序内部保留默认 fallback 值（如 `MAX_REACT_ROUNDS = 5`），pipeline 从 config 提取并传入，config 缺失时使用默认值
4. **重组 executor VLA 模型目录 + 工厂函数**

   - 新建 `executor/model/` 文件夹，每个 VLA 后端占一个子文件夹：
     ```
     executor/model/
       ├── base.py               # BaseVLA 抽象基类（从 executor/vla.py 迁出）
       ├── factory.py            # create_vla(config) 工厂函数（新增）
       ├── mock/
       │   └── mock_vla.py       # MockVLA（从 executor/vla.py 迁出）
       ├── llm_vla/
       │   └── llm_vla.py        # LLMVLA（Iteration 7 实现）
       ├── small_vla/
       │   └── small_vla.py      # SmallVLA（规划中）
       └── openvla/
           └── openvla_adapter.py # OpenVLA 适配器（规划中）
     ```
   - `executor/vla.py` 原文件删除，内容拆分到 `model/base.py` + `model/mock/`
   - `executor/model/factory.py` 新增 `create_vla(vla_config, llm_config=None) -> BaseVLA`
     - 根据 `vla_config.backend` 分派到对应子文件夹的实现
     - 支持 mock / llm_vla / small_vla / openvla 四种后端
   - `executor/__init__.py` 调整导入：从 `executor.model.base` 导 `BaseVLA`，从 `executor.model.factory` 导 `create_vla`，从 `executor.model.mock` 导 `MockVLA`
5. **更新测试**

   - 删除针对 lc_ 文件和 MiniMaxClient 的测试
   - 更新导入路径（`from executor.vla import X` → `from executor.model.base import X`）
   - 确保回归测试通过

### 交付物

- 重构后的代码（pipeline 模块 + 清理后的 tools/agents + 重组后的 executor/model/ 目录）
- `executor/model/factory.py` 工厂函数（支持 backend 分派）
- 通过的测试用例
- 更新后的 `configs/default.yaml`

---

## Iteration 2: 渲染与环境模式配置化

**状态**：✅ 已完成

**目标**：解决 M4 Mac 上 `p.getCameraImage()` 段错误问题，支持渲染器（GPU/CPU）和环境模式（DIRECT/GUI）均可配置。

### 背景

当前系统为了绕开 M4 Mac 上 PyBullet GPU 渲染段错误（SIGSEGV），在 observe 工具中强制 `include_rgb=False`，导致 LLM 大脑和 VLA 脊髓都看不到图片。这严重削弱了项目核心创新点——LLM 的原生多模态视觉感知能力无法使用。

根因分析：

- PyBullet 默认用 `p.BULLET_HARDWARE_OPENGL` 渲染器，走 OpenGL 离屏渲染 + `glReadPixels` 读回像素
- Apple Silicon 上 OpenGL 是 Metal 兼容层，离屏 FBO + 像素读回有 bug
- 解决方案：改用 `p.TINY_RENDERER`（纯 CPU 软渲染），绕开 OpenGL

但渲染器选择不应写死——不同设备情况不同：

- M4 Mac：必须用 CPU（TINY_RENDERER），GPU 会段错误
- Linux + NVIDIA GPU（如 4060 笔记本）：可用 GPU（OpenGL）加速，渲染快
- 无 GPU 服务器：用 CPU 软渲染

因此渲染器需要可配置，并支持"自动"模式根据平台判断。

### 任务

1. **渲染器配置化**

   - `EnvConfig` 新增 `renderer` 字段：`"auto"` | `"cpu"` | `"gpu"`，默认 `"auto"`
   - 渲染器映射：| 配置值     | 实际渲染器                   | 说明                                                               |
     | ---------- | ---------------------------- | ------------------------------------------------------------------ |
     | `"auto"` | 平台自动判断                 | M4 Mac →`p.TINY_RENDERER`；其他 → `p.BULLET_HARDWARE_OPENGL` |
     | `"cpu"`  | `p.TINY_RENDERER`          | 强制 CPU 软渲染，跨平台稳定                                        |
     | `"gpu"`  | `p.BULLET_HARDWARE_OPENGL` | 强制 GPU 渲染，需平台支持                                          |
   - `PyBulletPandaEnv` 内部实现 `_resolve_renderer(config_value) -> int`：
     - `"auto"` 时检测 `platform.system()` + `platform.processor()`，Apple Silicon 返回 `p.TINY_RENDERER`，其他返回 `p.BULLET_HARDWARE_OPENGL`
     - 显式 `"cpu"` / `"gpu"` 直接映射
   - `render()` 方法使用解析后的渲染器常量
   - 兼容性：`renderer` 缺失时 fallback 到 `"auto"`
2. **环境模式配置化**

   - `EnvConfig` 新增 `mode` 字段：`"direct"` | `"gui"`，默认 `"direct"`
   - `PyBulletPandaEnv.__init__` 根据 `mode` 选择 `p.connect(p.DIRECT)` 或 `p.connect(p.GUI)`
   - 原有的 `use_gui` 参数废弃，由 `mode` 替代
   - 注意：`mode` 控制是否开窗口，`renderer` 控制 `getCameraImage()` 用什么渲染——两者独立
3. **统一配置节**

   - `configs/default.yaml` 更新：
     ```yaml
     env:
       mode: direct                # "direct" | "gui"   是否开 GUI 窗口
       renderer: auto              # "auto" | "cpu" | "gpu"   渲染器选择
       camera_resolution: [640, 480]
     ```
   - 配置缺失时：`mode` fallback 到 `"direct"`，`renderer` fallback 到 `"auto"`
4. **render() 方法改造**

   - [pybullet_env.py](file:///Users/noah/项目/保研练习项目/ExAct/src/env/pybullet_env.py) 的 `render()` 中 `p.getCameraImage()` 调用使用 `self._renderer`（构造时解析好的常量）
   - 更新日志：动态标注实际使用的渲染器（"CPU/TINY_RENDERER" 或 "GPU/OPENGL"）
   - 验证：M4 Mac 上 `renderer=auto` 自动选 CPU，不段错误
5. **恢复 observe 工具的视觉感知**

   - [observe.py](file:///Users/noah/项目/保研练习项目/ExAct/src/tools/observe.py) 的 `include_rgb=False` 改为 `include_rgb=True`
   - 验证：observe 工具能正常获取 RGB 图像（不段错误）
   - 注意：此阶段 observe 返回的仍是文本，图片如何传给 LLM 在 Iteration 5 解决
6. **测试**

   - 单元测试：`_resolve_renderer("auto"/"cpu"/"gpu")` 在当前平台返回正确常量
   - 单元测试：`render()` 在 DIRECT + CPU 模式下返回正确 shape 的 ndarray
   - 集成测试：连续多次调用 `getCameraImage()` 不崩溃
   - 性能测试：记录 CPU 软渲染单帧耗时（基线参考，供后续对照）

### 交付物

- `EnvConfig.renderer` 配置字段（支持 auto/cpu/gpu）
- `EnvConfig.mode` 配置字段（支持 direct/gui）
- `_resolve_renderer()` 平台自动判断逻辑
- `render()` 方法使用可配置渲染器
- observe 工具恢复 `include_rgb=True`
- 渲染稳定性测试通过

---

## Iteration 3: 实验数据落地（ExperimentRecorder 一体化）

**状态**：✅ 已完成

**目标**：`src/experiment/recorder.py` 中的 `ExperimentRecorder` 类**同时负责收集数据和保存数据**——业务代码统一调 `recorder.emit(event, **fields)`，由 recorder 内部按事件类型落到 txt 日志、图片、视频占位。

### 背景

后续要做对照实验，需要能复盘每次实验。当前 pipeline / executor / tools 只负责跑通，**没有产物**。

需求简单：

- 一份 txt 日志，像终端输出一样记录全过程（启动信息、LLM 思考、工具调用、动作、错误、时间戳）
- 一系列图片，observe 工具每次调用的截图统一存放
- 视频暂时不做（后续迭代）

**反例（避免）**：把数据保存塞进 `PipelineResult`、`AgentResult`、`ExecResult` 等接口——业务代码被迫感知"我要被观测"，每个新模块都要传回调，污染面太大。

**正解**：业务代码只调 `recorder.emit(event, **fields)`，recorder 内部识别三类事件并落盘。**emit 入口、目录创建、文件写入都在 `ExperimentRecorder` 一个类里，不拆成独立框架。**

### 数据落地结构

```
ExAct/data/experiment/
  └── 20260813_153000/             # 时间戳命名的实验目录（每次实验一个）
       ├── experiment.log          # 整个实验过程的文本日志（类似终端输出）
       └── observer/               # observe 工具返回的图片
           ├── 001.png
           ├── 002.png
           └── ...
```

**目录命名规则**：`YYYYMMDD_HHMMSS`（本地时间）。同一秒内多次实验自动追加 `_001`、`_002` 后缀。

**视频占位**：本迭代不实现，但事件类型 `video_frame` 已在 recorder 里登记（直接 no-op），未来加视频实现即可启用，**业务代码零改动**。

### 三类事件契约

| event 名称        | 必带 fields                          | 落盘目标                          |
| ----------------- | ------------------------------------ | --------------------------------- |
| `log`             | `message: str`, `level: str = "INFO"` | 追加到 `experiment.log`        |
| `observe_image`   | `image: np.ndarray`, `idx: int`       | 保存到 `observer/{idx:03d}.png`   |
| `video_frame`     | `image: np.ndarray`, `timestamp: float` | **本迭代占位，no-op 不落盘** |

**其他事件一律 no-op**——recorder 收到不认识的 event 直接 return，不报错。这样未来想加新事件时，业务代码可以先 `recorder.emit(...)` 占位，等实现了再真正落盘。

### 任务

1. **ExperimentRecorder 类**（`src/experiment/recorder.py`，唯一模块）

   - 一个类搞定所有事，**不拆框架**：

     ```python
     class ExperimentRecorder:
         def __init__(self, root: Path, enabled: bool, log_to_stdout: bool):
             self.root = root          # ExAct/data/experiment
             self.enabled = enabled    # False 时所有方法 no-op
             self.log_to_stdout = log_to_stdout
             self.exp_dir: Path | None = None
             self.observer_dir: Path | None = None
             self._log_fh: TextIO | None = None

         def start(self) -> Path:
             """创建 {root}/{timestamp}/ 和 observer/，返回 exp_dir"""
             ...

         def emit(self, event: str, **fields) -> None:
             """统一入口，按 event 类型分发到 _handle_log / _handle_observe_image / _handle_video_frame"""
             ...

         def _handle_log(self, message: str, level: str = "INFO") -> None:
             """追加一行到 experiment.log"""
             ...

         def _handle_observe_image(self, image: np.ndarray, idx: int) -> None:
             """保存到 observer/{idx:03d}.png"""
             ...

         def _handle_video_frame(self, image: np.ndarray, timestamp: float) -> None:
             """本迭代占位，直接 return"""
             ...

         def finish(self, success: bool, summary: str) -> None:
             """emit('log', message=f'Pipeline finished, success={success}, summary={summary}')"""
             ...
     ```

   - **dispatch 表**：内部维护 `EVENT_HANDLERS = {"log": _handle_log, "observe_image": _handle_observe_image, "video_frame": _handle_video_frame}`，emit 直接查表分派
   - **enabled=False 行为**：所有方法（start / emit / finish）退化为 no-op，业务代码无需分支判断
   - **异常处理**：emit 内 try/except 吞掉异常记 warning，不让埋点崩业务
   - **进程级单例**：用模块全局变量 `_global_recorder: ExperimentRecorder | None = None`，业务代码通过 `get_recorder() -> ExperimentRecorder` 拿同一实例
2. **业务代码埋点（关键位置）**

   - 在以下位置加 `recorder.emit(...)`，**每处不超过一行**：

     | 埋点位置                    | event              | fields                                |
     | --------------------------- | ------------------ | ------------------------------------- |
     | pipeline 启动               | `log`              | `message="Pipeline started, ..."`     |
     | pipeline 结束               | `log`              | `message="Pipeline finished, ..."`    |
     | observe 工具拿到 ndarray 后 | `observe_image` + `log` | `image=ndarray, idx=call_count` / `message="[observe] saved observer/{idx:03d}.png"` |
     | 工具调用日志（可选）        | `log`              | `message="[tool:{name}] args={...}"`  |
     | LLM 响应（可选）            | `log`              | `message="[llm] {content 摘要}"`      |
     | action 调用（可选）         | `log`              | `message="[action] instruction={...}"`|

   - **约束**：埋点只读本地变量，不修改业务状态、不做异常处理
   - **约束**：业务代码只 `from src.experiment.recorder import get_recorder; recorder = get_recorder()`，不 import 任何 sink/event_bus 类
3. **pipeline 集成**

   - `pipeline/runner.py` 启动时：
     ```python
     from src.experiment.recorder import ExperimentRecorder
     recorder = ExperimentRecorder(
         root=config.experiment.root,
         enabled=config.experiment.enabled,
         log_to_stdout=config.experiment.log_to_stdout,
     )
     recorder.start()
     ```
   - 结束时调 `recorder.finish(success=..., summary=...)`
   - **不再扩展** `PipelineResult`、`AgentResult` 的字段
   - **不传** recorder 实例给下层模块——下层通过 `get_recorder()` 拿全局实例
4. **配置项**

   - `configs/default.yaml` 新增 `experiment` 节：
     ```yaml
     experiment:
       enabled: true                 # 默认开启，每次跑都存档
       root: "data/experiment"       # 相对项目根目录
       log_to_stdout: true           # log 事件是否同时输出到终端
     ```
   - `ExperimentConfig` dataclass（`enabled`, `root`, `log_to_stdout`）
   - `enabled=False` 时 recorder 所有方法 no-op（**业务代码无感知**）

### 交付物

- `src/experiment/recorder.py` 实现（一个类搞定：目录创建 + emit 分发 + 三类落地）
- 业务代码在 observe 工具、pipeline 启动/结束处埋点（`recorder.emit(...)`）
- 每次实验自动产出 `experiment.log` + `observer/*.png`
- `config.experiment` 配置节
- 单元测试：目录创建、三类事件分发、未知事件 no-op、enabled=False 时全 no-op

### 不在本迭代

- ❌ 视频录制（`video_frame` 事件占位 handler 是 no-op，等未来）
- ❌ 独立的 tracer/event_bus/Sink 框架（已合并进 recorder，不拆）
- ❌ 全量结构化事件采集（events.jsonl、其他类型事件）
- ❌ 实验索引查询脚本（grep `experiment.log` 即可）
- ❌ 实验报告自动生成（人工看 log + 图片复盘即可）
- ❌ 为观测而扩展 dataclass 字段

### 后续迭代如何用

- **人工复盘**：打开 `experiment.log` 看每步干了什么，看 `observer/*.png` 看环境如何变化
- **后续对照实验**：批量跑实验后 `grep -l "Pipeline finished, success=true" */experiment.log` 统计成功率
- **新事件类型**：在 `EVENT_HANDLERS` 表里加一行 + 一个 `_handle_xxx` 方法，业务代码即可调用

---

## Iteration 4: 图片存储器基础

**目标**：建立通用的图片存取与 URL 协议，让图片可以在模块间用 URL 引用，与具体存储后端解耦。**仅做存储，不做多模态接入。**

### 背景

Iteration 2 解决了渲染段错误，observe 工具能拿到 RGB 图像。但当前 observe 工具返回的是纯文本字符串，图片无法在模块间传递，后续要让 LLM 看到图、让 VLA 看到图，都需要一个统一的图片引用机制。

本迭代只做**最薄一层**：图片能存、能取、能用 URL 引用。LLM 能不能看到图、VLA 能不能拿到图，留给后续迭代。

### 任务

1. **图片存储器模块**（`src/utils/image_store/`）

   - 目录结构：
     ```
     utils/image_store/
       ├── __init__.py
       ├── store.py          # ImageStore 主类
       ├── backend.py        # 存储后端抽象（内存 / 文件系统）
       └── url_scheme.py     # URL 生成与解析
     ```
   - `ImageStore` 核心接口：
     - `save(image: np.ndarray, category: str = "observations") -> str`：保存图片，返回 URL
     - `load(url: str) -> np.ndarray`：通过 URL 加载图片
     - `exists(url: str) -> bool`：检查 URL 是否存在
     - `list(category: str | None = None) -> list[str]`：列出图片 URL
   - URL 格式：`img://{category}/{filename}`（如 `img://observations/20260813_153000_001.png`）
   - 存储后端：
     - `MemoryBackend`：图片存内存（默认，适合单次运行）
     - `FileBackend`：图片存文件系统（路径从 config 读）
   - 工厂函数 `create_image_store(config) -> ImageStore`
2. **配置扩展**

   - 新增 `image_store` 配置节：
     ```yaml
     image_store:
       backend: "memory"          # "memory" | "file"
       file_dir: "data/images"    # 仅 file 后端使用
     ```
3. **单元测试**

   - `MemoryBackend` save/load/exists/list 路径覆盖
   - `FileBackend` 跨进程持久化（写入文件 → 新实例读取）
   - URL scheme 解析边界（空、非法 category、含空格）
   - 工厂函数按 config 返回正确 backend

### 交付物

- `src/utils/image_store/` 模块（store + backend + url_scheme）
- `config.image_store` 配置节
- 单元测试通过

### 不在本迭代

- ❌ observe 工具改造（要等 Iteration 5）
- ❌ action 工具改造（要等 Iteration 6）
- ❌ executor 链路 image_url 传递（要等 Iteration 6）
- ❌ LLM 多模态输入（要等 Iteration 5）
- ❌ pipeline 集成（要等 Iteration 5/6）

---

## Iteration 5: LLM 多模态视觉接入

**状态**：✅ 已完成（2026-08-13）

**目标**：让 LLM 能通过 URL 看到 observe 工具返回的图片，实现原生多模态感知。**不动 VLA 链路。**

### 背景

Iteration 4 解决了图片的存储和引用，但 observe 工具仍然只返回文本。LLM 需要看到场景截图才能做探索判断、子任务完成度评估等高级决策。本迭代让 observe 工具返回"文本 + 图片 URL"结构化内容，LLM 通过原生多模态能力看到图片。

**范围限定**：本迭代只让 LLM 看到图，不动 action 工具、不动 executor 链路、不动 VLA。VLA 拿到图片是 Iteration 6 的事。

### 任务

1. **observe 工具改造**

   - `ObserveTool` 注入 `ImageStore` 实例
   - `observe._run()` 流程：
     1. 调 `env.render()` 拿 RGB ndarray
     2. 调 `image_store.save(image, category="observations")` 拿 URL
     3. 返回结构化内容：文本描述（ee_pos + object_info）+ 图片 URL
   - LangChain BaseTool 的 `content` 字段支持 list，含 `text` 和 `image_url` 块
   - **关键**：observe 仍可独立运行（不依赖 image_store 也能工作，只是拿不到图）
2. **Agent 多模态消息组装**

   - `agents/lc_agent.py`（或对应文件）改造 tool 消息组装：
     - tool 返回的 `content` 为 list 时，拆成 text + image_url 块传给 LLM
     - 单测覆盖 list / str 两种返回形态
3. **prompt 注入图片引用**

   - system prompt 不变（LLM 通过 image_url 块直接看图）
   - 不强制 LLM "必须看图"，保留灵活性
4. **端到端冒烟**

   - 跑一次任务，验证日志能看到 observe 返回的图片 URL
   - 验证 LLM 后续回复中确实在引用图片内容（如"我看到红色方块在..."）

### 交付物

- `ObserveTool` 返回结构化内容（text + image_url）
- Agent tool 消息组装支持多模态块
- 端到端验证 LLM 看到图

### 不在本迭代

- ❌ action 工具接收 image_url（要等 Iteration 6）
- ❌ executor.run_action(image_url)（要等 Iteration 6）
- ❌ VLA 接口改动（BaseVLA.predict 已经接 image，本迭代不调 VLA）

---

## Iteration 6: VLA 链路验证

**状态**：⏳ 待启动

**目标**：验证 VLA 已经能从 env 拿到图（链路通了）——**不需要 action 工具传递 image_url，VLA 直接接触环境**。

### 背景

之前 Iteration 6 的设计是"让 LLM 把看到的图 URL 传给 VLA"。反思后这个设计**反模式**：

- VLA 是控制系统，应该自主观察环境（闭环反馈），不应该被 LLM 喂一张"过时"的图
- 当前 `Executor.run_action` 已经在循环内每步从 `env.get_obs()["rgb"]` 拿图，通过 `build_vla_input` 喂给 VLA——**链路本就通的**
- 真正的需求不是"让 VLA 拿图"（已经能），而是"确认这条链路确实在工作"

### 职责分层重申

```
LLM 大脑：observe → 规划 → 下发子指令（语义层面）
VLA 脊髓：接收指令 → 自主观察 env → 执行动作（控制层面）
```

VLA 不应该被 LLM 喂图，LLM 也不应该传递图给 VLA。

### 任务

1. **加链路验证 print**

   - 在 `Executor.run_action` 循环内第一步 `vla.predict(...)` 之后加一个 print（或 logger.info），打印：
     - `image.shape` 和 `image.dtype`
     - `image` 是否与当前 `obs_before["rgb"]` 数值一致（`np.array_equal`）
   - 验证目标：**VLA 收到的 image 不是 None，且数值与 env.get_obs() 返回的 rgb 数组一致**
   - print 是临时的——本迭代结束后可以保留作为长期链路健康检查，或后续迁移到 ExperimentRecorder 的 log 事件

2. **跑一次完整 pipeline 验证**

   - 用 MockVLA backend 跑一次完整任务（任意简单目标）
   - 观察 print 输出，确认 VLA 收到的 image 是有效 ndarray 且与 env rgb 一致

3. **文档说明**

   - 在 `executor/__init__.py` 的 `run_action` docstring 里加一段说明：
     - "VLA 通过 `env.get_obs()["rgb"]` 获取图像——VLA 直接接触环境，不依赖 LLM 传入图片"
   - 在 `iteration-plan.md` 中记录反思："VLA 不应接收 LLM 的图 URL，由 VLA 自主观察 env"

### 交付物

- `Executor.run_action` 加链路验证 print（image 形状 + dtype + 与 env rgb 一致性断言）
- 跑一次 pipeline 确认 print 输出正确
- executor docstring 更新 + iteration-plan 反思记录

### 不在本迭代

- ❌ ActionTool 不加 image_url 字段（VLA 自己看 env）
- ❌ Executor 不加 URL 协议识别逻辑（不需要）
- ❌ ImageStore 不加 mm_file_id → img_url 索引（不需要）
- ❌ build_vla_input 不加 image 覆盖参数（不需要）
- ❌ 任何"LLM 把图传给 VLA"的链路设计

### 后续迭代如何用

- **Iteration 7（LLMVLA）**：实现 LLMVLA 时确认它从 env.get_obs 拿语义信息（ee_pos / object_info），不看 image
- **后续真实 VLA（SmallVLA / OpenVLA）**：接真实 VLA 时，链路已经验证通，executor 喂 env rgb 即可
- **链路 print**：可作为长期健康检查保留，或迁到 ExperimentRecorder 的 log 事件

---

## Iteration 7: LLM-VLA 适配器

**目标**：用同一个 MiniMax-M3 作为伪 VLA 脊髓，替换 MockVLA，让机械臂真正响应指令。

### 背景

当前 MockVLA 用哈希生成伪随机动作，机械臂几乎不动，无法作为探索机制研究的基座。但直接上 OpenVLA 需要 4060 GPU 环境 + 完善的探索机制，时机不成熟。

折中方案：利用 VLA 可插拔接口，**插入一个 LLM-VLA 适配器**——内部调用同一个 MiniMax-M3 API，通过严格的系统提示词约束它输出结构化的 7D 动作 JSON，而不是自然语言。

### 设计思路

```
LLMVLA.predict(image, instruction):
    组装 Prompt:
        System: "你是 VLA 脊髓，负责输出机械臂 7D 动作。严格输出 JSON，
                 格式: {"dx": float, "dy": float, "dz": float,
                         "drx": float, "dry": float, "drz": float,
                         "gripper": float}
                 dx/dy/dz 单位米，建议单次位移 <= 0.05m。
                 gripper: 0.0=闭，1.0=开。只输出 JSON，不许其他文字。"
        User: "当前末端位置: {ee_pos}\n物体列表: {object_info}\n指令: {instruction}"
    ↓
调用 MiniMax-M3 API → 解析 JSON → Action7D
```

**注意**：LLMVLA 可以不看图片（image 参数忽略），因为它接收的是语义化的 ee_pos + object_info。但接口遵守 `BaseVLA.predict(image, instruction)` 契约，image 参数保留。真正必须看图的是 SmallVLA/OpenVLA（后续规划）。

### 任务

1. **实现 LLMVLA 类**（`executor/model/llm_vla/llm_vla.py`）

   - 继承 `BaseVLA`，实现 `predict(image=None, instruction) -> Action7D`
   - 内部调用 ChatOpenAI（复用已有的 MiniMax-M3 连接，由 factory 传入）
   - 系统提示词严格约束输出 JSON 格式（Action7D 的 7 个字段）
   - 容错：JSON 解析失败时 fallback 到零动作（不移动，只报错）
   - 可选约束：`llm_vla_max_displacement` 配置限制单步最大位移（clipping）
2. **注册到工厂**（`executor/model/factory.py`）

   - backend == `"llm_vla"` 时，构造 `LLMVLA(llm_config, vla_config)` 实例返回
3. **扩展 VLAConfig**

   - config `vla.backend` 新增 `"llm_vla"` 选项
   - `VLAConfig` 可选增加 `llm_vla_max_displacement`（单步最大位移，如 0.05）等约束字段
4. **验证基础闭环**

   - 切换 backend=llm_vla，运行 `app.py`
   - 验证：LLM 下发"移动到红色方块上方" → LLMVLA 输出合理 dx/dy/dz → 机械臂移动
   - 对比 MockVLA：机械臂应有明显且朝目标方向的位移

### 交付物

- `src/executor/model/llm_vla/llm_vla.py` 实现
- `executor/model/factory.py` 中 llm_vla backend 注册
- config `backend: "llm_vla"` 选项
- 基础闭环 demo（机械臂能真正移动）

---

## Iteration 9: smolVLA 指令契约 + 基础探索

**状态**：⏳ 待启动

**目标**：让 LLM 大脑下发给 VLA 脊髓的指令符合 smolVLA 指令规范，同时引入最简探索工具（笔记本），让模型积累指令经验。为后续探索机制迭代打地基。

### 背景

`docs/knowledage/smolVLA-insturction.md` 定义了 VLA 任务标注的提示词原则——smolVLA 模型生成语言指令时必须遵守：

> 生成一句极简短、清晰、完整的单句，描述机械臂执行的动作（最多 30 个字符）。句子必须直接以动作动词开头，例如 "拿起""放置""打开" 等。不要加入冗余词汇。

当前链路中 LLM 大脑通过 `action` 工具下发指令，`ActionTool._run` 把 LLM 原话直接传给 `executor.run_action`，再经 `build_vla_input` 原样透传给 `VLA.predict`——**没有任何地方强制这条格式契约**。LLM 可能输出长句、多句、无动词开头的指令，与 VLA 脊髓（尤其 LLMVLA 及未来的 SmallVLA/OpenVLA）的输入形态不匹配。

本迭代做三件事：**拦截**（action 工具边界校验）、**约束**（指令要求写入系统提示词）、**探索 v0**（最简笔记本工具）。

### 任务

1. **action 工具基本拦截**

   - 在 `ActionTool._run` 内、`executor.run_action` 之前增加指令校验：
     - 规则 1：以动作动词开头（拿起 / 放置 / 打开 / 移动 / 推 / 夹取 等白名单或词性判断）
     - 规则 2：单句（不含句号/换行/连接词拼接的多句）
     - 规则 3：长度不超过 smolVLA 上限（30 字符；中文按字符计）
   - **不修指令，只拒绝**：判定不合规时返回错误消息（如"指令不符合 smolVLA 规范：不是动词开头 / 超长 / 多句，请重新下发简洁指令"），**不执行动作**，让 LLM 在下一轮重新调用
   - 校验规则收敛为纯函数（如 `validate_instruction(instruction) -> Optional[str]`，返回不合规原因或 None），便于单元测试
   - 保留 `parse_target_pos` 现有解析逻辑，两者独立、先后执行
2. **指令要求加入系统提示词**

   - 在 `src/agents/prompt.py` 的 `DEFAULT_SYSTEM_PROMPT` 中，`action` 工具描述处补充指令规范：
     - "action 的 instruction 必须是一句以动作动词开头的简短单句（≤30 字符），如『拿起红色方块』；不符合规范会被拒绝并返回错误，请重新下发"
   - 让 LLM 在源头尽量合规，减少拦截器命中次数（拦截器是契约强制点，提示词是第一道约束）
3. **最简探索工具（笔记本）**

   - 新增 `src/explore/` 模块，本迭代只做**最简可用**版本：
     - 工具名称就叫**探索工具**（`explore`），注册为 Agent 可调用工具（LangChain BaseTool），与 observe/action 平级
     - 功能非常简单，**就是个笔记本**：
       - `write(note)`：把探索到的经验写入笔记，如"指令『拿起红色方块』合规，『请拿起红色方块』动词未开头被拒绝"
       - `read()`：返回已有全部笔记，供 LLM 下发指令前查阅
     - 内部维护一个内存笔记列表（session 内有效），不落盘、不做结构化知识
   - 用途：让 LLM 在探索-执行中积累"哪些指令说法会被接受/拒绝"的经验，下发指令前先读笔记本，减少被拦截次数
   - 本迭代**不做**：驱动 env 的探针测试、物理特性探测、工作空间映射、结构化知识库、失败重规划（留给后续探索迭代）

### 交付物

- `ActionTool` 指令校验拦截（纯函数 + 拒绝式错误返回）
- `DEFAULT_SYSTEM_PROMPT` 增加指令规范约束
- `src/explore/` 最简笔记本工具（`explore`）
- 单元测试：校验规则各分支（动词开头 / 单句 / 长度上限）、笔记本 write/read 行为
- 端到端冒烟：LLM 下发不合规指令被拒绝后自我纠正、合规指令正常执行

### 不在本迭代

- ❌ 指令自动改写/归一化（只拒绝不修，改写留待后续迭代）
- ❌ 物理特性探测、工作空间映射、结构化环境知识（探索 v2）
- ❌ 评估框架与对照实验（顺延为 Iteration 11）
- ❌ 真实 SmallVLA / OpenVLA 接入（原 Iteration 11/12）

---

## Iteration 10: VLA chunk 接口契约重构

**状态**：⏳ 待启动

**目标**：把 `BaseVLA.predict` 从「返回单步动作」改为「返回 N 步轨迹」，executor 按 VLA 实际输出的步数执行，让 chunking VLA（smolVLA / ACT / Pi0 / Diffusion）的多步规划能力真正贯通到执行链路。同时保留单步 VLA 的兼容性。

### 背景

Iteration 7（LLMVLA）和 Iteration 9（smolVLA 接入）暴露了 **执行链路与 VLA 实际工作方式脱节** 的根因：

| 当前层级 | 契约/行为 | 问题 |
|---|---|---|
| `BaseVLA.predict`（`src/executor/model/base.py:36`） | 返回 `VLAOutput`，只承载单帧动作值 | 把 lerobot chunking VLA 的多步输出能力屏蔽成单步 |
| `LeRobotVLA.predict`（`src/executor/model/lerobot/lerobot_vla.py:560-577`） | 调 `self._policy.select_action(...)`，popleft 一个 | 走 smolVLA 的 deque 队列，跨 `action()` 调用残留旧 chunk |
| `Executor.run_action`（`src/executor/__init__.py`） | `for step in range(self.max_steps)` 硬编码 50 步 | 不知道当前 VLA 的 `chunk_size` 是几；与模型意图完全脱节 |
| `check_done`（`src/executor/check_done.py:46-58`） | 没 `target_pos` 时 `dist ≤ 0.01m` 判 done | 反向语义，单步即 break，截断 VLA 的多步轨迹 |
| Adapter（`src/utils/adapter/adapters/joint_to_joint.py`） | 单帧空间转换 | 不感知 chunk 维度（属于空间层，不是时间层） |

详细问题记录见 `docs/develop/2026-08-20-vla-chunk-execution-issues.md`。

### 核心设计

把 chunk 维度显式化，**让接口对 N 无感**：

```python
# BaseVLA.predict 改后
@dataclass
class VLAOutput:
    values: np.ndarray   # shape: (N, action_dim)，N 由 VLA 自己决定
    spec: ActionSpec

class BaseVLA(abc.ABC):
    @abc.abstractmethod
    def predict(self, image, instruction, state=None) -> VLAOutput:
        """返回 N 步完整轨迹。N 由后端决定：smolVLA=50, ACT=100, Diffusion=8, 单步 VLA=1, Mock=1。"""
```

各后端的 N 取值：

| 后端 | `values.shape` | Executor 跑几步 |
|---|---|---|
| smolVLA / ACT / Pi0 / Diffusion（chunking） | `(N, action_dim)`，N ∈ {8, 50, 100} | N 步 |
| LLMVLA（每次推理出 1 个 7D Action） | `(1, 7)` | 1 步 |
| MockVLA | `(1, 7)` | 1 步（保持兼容） |
| 未来单步策略 | `(1, ...)` | 1 步 |

**Executor 不知道 N 是几，照单全跑**：

```python
# src/executor/__init__.py:run_action 改后
vla_output = self.vla.predict(image, instruction, state=state_vec)  # VLAOutput
actions = adapter(vla_output.values, env)  # shape (N, action_dim)

obs_after = obs_before
for action in actions:    # 循环边界就是 len(actions)，不再写死 max_steps
    obs_after, _, _, _ = env.step(action)
    # 中间不再调 check_done，不干预 VLA 意图

return ExecResult(
    success=None,           # 语义成功由 LLM 看最终画面判断
    steps=len(actions),
    final_obs=obs_after,
    message=f"执行 VLA 规划的 {len(actions)} 步"
)
```

**Adapter 按第一维保留 N 转换**：

```python
# src/utils/adapter/adapters/joint_to_joint.py 改后
def joint_to_joint_transform(vla_output, env):
    arr = np.asarray(vla_output)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)   # 单步 → (1, dim)
    elif arr.ndim == 2:
        pass                       # chunk → (N, dim)
    else:
        raise ValueError(...)
    target_dim = env.input_spec.dim
    mapped = arr[:, :target_dim].copy()  # 切片第一维保留 N
    if has_gripper and arr.shape[1] == target_dim - 1:
        gripper_col = np.full((arr.shape[0], 1), _DEFAULT_GRIPPER)
        mapped = np.concatenate([mapped, gripper_col], axis=1)
    return mapped   # shape (N, target_dim)
```

### 任务

1. **改 `BaseVLA.predict` 契约**

   - `VLAOutput.values` 从「单帧动作」改为 `np.ndarray`，shape `(N, action_dim)`
   - 文档化：N 由后端决定，调用方按第一维迭代
   - mock 路径兼容：MockVLA 返回 `(1, 7)` 不破坏既有测试

2. **改 `LeRobotVLA.predict` 走 `predict_action_chunk`**

   - 不再调 `self._policy.select_action(preprocessed)`（带 deque）
   - 改为 `self._policy.predict_action_chunk(preprocessed)` 一次拿整 chunk
   - postprocessor 输出的 `(1, N, action_dim)` 去 batch 维 → `(N, action_dim)`
   - **好处**：彻底消除 deque 跨调用残留；每 chunk 1 次推理，不存在旧 chunk 污染新调用的问题

3. **改 `Executor.run_action` 按 VLA 输出执行**

   - 循环边界从 `range(self.max_steps)` 改为 `for action in actions`
   - 中间**不再调** `check_done`——VLA 的 chunk 是它对下一步的承诺，照单全收
   - 加 `max_chunk_steps` 参数（默认 200）作为异常 VLA 返回过大 N 的安全兜底
   - `ExecResult.success` 改为 `None` 或新增 `semantic_success` 字段让 LLM 自行判断

4. **改 `check_done` 只做执行后报告**

   - `check_done` 不再参与 executor 循环控制
   - 仅在 `ActionTool` 拿到 `ExecResult` 后**作为信息返回**，帮助 LLM 评估当前状态
   - 删除 `target_pos=None` 时的反向兜底逻辑（这个分支本身就是 bug）

5. **改 3 个 adapter 支持批量转换**

   - `joint_to_joint_transform`：第一维保留 N，第二维做空间转换（见上面代码示例）
   - `task_to_joint_hardcode_transform`：同理，第一维保留 N
   - `identity_transform`：shape `(N, dim) -> (N, dim)` 直接透传
   - 共用 `vla_output` 入参的 shape 自适应：ndim=1 → reshape (1, -1)，ndim=2 → 保持

6. **MockVLA 兼容 N=1**

   - `MockVLA.predict` 和 `JointMockVLA.predict` 改为返回 `VLAOutput(values=np.array([action]), ...)`
   - 旧的单步测试不需要改测试断言，只要返回 shape 变了就跟着调整

7. **ActionTool 适配新返回**

   - `ActionTool._run` 拿到的 `ExecResult.message` 改成「执行 VLA 规划的 N 步」结构
   - 增加返回 `final_obs["ee_pos"]` 给 LLM，让 LLM 自己判断「到位了没」
   - 检查 `success=None` 不再让 agent.py 的 `_parse_agent_result` 把空 final_answer 当作失败

### 验证可插拔性

跑同一段 executor 代码，分别测试 4 种 VLA：

```python
# 测试 1: smolVLA（chunking, N=50）
vla = LeRobotVLA(policy_type="smolvla", ...)
result = executor.run_action(env, "move forward")  # 内部跑 50 步

# 测试 2: ACT（chunking, N=100）
vla = LeRobotVLA(policy_type="act", ...)
result = executor.run_action(env, "move forward")  # 内部跑 100 步

# 测试 3: LLMVLA（单步, N=1）
vla = LLMVLA(llm_config, vla_config)
result = executor.run_action(env, "move forward")  # 内部跑 1 步

# 测试 4: MockVLA（单步, N=1）
vla = MockVLA(seed=0)
result = executor.run_action(env, "move forward")  # 内部跑 1 步
```

**executor 一行代码不改，4 种 VLA 都能跑**——这就是可插拔。

诊断脚本 `src/pipeline/script/diagnose_arm_movement.py` 同样按新契约调整（自己循环 `len(actions)` 而不是 `range(max_steps)`）。

### 交付物

- `BaseVLA.predict` 契约文档（明确 N 由后端决定）
- `LeRobotVLA.predict` 改走 `predict_action_chunk`
- `Executor.run_action` 按 VLA 输出步数执行，去掉硬编码 max_steps 循环与中间 check_done
- `check_done` 退化为执行后报告（不影响循环）
- 3 个 adapter 全部支持批量转换（`(N, dim)` 透传 / 切片 / 补 gripper）
- MockVLA / JointMockVLA / LLMVLA 返回 `shape (1, ...)` 兼容
- `ActionTool` 处理新返回结构
- 单元测试：4 种 VLA 后端的 `predict` 返回 shape 契约
- 端到端冒烟：
  - `diagnose_arm_movement.py --vla smolvla --max-steps 200` 跑完一个 chunk 看累计位移
  - `app.py` 跑 `task=把黄色海绵块夹起来`，对比 Iter 9 报告的「每次 0.001-0.003m」，验证 chunk 真正贯通

### 不在本迭代

- ❌ VLA 端到端的视觉能力提升（仍是当前 smolVLA / LLMVLA 模型）
- ❌ VLA 训练 / fine-tune（让模型对当前 cube 场景不 OOD）
- ❌ 时序集成（temporal ensemble）等 chunk 利用策略升级——本迭代先把 chunk 接通，利用策略留后续迭代
- ❌ 评估框架与对照实验（顺延为 Iteration 11）
- ❌ 真实 SmallVLA / OpenVLA 接入（顺延为 Iteration 12+）

---

## 技术决策

### 配置化原则

- 所有可调参数（机械臂、任务、Agent 限制、VLA 后端、环境模式、图片存储、埋点开关）统一在 YAML 配置
- 程序内部保留默认 fallback 值，config 缺失时使用默认值
- pipeline 从 config 提取参数，显式传入各模块（不在模块内直接读 config）

### 埋点解耦原则

**业务侧零侵入，观测侧一体化在 ExperimentRecorder**：

- 业务代码（pipeline / observe 工具等）只在关键位置 `recorder.emit(event, **fields)`，不感知数据落到哪里、什么格式
- `ExperimentRecorder` **唯一负责**收集与保存：建目录、emit 分发、写文件——一个类搞定，不拆成框架
- recorder **只识别三类事件**：`log` / `observe_image` / `video_frame`（视频占位）；其他事件一律 no-op
- `ExperimentConfig.enabled=False` 时 recorder 所有方法 no-op，业务代码无需分支判断
- 新增观测维度只需在对应位置加一行 `recorder.emit(...)`，无需修改任何接口、签名、返回值
- 不为"被观测"而扩展 dataclass 字段（`AgentResult` / `ExecResult` / `ActionInput` 等保持纯净）
- 数据落地最小化：每个实验目录三类文件（log txt + observer/*.png + 视频占位），人类可直接复盘

### 模块边界

```
config/        配置加载与 dataclass
env/           仿真环境（PyBullet），最底层
tools/         Agent 可调用的工具（LangChain BaseTool）
agents/        LLM 与 Agent 组装（LangGraph StateGraph）
executor/      动作执行循环（Executor + ExecResult + build_vla_input + check_done）
  └── model/   VLA 模型集合（按后端分子目录，工厂统一创建）
        ├── base.py                    BaseVLA 抽象
        ├── factory.py                 create_vla(config) 工厂
        ├── mock/mock_vla.py           MockVLA
        ├── llm_vla/llm_vla.py         LLMVLA（Iteration 7）
        ├── small_vla/small_vla.py     SmallVLA（规划中）
        └── openvla/openvla_adapter.py OpenVLA（规划中）
pipeline/      主流程编排（组装 env + executor + agent + recorder）
  └── runner.py     Pipeline 主流程（启动/结束时调 recorder.start() / finish()）
experiment/    实验数据收集与落地（Iteration 3，唯一模块）
  ├── __init__.py
  └── recorder.py   ExperimentRecorder：start 建目录、emit 分发三类事件、写文件
utils/
  ├── logging.py    日志
  ├── image.py      图像编解码工具
  └── image_store/  图片存储器（Iteration 4-6，分阶段实施）
        ├── __init__.py
        ├── store.py          ImageStore 主类
        ├── backend.py        存储后端（内存 / 文件系统）
        └── url_scheme.py     URL 生成与解析
explore/       探索机制相关（Iteration 9 启用，笔记本式探索工具）
```

### VLA 接口契约与可插拔性

**当前已经可插拔**，核心机制：

1. **统一抽象**：`BaseVLA.predict(image: np.ndarray, instruction: str) -> Action7D`（位于 `executor/model/base.py`）
2. **Executor 解耦**：构造函数接受任意 `BaseVLA` 子类，上层无感知
3. **后端切换**：`vla.backend` 配置字段控制
4. **工厂统一创建**：`create_vla(vla_config, llm_config=None) -> BaseVLA`（位于 `executor/model/factory.py`）
5. **新增后端步骤**：
   - 在 `executor/model/` 下新建子文件夹 `xxx/xxx_vla.py`
   - 继承 `BaseVLA` 实现 `predict`
   - 在 `factory.py` 中注册一个分支

当前已规划的后端：

| backend       | 实现路径                                      | 是否看图             | 作用                              |
| ------------- | --------------------------------------------- | -------------------- | --------------------------------- |
| `mock`      | `executor/model/mock/mock_vla.py`           | 否                   | 单元测试 / smoke test             |
| `llm_vla`   | `executor/model/llm_vla/llm_vla.py`         | 否（用文本语义信息） | 探索机制研究基座（Iteration 7）    |
| `small_vla` | `executor/model/small_vla/small_vla.py`     | **是**（必须） | 真实小 VLA 对照（规划中）         |
| `openvla`   | `executor/model/openvla/openvla_adapter.py` | **是**（必须） | 最终正式实验（规划中）            |

### 图片流转路径

```
env.render() (TINY_RENDERER CPU 软渲染)
    ↓ np.ndarray
ImageStore.save(image) → URL (img://observations/xxx.png)
    ↓ URL
observe 工具返回给 LLM (文本描述 + image_url)
    ↓ LLM 决策
action 工具接收 image_url 参数
    ↓ URL
executor.run_action(image_url=...)
    ↓ ImageStore.load(url) → np.ndarray
VLA.predict(image, instruction)
```

### 指令下发路径

```
LLM 大脑
    ↓ 下发前先读笔记本：explore.read() 查阅已积累的指令经验
    ↓
action 工具边界：validate_instruction 拦截（动词开头 / 单句 / ≤30 字符）
    ↓ 不通过 → 返回错误，LLM 重新下发
    ↓ 通过
Executor → build_vla_input → VLA.predict(规范指令)
    ↓
explore.write(note) 记录经验（如"动词未开头的说法被拒绝"）
```

### 实验数据落地路径（ExperimentRecorder 一体化）

```
pipeline 启动
    ↓
recorder = ExperimentRecorder(root="ExAct/data/experiment", enabled=True, log_to_stdout=True)
recorder.start()
    ↓ 创建 ExAct/data/experiment/{YYYYMMDD_HHMMSS}/ 和 observer/ 子目录
    ↓ recorder.emit("log", message="Pipeline started, ...")
                ↓
                recorder._handle_log → 追加到 experiment.log
业务代码关键节点（observe / llm / action / executor step）
    ↓
recorder.emit("observe_image", image=ndarray, idx=1)
                ↓
                recorder._handle_observe_image → 保存到 observer/001.png
recorder.emit("log", message="[observe] saved observer/001.png")
                ↓
                recorder._handle_log → 追加到 experiment.log
recorder.emit("log", message="[llm] ...")
recorder.emit("log", message="[action] instruction=...")
    ↓ 全部追加到 experiment.log
pipeline 结束
    ↓
recorder.finish(success=True, summary="steps=12, success=true")
    ↓ 写收尾日志到 experiment.log
```

**关键设计**：

- 业务代码只调 `recorder.emit(...)`，不感知内部 handler、目录路径、文件格式
- recorder 内部用 dispatch 表分派事件：`EVENT_HANDLERS = {"log": ..., "observe_image": ..., "video_frame": ...}`
- `video_frame` 事件本迭代占位（handler 直接 return），未来加视频实现时业务代码零改动

**目录结构**：

```
ExAct/data/experiment/
  └── {YYYYMMDD_HHMMSS}/
       ├── experiment.log          # 全程文本日志（含时间戳）
       └── observer/
            ├── 001.png
            ├── 002.png
            └── ...
```

**与扩展接口方案对比**：

| 维度           | 扩展接口方案（弃）                          | ExperimentRecorder 一体化（本迭代）     |
| -------------- | ------------------------------------------- | -------------------------------------- |
| 业务代码改动   | 改 dataclass 字段、改函数签名、传回调       | 仅在关键位置加 `recorder.emit(...)`    |
| 新增观测维度   | 改 PipelineResult + AgentResult + 调用链    | 在对应位置加一行 `recorder.emit(...)`  |
| 关闭观测       | 不支持（接口已固化）                        | `enabled=False` 时 recorder 全 no-op  |
| 落盘产物       | 紧耦合 result.json + trajectory.jsonl       | log txt + observer/*.png（可直接读）   |
| 复盘方式       | 写解析脚本读 JSONL                          | 打开 log + 看图片                      |
| 实现复杂度     | 事件总线 + 全量结构化事件 + 配置            | 一个类，dispatch 表 + 三个 handler      |
