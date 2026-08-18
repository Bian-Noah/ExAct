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
| 8     | 探索机制 v1 - 基础探索              | 先探索再执行的双阶段流程跑通                                     | LLMVLA             | M4 Air      | ⏳ 待启动     |
| 9     | 探索机制 v2 - 策略化探索与知识利用  | 提升探索质量，验证知识传递效果                                   | LLMVLA             | M4 Air      | ⏳ 待启动     |
| 10    | 评估框架与对照实验                  | 量化探索收益，跑首批对照实验                                     | LLMVLA → SmallVLA  | M4 Air      | ⏳ 待启动     |
| 11    | 小型 VLA 接入                       | 训练真实小型 VLA，替换 LLM-VLA                                   | **SmallVLA**       | M4 Air      | ⏳ 待启动     |
| 12    | OpenVLA 接入与正式实验              | 探索机制完善后，接入 OpenVLA 做正式对照                          | **OpenVLA-7B 4bit** | 4060 笔记本 | ⏳ 待启动     |

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
       │   └── small_vla.py      # SmallVLA（Iteration 11 实现）
       └── openvla/
           └── openvla_adapter.py # OpenVLA 适配器（Iteration 12 实现）
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

Iteration 10 要做对照实验，需要能复盘每次实验。当前 pipeline / executor / tools 只负责跑通，**没有产物**。

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
- **Iteration 10 对照实验**：批量跑实验后 `grep -l "Pipeline finished, success=true" */experiment.log` 统计成功率
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
- **Iteration 11/12（SmallVLA / OpenVLA）**：接真实 VLA 时，链路已经验证通，executor 喂 env rgb 即可
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

**注意**：LLMVLA 可以不看图片（image 参数忽略），因为它接收的是语义化的 ee_pos + object_info。但接口遵守 `BaseVLA.predict(image, instruction)` 契约，image 参数保留。真正必须看图的是 SmallVLA/OpenVLA（Iteration 11/12）。

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

## Iteration 8: 探索机制 v1 - 基础探索

**目标**：实现"先探索再执行"的双阶段流程，验证探索-执行闭环。

### 背景

项目核心创新点是探索-执行双阶段。Iteration 8 实现最基础的探索能力：让 LLM 能主动调用探索工具，记录环境知识，并在执行阶段利用这些知识。

### 任务

1. **新增探索工具**

   - `explore(direction: str)`：控制机械臂在指定方向移动，记录沿途观测
   - `probe(object: str, action: str)`：对特定物体执行探测操作（推/碰/试抓），记录结果
   - 工具继承 LangChain BaseTool，集成到 Agent
2. **实现探索知识库**

   - `ExploreState` 数据结构：记录探索历史（动作、观测、结果）
   - `KnowledgeBase` 类：`record()` / `get_summary()` / `summarize()`
   - 探索结束后 LLM 生成自然语言环境知识摘要
3. **双阶段 Pipeline**

   - `pipeline.run(explore_first=True)`：先探索再执行
   - 探索阶段：LLM 自主调用 explore/probe，积累环境知识
   - 执行阶段：将知识摘要注入系统提示词，指导 LLM 规划动作
   - `pipeline.run(explore_first=False)`：直接执行（对照组）
4. **扩展 Agent 系统提示词**

   - 探索阶段提示词：鼓励 LLM 主动探索物理特性、工作空间
   - 执行阶段提示词：注入探索知识，引导利用经验

### 交付物

- `src/explore/` 模块（工具 + 知识库）
- 双阶段 pipeline
- 探索知识摘要样例

---

## Iteration 9: 探索机制 v2 - 策略化探索与知识利用

**目标**：提升探索质量，验证探索知识对 VLA 执行的实质性提升。

### 任务

1. **物理特性探测**

   - 推力-位移关系：记录施加动作与物体位移的对应关系
   - 夹取测试：尝试不同夹爪力度，记录成功/失败
   - 摩擦系数估计：通过多次推动推断
2. **工作空间探测**

   - 机械臂运动边界映射：各方向最大可达位置
   - 可达性图谱：哪些区域机械臂能稳定操作
3. **结构化环境知识**

   - 物体特性表：每个物体的重量、摩擦、可抓性评分
   - 空间约束表：工作空间边界、可达区域
   - 操作策略表：针对不同物体的推荐操作方式
4. **探索-执行知识传递优化**

   - 子指令增强：将环境知识融入 LLM 下发给 VLA 的子指令
     - 无知识："移动到红色方块上方"
     - 有知识："缓慢移动到红色方块上方（方块较滑，避免碰推）"
   - 对比不同知识传递方式的效果
5. **失败重规划机制**

   - 执行失败后，LLM 分析失败原因（距离不够/碰落物体/夹爪未对准）
   - 基于探索经验调整策略（换角度/换力度/换路径）

### 交付物

- 策略化探索工具集
- 结构化环境知识表示
- 失败重规划 demo
- 知识传递效果对比记录

---

## Iteration 10: 评估框架与对照实验

**目标**：建立量化评估体系，跑首批对照实验，用 LLMVLA 验证探索机制的效果。

### 任务

1. **任务集定义**

   - 简单抓取放置："把红色方块放到蓝色区域"
   - 多物体操作："把两个方块分别放到左右两侧"
   - 精细操控："把方块叠到杯子上"
   - 工具使用："用推的方式把方块移到目标位置"
   - 每类 20 个任务实例（随机初始化）
2. **评估指标实现**

   - 任务成功率（done_criteria 判定）
   - 平均执行步数（VLA 调用次数）
   - 探索收益（有探索 vs 无探索成功率差）
   - 失败恢复率（失败后重规划并成功的比例）
3. **对照实验框架**

   | 实验组   | 探索阶段       | 是否利用知识 | 预期作用                         |
   | -------- | -------------- | ------------ | -------------------------------- |
   | 基线组   | 无             | 无           | 纯 VLA 泛化基线                  |
   | 实验组   | 有             | 有           | 验证探索-执行整体效果            |
   | 消融组 A | 有             | 无           | 验证提升来自"利用知识"而非"热身" |
   | 消融组 B | 无（多走步数） | 无           | 控制总步数，排除"多走几步"的干扰 |
4. **实验自动化**

   - 批量实验脚本：自动跑 N 次任务，记录结果
   - 结果统计与可视化：成功率柱状图、步数分布图
   - 实验报告生成：自动汇总数据到 markdown
   - 集成 Iteration 3 的数据保存能力，每次实验自动持久化

### 交付物

- `src/evaluation/` 模块（任务集 + 指标 + 实验运行器）
- 首批对照实验数据（基于 LLMVLA）
- 实验报告初稿

---

## Iteration 11: 小型 VLA 接入

**目标**：训练并接入真实的小型 VLA，替换 LLM-VLA。验证 LLMVLA 上验证过的探索机制在真实小 VLA 上是否同样有效。

### 背景

LLM-VLA 能跑通流程，但它本质上也是同一个 LLM，不是真正的"脊髓级"模型——它直接"看到" ee_pos 和 object_info 语义化信息，而真实 VLA 只能看图像 + 指令。Iteration 11 引入真正的小 VLA，保证研究结论的有效性。

### 任务

1. **设计轻量级 VLA 架构**

   - 基于 CNN（图像编码）+ MLP（指令 embedding + 动作解码）
   - 参数量 < 10M，M4 Air CPU 可推理（单次推理 < 100ms）
   - 输入：场景图像 (H,W,3) + 自然语言指令
   - 输出：7D 动作（dx/dy/dz/drx/dry/drz/gripper）
2. **训练数据生成**

   - 用 PyBullet 仿真自动生成 (图像, 指令, 动作) 三元组
   - 简单任务：移动到目标位置（用 IK 计算参考动作）
   - 数据增强：多视角、多物体颜色、随机初始位姿
3. **实现 SmallVLA 类**（`executor/model/small_vla/small_vla.py`）

   - 继承 `BaseVLA`，实现 `predict(image, instruction) -> Action7D`
   - **必须使用 image 参数**（与 LLMVLA 不同，SmallVLA 从像素提取信息）
   - 支持模型加载/保存（从 `vla_config.model_path` 读路径）
   - 注册到 factory：backend == `"small_vla"` 时构造返回
4. **验证基础闭环与探索迁移**

   - SmallVLA 基础闭环：LLM 下发指令 → SmallVLA 产生动作 → 机械臂移动
   - 探索机制迁移：将 Iteration 7 在 LLMVLA 上验证的探索流程应用到 SmallVLA
   - 对比 LLMVLA vs SmallVLA 在相同探索机制下的性能差异

### 交付物

- `src/executor/model/small_vla/small_vla.py` 实现
- `executor/model/factory.py` 中 small_vla backend 注册
- 训练脚本 `scripts/train_small_vla.py`
- 训练数据生成脚本
- SmallVLA 探索机制迁移验证报告

---

## Iteration 12: OpenVLA 接入与正式实验

**目标**：探索机制完全成熟后，在 4060 笔记本上接入 OpenVLA，做最终正式对照实验，形成结论。

### 任务

1. **实现 OpenVLA 适配器**（`executor/model/openvla/openvla_adapter.py`）

   - 继承 `BaseVLA`，封装 OpenVLA-7B 模型
   - 支持 4bit 量化加载（适配 8GB 显存，模型路径从 `vla_config.model_path` 读）
   - 注册到 factory：backend == `"openvla"` 时构造返回
2. **部署与测试**

   - 在 4060 笔记本部署 OpenVLA-7B 4bit
   - 验证基础闭环（LLM + OpenVLA）
3. **正式实验**

   - LLMVLA vs SmallVLA vs OpenVLA，各有探索 vs 无探索
   - 6 组对照，每组 20 次/任务类
   - 核心对比：探索机制对不同规模 VLA 的收益差异
     - LLM-VLA（本身也懂语义）：探索收益是否最小？
     - SmallVLA（能力最弱）：探索收益是否最大？
     - OpenVLA（能力强）：探索收益居中？
4. **结论与交付**

   - 录制 demo 视频
   - 形成最终实验报告
   - 总结探索机制的适用条件与局限

### 交付物

- `src/executor/model/openvla/openvla_adapter.py` 实现
- `executor/model/factory.py` 中 openvla backend 注册
- 正式实验数据（6 组对照）
- 最终实验报告
- demo 视频

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
        ├── small_vla/small_vla.py     SmallVLA（Iteration 11）
        └── openvla/openvla_adapter.py OpenVLA（Iteration 12）
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
explore/       探索机制相关（Iteration 8 启用）
evaluation/    评估框架（Iteration 10 启用）
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
| `llm_vla`   | `executor/model/llm_vla/llm_vla.py`         | 否（用文本语义信息） | 探索机制研究基座（Iteration 7-10） |
| `small_vla` | `executor/model/small_vla/small_vla.py`     | **是**（必须） | 真实小 VLA 对照（Iteration 11）   |
| `openvla`   | `executor/model/openvla/openvla_adapter.py` | **是**（必须） | 最终正式实验（Iteration 12）     |

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

### 探索知识传递路径

```
探索阶段：LLM 调用 explore/probe → KnowledgeBase.record()
                                    ↓
                    KnowledgeBase.summarize() → 自然语言环境摘要
                    ↓
执行阶段：环境摘要注入系统提示词 → LLM 规划时利用知识
                                    ↓
                    LLM 下发增强子指令 → VLA 执行
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
