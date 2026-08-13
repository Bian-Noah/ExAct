# ExAct 迭代计划

## 项目愿景

研究**大模型（LLM）配合 VLA**，在**探索机制**下是否能更好地完成具身操控任务。

核心假设：让 LLM 大脑先探索环境、积累经验，再指导 VLA 脊髓执行任务，能弥补 VLA 泛化能力不足的问题。

---

## 迭代路线图总览

| 迭代 | 名称                               | 目标                                                  | VLA 后端                  | 设备        |
| ---- | ---------------------------------- | ----------------------------------------------------- | ------------------------- | ----------- |
| 1    | 流水线重构与配置化                 | 清理技术债，建立可扩展结构                            | Mock（不替换）            | M4 Air      |
| 2    | 渲染与环境模式配置化               | 解决 M4 Mac 段错误，渲染器(GPU/CPU)与环境模式均可配置 | Mock                      | M4 Air      |
| 3    | 实验数据保存与 Pipeline 完善       | 支持实验轨迹/结果持久化，可复现                       | Mock                      | M4 Air      |
| 4    | 图片存储器与 LLM 视觉能力          | LLM 能看图、存图，URL 可传给 VLA                      | Mock                      | M4 Air      |
| 5    | LLM-VLA 适配器                     | 用 MiniMax-M3 本身作为伪 VLA，机械臂真正响应指令      | **LLMVLA**          | M4 Air      |
| 6    | 探索机制 v1 - 基础探索             | 先探索再执行的双阶段流程跑通                          | LLMVLA                    | M4 Air      |
| 7    | 探索机制 v2 - 策略化探索与知识利用 | 提升探索质量，验证知识传递效果                        | LLMVLA                    | M4 Air      |
| 8    | 评估框架与对照实验                 | 量化探索收益，跑首批对照实验                          | LLMVLA → SmallVLA        | M4 Air      |
| 9    | 小型 VLA 接入                      | 训练真实小型 VLA，替换 LLM-VLA                        | **SmallVLA**        | M4 Air      |
| 10   | OpenVLA 接入与正式实验             | 探索机制完善后，接入 OpenVLA 做正式对照               | **OpenVLA-7B 4bit** | 4060 笔记本 |

---

## Iteration 1: 流水线重构与配置化

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
       │   └── llm_vla.py        # LLMVLA（Iteration 5 实现）
       ├── small_vla/
       │   └── small_vla.py      # SmallVLA（Iteration 9 实现）
       └── openvla/
           └── openvla_adapter.py # OpenVLA 适配器（Iteration 10 实现）
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
   - 注意：此阶段 observe 返回的仍是文本，图片如何传给 LLM 在 Iteration 4 解决
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

## Iteration 3: 实验数据保存与 Pipeline 完善

**目标**：完善 pipeline，支持实验轨迹和结果的持久化保存，为后续对照实验打基础。

### 背景

当前 pipeline（Iteration 1 建立）只负责组装和运行，不保存实验数据。后续 Iteration 8 要做对照实验，需要：

- 每次实验的完整轨迹（工具调用记录、LLM 回答、动作序列）
- 实验配置快照（用了什么 VLA 后端、什么任务、是否探索）
- 结构化存储，便于后续统计分析和复现

### 任务

1. **设计 PipelineResult 数据结构**

   - 扩展 `AgentResult`，增加实验元信息：
     - `experiment_id`：实验唯一 ID（时间戳 + 任务名）
     - `config_snapshot`：实验时的配置快照（YAML dump）
     - `task_spec`：任务定义
     - `timestamp_start` / `timestamp_end`：起止时间
     - `env_mode`：direct / gui
     - `vla_backend`：mock / llm_vla / small_vla / openvla
   - 包含完整轨迹：`trajectory`（工具调用记录）+ `final_answer`
2. **实验数据持久化**

   - `pipeline/` 新增 `storage.py` 模块
   - 每次实验保存到 `experiments/{experiment_id}/` 目录：
     ```
     experiments/
       └── 20260813_153000_task_move_to_red/
         ├── config.yaml          # 配置快照
         ├── result.json          # PipelineResult 序列化
         ├── trajectory.jsonl     # 工具调用轨迹（每行一条）
         └── meta.json            # 实验元信息
     ```
   - `ExperimentStorage.save(result)` 方法：原子写入，失败不丢数据
3. **Pipeline 集成数据保存**

   - `pipeline.run()` 执行完成后自动调用 `storage.save(result)`
   - 支持配置开关：`config.experiment.save_to_disk`（默认 True）
   - 支持自定义输出目录：`config.experiment.output_dir`（默认 `experiments/`）
4. **实验配置扩展**

   - `config` 新增 `experiment` 节：
     ```yaml
     experiment:
       save_to_disk: true
       output_dir: "experiments"
       save_trajectory: true       # 是否保存详细轨迹
       save_config_snapshot: true  # 是否保存配置快照
     ```
5. **实验索引与查询**

   - `ExperimentStorage.list_experiments()`：列出所有实验
   - `ExperimentStorage.load(experiment_id)`：加载某个实验结果
   - 为 Iteration 8 的批量实验和统计分析打基础

### 交付物

- `src/pipeline/storage.py` 实现
- `PipelineResult` 数据结构（含实验元信息）
- `config.experiment` 配置节
- 实验数据自动保存到 `experiments/` 目录
- 数据加载与查询 API

---

## Iteration 4: 图片存储器与 LLM 视觉能力

**目标**：建立图片存储器模块，让 LLM 能看图片、存图片，并通过 URL 将图片传递给执行工具和 VLA 模型。

### 背景

Iteration 2 解决了渲染段错误，observe 工具能拿到 RGB 图像。但当前 observe 工具返回的是纯文本字符串，图片无法传给 LLM。本迭代建立图片存储器，实现图片的存储、URL 引用和跨模块传递。

### 核心设计

```
observe 工具获取图片
    ↓
ImageStore.save(image) → 生成 URL（如 "img://observations/20260813_153000_001.png"）
    ↓
observe 工具返回给 LLM 的消息中包含：
    1. 文本描述（ee_pos + object_info）
    2. 图片 URL（LLM 可通过 URL 访问图片内容）
    ↓
LLM 决策后调用 action 工具时，可传入 image_url
    ↓
action 工具将 image_url 传给 executor
    ↓
executor 通过 ImageStore.load(url) 获取图片
    ↓
传给 VLA.predict(image, instruction)
```

### 任务

1. **实现图片存储器模块**（`utils/image_store/`）

   - 目录结构：
     ```
     utils/image_store/
       ├── __init__.py
       ├── store.py          # ImageStore 主类
       ├── backend.py        # 存储后端抽象（内存 / 文件系统）
       └── url_scheme.py     # URL 生成与解析
     ```
   - `ImageStore` 类核心接口：
     - `save(image: np.ndarray, category: str = "observations") -> str`：保存图片，返回 URL
     - `load(url: str) -> np.ndarray`：通过 URL 加载图片
     - `exists(url: str) -> bool`：检查 URL 是否存在
     - `list(category: str = None) -> list[str]`：列出图片 URL
   - URL 格式设计：`img://{category}/{filename}`（如 `img://observations/20260813_153000_001.png`）
   - 存储后端：
     - `MemoryBackend`：图片存内存（默认，适合单次运行）
     - `FileBackend`：图片存文件系统（适合持久化，路径从 config 读）
   - 工厂函数 `create_image_store(config) -> ImageStore`
2. **observe 工具改造**

   - `ObserveTool` 注入 `ImageStore` 实例
   - `observe._run()` 获取 RGB 图像后：
     - 调用 `image_store.save(image)` 生成 URL
     - 返回结构化内容：文本描述 + 图片 URL
   - LangChain BaseTool 返回支持多模态内容（`content` 可以是 list，含 `text` 和 `image_url` 块）
   - LLM 通过 `image_url` 字段看到图片
3. **action 工具改造**

   - `ActionTool` 注入 `ImageStore` 实例
   - `ActionInput` 新增可选字段 `image_url: Optional[str]`
   - LLM 调用 action 时可传入 observe 返回的图片 URL
   - `action._run()` 将 `image_url` 传给 `executor.run_action()`
4. **executor 链路改造**

   - `Executor.run_action()` 新增可选参数 `image_url: str = None`
   - 若提供 `image_url`：通过 `image_store.load(url)` 获取图片，传给 VLA
   - 若未提供：走原有逻辑（`env.get_obs()` 获取当前图片）
   - `build_vla_input()` 支持 `image_url` 字段
5. **BaseVLA 接口验证**

   - `BaseVLA.predict(image, instruction)` 接口不变
   - LLMVLA（Iteration 5）可以忽略 image 参数
   - SmallVLA/OpenVLA（Iteration 9/10）必须使用 image 参数
   - 图片来源由上层（executor）决定，VLA 只接收 ndarray
6. **config 配置扩展**

   - 新增 `image_store` 配置节：
     ```yaml
     image_store:
       backend: "memory"          # "memory" | "file"
       file_dir: "data/images"    # 仅 file 后端使用
     ```
7. **pipeline 集成**

   - `pipeline/runner.py` 创建 `ImageStore` 实例，注入 observe 和 action 工具
   - 实验结束时（Iteration 3 的 storage）可选保存所有图片

### 交付物

- `src/utils/image_store/` 模块（store + backend + url_scheme）
- observe 工具改造（返回文本 + 图片 URL）
- action 工具改造（接受 image_url 参数）
- executor 链路改造（支持 image_url 传图）
- `config.image_store` 配置节
- 端到端验证：LLM 能通过 observe 看到图片，action 能传递图片 URL 给 VLA

---

## Iteration 5: LLM-VLA 适配器

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

**注意**：LLMVLA 可以不看图片（image 参数忽略），因为它接收的是语义化的 ee_pos + object_info。但接口遵守 `BaseVLA.predict(image, instruction)` 契约，image 参数保留。真正必须看图的是 SmallVLA/OpenVLA（Iteration 9/10）。

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

## Iteration 6: 探索机制 v1 - 基础探索

**目标**：实现"先探索再执行"的双阶段流程，验证探索-执行闭环。

### 背景

项目核心创新点是探索-执行双阶段。Iteration 6 实现最基础的探索能力：让 LLM 能主动调用探索工具，记录环境知识，并在执行阶段利用这些知识。

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

## Iteration 7: 探索机制 v2 - 策略化探索与知识利用

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

## Iteration 8: 评估框架与对照实验

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

## Iteration 9: 小型 VLA 接入

**目标**：训练并接入真实的小型 VLA，替换 LLM-VLA。验证 LLMVLA 上验证过的探索机制在真实小 VLA 上是否同样有效。

### 背景

LLM-VLA 能跑通流程，但它本质上也是同一个 LLM，不是真正的"脊髓级"模型——它直接"看到" ee_pos 和 object_info 语义化信息，而真实 VLA 只能看图像 + 指令。Iteration 9 引入真正的小 VLA，保证研究结论的有效性。

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

## Iteration 10: OpenVLA 接入与正式实验

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

- 所有可调参数（机械臂、任务、Agent 限制、VLA 后端、环境模式、图片存储）统一在 YAML 配置
- 程序内部保留默认 fallback 值，config 缺失时使用默认值
- pipeline 从 config 提取参数，显式传入各模块（不在模块内直接读 config）

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
        ├── llm_vla/llm_vla.py         LLMVLA（Iteration 5）
        ├── small_vla/small_vla.py     SmallVLA（Iteration 9）
        └── openvla/openvla_adapter.py OpenVLA（Iteration 10）
pipeline/      主流程编排（组装 env + executor + agent + storage）
  ├── runner.py     Pipeline 主流程
  └── storage.py    实验数据持久化（Iteration 3）
utils/
  ├── logging.py    日志
  ├── image.py      图像编解码工具
  └── image_store/  图片存储器（Iteration 4，大模块）
        ├── __init__.py
        ├── store.py          ImageStore 主类
        ├── backend.py        存储后端（内存 / 文件系统）
        └── url_scheme.py     URL 生成与解析
explore/       探索机制相关（Iteration 6 启用）
evaluation/    评估框架（Iteration 8 启用）
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
| `llm_vla`   | `executor/model/llm_vla/llm_vla.py`         | 否（用文本语义信息） | 探索机制研究基座（Iteration 5-8） |
| `small_vla` | `executor/model/small_vla/small_vla.py`     | **是**（必须） | 真实小 VLA 对照（Iteration 9）    |
| `openvla`   | `executor/model/openvla/openvla_adapter.py` | **是**（必须） | 最终正式实验（Iteration 10）      |

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

### 实验数据持久化路径

```
pipeline.run() 执行
    ↓
PipelineResult (含轨迹 + 配置快照 + 元信息)
    ↓
ExperimentStorage.save(result)
    ↓
experiments/{experiment_id}/
  ├── config.yaml          # 配置快照
  ├── result.json          # 完整结果
  ├── trajectory.jsonl     # 工具调用轨迹
  └── meta.json            # 实验元信息
```
