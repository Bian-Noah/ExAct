# ExAct

面向桌面任务的仿真机械臂具身智能体。用户输入一句指令，机械臂自主观察、规划、执行，可选地先探索环境再动手。

## 架构

```
   VLA 层（可插拔）        适配层               机器人层（可插拔）
┌──────────────────┐   ┌─────────────────┐   ┌─────────────────────┐
│ MockVLA          │   │  utils/adapter/ │   │  env/robot/panda/   │
│ LeRobotVLA       │──▶│  get_adapter()  │──▶│  env/robot/so101/   │
│ OpenVLA          │   │  按 spec 选适配  │   │  env/robot/widowx/  │
│ LLMVLA / SmallVLA│   └─────────────────┘   └─────────────────────┘
│ 自声明 output_spec│        spec 匹配            ▲ 自声明 input_spec
└──────────────────┘                              │
                              └── ActionTool._run 里调用 ──┘
```

数据流：用户在 `configs/local.yaml` 写 `task.default_user_goal` → `app.py` 调 `pipeline.run_pipeline` → 构造 env / VLA / LLM / 工具 / Agent → Agent 在主线程同步循环调 `observe` / `action` / `explore` 工具 → `ActionTool` 走 VLA chunk → adapter → env.step → 返回 obs。

## 核心约定

1. **VLA 自声明 `output_spec`**：`BaseVLA` 子类在内部声明自己输出的动作语义（空间、维度、每维含义），不通过 config 配置。
2. **机器人自声明 `input_spec`**：`Robot` 子类在 `input_spec` property 里声明自己消费的动作 spec。
3. **适配层按 spec 自动分派**：`get_adapter(vla.output_spec, env.input_spec)` 按 `(vla_space, env_space)` 查注册表返回转换函数。组合未注册时抛 `AdapterNotFoundError`。
4. **VLA chunk 契约**：`VLAOutput.values` shape 为 `(N, action_dim)`。N 由 VLA 后端决定（smolVLA=50, ACT=100, Mock=1 等）。Executor 整 chunk 接收，循环 N 次 `env.step`，循环中**不**调用 `check_done`，循环结束后做一次事后报告。

## 安装

安装命令取决于你想用哪种 VLA 后端。

### 项目内置两种 VLA

| VLA | 安装命令 | 适用场景 |
|-----|---------|---------|
| MockVLA | 无（项目自带） | 无 GPU 环境下的链路验证、Agent 流程调试 |
| OpenVLA | `pip install -r requirements-openvla.txt` | 真实权重推理（需 CUDA GPU；4bit 量化时 ~4GB 显存） |

### 其他 VLA 后端（LeRobot / LLMVLA / SmallVLA / 自定义）

项目仅维护上述两种内置 VLA。其他后端（如 LeRobot 的 ACT/SmolVLA/Pi0 等）需要自行完成三件事：
1. 在 `src/executor/model/<name>/` 下新增 `<name>_vla.py`，继承 `BaseVLA` 并实现 `output_spec` + `predict`
2. 在 `src/executor/model/factory.py` 的 `create_vla()` 加分派分支
3. 准备对应的 `requirements-<name>.txt` 并安装

详见下文「添加新的 VLA 模型」一节。

### 验证

```bash
python -c "import pybullet as p; p.connect(p.DIRECT); p.disconnect(); print('OK')"
python -c "import yaml, numpy, PIL, gymnasium, langchain, requests, pytest; print('all OK')"
```

M4 Mac 无 torch，依赖 torch 的 OpenVLA 用例会 skip；用 `backend: mock` 在本地完整跑通链路。


## 配置

完整配置节见 `configs/default.yaml`。以下逐节说明：

### `env`

```yaml
env:
  mode: direct          # "direct" | "gui"
  renderer: auto        # "auto" | "cpu" | "gpu"
  camera_resolution: [640, 480]
  cameras:              # iter11 多相机：每个相机作为 obs["rgb"] 的一个 key
    - name: observation.images.top       # 键名需与下游 VLA policy 一致
      target: [0.5, 0.0, 0.30]
      distance: 0.9
      yaw: 90
      pitch: -15
      roll: 0
      fov: 60
      resolution: [640, 480]
```

### `vla`

`backend` 字段决定分派目标：

| backend | 状态 | variant / policy / 参数 |
|---------|------|------------------------|
| `mock` | 内置 | `mock.variant: task` → MockVLA（7D task 空间）；`joint` → JointMockVLA（6D joint 空间） |
| `openvla` | 内置 | `openvla.unnorm_key / attn_impl / dtype / quantization` |
| `lerobot` | 历史遗留 | factory 分派仍存在但项目不再维护；如需使用自行补依赖与代码 |
| `llm_vla` | 预留 | factory 分派抛 `NotImplementedError`；需自行实现 |
| `small_vla` | 预留 | factory 分派抛 `NotImplementedError`；需自行实现 |

`model_path` 仅 openvla 需要；mock 设为 null。

### `robot`

```yaml
robot:
  type: panda            # panda | so101 | widowx
  urdf_path: "franka_panda/panda.urdf"
  base_position: [0.0, 0.0, 0.0]
  panda:                 # type=panda 时生效
    arm_joint_indices: [0, 1, 2, 3, 4, 5, 6]
    ee_link_index: 11
    finger_joint_indices: [9, 10]
  so101:                 # type=so101 时生效
    arm_joint_indices: [0, 1, 2, 3, 4, 5]
    ee_link_index: 6
  widowx:                # type=widowx 时生效
    arm_joint_indices: [0, 1, 2, 3, 4]
    ee_link_index: 11
    gripper_joint_indices: [9, 10]
    urdf_url: ...        # 首次运行时自动下载
    urdf_local_path: ...
```

旧平铺格式（`arm_joint_indices` 直接在 `robot:` 顶层）仍兼容：加载时自动合并进对应特化块。

### `llm`

```yaml
llm:
  api_key: "YOUR_API_KEY_HERE"   # ★ 必填，提交代码时保持占位符
  model: "MiniMax-M3"
  base_url: "https://api.minimax.chat/v1"
  max_tokens: 2048
```

通过 `langchain_openai.ChatOpenAI` 连接 OpenAI 兼容接口。

### `task`

```yaml
task:
  default_user_goal: "把机械臂移到红色方块上方"
  objects:
    - type: cube
      pos: [0.5, 0, 0.35]
      color: red
```

`default_user_goal` 在 `app.py` 未传 `user_goal` 参数时作为 Agent 的初始指令；`objects` 在 `task_spec=None` 时传入 `env.reset()`。

### `agent`

```yaml
agent:
  max_react_rounds: 5
  max_tool_calls: 3
  # extra_prompt: |    # 可选，替换默认系统提示词中的"后端模型 / 运行策略"段
  #   本场景运行 OpenVLA(bridge_orig)...
```

### `explore`

```yaml
explore:
  enabled: false        # 关闭时不构造 Explore、不挂 ExploreTool
  root: "data/explore"  # 探索笔记按日落盘到 {root}/{YYYY-MM-DD}.log
```

### `experiment`

```yaml
experiment:
  enabled: true
  root: "data/experiment"
  log_to_stdout: true
  video:
    enabled: true
    filename: "demo.mp4"
    fps: 15
    resolution: [640, 480]   # 必须为偶数宽高
    camera: null             # null = env.cameras[0]
    capture_every_n_steps: 1
```

### `image_store`

```yaml
image_store:
  backend: "memory"          # "memory" | "file"
  file_dir: "data/images"
  max_memory_items: 10       # memory backend 内存上限；溢出 spill 到 file_dir
```

## 运行

```bash
# 1. 复制 default.yaml 为 local.yaml，填入 llm.api_key
cp configs/default.yaml configs/local.yaml

# 2. 在 local.yaml 里按需覆盖（如 env.mode: gui / explore.enabled: true）

# 3. 运行
PYTHONPATH=src python src/app.py
```

`app.py` 二选一加载：`configs/local.yaml` 存在则用 local，否则用 default。local.yaml 缺省节走默认值。

## 工具

Agent 通过以下三个 LangChain `BaseTool` 子类与 env / Explore 交互：

### ObserveTool

观察当前场景。返回 LangChain 标准 content blocks：1 个 text 块（末端位置 + 物体列表）+ N 个 image 块（N = `env.cameras` 数量）。

### ActionTool

执行动作。两个字段：
- `instruction`：动作指令（OpenVLA 路线需符合 `bridge_orig` 训练分布：英文、单句、≤50 字符）。
- `operation`：`"vla"`（默认，走 VLA 推理）或 `"reset"`（瞬时把机械臂关节复位到 home，不走 VLA）。

调用流程：`_run` 入口用 `validate_instruction` 拦截不合规指令 → 构造 VLA 输入（`build_vla_input`）→ `Executor.run_action`（VLA 路径）或 `env.reset_arm_to_home`（reset 路径）→ 循环结束后 `check_done` 做事后报告。

### ExploreTool

探索笔记。`sub_action`：`"read_notes"`（查阅既有笔记）或 `"write_note"`（写入新笔记，单行）。详见「探索阶段」一节。

## 探索阶段

项目支持可选的探索阶段：在执行任务前由 Agent 主动调用 `explore.read_notes` 查阅经验、调用 `explore.write_note` 记录发现。

实现：
- `Explore` 类维护内存笔记列表；`enabled=True` 时构造并按 `flush()` 落盘到 `{root}/{YYYY-MM-DD}.log`，flush 后清空内存。
- `ExploreTool` 提供 `read_notes` / `write_note` 两个子动作，避免挤占 `max_tool_calls=3` 的预算。
- `configs/default.yaml` 的 `explore.enabled` 开关全局控制；关闭时不构造 `Explore`，不挂 `ExploreTool`。

## 实验录制

`ExperimentRecorder`（`src/experiment/recorder.py`）记录三类事件：

| 事件 | 落盘位置 | 触发点 |
|------|---------|--------|
| `log` | `experiment.log` | pipeline 启动 / 结束、各工具调用 |
| `observe_image` | `observer/{idx:03d}.png` | ObserveTool 调用 |
| `video_frame` | `process/{filename}.mp4` | 每个物理子步（按 `capture_every_n_steps` 节流） |

每次 pipeline 运行在 `{experiment.root}/{timestamp}/` 下产出独立目录。`enabled=False` 时所有方法 no-op。

视频录制需要 `cv2`（来自 `opencv-python`）。未安装时降级为 no-op，不影响业务主线。

## 添加新的 VLA 模型

只需 3 步：

1. **建后端文件**：`src/executor/model/<name>/<name>_vla.py`，继承 `BaseVLA`，实现 `output_spec` 与 `predict`：

   ```python
   from executor.model.base import BaseVLA, VLAOutput
   from env.base import ActionSpec

   class MyVLA(BaseVLA):
       @property
       def output_spec(self) -> ActionSpec:
           return ActionSpec("joint", ("joint",) * 6)

       def predict(
           self,
           image: dict[str, np.ndarray],   # 多相机 RGB dict（iter11 起）
           instruction: str,
           state: np.ndarray | None = None,  # 可选关节角 / 本体感知
       ) -> VLAOutput:
           values = ...                        # shape (N, action_dim)
           return VLAOutput(values=values, spec=self.output_spec)
   ```

2. **工厂分派**：`src/executor/model/factory.py` 的 `create_vla()` 加一个 `backend == "<name>"` 分支。

3. **config 声明**：`configs/default.yaml` 的 `vla.backend: <name>`。

适配层自动工作：`get_adapter(vla.output_spec, env.input_spec)` 按 spec 组合选 adapter，**无需改 adapter 代码**。若组合未注册（见下表），需新增 adapter。

### 当前 adapter 注册表（`src/utils/adapter/main.py`）

| vla 空间 → env 空间 | adapter | 用途 |
|---------------------|---------|------|
| `joint` → `joint` | `joint_to_joint_transform` | LeRobot → SO101 / Panda（数值映射：截取 / 补 gripper） |
| `task` → `joint` | `task_to_joint_hardcode_transform` | 当前是前 N 维硬编码占位，**未接入真实 IK** |
| `task` → `task` | `identity_transform` | Mock → Panda（同空间直通） |
| `joint` → `task` | 未实现 | 抛 `AdapterNotFoundError` |

## 添加新的机器人

只需 4 步：

1. **建机器人包**：`src/env/robot/<type>/<type>_robot.py`，实现 `Robot` 接口（构造 + `input_spec` + `step_action`）：

   ```python
   from env.base import ActionSpec

   class MyRobot:
       def __init__(self, robot_config: RobotConfig):
           self.arm_joint_indices = tuple(robot_config.myrobot.arm_joint_indices)

       @property
       def input_spec(self) -> ActionSpec:
           return ActionSpec("joint", ("joint",) * 6)

       def step_action(self, action, robot_id, client_id, on_substep=None):
           # 关节驱动 + stepSimulation
           ...
   ```

   参考：`panda/panda_robot.py`（7 臂 IK + 2 夹爪）、`so101/so101_robot.py`（5 臂 + gripper）、`widowx/widowx_robot.py`（5 臂 IK + 2 夹爪，带 URDF 自动下载）。

2. **注册分派**：`src/env/robot/__init__.py` 的 `_ROBOT_REGISTRY` 加 `"<type>": MyRobot`。

3. **config 特化配置**：`src/config/loader.py` 加 `<type>` 特化 dataclass，`RobotConfig` 加嵌套字段（`default_factory`）；`configs/default.yaml` 的 `robot.<type>` 节补特化块。

4. **切换**：`configs/*.yaml` 的 `robot.type: <type>` + `urdf_path` 指向 URDF。WidowX 还要在 `widowx.urdf_url` 给定远程 URL，首次运行自动下载 URDF + mesh（URDF 引用 `package://...` 的 mesh 缺失会让 pybullet 加载失败）。

调用方 `PyBulletEnv(env_config, robot_config)` **零改动**——env 按 `robot.type` 自构造对应 Robot 注入。

## 测试

```bash
# 单元测试（按模块子目录组织）
python -m pytest tests/

# 手动 smoke 脚本（不参与 pytest 收集）
PYTHONPATH=src python tests/scripts/smoke_robot_vla_adapter.py
```

测试目录与源码对应：
- `tests/adapter/` — adapter 注册表与各 adapter
- `tests/agents/` — LangGraph agent + 提示词
- `tests/config/` — 配置 dataclass + loader
- `tests/pipeline/` — pipeline + recorder 集成
- `tests/tools/` — action / observe / explore 工具
- `tests/utils/` — image_store / image / logging

`tests/scripts/manual/` 下是手动验证脚本（如 `test_act_predict.py`）。

## 常见问题

- **GUI 弹窗打扰**：所有脚本 / 测试 / 调试用 `env.mode: direct`。需要看仿真再临时改 `gui`。
- **`AdapterNotFoundError`**：VLA 输出与机器人输入 spec 组合未在注册表。检查两个 spec 的 `space`，或按上表新增 adapter。
- **`未知机器人类型`**：`robot.type` 不在 `_ROBOT_REGISTRY`。按"添加新的机器人"补注册。
- **M4 Mac 无 torch**：用 `backend: mock` 开发；真实权重推理需 CUDA GPU。
- **OpenVLA 指令被 ActionTool 拒绝**：指令不符合 `validate_instruction` 约束（非空 / 全英文 / 单句 / ≤50 字符）。按工具返回的错误原因修正。
- **WidowX 首次运行慢**：`widowx_robot.ensure_urdf()` 会下载 URDF + 多个 mesh 文件，受网络影响；后续运行本地已有则跳过。
- **`requirements.txt` 是历史 lerobot 路线**：项目默认走 `requirements-openvla.txt`。如果你要用 lerobot / 其他 VLA，需自行准备依赖文件。
