# ExAct

机器人实验项目：**可插拔的机器人 + 可插拔的 VLA 模型**，中间通过适配层自动对接。

```
   VLA 层（可插拔）        适配层               机器人层（可插拔）
┌──────────────────┐   ┌─────────────────┐   ┌─────────────────────┐
│ MockVLA          │   │  utils/adapter/ │   │  env/robot/panda/   │
│ LeRobotVLA(ACT)  │──▶│  get_adapter()  │──▶│  env/robot/so101/   │
│ OpenVLA (预留)   │   │  按 spec 选适配  │   │  ...                │
│ 自声明 output_spec│   └─────────────────┘   └─────────────────────┘
└──────────────────┘        spec 匹配            ▲ 自声明 input_spec
                              │                  │
                              └── ActionTool._run 里调用 ──┘
```

核心概念：
- **VLA 自声明 `output_spec`**：每个模型在类内部声明自己输出的动作语义（task/joint 空间、维度、每维含义），不落入 config。
- **机器人自声明 `input_spec`**：每类机器人在自己的文件里声明消费的动作 spec。
- **适配层按 spec 自动选择 adapter**：`get_adapter(vla_spec, env_spec)` 查注册表返回转换函数，把 VLA 输出转换为 env 原生动作。

## 目录结构

```
configs/            # 配置文件（default.yaml 提交；local.yaml 含 API key，不提交）
src/
  config/           # 配置加载（AppConfig / EnvConfig / RobotConfig / VLAConfig...）
  env/
    base.py         # ActionSpec / Action7D / BaseEnv（input_spec + ik/fk 接口）
    pybullet_env.py # PyBulletEnv 通用仿真：连接/渲染/相机/物体/观测，自构造 Robot 注入
    robot/          # 机器人包：__init__.py(build_robot 工厂) + panda/ + so101/
  executor/         # 主循环；model/ 下是各 VLA 后端（mock/lerobot/openvla）
  tools/            # ActionTool（注入 adapter）/ ObserveTool
  utils/adapter/    # 适配层：main.py(注册表) + adapters/(identity/joint_to_joint/task_to_joint)
tests/              # pytest 单元测试 + 功能测试（tests/scripts/smoke_*.py）
```

## 安装依赖

### 阶段一（Mac 本地，仿真 + Mock）

```bash
# 1. conda 环境
conda activate exact          # 或按项目实际环境名
# 2. 阶段一依赖
pip install -r requirements.txt
```

验证环境：

```bash
python -c "import pybullet as p; p.connect(p.DIRECT); p.disconnect(); print('OK')"
python -c "import yaml, numpy, PIL, gymnasium, langchain, requests, pytest; print('all OK')"
```

### 阶段二（Windows + RTX 4060，真实 VLA 权重）

> ⚠️ **不能直接 `pip install -r requirements-stage2.txt`**（会从 PyPI 拉 CPU 版 torch，
> 覆盖 conda 的 CUDA 版 PyTorch）。按文件头注释的标准流程走：

```bash
conda activate robot
# 1) 与 torch 配套的 CUDA torchvision（必须 conda，否则覆盖 conda 的 torch）
conda install -c pytorch torchvision==0.20.1
# 2) conda-forge 上的运行时工具（与 conda 的 CUDA torch 配套最稳）
conda install -c conda-forge accelerate bitsandbytes tokenizers
# 3) 其他小依赖走 pip，默认安装（不要 --no-deps，让 pip 自动拉传递依赖）
pip install draccus opencv-python-headless termcolor
# 4) lerobot 本身是唯一允许 --no-deps 的包（防 pip 抢走 conda 的 CUDA torch）
pip install --no-deps lerobot
# 5) 闭环校验
pip check
```

`requirements-stage2.txt` 只作为依赖清单参考（当前唯一启用项是 `lerobot>=0.6`；
OpenVLA 相关整段注释掉，未启用）。

❌ 禁止：
- `pip install -r requirements-stage2.txt`
- `pip install torch torchvision torchaudio`（覆盖 conda 的 CUDA 版）

### 运行测试

```bash
python -m pytest tests/
```

> 注：M4 Mac 无 torch，`test_lerobot_vla_output_spec` 等依赖 torch 的用例会 skip；阶段二（RTX4060）验证。

## 运行

```bash
# 1. 配置 API key：复制 default.yaml 为 local.yaml，填入 llm.api_key
cp configs/default.yaml configs/local.yaml

# 2. 修改 env.mode：local.yaml 里设 mode: direct（无 GUI）或 gui（看仿真窗口）

# 3. 运行（存在 local.yaml 用 local，否则用 default）
PYTHONPATH=src python src/app.py
```

> `app.py` 二选一加载：`configs/local.yaml` 存在则用 local，否则用 default。local.yaml 缺省节走默认值，可以只写要覆盖的字段（如 llm.api_key、env.mode）。

## 添加新的 VLA 模型

只需 3 步，不需要改适配层和 config：

1. **建后端文件**：`src/executor/model/<name>/<name>_vla.py`，继承 `BaseVLA`，实现两个成员：

   ```python
   from executor.model.base import BaseVLA, VLAOutput
   from env.base import ActionSpec

   class MyVLA(BaseVLA):
       @property
       def output_spec(self) -> ActionSpec:
           # 诚实声明输出语义：空间 + 每维含义
           return ActionSpec("joint", ("joint",) * 6)   # 例：joint 6 维

       def predict(self, image, instruction) -> VLAOutput:
           values = ...                                  # 模型推理
           return VLAOutput(values=values, spec=self.output_spec)
   ```

2. **工厂分派**：`src/executor/model/factory.py` 的 `create_vla()` 加一个 `backend == "<name>"` 分支返回该实例。

3. **config 声明**：`configs/default.yaml` 的 `vla.backend: <name>`。

适配层自动工作：`get_adapter(vla.output_spec, env.input_spec)` 按 spec 空间组合选 adapter，**无需改 adapter 代码**。若你的模型输出语义是当前注册表没有的组合（见下表），才需要新增一个 adapter。

当前 adapter 注册表（`src/utils/adapter/main.py`）：

| vla 空间 → env 空间 | adapter | 场景 |
|---------------------|---------|------|
| task → task | `identity_transform` | Mock → Panda（同空间直通） |
| joint → joint | `joint_to_joint_transform` | LeRobot(SO101 joint) → SO101 / Panda joint（数值映射：截取/补 gripper） |
| task → joint | `task_to_joint_transform` | OpenVLA → Panda（预留，依赖 env.ik） |
| 其他 | `AdapterNotFoundError` | 未覆盖组合，需新增 adapter |

## 添加新的机器人

只需 4 步，不需要改 env 通用逻辑：

1. **建机器人包**：`src/env/robot/<type>/`，放 `<type>_robot.py`，实现 `Robot` 接口（构造 + `input_spec` + `step_action(action, robot_id, client_id)`）：

   ```python
   import pybullet as p
   from config.loader import RobotConfig
   from env.base import ActionSpec

   class MyRobot:
       def __init__(self, robot_config: RobotConfig):
           self.arm_joint_indices = tuple(robot_config.myrobot.arm_joint_indices)

       @property
       def input_spec(self) -> ActionSpec:
           return ActionSpec("joint", ("joint",) * 6)   # 声明消费的动作 spec

       def step_action(self, action, robot_id, client_id):
           # 用 action 驱动关节 + stepSimulation
           ...
   ```

   参考：`src/env/robot/panda/panda_robot.py`（7 臂 IK + 2 夹爪）、`src/env/robot/so101/so101_robot.py`（6 臂无 gripper）。

2. **注册分派**：`src/env/robot/__init__.py` 的 `_ROBOT_REGISTRY` 加 `"<type>": MyRobot`。

3. **config 特化配置**：`src/config/loader.py` 加 `<type>` 特化 dataclass（如 `MyRobotConfig`），`RobotConfig` 加嵌套字段（default_factory）。`configs/default.yaml` 的 `robot` 节补特化块。

4. **切换**：`configs/*.yaml` 的 `robot.type: <type>` + `urdf_path` 指向该机器人 URDF。

调用方 `PyBulletEnv(env_config, robot_config)` **零改动**——env 按 `robot.type` 自我构造对应 Robot 注入，机器人类型对外透明。

### RobotConfig 结构（`configs/default.yaml`）

```yaml
robot:
  type: panda                     # 分派键（env/robot/<type>/ 下加载对应 Robot）
  urdf_path: "franka_panda/panda.urdf"   # 顶层公共字段（所有机器人通用）
  base_position: [0.0, 0.0, 0.0]
  panda:                          # Panda 特化配置
    arm_joint_indices: [0, 1, 2, 3, 4, 5, 6]
    ee_link_index: 11
    finger_joint_indices: [9, 10]
  so101:                          # SO101 特化配置
    arm_joint_indices: [0, 1, 2, 3, 4, 5]
    ee_link_index: 6
```

> 旧平铺格式（`arm_joint_indices` 直接在 `robot:` 顶层）仍兼容：加载时自动合并进对应特化块。

## 常见问题

- **GUI 弹窗打扰**：功能测试 / 脚本一律 `env.mode: direct`（无窗口）。需要看仿真再临时改 `gui`。
- **`AdapterNotFoundError`**：你的 VLA 输出与机器人输入 spec 组合没有对应 adapter。检查两个 spec 的 `space`，或按上表新增 adapter。
- **`未知机器人类型`**：`robot.type` 不在 `_ROBOT_REGISTRY`。按"添加新的机器人"补注册。
- **LeRobot 缺 torch**：M4 Mac 无 torch，用 `backend: mock` 开发；真实权重推理在阶段二（RTX4060）环境。
