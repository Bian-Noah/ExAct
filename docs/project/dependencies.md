# 依赖清单

> 本文档只列**顶层依赖**——你直接安装的那些包。pip 装这些时自动拉进来的小依赖（如 langsmith、orjson、filelock、sympy 等）不列。
>
> **大依赖例外**：即使是被自动拉进来的，但体量大、对运行环境有显著影响（需要装 CUDA、占 GB 级空间、需要独立环境等），仍列入。

---

## 一、基础运行环境

| 包 | 用途 | 备注 |
|---|---|---|
| `python` | 解释器 | 需 3.10+，项目实测 3.12.13 |
| `pip` | 包管理器 | 跟随 python |

> 标准库（json/os/math/io/typing/dataclasses/pathlib/threading 等）Python 自带，**不需要装**。

---

## 二、仿真物理（env）

| 包 | 用途 | 安装渠道 |
|---|---|---|
| `pybullet` | 物理仿真本体 | conda-forge 或 pip（macOS ARM64 需 conda） |
| `torch` | 张量运算、policy 推理（CUDA 推理本体） | **大依赖**：conda（pytorch channel）+ CUDA |
| `torchvision` | 视觉算子 | 同上，与 torch 配套 |

---

## 三、图像处理

| 包 | 用途 | 是否可选 |
|---|---|---|
| `Pillow` | PNG/JPEG 编码解码 | 必需 |
| `imageio` | 异步 PNG IO（image_store 持久化时用） | 走 image_store 时必需；纯内存模式可省 |

---

## 四、配置加载

| 包 | 用途 | 是否可选 |
|---|---|---|
| `PyYAML` | 加载 yaml 配置文件 | 必需 |

---

## 五、LLM / Agent 编排

| 包 | 用途 | 是否可选 |
|---|---|---|
| `langchain-core` | `BaseTool`、Message 类型（Agent 工具契约） | 必需 |
| `langchain-openai` | `ChatOpenAI`（兼容 OpenAI 接口的 LLM） | 必需 |
| `langgraph` | Agent 的状态图运行时 | 必需 |
| `pydantic` | 工具入参的 schema 校验 | 必需 |

> 注：`typing_extensions.TypedDict` 也是项目用到，但**已被 langchain-core 拉进来**（属于"自动装的小依赖"），不用单独装。

---

## 六、VLA 模型后端

| 包 | 用途 | 是否可选 |
|---|---|---|
| `lerobot` | VLA 框架本体（ACT / SmolVLA / pi0 等 policy 的推理封装） | 启用 `vla.backend=lerobot` 时必需 |

---

## 七、VLA 量化（8GB 卡等显存吃紧时）

| 包 | 用途 | 备注 |
|---|---|---|
| `bitsandbytes` | 4bit / 8bit 线性层量化 | `quantization: 4bit` 或 `8bit` 时必需 |
| `accelerate` | 推理设备 / 量化加载辅助 | 装 SmolVLA 时必需 |

---

## 八、VLA 大模型链（仅 SmolVLA / pi0 等 VLM-based policy）

| 包 | 用途 | 备注 |
|---|---|---|
| `transformers` | SmolVLM2 等 VLM backbone | `policy_type=smolvla`/`pi0` 时必需（lerobot 某次安装会带，但版本敏感） |
| `num2words` | 数字转英文（VLM prompt 用） | 同上 |

---

## 九、测试

| 包 | 用途 | 是否可选 |
|---|---|---|
| `pytest` | 测试框架 | 仅 dev / CI |
| `pytest-cov` | 覆盖率 | 仅 dev / CI |

---

## 速查：完整一次性安装（conda env 内）

```bash
# 1. PyTorch + 视觉
conda install -c pytorch "torch>=2.7" "torchvision>=0.22"

# 2. PyBullet
conda install -c conda-forge pybullet

# 3. 工具/通用
conda install -c conda-forge pyyaml pillow numpy imageio requests accelerate bitsandbytes

# 4. LangChain 栈
pip install "langchain>=0.3" "langchain-core>=0.3" "langchain-openai>=0.3" langgraph pydantic

# 5. LeRobot（按 policy 选择）
pip install "lerobot[smolvla]"   # SmolVLA / pi0
# 或
pip install lerobot              # ACT / diffusion / vqbet

# 6. 测试
pip install "pytest>=8.0" "pytest-cov>=7.0"
```

> **速查仅是参考**。项目自带 `requirements.txt` / `requirements-stage2.txt` 才是当前权威清单，且文件头有详细的安装约定（哪些走 conda、哪些走 pip `--no-deps`、哪些不能 `pip install torch`）。

---

## 速查：分阶段最小集

**只想跑通 `mock` backend 跑通 pipeline**：

```bash
pip install numpy pyyaml pillow imageio requests langchain langchain-core langchain-openai langgraph pydantic gymnasium
```

**加 ACT 跑 lerobot 后端**：

```bash
# 上面 + conda 安装 pybullet / torch / torchvision，外加
pip install lerobot termcolor draccus opencv-python-headless
```

**加 SmolVLA + 4bit 量化**：

```bash
# 上面 + 
pip install "transformers>=5.4.0" accelerate bitsandbytes num2words
```
