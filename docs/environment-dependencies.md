# ExAct 当前 Conda 环境依赖清单（robot env）

> 状态快照时间：2026-08-16
>
> 生成方式：`pip list` + `pip check` + 读取 `requirements*.txt` 推导"谁是谁的依赖"
>
> 本文档**只描述现状、不动任何安装动作**。

## 0. 环境概览

| 项 | 值 |
|---|---|
| Python | 3.12.13 |
| pip | 26.1.2 |
| 平台 | Linux-6.18.33.2-microsoft-standard-WSL2-x86_64 |
| Conda env 名 | robot |
| 已装包总数 | 106 |
| torch 版本 | 2.5.1（CUDA 12.4 编译，**运行时 GPU 不可用**） |
| CUDA available | False（WSL2 GPU 直通被 OS 屏蔽） |
| 网络出站 | 受限（pypi.org / huggingface.co 均无法直连） |

---

## 1. 顶层依赖（用户主动安装 / 在 requirements 中登记）

> 顶层 = "没有任何其它包把它写进 Requires-Dist" 或者 "在项目 requirements*.txt 中显式声明"。
> 顺序按分组列。

### 1.1 `requirements.txt` 声明（阶段一，M4 Mac）

| 包 | 版本约束 | 当前安装版本 | 安装渠道 |
|---|---|---|---|
| `pyyaml` | `>=6.0` | 6.0.3 | pip |
| `numpy` | `>=1.24`（阶段一）/ 2.0.1+（阶段二 lerobot 要求 `<2.3.0`） | 2.0.1 | pip |
| `Pillow` | `>=10.0` | 12.3.0 | pip |
| `gymnasium` | `>=0.29` | 1.3.0 | pip |
| `langchain` | `>=0.3` | 1.3.15 | pip |
| `langchain-core` | `>=0.3` | 1.5.4 | pip |
| `langchain-openai` | `>=0.3` | 1.5.0 | pip |
| `requests` | `>=2.31` | 2.34.2 | pip |
| `pytest` | `>=8.0` | 9.1.1 | pip |
| `pytest-cov` | `>=7.0` | 7.1.0 | pip |
| `imageio` | `>=2.30` | 2.37.4（PyPI 名 `ImageIO`） | pip |

### 1.2 `requirements-stage2.txt` 声明（阶段二，RTX 4060 + CUDA）

| 包 | 当前安装版本 | 安装方式约定（来自文件头注释） |
|---|---|---|
| `torch` | 2.5.1 | conda（pytorch channel）+ cu12.4，**pip 严格禁止重装** |
| `torchvision` | 0.20.1 | conda 与 torch 配套 |
| `lerobot` | 0.6.1 | pip `--no-deps`（防 pip 抢 torch） |
| `pybullet` | 3.2.5 | conda-forge |
| `draccus` | 0.11.6 | pip 或 conda-forge |
| `opencv-python-headless` | 4.13.0.92 | pip 或 conda-forge |
| `termcolor` | 3.3.0 | pip 或 conda-forge |
| `accelerate` | 1.14.0 | conda-forge（供 VLA 量化用） |
| `bitsandbytes` | （未装） | 阶段二注释未启用 |
| `tokenizers` | （未装） | transformers 的硬依赖，但本 env 未装 |

### 1.3 项目运行实际需要、但未登记的顶层包

> 这一类要么是 `lerobot[smolvla]` extra 装上来的，要么是历史安装残留，不在 requirements*.txt 中。

| 包 | 当前安装版本 | 来源 | 说明 |
|---|---|---|---|
| `transformers` | 5.5.4 | `pip install "lerobot[smolvla]"` 时被装上（最近一次） | 实际**未被任何运行时路径调用**（ACT 不需要它；SmolVLA 还没跑通） |
| `num2words` | 0.5.14 | 同上 | SmolVLA extra 的依赖 |
| `huggingface_hub` | 1.27.0 | 自动 | lerobot 自身的依赖，但未被顶层包显式声明 |
| `safetensors` | 0.8.0 | 自动 | 同上 |
| `draccus` | 0.11.6 | 已登记（阶段二注释里提到） | — |
| `diffusers` | 0.39.0 | 历史残留？ | 与本项目功能无关，疑似先前的环境遗留 |

### 1.4 历史可见的顶层包（pip 视角无 Requires-Dist，疑似独立安装/未受管）

| 包 | 版本 | 备注 |
|---|---|---|
| `triton` | 3.1.0 | torch 自带的子包，独立分发版本号 |
| `mkl_fft` / `mkl_random` / `mkl-service` | 1.3.11 / 1.2.8 / 2.4.0 | Intel MKL 加速库（numpy 的优化后端），随 conda numpy 带进来 |
| `pydantic_core` | 2.46.4 | pydantic 的 native 部分，被 langchain 拉来 |
| `langchain` | 1.3.15 | 已在 1.1 节登记，但分析算法识别它无上游需要它 → 双重身份 |
| `backports.zstd` | 1.6.0 | 不明来源，可能随 zstandard 引入 |
| `mypy_extensions` | 1.1.0 | 早期 typing 兼容，多数项目用不到 |
| `uuid_utils` | 0.17.0 | uuid 加速实现，langsmith 带入 |
| `pybullet` | 3.2.5 | 已在 1.2 节登记 |
| `charset-normalizer` | 3.5.0 | requests 的依赖，独立分发版本号 |

---

## 2. 底层依赖（pip 自动拉取的传递依赖）

> 这一类 = "是被别人 requires 进来的"。按被哪个顶层包需要分组。

### 2.1 `lerobot 0.6.1` 的传递依赖

lerobot 自身在 `Requires-Dist` 声明的直接依赖有：  
`torch`, `torchvision`, `numpy`, `opencv-python-headless`, `Pillow`, `einops`, `draccus`, `huggingface-hub`, `requests`, `gymnasium`, `safetensors`, `packaging`, `termcolor`, `tqdm`, `cmake`, `setuptools`，外加可选 extra：`datasets`/`pandas`/`pyarrow`/`torchcodec`/`jsonlines`/`transformers`/`accelerate`/`num2words`/`wandb` 等。

实际被拉进来（且仍存在）的：

| 依赖 | 用途 |
|---|---|
| `draccus 0.11.6` | 配置系统 |
| `einops 0.8.2` | tensor 操作 |
| `gymnasium 1.3.0` | env API 兼容层 |
| `huggingface_hub 1.27.0` | 模型/数据集下载 |
| `safetensors 0.8.0` | 权重格式 |
| `mergedeep 1.3.4` | draccus 的依赖 |
| `typing-inspect 0.9.0` | draccus 的依赖 |
| `toml 0.10.2` | draccus 的依赖 |
| `Farama-Notifications 0.0.6` | gymnasium 的依赖 |
| `cloudpickle 3.1.2` | gymnasium 的依赖 |
| `networkx 3.6.1` | torch 自带（被算进 lerobot 依赖图） |
| `Jinja2 3.1.6` | torch 自带 |
| `sympy 1.14.0` | torch 自带 |
| `mpmath 1.3.0` | sympy 的依赖 |
| `markupsafe 3.0.3` | Jinja2 的依赖 |
| `av 18.0.0` | lerobot 数据加载（PyAV），自动带入 |
| `transformers 5.5.4` | `lerobot[smolvla]` extra 装上 |
| `num2words 0.5.14` | `lerobot[smolvla]` extra 装上 |
| `accelerate 1.14.0` | VLA 量化依赖 |

### 2.2 `langchain` 系列传递依赖

`langchain 1.3.15` 顶层 + `langchain-core 1.5.4`、`langchain-openai 1.5.0` 直接：

| 顶层 | 直接 requires |
|---|---|
| `langchain` | `langchain-core`, `langgraph`, `pydantic`, `langchain-openai` |

间接（被上面再拉进来的）：

```
distro 1.9.0
jsonpatch 1.33 / jsonpointer 3.1.1   ← langgraph-checkpoint 带入
langchain-protocol 0.0.18
langgraph-checkpoint 4.2.0
langgraph-prebuilt 1.1.0
langgraph-sdk 0.4.2
langsmith 0.10.18
orjson 3.11.9
ormsgpack 1.12.2
regex 2026.7.19
requests-toolbelt 1.0.0
sniffio 1.3.1
tenacity 9.1.4
tiktoken 0.13.0
websockets 15.0.1
xxhash 4.0.0
```

`langchain-openai` 拉的：

```
openai 3.0.0
jiter 0.16.0
tqdm 4.70.0
truststore 0.10.4
httpx 0.28.1 / httpx2 2.10.0
httpcore 1.0.9 / httpcore2 2.10.0
h11 0.16.0 / h2 4.4.1 / hpack 4.2.0 / hyperframe 6.1.0
anyio 4.14.2 / idna 3.18 / PySocks 1.7.1 / socksio 1.0.0
```

### 2.3 `langchain-core 1.5.4` 拉的

```
pydantic 2.13.4
pydantic_core 2.46.4
annotated-types 0.8.0
typing-inspection 0.4.4
typing_extensions 4.16.0
uuid_utils 0.17.0
```

### 2.4 `huggingface_hub 1.27.0` 拉的

```
hf-xet 1.5.2
fsspec 2026.7.0
filelock 3.29.4
click 8.4.2
```

### 2.5 `pytest 9.1.1` 拉的

```
pluggy 1.6.0
iniconfig 2.3.0
exceptiongroup 1.3.1
coverage 7.15.4     ← 由 pytest-cov 带入
Pygments 2.20.0     ← 仅在覆盖率报告渲染时用
psutil 7.2.2        ← 某些插件用
```

### 2.6 `requests 2.34.2` 拉的

```
charset-normalizer 3.5.0
urllib3 2.7.0
certifi 2026.7.22
```

### 2.7 `pybullet 3.2.5` 拉的

无（独立 wheel）。

### 2.8 `transformers 5.5.4` 拉的（但 env 中缺一部分）

transformers 5.5.4 自身 Requires-Dist 中要求：

| 依赖 | 是否已装 |
|---|---|
| `huggingface_hub` | ✓ 1.27.0 |
| `numpy` | ✓ 2.0.1 |
| `packaging` | ✓（但 26.2 略高于 lerobot 上限） |
| `pyyaml` | ✓ 6.0.3 |
| `regex` | ✓ |
| `requests` | ✓ |
| `safetensors` | ✓ |
| `tokenizers` | ✗ **未装**（pip check 报告 "required but not installed"） |
| `typer` | ✗ **未装**（同上） |
| `filelock` | ✓ |
| `importlib_metadata` | ✓ 9.0.0 |
| `pillow` | ✓ |

### 2.9 `num2words 0.5.14` 拉的

| 依赖 | 是否已装 |
|---|---|
| `docopt` | ✗ **未装**（pip check 报告） |

### 2.10 其它系统性依赖

| 依赖 | 来源 |
|---|---|
| `pip 26.1.2` | 自带 |
| `setuptools 83.0.0` | 自带（lerobot 想要 <82，但 pip 自动解到 83） |
| `wheel 0.47.0` | 自带 |
| `importlib_metadata 9.0.0` | 自带（transformers 还需它） |
| `zipp 4.1.0` | importlib_metadata 依赖 |
| `zstandard 0.25.0` | zstd 系列 |
| `gmpy2 2.3.1` | 某些数值运算 |
| `Backports.zstd 1.6.0` | — |
| `Brotli 1.2.0` | zstandard 依赖 |
| `certifi 2026.7.22` | requests 依赖 |

---

## 3. 已知约束冲突（`pip check` 输出）

```
lerobot 0.6.1 requires cmake, which is not installed.
num2words 0.5.14 requires docopt, which is not installed.
transformers 5.5.4 requires tokenizers, which is not installed.
transformers 5.5.4 requires typer, which is not installed.
lerobot 0.6.1 has requirement packaging<26.0,>=24.2, but you have packaging 26.2.
lerobot 0.6.1 has requirement setuptools<82.0.0,>=71.0.0, but you have setuptools 83.0.0.
lerobot 0.6.1 has requirement torch<2.12.0,>=2.7, but you have torch 2.5.1.
lerobot 0.6.1 has requirement torchvision<0.27.0,>=0.22.0, but you have torchvision 0.20.1.
torch 2.5.1 requires sympy==1.13.1; python_version >= "3.9", but you have sympy 1.14.0.
```

分门别类：

| 类型 | 触发包 | 期望 | 实际 | 严重度 |
|---|---|---|---|---|
| 版本过低（torch/torchvision） | lerobot | torch ≥2.7、torchvision ≥0.22.0 | torch 2.5.1、torchvision 0.20.1 | 低（ACT 跑通；SmolVLA 可能踩到新 API） |
| 版本过高（packaging/setuptools/sympy） | lerobot / torch | 上限限制 | packaging 26.2 / setuptools 83.0.0 / sympy 1.14.0 | 低（语义兼容范围内） |
| 直接缺失依赖 | lerobot | cmake | 无 | 中（lerobot 内部某些构建场景用到） |
| 直接缺失依赖 | num2words | docopt | 无 | 低（运行时大概率不触发） |
| 直接缺失依赖 | transformers | tokenizers, typer | 无 | **高**（import AutoTokenizer、AutoModel 时会 ImportError） |

---

## 4. 顶层 vs 底层 一眼总览

| 分组 | 顶层（用户/项目声明） | 底层（自动拉取） |
|---|---|---|
| 仿真/物理 | `pybullet`, `torch`, `torchvision` | triton, mkl_*, networkx, sympy, mpmath, Jinja2, markupsafe, av |
| 数据/格式 | `numpy`, `Pillow`, `imageio` | — |
| LLM 调用栈 | `langchain`, `langchain-core`, `langchain-openai`, `requests` | openai, jiter, tiktoken, httpx(httpcore), orjson, ormsgpack, regex, tenacity, langsmith, langgraph*, langchain-protocol, jsonpatch/jsonpointer, websockets, xxhash, distro, requests-toolbelt, truststore, anyio, idna, PySocks, socksio, h11/h2/hpack/hyperframe, sniffio |
| 测试 | `pytest`, `pytest-cov` | pluggy, iniconfig, exceptiongroup, coverage, Pygments, psutil |
| VLA/lerobot | `lerobot`, `transformers`, `num2words`, `accelerate`, `draccus`, `opencv-python-headless`, `termcolor` | einops, gymnasium, Farama-Notifications, cloudpickle, huggingface_hub, safetensors, mergedeep, typing-inspect, toml |
| 通用工具 | `huggingface_hub`, `safetensors`, `pyyaml`, `setuptools`, `pip`, `wheel`, `diffusers`（不明）, `pydantic`（via langchain）, `packaging`（自升） | filelock, fsspec, hf-xet, click, charset-normalizer, urllib3, certifi, importlib_metadata, zipp, zstandard, Brotli, gmpy2 |

---

## 5. 文档作者注

- 本文档**只描述当前 env 快照**，不修复任何版本冲突。
- `transformers 5.5.4`、`num2words 0.5.14` 这两个包是**最近一次 `pip install "lerobot[smolvla]"`** 部分成功留下的（pip 收集了 torch/CUDA 13 的 metadata 但未实际装上，可能 Ctrl+C 在 installs 阶段之前，或 pip 的 dry-run 行为）。
- **网络受限**（pypi/hf 均不可直连），任何补包（cmake/docopt/tokenizers/typer）都需离线 wheel。
- **GPU 不可用**（WSL2 直通问题），任何 VLA 推理实际都在 CPU。
