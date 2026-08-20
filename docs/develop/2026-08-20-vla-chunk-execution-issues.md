# ExAct VLA chunk 执行链路问题记录（2026-08-20）

> 起因：在 `app.py` 跑 `task=把黄色海绵块夹起来`（cube 在 (0.3, 0, 0.05)，SO101 + smolVLA 通用微调模型 `LeRobot-SO101-SmolVLA-universal-all_bs128_s20000/pretrained_model`）时观察到：第一次 `pick yellow cube` 末端从 (0.318, 0, 0.235) 退到 (0.022, 0.051, 0.077)，之后每次 `action` 调用末端只位移 0.001-0.003m。任务失败归因为「自身指令问题, 执行器问题」。
>
> 本文记录**执行链路层面**的真实问题。这些问题与 VLA OOD 行为叠加，导致模型的方向错误根本无法被 LLM 观察到。前提：**程序是死的**，任务是否真正完成由 LLM 输出与最终验证判断，executor 不应在中间用阈值判定语义层面的「到位」。

## 1. BaseVLA.predict 接口契约把 chunking 模型的能力屏蔽成「单步」

**现象**：`src/executor/model/base.py:36` 强制所有 VLA 后端实现：

```python
@abc.abstractmethod
def predict(self, image, instruction, state=None) -> VLAOutput:
    """输入图片 + 指令，输出带 spec 的动作（单个）。"""
```

返回值 `VLAOutput` 也只承载单个动作值。但 lerobot 里所有支持 action chunking 的 VLA 模型（smolVLA / ACT / Pi0 / Pi0Fast / Diffusion 等）在内部都是**一次性输出 N 步连贯轨迹**：

- smolVLA `_get_action_chunk` → `shape (1, 50, 6)`（基于 `chunk_size=50`）
- ACT 同上 → `(1, 100, ...)`
- Diffusion 同上 → `(1, 8, ...)`
- Pi0 同上 → `(1, 50, ...)`**这层契约直接抹平了「模型输出多步轨迹」这件事**，外部代码看不到 chunk 概念。

**影响**：调用方只能拿到 1 个动作，不知道模型本来规划了 50 步连贯动作；deque 机制把 chunk 残骸藏在 `policy._queues` 里，调用链完全不感知。

## 2. max_steps 与 chunk_size 完全脱节

**现象**：

```yaml
# configs/local.yaml
vla:
  max_steps: 50          # executor 循环上限（写死在 config）
```

```json
// 模型 config.json
{
  "chunk_size": 50,      // 模型一次推理输出多少步
  "n_action_steps": 50   // deque 缓存多少步
}
```

两个数字来源完全独立，没有任何代码把它们关联起来。

**后果对照表**：

| VLA | chunk_size | vla.max_steps | 实际行为 |
|---|---|---|---|
| smolVLA（当前） | 50 | 50 | 碰巧对齐 |
| ACT | 100 | 50 | 永远跑不完一个 chunk → deque 残留 50 个没用上 |
| Diffusion | 8 | 50 | 跑 8 步后必须重新预测 5 次 |
| 改成 80 的 smolVLA | 80 | 50 | 永远跑不完 → deque 永远残留旧 chunk |

**影响**：executor 不知道当前 VLA 的 `chunk_size`，循环步数与模型意图完全无关。

## 3. check_done 兜底逻辑反了：单步 ≤0.01m 反而判定 done

**现象**：`src/executor/check_done.py:46-58`：

```python
# 没给 target_pos 时的兜底分支
dist = math.dist(obs_before["ee_pos"], obs_after["ee_pos"])
if dist <= REACHED_THRESHOLD:    # 0.01 m
    return (True, "ee_pos 前后位移 X m，小于阈值 0.01")
return (False, "ee_pos 前后位移 X m，大于阈值 0.01")
```

**反向语义**：本意应该是「动作基本没生效 → 失败/卡住」，但代码写成「单步位移小 → 算 done → 让循环 break」。

**对 smolVLA 这种关节增量控制的影响**：

smolVLA 单步输出的关节增量约 ±0.05 rad，对应末端位移通常 1~3 mm。**几乎每一步都满足 `dist ≤ 0.01m` → 几乎每一步都判 done**。executor 的 50 步循环在**第 1 步就 break**。

**影响**：每次 `action` 工具调用内部只跑了 1 步物理仿真，末端位移 0.002 m。3 次 action × 0.002 m = 0.006 m，对比任务需要的 0.27 m 远远不够。

## 4. done_criteria 写死 "reached"，无法区分动作语义

**现象**：`src/executor/__init__.py:run_action` 把 `done_criteria="reached"` 写死：

```python
done, reason = check_done(obs_before, obs_after, "reached", target_pos=target_pos)
```

而 `check_done` 只认真实现了 `reached` 分支；`grasped` / `placed` 等都是占位返回 `(False, "grasped 判断未实现，需感知支持")`。

**影响**：LLM 发 "pick yellow cube" / "grab the yellow cube" / "release" 等不同语义的动作时，executor 都套用「末端到没到目标位置」的判定逻辑。**没有「夹取」概念、也没有「夹爪开合度」判定**。

## 5. parse_target_pos 几乎永远返回 None，导致兜底分支总被触发

**现象**：`src/tools/action.py:18-58` 只识别两类格式：

```python
pattern1 = r'[xyz]\s*[=:]\s*([\-0-9.]+)'         # "x=0.5, y=0.0, z=0.4"
pattern2 = r'[\(（]([\-0-9.]+)\s*,\s*([\-0-9.]+)\s*,\s*([\-0-9.]+)[\)）]'  # "(0.5, 0, 0.4)"
```

LLM 写的英文动词指令（"pick yellow cube"、"move forward"、"grab"）**没有任何坐标字符串** → `target_pos=None` → check_done 永远走兜底分支。

**影响**：和第 3 条叠加，兜底分支被触发 → 单步判 done → 循环 1 步 break。

## 6. smolVLA 的 deque 跨 LLM action() 调用残留，导致旧 chunk 污染新调用

**现象**：lerobot `SmolVLAPolicy.select_action` 内部维护 `deque(maxlen=50)`：

```python
# lerobot/policies/smolvla/modeling_smolvla.py
def select_action(self, batch, ...):
    self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
    if self._check_get_actions_condition():    # deque 空才预测
        actions = self._get_action_chunk(batch, noise)  # 跑模型，输出 50 个
        self._queues[ACTION].extend(actions.transpose(0, 1)[:50])
    return self._queues[ACTION].popleft()      # 出队一个
```

判断是否重新调模型的**唯一条件是 `len(_queues[ACTION]) == 0`**。

**配合 executor 循环的实际行为**：

```
LLM action("pick yellow cube")  ← 第 1 次 action
  step 1: predict() → deque 空 → 跑模型，填 50 个 → 返回 chunk[0]
          env.step → 位移 0.3m（OOD）
          check_done: 0.3 > 0.01 → False，继续
  step 2: predict() → deque 不空 → 不跑模型 → 返回 chunk[1]
          env.step → 位移 0.002m
          check_done: 0.002 ≤ 0.01 → True → break
   ⚠️ deque 还剩 49 个 chunk[1..49]（基于 obs_1 预测的旧轨迹）

LLM action("move forward")  ← 第 2 次 action
  step 1: predict() → deque 还剩 49 个 → 不跑模型 → 返回旧 chunk[1]
          ⚠️ chunk[1] 是「假设 chunk[0] 已执行完」的下一步动作
          在当前真实状态下单独执行 chunk[1] 几乎无效
          env.step → 位移 0.002m
          check_done: 0.002 ≤ 0.01 → True → break
   ⚠️ deque 还剩 48 个 chunk[2..49]（还是旧的）
```

**形成的恶性循环**：

```
旧 chunk 残留 → 取到不连贯的中间动作 → 单步几乎无效 → check_done done → 队列还不清
 ↑ ↓
   ← 下次 action 又取到下一个不连贯的中间动作 ←
```

**这是「每次 0.001-0.003m」的精确机制**：不是 VLA 不想动，是取到了「假设前面动作都已做完的中间步」，单独执行近乎无效。

## 7. Executor 写死循环结构，不与 VLA/Robot 的插拔性匹配

**现象**：

| 组件 | 插拔性 | 工厂位置 |
|---|---|---|
| VLA | ✅ | `executor/model/factory.py:create_vla`（5 个 backend 注册） |
| Robot | ✅ | `env/robot/__init__.py:_ROBOT_REGISTRY`（panda / so101） |
| Adapter | ✅ | `utils/adapter/main.py:_ADAPTER_REGISTRY`（3 种空间组合） |
| Executor | ❌ | `executor/__init__.py:Executor.run_action` 是写死的循环 |

**Executor 的循环结构**（`src/executor/__init__.py:run_action`）：

```python
for step in range(self.max_steps):    # 硬编码
    obs_before = env.get_obs()
    vla_output = self.vla.predict(...)  # 写死每步调一次
    ...
    obs_after, _, _, _ = env.step(action)
    done, reason = check_done(...)     # 写死每步判一次
    if done:
        break
```

不管 VLA 是 chunking 模型还是单步模型、不管 chunk_size 是 8 还是 100、不管动作语义是 reach 还是 grasp——**所有 VLA / Robot 组合都按这个模式跑**。

**影响**：换 ACT 时这个循环就跑不完了；换 diffusion 时这个循环浪费大量预测；想给 grasp 类动作加「夹爪力反馈」也无从插入。

## 8. 模型 OOD 行为被 check_done 截断隐藏，LLM 看不到方向错误

**现象**：smolVLA 对 "pick yellow cube" 输出错误方向的关节增量（OOD 行为），第 1 步就把臂推到 (0.022, 0.051, 0.077)，离 cube 0.27 m 远。

按设计意图，**应该让模型把这条轨迹走完**，LLM 看到画面后发现「方向错了」，主动改指令。

但 check_done 的兜底逻辑让循环在第 1 步就 break，LLM 收到 "位移 0.002 m，小于阈值 0.01" 这种**误导性消息**——它根本看不到「臂已经偏离 cube 0.3 m」的事实。后续重发的 "move forward"、"grab the yellow cube" 等指令，物理上还在远离 cube 的方向上做微调。

**影响**：模型的方向错误被死程序的阈值判断隐藏，LLM 拿不到完整轨迹的失败信号，只能在错误状态上反复微调。

## 9. 死规则判定（关键词 + 物体位置）的局限

**现象**：如果想用「关键词查表 + 物体位置查表」等方式做意图校验：

```python
# 设想的方向校验
expected_dir = direction_from_keywords(instruction) or direction_to_object(instruction, obs)
actual_dir = ee_delta / ||ee_delta||
if cosine(expected_dir, actual_dir) < 0.5: warn(...)
```

**局限**：

| 情况 | expected_dir 能算出吗 |
|---|---|
| "move forward" / "up" / "down" 等含方向词 | ✅ 关键词查表 |
| "pick the yellow cube" 含物体名 | ✅ 物体位置查表 |
| "do it" / "help" / "fix this" | ❌ 无关键词无物体 → 返回 None → 跳过 |
| "look up"（抬头 vs 向上看）歧义 | ❌ 简单约定无法消歧 |
| 物体没出现在 obs 里（识别失败） | ❌ 物体查表查不到 |
| "open the gripper" / "release" 等末端不动指令 | ⚠️ 需要「不动末端」动词白名单配合 |

**影响**：死规则校验只能覆盖模板化指令，对 LLM 的灵活表达容易误报或漏报。**死程序无法理解语义，这种约束会把模型能力锁死在固定模式里**，不是治本方案。

## 10. 接口契约应该改成什么

**当前**：

```python
class BaseVLA(abc.ABC):
    @abc.abstractmethod
    def predict(self, image, instruction, state=None) -> VLAOutput:
        """返回单个动作值。"""
```

**应该改成**（暴露 chunk 概念）：

```python
class VLAOutput:
    values: np.ndarray   # shape (N, action_dim) — VLA 规划的整条轨迹
    spec: ActionSpec

class BaseVLA(abc.ABC):
    @abc.abstractmethod
    def predict(self, image, instruction, state=None) -> VLAOutput:
        """返回 N 步完整轨迹。N 由 VLA 自己决定（smolVLA=50, ACT=100, ...）。"""
```

```python
class Executor:
    def run_action(self, env, instruction, ...) -> ExecResult:
        vla_output = self.vla.predict(image, instruction, state)  # 一次拿全部
        actions = vla_output.values  # shape (N, action_dim)
        
        for action in actions:        # 严格按 VLA 输出的步数执行
            env.step(action)
            # ← 这里不再调 check_done，不干预 VLA 意图
        
        # 全部执行完后再给 LLM 一个反馈
        return ExecResult(
            success=???,
            steps=len(actions),
            message=f"执行 VLA 规划的全部 {len(actions)} 步"
        )
```

**原则**：

- VLA 的 chunk = 它对「接下来要做的事」的承诺，executor 应该照单全收
- executor 不在中间做语义判断，不在中间截断
- LLM（外层）观察结果后决定下一步指令
- 模型的方向错误必须让模型完整执行完才能被识别、反馈给 LLM

## 11. 总结

| 层次 | 现状 | 问题 |
|---|---|---|
| 接口契约 | predict() 返回 1 个动作 | 屏蔽了 chunk 概念 |
| 执行循环 | `for step in range(max_steps)` | 与 VLA chunk_size 完全脱节 |
| 中间干预 | check_done 每步判断 | 隐藏模型错误，截断轨迹 |
| 信息流 | deque 跨调用残留 | 旧 chunk 污染新调用 |
| 反馈延迟 | LLM 第 1 步拿到"完成"消息 | 看不到完整轨迹结果 |

**根因：接口设计与 chunking 模型不匹配**。所有 chunking 模型（ACT / Diffusion / smolVLA / Pi0）设计都是「输出 N 步连贯轨迹」，但 BaseVLA.predict 把这个契约抹平成「单步」，executor 用 max_steps 假装在循环，check_done 在中间擅自叫停——三个环节都在和 VLA 的实际工作方式打架。

## 12. 应对方向汇总

1. **改 BaseVLA.predict 契约**：返回 `shape (N, action_dim)` 的整条轨迹，N 由 VLA 自己决定
2. **Executor 改成「按 VLA 输出的轨迹长度执行」**：去掉硬编码 `max_steps` 循环，去掉中间 `check_done`
3. **去掉 check_done 兜底分支**：要么 `target_pos` 必填，要么没 `target_pos` 直接跳过成功判定（让 LLM 看最后状态）
4. **executor 返回结构化反馈**：包含执行步数、最终 ee_pos、是否到达目标，让 LLM 自己判断
5. **接口契约暴露 chunk_size**：让 executor / 配置层知道当前 VLA 的轨迹长度，避免 max_steps 与 chunk_size 脱节
6. **VLA OOD 不是 executor 的问题**：接受模型方向可能错，让 LLM 通过完整执行结果识别并纠正，而不是用阈值把错误隐藏掉
