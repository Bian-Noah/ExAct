# ExAct 真实问题记录（2026-08-16）

> 起因：`task=把黄色海绵块夹起来放到指定位置`，cube 在 (0.3, 0, 0.05)，模型路径 `models/LeRobot-SO101-ACT-task1-unknown_bs32_s60000/pretrained_model`，21 次工具调用后任务未能完成。
>
> 这份文档记录**真实存在的问题**和**当前应该做的应对**。前提：**程序是死的**，任务是否真正完成由 LLM 输出与最终验证判断，不应交给程序做语义判定。

## 1. VLA 接收的信息缺失（缺少目标等任务级信息）

**现象**：`lerobot_vla.py:predict` 接收的只有 `(image, instruction, state)` 三样东西。instruction 里只有 LLM 写的自然语言，场景里 cube 坐标、drop zone 坐标这种任务级信息不会被显式拼接到输入里。

**影响**：模型理论上能听到"移到 (0.3, 0, 0.05)"，但 ACT 训练时大概率就没见过带坐标的指令，数字对它来说接近噪声。即便 instruction 写到了，**环境里那些已知坐标并没有作为契约字段喂进去**。

**应对方向**：
- 把 `object_info` / `target_zone` 序列化成文本追加进 instruction；
- 或者切到能在末端空间工作的模型（不在本迭代范围）。

## 2. VLA 接收的信息和数据集分布不一致

**现象**：模型名带 `task1-unknown`，没法认定训练分布。当前的 cube 颜色 / 位置 / 相机视角大概率与训练时不一致。

**影响**：模型进入 OOD（Out-of-Distribution）状态，输出无规律。第一次 action 调用末端从 (0.318, 0, 0.235) 直接飞到 (0.026, -0.055, 0.076)，跟 LLM 写的"移到正上方"完全无关——这是典型 OOD 行为。

**应对方向**：
- 短期：接受 OOD 现状，靠外层 reset / re-plan 收敛；
- 长期：在 SO101 + 当前 cube 场景下重新采集一份数据集并 fine-tune，或换用已经在该场景练过的模型。

## 3. 当前使用的 ACT 模型太小、能力上限受限

**现象**：默认 backend 是 ACT（轻量模仿学习 policy）。它能做的事有限，只能在它练过的"相似视觉 + 相似动作序列"里凑合跑。

**影响**：本场景和训练分布不完全一致 + ACT 的容量本身就小，模型偏离训练的输出幅度大、缺鲁棒性。

**应对方向**：
- 短期：在 orchestrator/agent 层用重试和 plan fallback 兜底；
- 长期：换 smolvla / pi0 这类更强的 VLA 模型（已被 `lerobot` 接口支持，需要更新 `configs/`）。

## 4. VLA 输出的关节目标在物理上可能无法实现

**现象**：ACT 吐 6 维关节角（5 臂 + 1 夹爪）。Bullet 接 `POSITION_CONTROL` 把这当作关节目标角度去插值执行。

**两种失效**：
1. **可达但不达意**：关节能转到、也确实转到了，但 FK（正运动学）算出的末端位置跟任务意图无关（5DOF SO101 够 3D 末端，IK 有多解，选错解是常见现象）。
2. **完全不可达**：目标关节角被关节限位（joint limit）clamp，电机无法全力执行 → 末端原地不动，表现为"位移 0.0000 m"。

**应对方向**：
- 在 executor 外层加 joint limit 检测，发现 clamp 时主动 reset 回 home pose；
- 在 prompt 提示 LLM 看到"卡住"反馈时主动改换方向。

## 5. 输出处理没做好——chunk_size=100 没被好好使用

**现象**：模型配置 `chunk_size=100, n_action_steps=100, temporal_ensemble_coeff=null`，`lerobot_vla.py:predict` 每次调用只取 chunk 的第一个动作，剩下 99 个当场丢弃，并且不维护跨调用上下文。

**影响**：模型理论上能一次"规划"100 步，但当前实现等同于把它当一次性回归使用，丢失时序连贯性、浪费推理预算。

**应对方向**：
- 在 `predict` 内部维护一个动作队列（长度 < chunk_size），消耗完再调模型；
- 启用 temporal ensemble，对重叠窗口里的多个预测做加权平均；
- 即便不给 LLM，改动对连贯性提升是直接的。

## 6. 程序端无法准确判断成功，只能做几何 proxy

**现象**：`check_done` 当前用 `dist(ee_pos, target_pos) < 0.01m` 判定 reached；`grasped` 是空壳；不存在 `placed` 分支。

**真实判断**：
- 程序是死的，**没有力觉夹爪力反馈、也没有物体位移感知**，任何"是否夹到"都是粗 proxy。
- 目标-位置重合判据对末端控制够用，对"抓 + 放"完全不够。
- 把 `dist < 0.01m` 这种判据当成功判定会骗 LLM，必须改。

**应对方向**：
- 把 `reached` 分支在没 `target_pos` 时的 fallback 删掉（不在没坐标时假装成功）；
- `max_steps=50` 改成更短（10–15 步就够），50 步只是让无效动作重复施加；
- 不强求程序判定真正的成功，那是死胡同；要"看见完成"靠的是带感知的 actuator 或视觉 cube 状态估计。

## 7. 执行器返回信息不全面

**现象**：executor 返回的 message 仅有"位移多少 / 距目标多少"，没有说明：
- 是否在关节限位上；
- 是否看起来已经卡住；
- 这一段是 OOD 行为还是正常动作；
- 末端是否在朝目标位移（速度方向）。

**影响**：LLM 收到 fallback 那条"位移 0.0000 m"会觉得成功，浪费后续工具调用。

**应对方向**：
- executor 返回信息扩成结构化 dict：`{moved, distance_to_target, joint_saturated:bool, velocity_estimate, status: "converging|stuck|diverging|ood"}`；
- 这样 LLM 拿到"stuck"信号后会改方向，不会傻傻重复同一指令。

## 8. LLM 收到的任务信息不完整（任务表述模糊）

**现象**：`configs/local.yaml:task.default_user_goal = "把黄色海绵块夹起来放到指定位置"`，但 `objects` 字段只有 cube，没有 `target_zone`。

**澄清的角色分工**：
- **不该由程序去硬规定"指定位置"在哪**——那会变成死规则；
- LLM 自己应该主动调 observe 把场景捋清楚，然后自己提出一个放置坐标，落进 instruction 里交给 executor。

**应对方向**：
- 在 Agent prompt 中加一段指引：任务模糊时必须先 observe 提取所有物体与场景结构，再把指令清晰化；
- observe 输出已经包含 cube pos，这层信息够 LLM 用，可以再扩一段"场景地形 / drop 候选区域"提示。

## 9. observe 工具发送图片的频率由调用方决定

**现象**：`observe_tool._run` 默认始终追加 image block。一轮 run 里 4–10 次 observe 会带 4–10 张几乎一样的图，浪费几千 token。

**应对方向**：
- 加一个 `include_image: bool` 参数，默认 `False`，LLM 主动开；
- 或者 observe 内部做 hash 去重，只在画面真变了或距上次发图 N 步后发图；
- 不破坏 LLM 当前用 API，但立刻能省 token。

## 10. 关节物理卡死

**现象**：实际执行中末端停在某个奇葩姿态附近，policy 给出的目标关节角被 joint limit clamp，关节不再移动，末端位移恒 0。这种现象轨迹里出现 10+ 次。

**应对方向**：
- executor 加 early stop：`连续 5 步 ee_pos 不动即终止`；
- 给 env 加 `home()` 接口：检测到卡死时调一下回到初始姿态，避免越卡越死；
- 让 LLM 拿到"卡死"信号后能选择 plan 重置。

---

## 总览：当前该做与不该做

**该做（短期，低成本，立刻可落地）**：
- 扩 executor 的返回信息（#7），加入"stuck / joint_saturated"信号；
- 把 `vla.max_steps` 从 50 降到 10–15；
- 删除 `check_done` 没 `target_pos` 时的 fallback；
- 加早期停止 + home reset（#10）；
- 让 observe 支持按需发图（#9）。

**不该做（被称作"程序判断成功"的事别投入精力）**：
- 不要去把 `grasped` / `placed` 写"真"判定——没有感知就是做不到；
- 不要把 `success` 字段当可信信号；
- 让 LLM 自己用结构化反馈（status 字段、cube 坐标、joint 状态）来判。

**中期**：
- 启用 ACT chunk 队列 + temporal ensemble（#5）；
- 给 LLM 系统提示加"任务模糊先观察-再清晰化"指引（#8）。

**长期**：
- 换/训一个真在本场景下练过的模型（#2，#3）。
