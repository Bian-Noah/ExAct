# llm_vla 后端（占位）

**Iteration 启用时间**：Iteration 5

本目录是 LLMVLA 后端的预留位置，**当前迭代（iter1-pipeline-refactor-config）仅为占位**，
本目录里没有任何实现代码。

## 计划职责

`LLMVLA` 后端接受一张图片 + 自然语言指令，调用 LLM（大语言模型）让模型直接生成 7D 动作，
而不是依赖传统的 VLA（Vision-Language-Action）模型权重。

## 待实现内容（Iteration 5）

- `LLMVLA(BaseVLA)` 类，继承自 `executor.model.base.BaseVLA`
- `predict(image, instruction)` 调用 `create_llm(config.llm)` + 自定义 prompt 解析 7D 动作
- 错误处理：LLM 输出格式异常时返回默认动作或重试

## 后续清理

Iteration 5 完成本后端实现后，请删除本 README 并替换为实际 Python 模块。