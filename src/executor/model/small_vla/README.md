# small_vla 后端（占位）

**Iteration 启用时间**：Iteration 9

本目录是 SmallVLA 后端的预留位置，**当前迭代（iter1-pipeline-refactor-config）仅为占位**，
本目录里没有任何实现代码。

## 计划职责

`SmallVLA` 后端是一个轻量级本地 VLA 模型（如 OpenVLA-7B 的蒸馏版本 / 简化版），
相比完整 OpenVLA 在 M4 Mac 这种无 GPU 环境下也能跑得动。

## 待实现内容（Iteration 9）

- `SmallVLA(BaseVLA)` 类，继承自 `executor.model.base.BaseVLA`
- 模型权重路径解析与加载
- 推理性能优化（量化、批处理）

## 后续清理

Iteration 9 完成本后端实现后，请删除本 README 并替换为实际 Python 模块。