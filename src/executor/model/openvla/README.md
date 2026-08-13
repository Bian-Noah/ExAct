# openvla 后端（占位）

**Iteration 启用时间**：Iteration 10

本目录是 OpenVLA 后端的预留位置，**当前迭代（iter1-pipeline-refactor-config）仅为占位**，
本目录里没有任何实现代码。

## 计划职责

`OpenVLA` 后端调用完整的 OpenVLA-7B 模型（VLA 领域 SOTA 开源模型），
依赖 PyTorch + Transformers + bitsandbytes 量化。

## 待实现内容（Iteration 10）

- `OpenVLA(BaseVLA)` 类，继承自 `executor.model.base.BaseVLA`
- 模型权重加载与 bf16 量化
- 推理入口（图像预处理 + tokenization + model.generate + 动作解码）

## 依赖

本后端需要 `requirements-stage2.txt` 中的 torch / transformers / bitsandbytes / accelerate，
**仅在阶段二（RTX 4060, CUDA）启用**，M4 Mac 阶段一不安装。

## 后续清理

Iteration 10 完成本后端实现后，请删除本 README 并替换为实际 Python 模块。