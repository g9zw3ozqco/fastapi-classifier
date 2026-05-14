---
name: train-model-with-ai
description: "使用场景：提出或实现新的训练流程或模型更新。"
inputs:
  - name: goal
    description: "希望改进或改变什么？"
    default: "在不增加推理延迟的前提下提升准确率。"
  - name: constraints
    description: "约束条件（时间、硬件、CI 限制）？"
    default: "保持 CI 快速；测试中避免下载完整数据集。"
---

你正在为本项目更新 MNIST 模型训练流程。

背景：
- 参考代码位于 ai-reference/autoresearch，借鉴其实验结构和记录方式。
- 训练代码在 model/train.py；模型类在 app/model_loader.py。
- CI 准确率门槛在 tests/test_model.py，使用 data/ci_eval_subset.pt。

任务：
1) 先阅读当前训练与模型代码。
2) 提出与 {{goal}} 一致的“模型结构/算法”改进方案（训练流程保持不变）。
3) 在 app/model_loader.py 中更新模型结构；必要时同步到 model/train.py 的模型创建逻辑。
4) 推理行为需保持与 app/model_loader.py 兼容。
5) 若准确率基线变化，更新 tests/test_model.py 并说明新阈值。

输出：
- 列出修改的文件及原因。
- 给出新增超参数与预期准确率影响。
