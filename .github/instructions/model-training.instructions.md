---
description: "使用场景：编辑 MNIST 分类器的训练或评估代码。"
applyTo: "model/**/*.py"
---

编辑训练或评估代码时请遵循：
- 实验结构和进度记录参考 ai-reference/autoresearch。
- 训练需可复现（设置随机种子），CI 中避免长时间训练。
- 模型产物保存到 model/model.pth，并保持与 app/model_loader.py 兼容。
- 不要修改 CI 评估数据集格式（data/ci_eval_subset.pt）。
- 如果调整准确率基线，更新 tests/test_model.py 并注明新阈值。
