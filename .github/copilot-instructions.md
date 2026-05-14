# fastapi-classifier 的 Copilot 指令

你正在开发一个 FastAPI + PyTorch 的 MNIST 分类项目。

项目约束与建议：
- 训练尽量小且可复现；CI 要快，测试时不要下载完整数据集。
- 准确率评估使用 data/ci_eval_subset.pt 这份固定子集。
- 模型保存到 model/model.pth，加载逻辑需与 app/model_loader.py 兼容。
- 提出新模型或训练流程时，参考 ai-reference/autoresearch 的实验结构和记录方式。
- 如果调整了准确率阈值，需要更新 tests/test_model.py，并说明原因。
