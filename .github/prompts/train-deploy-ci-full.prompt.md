---
name: train-deploy-ci-full
description: "使用场景：覆盖训练 + 部署 + CI 的全流程优化或变更。"
inputs:
  - name: goal
    description: "目标是什么？"
    default: "提升模型效果并确保 CI/CD 可稳定通过。"
  - name: constraints
    description: "约束条件（时间、硬件、CI 限制、部署环境）？"
    default: "CI 需在 5-10 分钟内完成；部署使用 Docker。"
  - name: scope
    description: "本次调整范围（训练/服务/CI/部署）？"
    default: "训练 + CI + Docker 部署"
---

你负责本项目“训练 + 部署 + CI 全流程”的改造或优化。

背景：
- 参考代码：ai-reference/autoresearch（借鉴实验结构和记录方式）
- 训练代码：model/train.py
- 模型加载：app/model_loader.py
- FastAPI 服务：app/main.py
- CI 准确率门槛：tests/test_model.py（使用 data/ci_eval_subset.pt）
- Docker：Dockerfile / deploy.sh
- Gitee 流水线：.gitee/go.yml
- GitHub Actions：.github/workflows/ci-cd.yml

任务：
1) 先阅读训练、服务、CI、部署相关代码，确认当前流程。
2) 按 {{goal}} 和 {{constraints}} 提出可执行的改进方案，覆盖 {{scope}}。
3) 进行必要的代码修改：
   - 训练/评估逻辑可改，但必须保持模型产物兼容 app/model_loader.py。
   - CI 测试要快且稳定，不下载完整数据集。
   - Docker 构建与部署脚本要可执行。
4) 若准确率阈值变化，更新 tests/test_model.py 并说明原因。
5) 更新或新增文档说明（README.md）以便团队理解流程。

输出要求：
- 列出修改的文件及原因。
- 给出关键参数与预期效果。
- 若涉及 CI/CD，说明如何验证与回滚。
