import torch
import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model_loader import SimpleNN

def test_model_accuracy():
    """
    自动化测试：验证模型的准确率是否符合部署标准
    要求：在测试集上的准确率不得低于 0.85
    （注：为了演示快速运行，我们设置 0.85，因为上面只训练了 1 个 Epoch）
    """
    MIN_ACCURACY = 0.85 
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'model.pth')
    
    # 1. 确保模型文件存在
    assert os.path.exists(model_path), f"找不到模型文件：{model_path}，请先运行训练脚本。"
    
    # 2. 加载模型
    model = SimpleNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # 3. 读取仓库内固化的评估样本，避免 CI 运行时下载数据集
    subset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'ci_eval_subset.pt')
    assert os.path.exists(subset_path), f"找不到评估样本文件：{subset_path}"
    subset_payload = torch.load(subset_path, map_location=torch.device('cpu'))
    images = subset_payload["images"]
    labels = subset_payload["labels"]
    subset_size = len(labels)
    dataset = torch.utils.data.TensorDataset(images, labels)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=250, shuffle=False)

    correct = 0
    total = 0
    
    # 4. 执行推理并计算准确率
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"\nModel Accuracy on {subset_size} test images: {accuracy * 100:.2f}%")
    
    # 5. 断言准确率达标，否则阻止流水线继续部署
    assert accuracy >= MIN_ACCURACY, f"部署失败：模型准确率 {accuracy:.4f} 低于最低要求 {MIN_ACCURACY}"
