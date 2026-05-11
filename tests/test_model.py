import torch
import torchvision
import torchvision.transforms as transforms
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

    # 3. 准备测试数据 (MNIST Test Set)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 只下载测试集进行验证。CI 中抽样一部分数据，避免云端执行过慢。
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    subset_size = 2000
    subset = torch.utils.data.Subset(testset, range(subset_size))
    testloader = torch.utils.data.DataLoader(subset, batch_size=500, shuffle=False)

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
