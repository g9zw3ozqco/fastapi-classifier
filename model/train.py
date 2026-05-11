import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import sys

# 将上一级目录加入路径，以便导入 app.model_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model_loader import SimpleNN

def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("正在下载/加载 MNIST 数据集...")
    # MNIST 自动划分为 train 和 test，此处加载 train 集合进行训练
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("开始训练...")
    # 这里为了演示和 CI 快速通过，仅训练 1 个 Epoch。实际业务中可设置为 10-20
    epochs = 1  
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 200 == 199:
                print(f'[Epoch {epoch + 1}, Batch {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    # 保存模型
    os.makedirs('./model', exist_ok=True)
    torch.save(model.state_dict(), './model/model.pth')
    print("训练完成，模型已保存至 ./model/model.pth")

if __name__ == '__main__':
    train()
