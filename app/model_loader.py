import torch
import torch.nn as nn
import torch.nn.functional as F
import io
from PIL import Image
import torchvision.transforms as transforms

# 简单的多层感知机（MLP），用于 MNIST 手写数字分类
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = None
# MNIST 数据集的预处理流程
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def load_model(model_path="model/model.pth"):
    """
    加载已训练的 PyTorch 模型
    """
    global model
    model = SimpleNN()
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"警告：无法加载模型 {model_path}。原因: {e}")

def predict_image(image_bytes: bytes) -> int:
    """
    接收图像字节流，预处理并进行分类预测
    """
    if model is None:
        raise RuntimeError("模型尚未加载！")
    
    # 将字节流转换为图像对象
    image = Image.open(io.BytesIO(image_bytes))
    
    # 预处理
    tensor = transform(image).unsqueeze(0)  # 增加 batch 维度: (1, 1, 28, 28)
    
    # 预测
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs.data, 1)
        
    return predicted.item()
