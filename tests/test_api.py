import os
import sys
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 在测试中，我们需要在导入 main 前确保模型加载逻辑能够找到 mock 的模型或真实模型
# 因为是在集成测试，直接导入 app
from app.main import app
from app.model_loader import load_model

# 初始化 TestClient，它会触发 lifespan 事件（但有时不会，所以我们手动 load_model 保底）
client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_model():
    """测试前置条件：确保模型被加载。"""
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'model.pth')
    if os.path.exists(model_path):
        load_model(model_path)
    yield

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to FastAPI Image Classifier"}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_invalid_file_type():
    """
    边缘情况测试：上传非图片文件
    """
    files = {'file': ('test.txt', b'this is a plain text file', 'text/plain')}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 400
    assert "请上传图片格式的文件" in response.json()['detail']

def test_predict_valid_image():
    """
    正常调用测试：上传合法的 28x28 灰度图片，并断言返回了预测结果
    """
    # 在内存中构造一张 28x28 的全黑图片（模拟手写图片）
    image = Image.new('L', (28, 28), color=0)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {'file': ('dummy.png', img_byte_arr, 'image/png')}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "status" in data
    assert data["status"] == "success"
    # 因为输出是 0-9 之一
    assert 0 <= data["prediction"] <= 9
