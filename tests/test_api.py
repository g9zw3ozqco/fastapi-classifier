import os
import sys
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.model_loader import load_model

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
    """边缘情况测试：上传非图片文件"""
    files = {'file': ('test.txt', b'this is a plain text file', 'text/plain')}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 400
    assert "请上传图片格式的文件" in response.json()['detail']

def test_predict_valid_image():
    """正常调用测试：上传合法的 28x28 灰度图片"""
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
    assert 0 <= data["prediction"] <= 9

def test_predict_empty_file():
    """边缘情况测试：上传空文件"""
    files = {'file': ('empty.png', b'', 'image/png')}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 500

def test_predict_large_image():
    """边缘情况测试：上传过大的图片（非28x28尺寸）"""
    image = Image.new('L', (100, 100), color=128)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {'file': ('large.png', img_byte_arr, 'image/png')}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["status"] == "success"

def test_predict_color_image():
    """正常调用测试：上传彩色图片（应自动转换为灰度）"""
    image = Image.new('RGB', (28, 28), color=(255, 0, 0))
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {'file': ('color.png', img_byte_arr, 'image/png')}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["status"] == "success"

def test_predict_jpeg_format():
    """正常调用测试：上传 JPEG 格式图片"""
    image = Image.new('L', (28, 28), color=128)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["status"] == "success"

def test_predict_no_file():
    """边缘情况测试：未上传任何文件"""
    response = client.post("/predict")
    
    assert response.status_code == 422

def test_predict_invalid_image_data():
    """边缘情况测试：上传损坏的图片数据"""
    files = {'file': ('corrupt.png', b'invalid image data', 'image/png')}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 500

def test_api_endpoint_not_found():
    """测试：访问不存在的端点"""
    response = client.get("/nonexistent")
    assert response.status_code == 404

def test_predict_filename_special_chars():
    """测试：上传文件名包含特殊字符的图片"""
    image = Image.new('L', (28, 28), color=0)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {'file': ('test@#$%.png', img_byte_arr, 'image/png')}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == 'test@#$%.png'
    assert data["status"] == "success"
