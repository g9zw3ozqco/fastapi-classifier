from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import os
import sys

# 确保可以正确导入同级目录下的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model_loader import load_model, predict_image

# 寿命周期管理器，启动时加载模型
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 假设应用运行在根目录下
    model_path = os.path.join("model", "model.pth")
    if os.path.exists(model_path):
        load_model(model_path)
    else:
        print(f"Warning: {model_path} 不存在，如果是在测试或训练阶段可以忽略。")
    yield
    # 可以在此处执行清理操作（如释放资源）

app = FastAPI(lifespan=lifespan, title="FastAPI Image Classifier", version="1.0.0")

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI Image Classifier"}

@app.get("/health")
def health_check():
    """用于 Kubernetes 或 Docker 的健康检查接口"""
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """接收图片上传，并返回模型预测结果"""
    # 验证文件类型（边缘情况：上传非图片）
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片格式的文件 (image/*)")
    
    try:
        # 读取图片字节流
        contents = await file.read()
        # 预测
        prediction = predict_image(contents)
        return {
            "filename": file.filename, 
            "prediction": prediction,
            "status": "success"
        }
    except Exception as e:
        # 捕获推理过程中的异常并返回 500
        raise HTTPException(status_code=500, detail=f"预测过程中发生错误: {str(e)}")
