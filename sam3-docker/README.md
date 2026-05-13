# SAM 3 Docker 环境

Segment Anything Model 3 (SAM 3) 的 Docker 容器化部署方案。

## 🚀 快速开始

### 前置要求

- Docker
- NVIDIA GPU (推荐)
- NVIDIA Container Toolkit

### 构建和运行

```bash
# 使用 docker-compose（推荐）
docker-compose up --build

# 或者使用 docker build
docker build -t sam3 .
docker run --gpus all -it -p 8888:8888 -v $(pwd)/data:/app/data sam3
```

### 访问 Jupyter

打开浏览器访问：`http://localhost:8888`

## 📋 SAM 3 申请访问权限

SAM 3 模型需要从 HuggingFace 申请访问权限：

1. 访问：https://huggingface.co/facebook/sam3
2. 点击 "Request access"
3. 申请通过后，获取 Token：https://huggingface.co/settings/tokens
4. 设置环境变量：`HF_TOKEN=your_token_here`

## 🎯 先使用 SAM 2

如果没有 SAM 3 权限，可以先使用 SAM 2：

```python
from ultralytics import SAM

# SAM 2（不需要特殊权限）
model = SAM('sam2.pt')

# 运行分割
results = model('image.jpg')
results[0].save('result.jpg')
```

## 📁 目录结构

```
sam3-docker/
├── Dockerfile
├── requirements.txt
├── docker-compose.yml
├── example_sam2.py
├── data/           # 放置测试图片
├── models/         # 模型文件自动下载
├── notebooks/      # Jupyter notebooks
└── output/         # 输出结果
```

## 🔧 配置说明

### 环境变量

- `HF_TOKEN`: HuggingFace 访问令牌（用于 SAM 3）

### 端口映射

- `8888`: Jupyter Notebook
- `8000`: 可用于 API 服务

## 📚 SAM 3 主要功能

| 功能 | 说明 |
|------|------|
| 文本提示分割 | 使用文本描述分割对象 |
| 图像示例分割 | 使用参考图像分割 |
| 视频跟踪 | 跟踪视频中的对象 |
| 视觉提示 | 点、框、掩码提示 |

## 🔍 故障排除

### 模型下载失败

检查网络连接，或使用代理：
```bash
export https_proxy=http://your-proxy:port
```

### CUDA 错误

确保：
- 安装了 NVIDIA Container Toolkit
- 使用 `--gpus all` 参数启动容器
