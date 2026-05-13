# FastAPI Classifier

A FastAPI backend for MNIST handwritten digit classification using PyTorch.

## Features

- ✅ FastAPI backend
- ✅ PyTorch MNIST classifier (93.8% accuracy)
- ✅ Docker deployment ready
- ✅ GitHub Actions CI/CD pipeline
- ✅ Automated testing with pytest

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run tests
python -m pytest tests/ -v
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Health status |
| `/predict` | POST | Predict digit from image |

## CI/CD Pipeline

GitHub Actions automatically runs:
1. Tests on push to main branch
2. Docker build and push if tests pass

## Model Accuracy

- Training accuracy: 93.8%
- Test accuracy: 93.8% (threshold for deployment)
