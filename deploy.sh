#!/bin/bash
# 自动部署脚本：拉取最新镜像并重启服务
# 需在服务器上配置DOCKERHUB_USERNAME、DOCKERHUB_TOKEN等环境变量

set -e

IMAGE_NAME="fastapi-classifier"
DOCKERHUB_USER=${DOCKERHUB_USERNAME:-your_dockerhub_username}

# 登录DockerHub（如已登录可省略）
echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USER" --password-stdin

# 拉取最新镜像
docker pull "$DOCKERHUB_USER/$IMAGE_NAME:latest"

# 停止并删除旧容器（如存在）
docker stop $IMAGE_NAME || true
docker rm $IMAGE_NAME || true

# 启动新容器
docker run -d --name $IMAGE_NAME -p 8000:8000 \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/data:/app/data \
  $DOCKERHUB_USER/$IMAGE_NAME:latest

echo "部署完成，服务已启动在8000端口。"
