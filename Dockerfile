FROM python:3.12-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY config.py agents.py pipeline.py app.py ./

# 暴露端口
EXPOSE 8000

# 启动 (API Key 通过环境变量注入)
CMD ["python", "app.py"]
