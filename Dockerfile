FROM python:3.10-slim

# 安装libgomp1库
RUN apt-get update && apt-get install -y libgomp1 && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# 清除缓存并强制重新安装依赖
RUN pip cache purge && pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]