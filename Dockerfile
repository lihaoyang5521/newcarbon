FROM python:3.10-slim

# 安装libgomp1库，LightGBM需要
RUN apt-get update && apt-get install -y libgomp1 && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# 强制安装确切的依赖版本
RUN pip cache purge && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir scikit-learn==1.6.1

COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]