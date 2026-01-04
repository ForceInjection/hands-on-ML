FROM jupyter/scipy-notebook:latest

USER root

# 安装 LightGBM 和 XGBoost 的系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgomp1 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER jovyan

# 安装 Python 库
RUN pip install --no-cache-dir --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ && \
    pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ \
    numba \
    lightgbm \
    xgboost \
    mlxtend
