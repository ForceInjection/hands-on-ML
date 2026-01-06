FROM jupyter/scipy-notebook:python-3.11

USER root

# 配置阿里云镜像源并安装 LightGBM 和 XGBoost 的系统依赖
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/ports.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgomp1 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER jovyan

# 安装 Python 库
# 改用 pip 安装以避免 mamba 求解依赖时的内存溢出 (OOM)
# 使用阿里云 PyPI 镜像源
# 强制重装核心库以确保与 NumPy 2.x 的二进制兼容性
RUN pip install --no-cache-dir --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ && \
    pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ \
    --force-reinstall \
    'jupyterlab' \
    'numpy' \
    'pandas' \
    'scipy' \
    'scikit-learn' \
    'matplotlib' \
    'numexpr' \
    'bottleneck' \
    'pyarrow' \
    'numba' \
    'lightgbm' \
    'xgboost' \
    'mlxtend' \
    'nltk' \
    'shap' \
    'gensim'

# 安装 imbalanced-learn
# 阿里云镜像源可能存在同步延迟，改用清华源以避免连接 pypi.org 超时
RUN pip install -U \
    'imbalanced-learn'