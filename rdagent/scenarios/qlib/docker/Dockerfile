# 使用 Ubuntu 20.04 Focal 基础镜像 (ARM64)
FROM arm64v8/ubuntu:focal

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# 配置 Ubuntu 软件源（清华大学镜像）
RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list && \
    sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list

# 安装系统依赖（增加编译 SciPy 所需的额外依赖）
RUN apt-get update && apt-get install -y \
    curl \
    vim \
    git \
    redis-server \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-venv \
    libatlas-base-dev \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置Python 3为默认Python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 配置 pip 使用国内源（清华大学镜像）
RUN mkdir -p /root/.config/pip && \
    echo "[global]" > /root/.config/pip/pip.conf && \
    echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> /root/.config/pip/pip.conf && \
    echo "trusted-host = pypi.tuna.tsinghua.edu.cn" >> /root/.config/pip/pip.conf

# 升级pip并安装必要的构建依赖
RUN pip install --upgrade pip setuptools wheel cython numpy

# 克隆QLib代码
RUN git clone https://gitee.com/mirrors_microsoft/qlib.git /workspace/qlib

WORKDIR /workspace/qlib

# 重置到指定提交
RUN git fetch && git reset --hard 3e72593b8c985f01979bebcf646658002ac43b00

# 安装 QLib 的核心依赖（手动指定，因为可能没有 requirements.txt）
RUN pip install pandas numpy scikit-learn lightgbm requests ruamel.yaml

# 安装 SciPy - 使用较新但兼容的版本（ARM64 可用）
RUN pip install scipy==1.10.1 || \
    (echo "预编译版本不可用，从源码编译 SciPy..." && \
     pip install --no-binary scipy scipy==1.10.1)

# 安装PyTorch CPU版本 (ARM64兼容)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装QLib
RUN pip install -e .

# 安装其他依赖
RUN pip install catboost xgboost tables python-redis-lock

# 清理缓存
RUN apt-get clean && \
    pip cache purge && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
