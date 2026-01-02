# JupyterLab 安装与使用指南

本文档旨在提供 JupyterLab 在不同操作系统（Mac、Windows、Linux）下的安装与使用指南。涵盖本地环境安装（基于 Python 或 Conda）以及 Docker 容器化部署两种方式。

## 1. 简介

JupyterLab 是 Project Jupyter 的下一代基于 Web 的用户界面。它提供了灵活且强大的数据科学环境，支持在同一窗口中处理文档、代码编辑器、终端和自定义组件。相比经典的 Jupyter Notebook，JupyterLab 提供了更好的模块化体验和扩展能力。

---

## 2. 本地安装（Local Installation）

本地安装是最常见的部署方式，适用于个人开发环境。无论您使用的是 macOS、Windows 还是 Linux，安装步骤基本一致。

### 2.1 前置条件

在安装 JupyterLab 之前，请确保您的系统中已安装 Python 环境。

- **推荐**：安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)，它们集成了 Python 和常用的数据科学包。
- **或者**：直接安装 [Python 3.8+](https://www.python.org/)。

### 2.2 使用 Conda 安装（推荐）

如果您使用的 Miniconda，建议使用 `conda` 命令进行安装，这样可以更好地管理依赖包。

打开终端（Mac/Linux）或 Anaconda Prompt（Windows），执行以下命令：

```bash
# 1. (可选) 创建一个新的虚拟环境，建议为每个项目创建独立环境
conda create -n jupyter_env python=3.9

# 2. 激活环境
conda activate jupyter_env

# 3. 安装 JupyterLab
# -c conda-forge 指定从 conda-forge 频道下载，通常更新更快
conda install -c conda-forge jupyterlab
```

### 2.3 使用 pip 安装

如果您使用的是标准 Python 环境，可以使用 `pip` 包管理器进行安装。

打开终端（Mac/Linux）或命令提示符/PowerShell（Windows），执行以下命令：

```bash
# 建议先升级 pip
pip install --upgrade pip

# 安装 JupyterLab
pip install jupyterlab
```

---

## 3. Docker 安装

使用 Docker 安装可以确保环境的一致性，避免本地环境依赖冲突，非常适合团队协作或快速体验。

### 3.1 前置条件

请确保您的系统已安装并启动了 Docker Desktop（Mac/Windows）或 Docker Engine（Linux）。

### 3.2 拉取并运行镜像

Jupyter 官方提供了多个 Docker 镜像栈。常用的有：

- `jupyter/base-notebook`: 基础镜像，仅包含 Python 和 JupyterLab。
- `jupyter/scipy-notebook`: 包含常用的科学计算包（Pandas, NumPy, Scikit-learn 等）。
- `jupyter/datascience-notebook`: 包含 Python, R, Julia 及相关数据科学包。

**运行命令示例：**

在终端中执行以下命令：

```bash
# 拉取并启动 jupyter/scipy-notebook 镜像
# -p 8888:8888 将容器的 8888 端口映射到宿主机的 8888 端口
# -v "$PWD":/home/jovyan/work 将当前工作目录挂载到容器内的 /home/jovyan/work 目录，实现数据持久化
# --name my-jupyter 指定容器名称
docker run -p 8888:8888 \
    -v "$PWD":/home/jovyan/work \
    --name my-jupyter \
    jupyter/scipy-notebook:latest
```

_注意：Windows 用户在 PowerShell 中使用挂载路径时，可能需要将 `"$PWD"` 替换为绝对路径，例如 `c:/Users/YourName/Project:/home/jovyan/work`。_

启动后，终端会输出包含 Token 的 URL，复制该 URL 到浏览器即可访问。

---

## 4. 启动与使用

### 4.1 启动 JupyterLab

**本地安装启动：**

在终端或命令提示符中，进入您的项目目录，然后输入：

```bash
jupyter lab
```

执行后，默认浏览器会自动打开 `http://localhost:8888/lab`。如果未自动打开，请复制终端中显示的 URL（包含 token）访问。

**Docker 安装启动：**

如上节所述，容器启动后直接访问终端输出的 URL。

### 4.2 界面概览与基本操作

JupyterLab 的界面主要分为以下几个区域：

1. **菜单栏 (Menu Bar)**：顶部区域，包含文件、编辑、视图、运行等常用命令。
2. **左侧边栏 (Left Sidebar)**：
   - **文件浏览器 (File Browser)**：管理工作目录下的文件和文件夹。
   - **运行中的终端和内核 (Running Terminals and Kernels)**：查看并管理当前活跃的会话。
   - **命令面板 (Command Palette)**：搜索并执行 JupyterLab 的所有命令。
   - **插件管理器 (Extension Manager)**：安装和管理第三方插件。
3. **主工作区 (Main Work Area)**：用于打开 Notebook、编辑器、终端等。支持拖拽标签页进行分屏显示。

### 4.3 常用快捷键

- `Shift + Enter`: 运行当前单元格并选中下一个单元格。
- `Ctrl + Enter` (Mac: `Cmd + Enter`): 运行当前单元格并保持选中。
- `A`: 在上方插入新单元格。
- `B`: 在下方插入新单元格。
- `M`: 将当前单元格转换为 Markdown 模式。
- `Y`: 将当前单元格转换为 Code 模式。
- `D, D` (按两次 D): 删除当前单元格。

---

## 5. 进阶配置

### 5.1 安装中文语言包

JupyterLab 默认是英文界面，可以通过安装语言包切换为中文。

```bash
# 安装中文语言包
pip install jupyterlab-language-pack-zh-CN
```

安装完成后，刷新页面，在 `Settings` -> `Language` 中选择 `Chinese (Simplified, China)` 即可。

### 5.2 常用插件推荐

- **jupyterlab-git**: 提供 Git 版本控制的图形化界面。
- **jupyterlab-lsp**: 提供代码自动补全、跳转定义等 IDE 功能（需要配合 Python LSP Server）。
- **jupyterlab-drawio**: 在 JupyterLab 中绘制流程图。

可以通过左侧的插件管理器搜索并安装这些插件（部分插件可能需要 Node.js 环境）。
