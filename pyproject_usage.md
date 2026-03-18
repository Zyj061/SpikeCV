# pyproject.toml 使用说明

## 为什么使用 pyproject.toml？

### 从 setup.py 到 pyproject.toml 的转变

SpikeCV 项目从传统的 `setup.py` 配置方式迁移到现代的 `pyproject.toml` 标准，會带来以下优势：

#### pyproject.toml 的优势

1. **标准化配置**
   - 遵循 PEP 518/621 标准，是 Python 社区的推荐做法
   - 统一的配置格式，易于维护和理解
   - 与现代 Python 工具链更好的兼容性
2. **更好的依赖管理**
   - 清晰区分核心依赖和可选依赖
   - 支持更灵活的版本约束
   - 更好的依赖解析和冲突检测
3. **构建系统现代化**
   - 使用 setuptools 作为构建后端
   - 支持多种构建系统（setuptools、poetry、flit 等）
   - 更快的构建速度和更好的缓存机制
4. **开发体验提升**
   - 更好的 IDE 支持（自动补全、类型检查）
   - 统一的项目元数据管理
   - 与现代工具（如 uv、poetry）的无缝集成

#### 配置对比

**旧的 setup.py 方式**：

```python
from setuptools import setup, find_packages

setup(
    name="SpikeCV",
    version="0.1a",
    packages=find_packages(),
    install_requires=[
        "numpy<2.0",
        "torch",
        # ... 更多依赖
    ],
    extras_require={
        "tracking": ["motmetrics>=1.2.0"],
        # ... 更多可选依赖
    }
)
```

**新的 pyproject.toml 方式**：

```toml
[project]
name = "SpikeCV"
version = "0.1a"
dependencies = [
    "numpy<2.0",
    "torch",
]

[project.optional-dependencies]
tracking = ["motmetrics>=1.2.0"]
```

#### 迁移影响

- **向后兼容**：现有的安装方式仍然有效
- **命令一致**：`pip install` 命令保持不变
- **配置简化**：所有配置集中在一个文件中
- **工具支持**：支持更多现代 Python 工具

## 采用 pyproject.toml 后如何安装 SpikeCV

### 基础安装

#### 可编辑模式安装（推荐用于开发）

```bash
# 安装为可编辑模式
pip install -e .
```

#### 标准模式安装（推荐用于生产环境）

```bash
# 标准安装（不可编辑）
pip install .

# 或从 PyPI 安装（如果已发布）
pip install SpikeCV
```

**可编辑模式 vs 标准模式**：

- **可编辑模式** (`-e`)：代码修改即时生效，适合开发
- **标准模式**：代码需要重新安装才能生效，适合生产环境

### 按功能模块安装

```bash
# 重构算法
pip install -e ".[reconstruction]"

# 深度估计
pip install -e ".[depth_estimation]"

# 光流估计
pip install -e ".[optical_flow]"

# 目标检测
pip install -e ".[detection]"

# 目标识别
pip install -e ".[recognition]"

# 目标跟踪
pip install -e ".[tracking]"

# 文档构建
pip install -e ".[docs]"

# 测试工具
pip install -e ".[test]"

# 安装多個可选依赖
pip install -e ".[reconstruction,depth_estimation,optical_flow,detection,recognition,tracking,docs,test]"
```

### 使用 uv 安装（推荐）

`uv` 是一个快速的 Python 包管理器，可以显著提升安装速度。

#### uv 的优势

- **速度快**：比 pip 快 10-100 倍
- **依赖解析**：更准确的依赖冲突检测
- **缓存机制**：智能缓存，减少重复下载
- **兼容性**：完全兼容 pip 命令
- **现代化**：支持 lock 文件，确保依赖一致性

#### 安装 uv

```bash
# 在 Linux/macOS 上
curl -LsSf https://astral.sh/uv/install.sh | sh

# 在 Windows 上
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 或使用 pip
pip install uv
```

#### 使用 uv 安装 SpikeCV

**方式一：使用 uv pip（兼容 pip 命令）**

```bash
# 可编辑模式安装
uv pip install -e .

# 按功能模块安装
uv pip install -e ".[tracking]"
```

**方式二：使用 uv sync（推荐）**

`uv sync` 是 uv 的现代依赖管理方式，它会根据 `pyproject.toml` 和 `uv.lock` 文件自动同步依赖。

```bash
# 首次同步（安装所有依赖）
uv sync

# 同步特定依赖组
uv sync --extra tracking
uv sync --extra reconstruction,tracking

# 同步所有可选依赖
uv sync --all-extras

# 开发模式同步（包含 dev 依赖）
uv sync --dev
```

**uv sync 的优势**：

- **自动锁定**：生成 `uv.lock` 文件，确保依赖版本一致
- **虚拟环境管理**：自动创建和管理虚拟环境
- **增量更新**：只更新变化的依赖，速度更快
- **跨平台支持**：自动处理不同平台的依赖差异

## 當前项目 pyproject.toml 配置

### 基本信息

- **项目名称**: SpikeCV
- **版本**: 0.1a
- **Python 要求**: >= 3.10

### 核心依赖

- torch, torchvision
- numpy (< 2.0)
- matplotlib, scipy
- scikit-learn, scikit-image
- tensorboardX, tqdm
- 其他视觉处理库

### 可选依赖

按算法类型分类的可选依赖，可根据需要选择性安装。

## 注意事项

- Python 版本要求 >= 3.10
- numpy 版本限制 < 2.0，以保持更好的兼容性
- **建议使用虚拟环境**

## 如何修改 pyproject.toml

### 修改版本号

```toml
[project]
version = "0.1a"  # 修改为新版本号，如 "0.2.0"
```

### 添加/修改核心依赖

```toml
[project]
dependencies = [
    "numpy<2.0",           # 修改版本限制
    "torch>=2.0.0",        # 添加新依赖或修改版本
    "your-package>=1.0.0",  # 添加新的包
]
```

### 添加新的可选依赖组

```toml
[project.optional-dependencies]
# 添加新的算法类型
your_algorithm = [
    "package1>=1.0.0",
    "package2>=2.0.0",
]

# 修改现有的可选依赖
tracking = [
    "motmetrics>=1.2.0",
    "filterpy>=1.4.0",
    "new-tracking-package>=1.0.0",  # 添加新依赖
]
```

### 修改项目信息

```toml
[project]
name = "SpikeCV"
description = "An open-source framework for Spiking computer vision"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
urls = {repository = "https://github.com/yourusername/SpikeCV.git"}
```

### 修改 Python 版本要求

```toml
[project]
requires-python = ">=3.10"  # 修改为其他版本，如 ">=3.11"
```

### 常见修改场景

#### 场景 1: 为新算法添加依赖

```toml
[project.optional-dependencies]
your_new_algorithm = [
    "required-package>=1.0.0",
    "optional-package>=2.0.0",
]
```

#### 场景 2: 更新依赖版本

```toml
[project]
dependencies = [
    "numpy>=1.24.0,<2.0",  # 更新 numpy 版本范围
    "torch>=2.1.0",         # 更新 torch 版本
]
```

#### 场景 3: 添加开发工具

```toml
[project.optional-dependencies]
dev = [
    "black>=23.0.0",        # 代码格式化
    "flake8>=6.0.0",       # 代码检查
    "mypy>=1.0.0",         # 类型检查
]
```

### 修改后重新安装

#### 使用 pip

```bash
# 修改后重新安装
pip install -e .

# 或重新安装特定模块
pip install -e ".[your_new_algorithm]"
```

#### 使用 uv

```bash
# 使用 uv pip
uv pip install -e .
uv pip install -e ".[your_new_algorithm]"

# 使用 uv sync（推荐）
uv sync
uv sync --extra your_new_algorithm
uv sync --all-extras
```

### 验证配置

```bash
# 使用 pip
pip check

# 使用 uv
uv pip check
```

