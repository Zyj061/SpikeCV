# pyproject.toml Usage Guide

## Why Use pyproject.toml?

### The Transition from setup.py to pyproject.toml

The SpikeCV project has migrated from the traditional `setup.py` configuration method to the modern `pyproject.toml` standard, bringing the following advantages:

#### Advantages of pyproject.toml

1. **Standardized Configuration**
   - Follows PEP 518/621 standards, the recommended practice in the Python community
   - Unified configuration format, easy to maintain and understand
   - Better compatibility with modern Python toolchains

2. **Improved Dependency Management**
   - Clearly distinguishes core dependencies from optional dependencies
   - Supports more flexible version constraints
   - Better dependency resolution and conflict detection

3. **Modernized Build System**
   - Uses setuptools as the build backend
   - Supports multiple build systems (setuptools, poetry, flit, etc.)
   - Faster build speeds and better caching mechanisms

4. **Enhanced Developer Experience**
   - Better IDE support (autocompletion, type checking)
   - Unified project metadata management
   - Seamless integration with modern tools (such as uv, poetry)

#### Configuration Comparison

**Old setup.py approach**:

```python
from setuptools import setup, find_packages

setup(
    name="SpikeCV",
    version="0.1a",
    packages=find_packages(),
    install_requires=[
        "numpy<2.0",
        "torch",
        # ... more dependencies
    ],
    extras_require={
        "tracking": ["motmetrics>=1.2.0"],
        # ... more optional dependencies
    }
)
```

**New pyproject.toml approach**:

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

#### Migration Impact

- **Backward Compatibility**: Existing installation methods remain valid
- **Consistent Commands**: `pip install` commands remain unchanged
- **Simplified Configuration**: All configuration is centralized in one file
- **Tool Support**: Supports more modern Python tools

## How to Install SpikeCV After Adopting pyproject.toml

### Basic Installation

#### Editable Mode Installation (Recommended for Development)

```bash
# Install in editable mode
pip install -e .
```

#### Standard Mode Installation (Recommended for Production)

```bash
# Standard installation (non-editable)
pip install .

# Or install from PyPI (if published)
pip install SpikeCV
```

**Editable Mode vs Standard Mode**:

- **Editable Mode** (`-e`): Code changes take effect immediately, suitable for development
- **Standard Mode**: Code needs to be reinstalled to take effect, suitable for production

### Installing by Feature Modules

```bash
# Reconstruction algorithms
pip install -e ".[reconstruction]"

# Depth estimation
pip install -e ".[depth_estimation]"

# Optical flow estimation
pip install -e ".[optical_flow]"

# Object detection
pip install -e ".[detection]"

# Object recognition
pip install -e ".[recognition]"

# Object tracking
pip install -e ".[tracking]"

# Documentation building
pip install -e ".[docs]"

# Testing tools
pip install -e ".[test]"

# Install multiple optional dependencies
pip install -e ".[reconstruction,depth_estimation,optical_flow,detection,recognition,tracking,docs,test]"
```

### Installing with uv (Recommended)

`uv` is a fast Python package manager that significantly speeds up installation.

#### Advantages of uv

- **Speed**: 10-100 times faster than pip
- **Dependency Resolution**: More accurate dependency conflict detection
- **Caching**: Intelligent caching reduces redundant downloads
- **Compatibility**: Fully compatible with pip commands
- **Modern**: Supports lock files to ensure dependency consistency

#### Installing uv

```bash
# On Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

#### Installing SpikeCV with uv

**Method 1: Using uv pip (Compatible with pip commands)**

```bash
# Editable mode installation
uv pip install -e .

# Install by feature modules
uv pip install -e ".[tracking]"
```

**Method 2: Using uv sync (Recommended)**

`uv sync` is uv's modern dependency management approach. It automatically synchronizes dependencies based on `pyproject.toml` and `uv.lock` files.

```bash
# First sync (installs all dependencies)
uv sync

# Sync specific dependency groups
uv sync --extra tracking
uv sync --extra reconstruction,tracking

# Sync all optional dependencies
uv sync --all-extras

# Sync in development mode (includes dev dependencies)
uv sync --dev
```

**Advantages of uv sync**:

- **Automatic Locking**: Generates a `uv.lock` file to ensure consistent dependency versions
- **Virtual Environment Management**: Automatically creates and manages virtual environments
- **Incremental Updates**: Only updates changed dependencies, faster
- **Cross-Platform Support**: Automatically handles platform-specific dependency differences

## Current pyproject.toml Configuration

### Basic Information

- **Project Name**: SpikeCV
- **Version**: 0.1a
- **Python Requirement**: >= 3.10

### Core Dependencies

- torch, torchvision
- numpy (< 2.0)
- matplotlib, scipy
- scikit-learn, scikit-image
- tensorboardX, tqdm
- Other vision processing libraries

### Optional Dependencies

Optional dependencies categorized by algorithm type, can be selectively installed as needed.

## Important Notes

- Python version requirement >= 3.10
- numpy version constrained to < 2.0 for better compatibility
- **It is recommended to use a virtual environment**

## How to Modify pyproject.toml

### Modifying the Version Number

```toml
[project]
version = "0.1a"  # Change to the new version number, e.g., "0.2.0"
```

### Adding/Modifying Core Dependencies

```toml
[project]
dependencies = [
    "numpy<2.0",           # Modify version constraint
    "torch>=2.0.0",        # Add new dependency or modify version
    "your-package>=1.0.0",  # Add a new package
]
```

### Adding a New Optional Dependency Group

```toml
[project.optional-dependencies]
# Add a new algorithm type
your_algorithm = [
    "package1>=1.0.0",
    "package2>=2.0.0",
]

# Modify existing optional dependencies
tracking = [
    "motmetrics>=1.2.0",
    "filterpy>=1.4.0",
    "new-tracking-package>=1.0.0",  # Add new dependency
]
```

### Modifying Project Information

```toml
[project]
name = "SpikeCV"
description = "An open-source framework for Spiking computer vision"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
urls = {repository = "https://github.com/yourusername/SpikeCV.git"}
```

### Modifying Python Version Requirement

```toml
[project]
requires-python = ">=3.10"  # Change to another version, e.g., ">=3.11"
```

### Common Modification Scenarios

#### Scenario 1: Adding Dependencies for a New Algorithm

```toml
[project.optional-dependencies]
your_new_algorithm = [
    "required-package>=1.0.0",
    "optional-package>=2.0.0",
]
```

#### Scenario 2: Updating Dependency Versions

```toml
[project]
dependencies = [
    "numpy>=1.24.0,<2.0",  # Update numpy version range
    "torch>=2.1.0",         # Update torch version
]
```

#### Scenario 3: Adding Development Tools

```toml
[project.optional-dependencies]
dev = [
    "black>=23.0.0",        # Code formatting
    "flake8>=6.0.0",       # Code linting
    "mypy>=1.0.0",         # Type checking
]
```

### Reinstalling After Modifications

#### Using pip

```bash
# Reinstall after modification
pip install -e .

# Or reinstall specific modules
pip install -e ".[your_new_algorithm]"
```

#### Using uv

```bash
# Using uv pip
uv pip install -e .
uv pip install -e ".[your_new_algorithm]"

# Using uv sync (recommended)
uv sync
uv sync --extra your_new_algorithm
uv sync --all-extras
```

### Verifying the Configuration

```bash
# Using pip
pip check

# Using uv
uv pip check
```
