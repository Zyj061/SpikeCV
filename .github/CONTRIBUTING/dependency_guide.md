# 依赖管理指南

本文档详细说明了 SpikeCV 项目的依赖管理规范和最佳实践。

## 1. 依赖分类

SpikeCV 的依赖分为两类：

### 核心依赖

所有或大部分算法都需要的基础依赖，包括：

- torch, torchvision（深度学习框架）
- numpy（数值计算）
- matplotlib, scipy（科学计算和可视化）
- scikit-learn, scikit-image（机器学习和图像处理）
- tensorboardX, tqdm（训练和进度显示）
- 其他视觉处理库

.. warning::
**重要提示**：核心依赖的修改需要谨慎处理！

- 核心依赖影响所有算法，修改前必须确保不会破坏现有功能
- 修改核心依赖版本需要经过充分的测试和验证
- 建议在 PR 中详细说明修改原因和影响范围
- 核心依赖的修改可能需要维护者的特别批准
- 如果只是您的算法需要特定依赖，请添加到可选依赖中

### 可选依赖

按算法类型分类的可选依赖：

- `reconstruction`: 重构算法相关依赖
- `depth_estimation`: 深度估计相关依赖
- `optical_flow`: 光流估计相关依赖
- `detection`: 目标检测相关依赖
- `recognition`: 目标识别相关依赖
- `tracking`: 目标跟踪相关依赖（如 motmetrics, filterpy）
- `docs`: 文档构建相关依赖（如 sphinx）
- `test`: 测试工具相关依赖（如 pytest）

## 2. 添加新依赖

### 添加核心依赖

.. warning::
**强烈建议**：优先考虑将依赖添加到可选依赖中！

- 核心依赖会影响所有算法，增加不必要的依赖会使项目变得臃肿
- 如果只是您的算法需要某个包，请将其添加到对应的可选依赖组
- 只有在确实需要所有算法都使用某个包时，才考虑添加到核心依赖
- 添加核心依赖前，请先在 Issue 中讨论，获得维护者的同意

如果您的算法确实需要新的核心依赖（经过讨论和批准后）：

```toml
[project]
dependencies = [
    "numpy<2.0",
    "torch",
    "your-new-package>=1.0.0",  # 添加新的核心依赖
]
```

### 添加可选依赖

如果您的算法只需要特定功能的依赖：

```toml
[project.optional-dependencies]
# 为新算法类型添加依赖
your_algorithm_type = [
    "package1>=1.0.0",
    "package2>=2.0.0",
]

# 或添加到现有的算法类型
tracking = [
    "motmetrics>=1.2.0",
    "filterpy>=1.4.0",
    "your-tracking-package>=1.0.0",  # 添加新的跟踪相关依赖
]
```

### 局部依赖配置（特殊情况）

.. note::
**特殊情况处理**：如果您的算法依赖与核心依赖存在版本冲突

虽然不建议修改核心依赖，但如果您的算法确实需要特定版本的依赖，而该版本与核心依赖冲突，您可以在自己的算法目录下创建一个局部依赖配置文件。

**适用场景**：

- 您的算法需要某个包的特定版本，但该版本与核心依赖冲突
- 临时解决方案，等待核心依赖更新后再统一
- 实验性算法，不影响其他功能

**实现方式**：

在您的算法目录下创建 `requirements.txt` 或 `pyproject.toml` 文件：

```
SpikeCV/
├── SpikeCV/
│   ├── spkProc/
│   │   ├── AlgorithmType/          # 算法类型目录（如 tracking）
│   │   │   ├── your_algorithm/    # 您的算法目录
│   │   │   │   ├── your_algorithm.py
│   │   │   │   ├── requirements.txt      # 局部依赖文件
│   │   │   │   └── __init__.py
```

**requirements.txt 示例**：

```txt
# 您的算法需要的特定版本依赖
package-name==1.5.0
another-package>=2.0.0,<3.0.0
```

**pyproject.toml 示例**：

```toml
[project]
dependencies = [
    "package-name==1.5.0",
    "another-package>=2.0.0,<3.0.0",
]
```

也可以使用`uv`等工具来辅助导出依赖 `uv export --format requirements-txt > requirements.txt`。

**使用方法**：

```bash
# 在运行您的算法前，先安装局部依赖
cd SpikeCV/spkProc/AlgorithmType/your_algorithm
pip install -r requirements.txt
# 或
pip install -e .
```

**注意事项**：

- 局部依赖只影响您的算法，不影响其他算法
- 请在文档中明确说明如何安装局部依赖
- 建议在 PR 中说明冲突原因和解决方案
- **长期来看，应该与维护者讨论如何统一依赖版本**

**不推荐的做法**：

- 不要在算法代码中动态修改 sys.path 来解决依赖冲突
- 不要在算法中强制安装特定版本的依赖
- 不要在全局环境中覆盖核心依赖

## 3. 依赖版本规范

### 版本指定方式

```toml
# 精确版本
"package==1.0.0"

# 最低版本
"package>=1.0.0"

# 版本范围
"package>=1.0.0,<2.0"

# 排除特定版本
"package>=1.0.0,!=1.5.0"
```

### 常见版本限制

```toml
# numpy 版本限制（避免与现有代码不兼容）
"numpy>=1.24.0,<2.0"

# torch 版本建议
"torch>=2.0.0"

# 不指定版本（不推荐，可能导致兼容性问题）
"package"  # ❌ 不推荐
```

## 4. 依赖管理最佳实践

### 最小化依赖

- 只添加必要的依赖
- 避免重复依赖（功能相似的包）
- 优先使用已有的核心依赖

### 版本兼容性

- 测试不同版本的兼容性
- 遵循语义化版本规范
- 考虑与现有代码的兼容性

### 依赖分组

```toml
[project.optional-dependencies]
# 按算法类型分组
reconstruction = [
    "package1>=1.0.0",
]

# 按功能分组
dev = [
    "black>=23.0.0",     # 代码格式化
    "flake8>=6.0.0",    # 代码检查
    "mypy>=1.0.0",      # 类型检查
]
```

## 5. 验证依赖

### 检查依赖冲突

```bash
# 检查依赖是否有冲突
pip check
```

### 查看依赖树

```bash
# 查看依赖关系
pipdeptree
```

### 测试安装

```bash
# 测试安装核心依赖
pip install -e .

# 测试安装特定模块
pip install -e ".[your_algorithm_type]"

# 测试安装所有依赖
pip install -e ".[reconstruction,depth_estimation,optical_flow,detection,recognition,tracking,docs,test]"
```

## 6. 更新依赖的流程

1. **确定依赖类型**
   - 判断是核心依赖还是可选依赖
   - 确定所属的算法类型
   - 检查是否与核心依赖存在版本冲突
2. **添加依赖到 pyproject.toml**
   - 在对应的 `[project.dependencies]` 或 `[project.optional-dependencies]` 中添加
   - 指定合适的版本范围
   - 如果存在版本冲突，考虑使用局部依赖配置
3. **测试安装**
   ```bash
   # 测试标准依赖安装
   pip install -e ".[your_algorithm_type]"

   # 如果使用局部依赖
   cd SpikeCV/spkProc/AlgorithmType/your_algorithm
   pip install -r requirements.txt
   ```
4. **验证功能**
   - 确保依赖可以正常导入
   - 测试算法功能是否正常
   - 确认不会影响其他算法的功能
5. **提交更改**
   ```bash
   git add pyproject.toml
   git commit -m "Add dependency: your-package for your-algorithm"
   ```

## 7. 常见问题

### Q: 如何选择依赖的版本？

A: 建议使用最低版本限制（`>=1.0.0`），并在测试中验证兼容性。对于关键依赖，可以指定版本范围（`>=1.0.0,<2.0`）。

### Q: 是否需要将所有依赖都添加到 pyproject.toml？

A: 只需要添加直接依赖。间接依赖会由 pip 自动解析和安装。

### Q: 如何处理依赖冲突？

A: 如果您的算法依赖与核心依赖存在版本冲突，有以下几种处理方式：

1. **优先使用可选依赖**：将您的依赖添加到对应的可选依赖组
2. **使用局部依赖配置**：在您的算法目录下创建 `requirements.txt` 或 `pyproject.toml` 文件，只影响您的算法
3. **联系维护者**：在 Issue 中讨论，寻求统一的解决方案
4. **等待核心依赖更新**：如果冲突是暂时的，可以等待核心依赖更新后再统一

**推荐顺序**：可选依赖 → 局部依赖配置 → 联系维护者

### Q: 可以移除旧的依赖吗？

A: 如果某个依赖不再被任何算法使用，可以移除。但需要确保不会影响其他功能。

### Q: 我可以修改核心依赖吗？

A: **不建议随意修改核心依赖**！核心依赖影响所有算法，修改可能导致现有功能失效。如果您的算法需要特定依赖，请添加到对应的可选依赖组。只有在确实需要所有算法都使用某个包时，才考虑修改核心依赖，并且需要先在 Issue 中讨论，获得维护者的批准。

### Q: 什么时候应该使用核心依赖而不是可选依赖？

A: 只有在以下情况下才考虑添加到核心依赖：

- 该依赖被多个算法类型使用
- 该依赖是项目基础设施的一部分（如日志、配置管理等）
- 经过讨论，维护者同意将其作为核心依赖
- 否则，请始终使用可选依赖

## 8. 示例：为新算法添加依赖

假设您添加了一个新的重构算法 `YourReconstruction`，需要 `your-recon-package` 依赖：

```toml
[project.optional-dependencies]
# 添加到现有的重构算法依赖组
reconstruction = [
    "your-recon-package>=1.0.0",  # 新增依赖
]
```

然后在 PR 描述中说明：

```
## 更改内容
- [ ] 添加了新的算法实现
- [ ] 添加了使用示范文件
- [ ] 更新了核心操作.rst
- [ ] 更新了使用例子.rst
- [ ] 添加了必要的图片资源
- [-] 更新了 pyproject.toml 依赖

## 依赖更新
- 添加 `your-recon-package>=1.0.0` 到 `reconstruction` 可选依赖组
```

