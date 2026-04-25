# Contributing to SpikeCV

[English Version](./CONTRIBUTING_en.md)

感谢您对 SpikeCV 项目的关注！我们欢迎以下形式的贡献：

- **新算法**：添加新的脉冲视觉算法
- **算法 Bug 修复**：修复现有算法的问题
- **文档改进**：完善算法文档和使用说明

## 目录

- [算法贡献流程](#算法贡献流程)
- [代码结构要求](#代码结构要求)
- [依赖管理](#依赖管理)
- [文档贡献](#文档贡献)
- [提交 Pull Request](#提交-pull-request)
- [代码审查流程](#代码审查流程)

## 算法贡献流程

### 1. Fork 项目

首先，您需要 Fork SpikeCV 仓库到您的 GitHub 账户。

### 2. 创建分支

从 `main` 分支创建一个新的功能分支：

```bash
git checkout -b feature/your-algorithm-name
```

分支命名建议：

- 新算法：`feature/algorithm-name`
- Bug 修复：`fix/algorithm-bug-description`
- 文档更新：`docs/algorithm-docs-update`

### 3. 根据 `README.md` 下载依赖和配置环境

### 4. 开发新算法

在相应的模块目录下开发您的算法：

```
SpikeCV/
├── SpikeCV/
│   ├── spkProc/          # 算法实现
│   │   ├── filters/          # 滤波器
│   │   ├── reconstruction/   # 重构算法
│   │   ├── detection/        # 物体检测
│   │   ├── tracking/         # 目标跟踪
│   │   ├── recognition/      # 物体识别
│   │   └── motion/           # 运动估计
│   ├── examples/             # 使用示范示例
│   ├── metrics/              # 评估指标
│   ├── spkData/              # 数据加载
│   ├── utils/                # 工具函数
│   └── visualization/        # 可视化工具
```

## 算法代码结构要求

### 1. 算法实现文件 (\*.py)

#### 文件位置

根据算法类型将代码放在对应的目录中：

- 滤波器：`SpikeCV/spkProc/filters/`
- 重构算法：`SpikeCV/spkProc/reconstruction/`
- 物体检测：`SpikeCV/spkProc/detection/`
- 目标跟踪：`SpikeCV/spkProc/tracking/`
- 物体识别：`SpikeCV/spkProc/recognition/`
- 运动估计：`SpikeCV/spkProc/motion/`

#### 文件命名

- 使用小写字母和下划线：`your_algorithm.py`
- 避免使用特殊字符和空格

#### 代码规范

```python
# -*- coding: utf-8 -*-
# @Time : YYYY/MM/DD HH:MM
# @Author : Your Name
# @Email: your.email@example.com
# @File : your_algorithm.py

import numpy as np
import torch

class YourAlgorithm:
    """
    算法简要描述
    
    详细描述算法的核心思想、输入输出、主要参数等。
    """
    
    def __init__(self, param1, param2, device, **kwargs):
        """
        初始化算法
        
        Parameters
        ----------
        param1 : type
            参数1的描述
        param2 : type
            参数2的描述
        device : torch.device
            使用的处理器类型，'cpu' 或 'cuda'
        **kwargs : dict
            其他可选参数
        """
        self.param1 = param1
        self.param2 = param2
        self.device = device
        
    def process(self, spikes):
        """
        处理输入数据
        
        Parameters
        ----------
        spikes : type
            输入数据描述
            
        Returns
        -------
        output_data : type
            输出数据描述
        """
        # 实现算法逻辑
        pass
```

#### 代码风格建议

- 遵循 PEP 8 代码风格
- **包的导入**：导入 SpikeCV 内置模块或算法时，请务必以 `SpikeCV` 或 `spikecv` 开头进行绝对导入（例如：`from spikecv.spkData import load_dat`）。切勿使用相对导入（如 `from .. import`）。
- 添加必要的注释和文档字符串
- 使用类型提示（Type Hints）
- 处理边界情况和错误
- 确保代码可读性和可维护性

#### 进度显示建议

在算法执行过程中，建议使用 `tqdm` 显示进度，提升用户体验。`tqdm` 是项目核心依赖之一，可以直接使用。

**使用场景**：

- 处理多帧数据时显示帧进度
- 处理多个数据文件时显示文件进度
- 迭代处理时显示迭代进度
- 任何耗时操作的进度显示

**示例代码**：

```python
from tqdm import tqdm
import torch
import numpy as np

class YourAlgorithm:
    def process(self, spikes):
        '''
        Process spike data and return tracking results
        
        spikes : np.ndarray
            Input spike data, shape (length, height, width)
            
        Returns
        -------
        results : list
            Tracking results for each frame
        '''
        timestamps = spikes.shape[0]
        results = []
        
        # Use tqdm to display processing progress
        for t in tqdm(range(timestamps), desc="Tracking"):
            try:
                # Process each frame
                input_spk = torch.from_numpy(spikes[t, :, :].copy()).to(self.device)
                
                # Your algorithm logic here
                
                # Store result
                results.append(tracking_result)
                
            except RuntimeError as exception:
                # handle error
                pass
        
        return results
```

**tqdm 常用参数**：

```python
# 基本用法
for i in tqdm(range(100)):
    pass

# 添加描述
for i in tqdm(range(100), desc="Processing"):
    pass

# 显示额外信息
for i in tqdm(range(100), desc="Processing", unit="frame"):
    pass

# 嵌套进度条
for file in tqdm(files, desc="Files"):
    for frame in tqdm(frames, desc=f"Processing {file}", leave=False):
        process_frame(frame)
```

**注意事项**：

- 在长时间运行的操作中使用 tqdm，让用户了解进度
- 使用清晰的描述信息（`desc` 参数）
- 合理设置单位（`unit` 参数）
- 嵌套进度条时使用 `leave=False` 避免混乱
- 在 Jupyter notebook 中可以使用 `from tqdm.notebook import tqdm`

### 2. 使用示范文件 (test\_\*.py)

#### 文件性质说明

虽然文件命名使用 `test_*.py` 格式，但这些文件实际上是**使用示范脚本**，用于展示如何使用各个算法。它们不是单元测试，而是完整的使用示例，帮助用户理解算法的正确使用方法。

#### 文件位置

将使用示范文件放在 `SpikeCV/examples/` 目录下：

```
examples/
├── test_your_algorithm.py
```

#### 使用示范文件结构

```python
# -*- coding: utf-8 -*-
# @Time : YYYY/MM/DD HH:MM
# @Author : Your Name
# @Email: your.email@example.com
# @File : test_your_algorithm.py

import sys
import os
sys.path.append("..")

import torch
import numpy as np
from spkData.load_dat import SpikeStream,data_parameter_dict
from spkProc.your_module.your_algorithm import YourAlgorithm

"""
演示如何使用您的算法

该函数展示如何使用 YourAlgorithm 类进行完整的使用示范，
包括数据加载、算法初始化、执行和结果保存。
这是一个使用示例，不是单元测试。
"""

# 加载数据
data_filename = "path/to/your/data.dat" # e.g. recVidarReal2019/classA/car-100kmh
label_type = 'your_label_type' # e.g. 'raw'

paraDict = data_parameter_dict(data_filename, label_type)
pprint(paraDict)

spike_stream = SpikeStream(**dataDict)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 算法初始化和执行
algorithm = YourAlgorithm(
    param1=value1,
    param2=value2,
    device=device
)
results = algorithm.process(spikes)

# 保存结果至文件、图像、影片等
if not os.path.exists('results'):
    os.makedirs('results')

# ...

print('Demo completed successfully!')

```

**注意**：使用示范文件位于 `SpikeCV/examples/` 目录，需要使用 `sys.path.append("..")` 来访问 `SpikeCV/` 目录下的其他模块（如 `spkProc`、`spkData`、`utils`、`visualization` 等）。

#### 使用示范要求

- 使用示范文件名以 `test_` 开头（保持命名一致性）
- 包含完整的使用示例，展示算法的典型使用场景
- 展示结果的可视化和保存方法
- 添加详细的注释说明每个步骤
- 确保代码可以直接运行（在提供正确数据的前提下）
- 展示算法的实际应用效果

## CLI工具贡献规范

如果您想将您的算法集成到 SpikeCV 的命令行工具（CLI）中，或提交关于 CLI 的改进，请遵循以下规范。CLI 相关代码位于 `SpikeCV/cli/` 目录下。

### 1. 文件位置
- 数据处理相关命令：修改 `SpikeCV/cli/data.py`
- 算法执行相关命令：修改 `SpikeCV/cli/proc.py`

### 2. 接口设计与开发模板
我们使用 `typer` 库构建分级 CLI。如果该类别下已有子命令（如 `track`, `reconst`），请向对应的子 Typer 程序中添加算法命令：

- 跟踪算法：修改 `SpikeCV/cli/proc/track.py`
- 重构算法：修改 `SpikeCV/cli/proc/reconst.py`

**开发规范：**
1. **函数内部导入**：务必在函数内部导入算法库，避免启动延迟。
2. **Panel 分组**：使用 `rich_help_panel="Algorithms"` 将算法命令归类。
3. **Agent 支持**：保留 `agent_used` 参数及 JSON 输出逻辑。

```python
import typer

# 在对应的 track.py 或 reconst.py 中添加
@app.command(name="your-algo", 
             help="Detailed description.", 
             rich_help_panel="Algorithms")
def your_algo(
    data_path: str = typer.Option(..., "--data-path", "-d"),
    agent_used: bool = typer.Option(False, "--agent-used", "-agent")
):
    import json
    import sys
    import contextlib
    
    # 强制要求：在函数内部导入算法库，以避免不必要的包预加载导致 CLI 启动延迟
    # 同时使用 SpikeCV/spikecv 开头进行内部导入，如：
    # from spikecv.examples import your_run_script
    
    result_dict = {"status": "error", "message": "Unknown error", "result": None} 
    err_msg = ""
    try:
        # CLI 业务逻辑...
        typer.echo("Running your task...", err=True)
        
        # 将标准输出重定向到 stderr，确保 stdout 仅用于纯净结构化数据返回（如 agent_used 时）
        with contextlib.redirect_stdout(sys.stderr):
            with contextlib.redirect_stderr(sys.stderr):
                # result_dict["result"] = your_run_script.main(args)
                pass
                
        result_dict.update({"status": "success", "message": "Done."})
    except Exception as e:
        final_msg = err_msg if err_msg else str(e)
        result_dict.update({"status": "error", "message": final_msg})
        typer.echo(final_msg, err=True)
        if not agent_used:
            raise typer.Exit(1)
    finally:
        if agent_used:
            # 专为 agent 提供结构化 JSON 格式输出在控制台
            typer.echo(json.dumps(result_dict))
```

### 3. PR 模板说明
CLI 相关 Pull Request 请直接使用下方“提交 Pull Request”章节中的统一 PR 描述模板。

如包含 CLI 改动，请在统一模板中的 “CLI 检查清单（如适用）” 部分勾选并补充对应验证信息。

## 依赖管理


### 项目环境管理

关于 `pyproject.toml` 详细的依赖管理指南请参考：[`.github/CONTRIBUTING/dependency_guide.md`](.github/CONTRIBUTING/dependency_guide.md)

该指南包含以下内容：

- 依赖分类（核心依赖和可选依赖）
- 如何添加新依赖
- 局部依赖配置（处理版本冲突）
- 依赖版本规范
- 依赖管理最佳实践
- 验证依赖的方法
- 更新依赖的流程
- 常见问题解答


### 环境依赖记录

为了避免"在我的电脑能跑"问题，我们强烈建议在提交 PR 前记录您的实际环境依赖：

#### 1. 记录 Python 版本

在您的算法文件夹中创建 `.python-version` 文件：

```
3.10
```

**说明**：

- 该文件用于 pyenv 等版本管理工具
- 确保其他开发者使用相同的 Python 版本
- 避免因 Python 版本差异导致的问题

#### 2. 记录依赖包版本

在您的算法文件夹中运行以下命令：

```bash
pip freeze > requirement.txt
```

**说明**：

- 该文件列出了您环境中所有已安装的包及其精确版本
- 帮助其他开发者复现您的环境
- 避免因依赖版本差异导致的问题

**示例文件结构**：

```
SpikeCV/
├── spkProc/
│   └── tracking/
│       └── SNN_Tracker/
│           ├── .python-version      # Python 版本
│           ├── requirement.txt      # 依赖包版本
│           ├── snn_tracker.py     # 算法实现
│           └── __init__.py
```

**使用方法**：

其他开发者可以使用以下命令复现您的环境：

```bash
# 使用 .python-version 设置 Python 版本
pyenv local 3.10

# 使用 requirement.txt 安装依赖
pip install -r requirement.txt

# 或者使用 uv 进行快速安装
uv pip install -r requirement.txt
```

**注意事项**：

- `pip freeze` 会列出所有已安装的包，包括核心依赖和可选依赖
- 如果您的算法有特殊的依赖需求，可以在 requirement.txt 中添加注释说明
- 建议在虚拟环境中运行 `pip freeze`，避免包含系统级的包
- 如果使用 conda，也可以使用 `conda env export > environment.yml` 导出环境
- **虽然可以记录依赖包，但这只是方便 work around 和 环境依赖协调根据，我们建议所有的环境依赖都通过项目根目录的** **`pyproject.toml`** **统一管理，在项目初始化后您若有任何依赖项变化（新增/版本限制改变），应该在提交 PR 前更新 `pyproject.toml`**

## 文档贡献

### 文档贡献类型

我们接受以下类型的文档贡献：

1. **新算法文档**：为新实现的算法添加完整的文档
2. **算法文档改进**：完善现有算法的文档，包括：
   - 修正文档中的错误或不准确之处
   - 补充缺失的算法描述或参数说明
   - 添加更多的使用示例
   - 改进文档的可读性和结构
3. **使用示例改进**：改进算法的使用示范文件

### 1. 更新核心操作.rst

在 `docs/source/核心操作.rst` 中添加您的算法文档。

#### 文档结构参考

```rst
找到对应的算法类别（如 重构算法 ）
--------

基于您的算法名称
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``spkProc.your_module.your_algorithm``\ 中基于[您的算法描述]\ ``YourAlgorithm``\ 。其核心算法思想是[简要描述算法的核心思想]。使用[您的算法]可先通过实例化\ ``spkProc.your_module.your_algorithm``\ 中的\ ``YourAlgorithm``\ 类，其采用的数据类型为 *pytorch* 的张量形式，初始化时需提供[必要的参数]。例如，通过以下例子进行创建算法实例：

.. code-block:: python

   from spkProc.your_module.your_algorithm import YourAlgorithm
   import torch

   device = torch.device('cuda')
   algorithm = YourAlgorithm(param1=value1, param2=value2, device=device)

YourAlgorithm类中的变量
~~~~~~~~~~~~~~~~~~~~~~~~

``your_algorithm.py``\ 中您的算法对应的类\ ``YourAlgorithm``\ 具有以下几种变量：

* ``param1``\ ：参数1的描述
* ``param2``\ ：参数2的描述
* ``device``\ ：所使用的处理器类型，\ ``cpu``\ 或者\ ``cuda``
* ``variable1``\ ：变量1的描述
* ``variable2``\ ：变量2的描述

YourAlgorithm类中的函数
~~~~~~~~~~~~~~~~~~~~~~~~

``your_algorithm.YourAlgorithm``\ 中包含以下函数：

#. 
   ``__init__(param1, param2, device, **kwargs)``\ ：初始化算法实例。

   参数说明：

   * ``param1``\ ：参数1的描述
   * ``param2``\ ：参数2的描述
   * ``device``\ ：处理器类型，\ ``cpu``\ 或者\ ``cuda``
   * ``**kwargs``\ ：其他可选参数

   其调用方式如下：

   .. code-block:: python

      from spkProc.your_module.your_algorithm import YourAlgorithm
      import torch

      device = torch.device('cuda')
      algorithm = YourAlgorithm(param1=value1, param2=value2, device=device)

#. 
   ``process(input_data)``\ ：处理输入数据并返回结果。

   参数说明：

   * ``input_data``\ ：输入数据描述

   返回值：

   * ``output_data``\ ：输出数据描述

   其调用方式如下：

   .. code-block:: python

      results = algorithm.process(spikes)

相关论文
~~~~~~~~

更多关于算法的细节可参考论文：

#. Author1, Author2, et al. Paper Title[J]. Journal Name, Year.
#. Author1, Author2, et al. Paper Title[C]//Conference Name. Year: pages.
```

更多可以参考`核心操作.rst `     中的实际例子。

#### 文档要求

- 使用标准的 RST 语法
- 包含算法的核心思想描述
- 详细说明类变量和函数
- 提供完整的使用示例
- 添加相关的论文引用
- 使用正确的代码块格式
- 添加必要的注意事项

### 2. 更新使用例子.rst

在 `docs/source/使用例子.rst` 中添加您的算法使用示例。

#### 文档要求

- 说明使用的数据集
- 展示使用示范脚本名称
- 提供代码示例
- 包含结果可视化
- 说明结果保存位置

### 3. 添加图片资源

如果您的算法需要展示结果图片：

1. 将图片放在 `docs/source/media/` 目录下
2. 图片命名建议：`algorithm_name_result.gif` 或 `algorithm_name_result.png`
3. 在文档中正确引用图片路径

**示例**：

```rst
重构结果：

.. image:: ./media/your_algorithm_result.gif
   :target: ./media/your_algorithm_result.gif
   :alt: your_algorithm_result
```

### 4. 文档间链接

在 `核心操作.rst` 中链接到 `使用例子.rst` 中的对应 section：

\*\* 使用标准 RST 引用\*\*

在 `使用例子.rst` 中为您的算法 section 添加锚点：

```rst
您的算法名称
--------------------------

.. _your-algorithm-usage:

使用\ ``dataset_name``\ 数据集...
```

然后在 `核心操作.rst` 中引用：

```rst
使用示例
~~~~~~~~

查看完整的使用示例，请参考：:ref:`alt-text <your-algorithm-usage>`。
```

## 提交 Pull Request

### 1. 提交代码

在完成开发和文档编写后，提交您的更改：

```bash
# 提交算法实现
git add SpikeCV/spkProc/your_module/your_algorithm.py

# 提交使用示范文件
git add SpikeCV/examples/test_your_algorithm.py

# 提交文档更新
git add docs/source/核心操作.rst
git add docs/source/使用例子.rst

# 提交图片资源（如果有）
git add docs/source/media/your_result.gif

# 提交依赖更新（如果有）
git add pyproject.toml

# 提交更改
git commit -m "Add YourAlgorithm: brief description of the algorithm"
```

#### Commit 消息规范

- 使用清晰、简洁的描述
- 首行简短描述（50字符以内）
- 空一行后添加详细描述
- 引用相关的 Issue（如果有）

示例：

```
feature(Add SNNTracker): spiking neural network based multi-object tracking

- Implement SNNTracker class with STP filter, DNF detection, and STDP clustering
- Add comprehensive documentation in 核心操作.rst
- Add usage examples in 使用例子.rst
- Add test_snntracker.py for algorithm validation

Closes #123
```

### 2. 推送到远程仓库

```bash
git push origin feature/your-algorithm-name
```

### 3. 创建 Pull Request

1. 访问您 Fork 的仓库页面
2. 点击 "New Pull Request" 按钮
3. **选择正确的分支**：
   - **base repository**: 选择 `Zyj061/SpikeCV`（原始仓库）
   - **base branch**: 选择 `main` 分支
   - **head repository**: 选择您自己的 Fork 仓库
   - **compare branch**: 选择您的功能分支（如 `feature/your-algorithm-name`）
4. 填写 PR 标题和描述
5. 关联相关的 Issue（如果有）
6. 提交 PR

#### PR 描述模板

```markdown
## PR类型
- [ ] 新算法
- [ ] 算法 Bug 修复
- [ ] 文档改进
- [ ] CLI 功能改进

## PR描述
简要描述您的算法功能、核心思想或修复的问题。

## 更改内容
- [ ] 添加了新的算法实现
- [ ] 修复了算法 Bug
- [ ] 添加了使用示范文件
- [ ] 更新了核心操作.rst
- [ ] 更新了使用例子.rst
- [ ] 添加了必要的图片资源
- [ ] 更新了 pyproject.toml 依赖（如有需要）
- [ ] 添加或修改了 CLI 命令（如有）

## CLI 检查清单（如适用）
- [ ] 算法核心库导入已放置在函数内部 (Lazy Load)
- [ ] 中间过程输出、进度条均通过 `stderr` 输出（例如 `typer.echo(..., err=True)`、`tqdm(..., file=sys.stderr)` 或 I/O 重定向）
- [ ] 仅在 `agent_used` 为 True 时，通过 `stdout` 输出结构化 JSON 结果
- [ ] CLI 基本功能：通过 `spikecv proc ...` 成功运行
- [ ] 异常处理：已测试路径不存在或参数错误时的报错提示
- [ ] 设备兼容性：已在 [CPU/CUDA] 环境下验证
- [ ] Agent 模式：通过 `--agent-used` 验证了 JSON 输出格式正确性

## 算法验证
描述您如何验证算法的正确性：
- 本机的环境
- 在数据集 X 上演示/测试了算法
- 验证了输出结果的正确性
- 确认了文档的完整性
- 确保使用示范文件可以正常运行
- 测试了不同参数配置

## CLI 测试命令（如适用）
`spikecv proc your-cmd -d /path/to/data`

## 相关 Issue
关联相关的 Issue 编号（如果有）。
```

## 代码审查流程

1. 自动化检查

您的 PR 会自动运行 Github Actions 部署的检查。

1. 人工审查

维护者会审查您的代码，主要关注：

- 代码质量和可读性
- 算法的正确性
- 文档的完整性和准确性
- 与现有代码的兼容性

1. 修改要求

如果审查过程中发现问题，维护者会：

- 在 PR 中添加评论
- 请求您进行修改
- 提供具体的修改建议

1. 合并

当 PR 通过所有检查和审查后，维护者会：

- 将您的代码合并到 `main` 分支
- 关闭相关的 Issue（如果有）
- 感谢您的贡献

## 获取帮助

如果您在贡献过程中遇到问题：

1. 查看 [Issues](https://github.com/your-repo/issues) 中是否有类似问题
2. 在 Issue 中提问，描述清楚您的问题
3. 参考现有的代码和文档
4. 联系维护者获取帮助

## 许可证

通过贡献代码，您同意您的贡献将使用与 SpikeCV 项目相同的许可证。

## 再次感谢

感谢您对 SpikeCV 项目的贡献！您的贡献将帮助更多人使用和改进脉冲相机视觉算法。
