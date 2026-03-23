# Contributing to SpikeCV

[中文版](./CONTRIBUTING.md)

Thank you for your interest in the SpikeCV project! We welcome the following forms of contributions:

- **New Algorithms**: Add new spike-based vision algorithms
- **Algorithm Bug Fixes**: Fix issues in existing algorithms
- **Documentation Improvements**: Enhance algorithm documentation and usage instructions

## Table of Contents

- [Algorithm Contribution Workflow](#algorithm-contribution-workflow)
- [Code Structure Requirements](#code-structure-requirements)
- [Dependency Management](#dependency-management)
- [Documentation Contribution](#documentation-contribution)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Code Review Process](#code-review-process)

## Algorithm Contribution Workflow

### 1. Fork the Project

First, you need to fork the SpikeCV repository to your GitHub account.

### 2. Create a Branch

Create a new feature branch from the `main` branch:

```bash
git checkout -b feature/your-algorithm-name
```

Branch naming suggestions:

- New algorithm: `feature/algorithm-name`
- Bug fix: `fix/algorithm-bug-description`
- Documentation update: `docs/algorithm-docs-update`

### 3. Download Dependencies and Configure the Environment According to `README_en.md`

### 4. Develop the New Algorithm

Develop your algorithm in the corresponding module directory:

```
SpikeCV/
├── SpikeCV/
│   ├── spkProc/          # Algorithm implementations
│   │   ├── filters/          # Filters
│   │   ├── reconstruction/   # Reconstruction algorithms
│   │   ├── detection/        # Object detection
│   │   ├── tracking/         # Object tracking
│   │   ├── recognition/      # Object recognition
│   │   └── motion/           # Motion estimation
│   ├── examples/             # Usage example files
│   ├── metrics/              # Evaluation metrics
│   ├── spkData/              # Data loading
│   ├── utils/                # Utility functions
│   └── visualization/        # Visualization tools
```

## Algorithm Code Structure Requirements

### 1. Algorithm Implementation File (\*.py)

#### File Location

Place the code in the corresponding directory based on the algorithm type:

- Filters: `SpikeCV/spkProc/filters/`
- Reconstruction: `SpikeCV/spkProc/reconstruction/`
- Object Detection: `SpikeCV/spkProc/detection/`
- Object Tracking: `SpikeCV/spkProc/tracking/`
- Object Recognition: `SpikeCV/spkProc/recognition/`
- Motion Estimation: `SpikeCV/spkProc/motion/`

#### File Naming

- Use lowercase letters and underscores: `your_algorithm.py`
- Avoid special characters and spaces

#### Code Standards

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
    Brief description of the algorithm
    
    Detailed description of the algorithm's core idea, input/output, main parameters, etc.
    """
    
    def __init__(self, param1, param2, device, **kwargs):
        """
        Initialize the algorithm
        
        Parameters
        ----------
        param1 : type
            Description of param1
        param2 : type
            Description of param2
        device : torch.device
            Device type to use, 'cpu' or 'cuda'
        **kwargs : dict
            Other optional parameters
        """
        self.param1 = param1
        self.param2 = param2
        self.device = device
        
    def process(self, spikes):
        """
        Process input data
        
        Parameters
        ----------
        spikes : type
            Description of input data
            
        Returns
        -------
        output_data : type
            Description of output data
        """
        # Implement algorithm logic
        pass
```

#### Code Style Suggestions

- Follow PEP 8 code style
- Add necessary comments and docstrings
- Use type hints
- Handle edge cases and errors
- Ensure code readability and maintainability

#### Progress Display Suggestions

It is recommended to use `tqdm` to display progress during algorithm execution to enhance user experience. `tqdm` is a core dependency of the project and can be used directly.

**Usage Scenarios**:

- Display frame progress when processing multiple frames
- Display file progress when processing multiple data files
- Display iteration progress during iterative processing
- Progress display for any time-consuming operations

**Example Code**:

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

**Common tqdm Parameters**:

```python
# Basic usage
for i in tqdm(range(100)):
    pass

# Add description
for i in tqdm(range(100), desc="Processing"):
    pass

# Display additional info
for i in tqdm(range(100), desc="Processing", unit="frame"):
    pass

# Nested progress bars
for file in tqdm(files, desc="Files"):
    for frame in tqdm(frames, desc=f"Processing {file}", leave=False):
        process_frame(frame)
```

**Important Notes**:

- Use tqdm for long-running operations to keep users informed of progress
- Use clear descriptions (`desc` parameter)
- Set the unit appropriately (`unit` parameter)
- Use `leave=False` for nested progress bars to avoid clutter
- In Jupyter notebooks, you can use `from tqdm.notebook import tqdm`

### 2. Usage Example File (test\_\*.py)

#### File Nature Explanation

Although these files are named with the `test_*.py` format, they are actually **usage example scripts**. They are not unit tests but complete usage examples designed to help users understand how to use each algorithm correctly.

#### File Location

Place the usage example files in the `SpikeCV/examples/` directory:

```
examples/
├── test_your_algorithm.py
```

#### Usage Example File Structure

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
from spkData.load_dat import SpikeStream, data_parameter_dict
from spkProc.your_module.your_algorithm import YourAlgorithm

"""
Demonstrates how to use your algorithm

This function shows how to use the YourAlgorithm class for a complete usage example,
including data loading, algorithm initialization, execution, and result saving.
This is a usage example, not a unit test.
"""

# Load data
data_filename = "path/to/your/data.dat" # e.g. recVidarReal2019/classA/car-100kmh
label_type = 'your_label_type' # e.g. 'raw'

paraDict = data_parameter_dict(data_filename, label_type)
pprint(paraDict)

spike_stream = SpikeStream(**dataDict)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Algorithm initialization and execution
algorithm = YourAlgorithm(
    param1=value1,
    param2=value2,
    device=device
)
results = algorithm.process(spikes)

# Save results to files, images, videos, etc.
if not os.path.exists('results'):
    os.makedirs('results')

# ...

print('Demo completed successfully!')

```

**Note**: Usage example files are located in the `SpikeCV/examples/` directory and need to use `sys.path.append("..")` to access other modules in the `SpikeCV/` directory (such as `spkProc`, `spkData`, `utils`, `visualization`, etc.).

#### Usage Example Requirements

- Usage example filenames start with `test_` (to maintain naming consistency)
- Include a complete usage example demonstrating typical usage scenarios of the algorithm
- Show result visualization and saving methods
- Add detailed comments explaining each step
- Ensure the code can run directly (provided the correct data is available)
- Demonstrate the practical application effect of the algorithm

## Dependency Management

### Project Environment Management

For detailed dependency management guidelines regarding `pyproject.toml`, please refer to: [`.github/CONTRIBUTING/dependency_guide_en.md`](.github/CONTRIBUTING/dependency_guide.md)

This guide includes:

- Dependency classification (core dependencies and optional dependencies)
- How to add new dependencies
- Local dependency configuration (handling version conflicts)
- Dependency version specifications
- Best practices for dependency management
- Methods to verify dependencies
- Process for updating dependencies
- Frequently Asked Questions

### Environment Dependency Recording

To avoid "it works on my machine" issues, we strongly recommend recording your actual environment dependencies before submitting a PR:

#### 1. Record Python Version

Create a `.python-version` file in your algorithm folder:

```
3.10
```

**Explanation**:

- This file is used by version management tools like pyenv
- Ensures other developers use the same Python version
- Avoids issues caused by Python version differences

#### 2. Record Dependency Package Versions

Run the following command in your algorithm folder:

```bash
pip freeze > requirement.txt
```

**Explanation**:

- This file lists all installed packages in your environment along with their exact versions
- Helps other developers reproduce your environment
- Avoids issues caused by dependency version differences

**Example File Structure**:

```
SpikeCV/
├── spkProc/
│   └── tracking/
│       └── SNN_Tracker/
│           ├── .python-version      # Python version
│           ├── requirement.txt      # Dependency package versions
│           ├── snn_tracker.py       # Algorithm implementation
│           └── __init__.py
```

**How to Use**:

Other developers can use the following commands to reproduce your environment:

```bash
# Set Python version using .python-version
pyenv local 3.10

# Install dependencies using requirement.txt
pip install -r requirement.txt

# Or use uv for faster installation
uv pip install -r requirement.txt
```

**Important Notes**:

- `pip freeze` lists all installed packages, including core and optional dependencies
- If your algorithm has special dependency requirements, you can add comments in requirement.txt
- It is recommended to run `pip freeze` in a virtual environment to avoid including system-level packages
- If using conda, you can also export the environment using `conda env export > environment.yml`
- **Although you can record dependency packages, this is only a workaround and helps coordinate environment dependencies. We strongly recommend managing all environment dependencies centrally through the project root's `pyproject.toml`. If you have any dependency changes (additions/version restrictions) after project initialization, you should update `pyproject.toml` before submitting a PR.**

## Documentation Contribution

### Types of Documentation Contributions

We accept the following types of documentation contributions:

1. **New Algorithm Documentation**: Add complete documentation for newly implemented algorithms
2. **Algorithm Documentation Improvements**: Enhance existing algorithm documentation, including:
   - Correcting errors or inaccuracies in the documentation
   - Supplementing missing algorithm descriptions or parameter explanations
   - Adding more usage examples
   - Improving documentation readability and structure
3. **Usage Example Improvements**: Enhance algorithm usage example files

### 1. Update core_operations.rst

Add your algorithm documentation to `docs/source/核心操作.rst`.

#### Documentation Structure Reference

```rst
Locate the corresponding algorithm category (e.g., Reconstruction Algorithms)
--------

Based on Your Algorithm Name
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``spkProc.your_module.your_algorithm``\ [Your algorithm description]\ ``YourAlgorithm``\ 。[Briefly describe the core idea of the algorithm]。To use [Your algorithm], [Describe the Process to Initialize]：

.. code-block:: python

   from spkProc.your_module.your_algorithm import YourAlgorithm
   import torch

   device = torch.device('cuda')
   algorithm = YourAlgorithm(param1=value1, param2=value2, device=device)

Variables in the YourAlgorithm Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The class ``YourAlgorithm`` in ``your_algorithm.py`` has the following variables:

* ``param1``\ ：Description of param1
* ``param2``\ ：Description of param2
* ``device``\ ：Device type to use, ``cpu`` or ``cuda``
* ``variable1``\ ：Description of variable1
* ``variable2``\ ：Description of variable2

Functions in the YourAlgorithm Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``your_algorithm.YourAlgorithm`` contains the following functions:

#. 
   ``__init__(param1, param2, device, **kwargs)``\ ：Initialize the algorithm instance.

   Parameter description:

   * ``param1``\ ：Description of param1
   * ``param2``\ ：Description of param2
   * ``device``\ ：Device type, ``cpu`` or ``cuda``
   * ``**kwargs``\ ：Other optional parameters

   Example call:

   .. code-block:: python

      from spkProc.your_module.your_algorithm import YourAlgorithm
      import torch

      device = torch.device('cuda')
      algorithm = YourAlgorithm(param1=value1, param2=value2, device=device)

#. 
   ``process(input_data)``\ ：Process input data and return results.

   Parameter description:

   * ``input_data``\ ：Description of input data

   Return value:

   * ``output_data``\ ：Description of output data

   Example call:

   .. code-block:: python

      results = algorithm.process(spikes)

Related Papers
~~~~~~~~

For more details about the algorithm, please refer to the papers:

#. Author1, Author2, et al. Paper Title[J]. Journal Name, Year.
#. Author1, Author2, et al. Paper Title[C]//Conference Name. Year: pages.
```

For more examples, refer to the actual examples in `核心操作.rst`.

#### Documentation Requirements

- Use standard RST syntax
- Include a description of the algorithm's core idea
- Detail class variables and functions
- Provide complete usage examples
- Add relevant paper citations
- Use correct code block formatting
- Add necessary notes

### 2. Update usage_examples.rst

Add your algorithm usage examples to `docs/source/使用例子.rst`.

#### Documentation Requirements

- Describe the dataset used
- Show the name of the usage example script
- Provide code examples
- Include result visualization
- Describe where results are saved

### 3. Add Image Resources

If your algorithm requires result images:

1. Place images in the `docs/source/media/` directory
2. Image naming suggestion: `algorithm_name_result.gif` or `algorithm_name_result.png`
3. Correctly reference the image paths in the documentation

**Example**:

```rst
Reconstruction results:

.. image:: ./media/your_algorithm_result.gif
   :target: ./media/your_algorithm_result.gif
   :alt: your_algorithm_result
```

### 4. Cross-Linking Documentation

Link from `core_operations.rst` to the corresponding section in `usage_examples.rst`:

**Use standard RST references**

Add an anchor for your algorithm section in `usage_examples.rst`:

```rst
Your Algorithm Name
--------------------------

.. _your-algorithm-usage:

Using the ``dataset_name`` dataset...
```

Then reference it in `core_operations.rst`:

```rst
Usage Examples
~~~~~~~~

For complete usage examples, please refer to: :ref:`alt-text <your-algorithm-usage>`.
```

## Submitting a Pull Request

### 1. Commit Code

After completing development and documentation, commit your changes:

```bash
# Commit algorithm implementation
git add SpikeCV/spkProc/your_module/your_algorithm.py

# Commit usage example file
git add SpikeCV/examples/test_your_algorithm.py

# Commit documentation updates
git add docs/source/核心操作.rst
git add docs/source/使用例子.rst

# Commit image resources (if any)
git add docs/source/media/your_result.gif

# Commit dependency updates (if any)
git add pyproject.toml

# Commit changes
git commit -m "Add YourAlgorithm: brief description of the algorithm"
```

#### Commit Message Guidelines

- Use clear, concise descriptions
- First line short description (within 50 characters)
- Leave a blank line, then add detailed description
- Reference related issues (if any)

Example:

```
feature(Add SNNTracker): spiking neural network based multi-object tracking

- Implement SNNTracker class with STP filter, DNF detection, and STDP clustering
- Add comprehensive documentation in core_operations.rst
- Add usage examples in usage_examples.rst
- Add test_snntracker.py for algorithm validation

Closes #123
```

### 2. Push to Remote Repository

```bash
git push origin feature/your-algorithm-name
```

### 3. Create a Pull Request

1. Go to your forked repository page
2. Click the "New Pull Request" button
3. **Select the correct branches**:
   - **base repository**: Select `Zyj061/SpikeCV` (original repository)
   - **base branch**: Select the `main` branch
   - **head repository**: Select your forked repository
   - **compare branch**: Select your feature branch (e.g., `feature/your-algorithm-name`)
4. Fill in the PR title and description
5. Link related issues (if any)
6. Submit the PR

#### PR Description Template

```markdown
## PR Type
- [ ] New Algorithm
- [ ] Algorithm Bug Fix
- [ ] Documentation Improvement

## PR Description
Briefly describe your algorithm's functionality, core ideas, or the issues fixed.

## Changes Made
- [ ] Added new algorithm implementation
- [ ] Fixed algorithm bug
- [ ] Added usage example file
- [ ] Updated core_operations.rst
- [ ] Updated usage_examples.rst
- [ ] Added necessary image resources
- [ ] Updated pyproject.toml dependencies (if required)

## Algorithm Validation
Describe how you validated the algorithm's correctness:
- Environment used
- Demonstrated/tested the algorithm on dataset X
- Verified the correctness of output results
- Confirmed documentation completeness
- Ensured usage example files run correctly
- Tested different parameter configurations

## Related Issues
Link related issue numbers (if any).
```

## Code Review Process

1. **Automated Checks**

   Your PR will automatically run checks deployed via GitHub Actions.

2. **Manual Review**

   Maintainers will review your code, focusing on:

   - Code quality and readability
   - Algorithm correctness
   - Documentation completeness and accuracy
   - Compatibility with existing code

3. **Revision Requests**

   If issues are found during review, maintainers will:

   - Add comments on the PR
   - Request modifications
   - Provide specific suggestions for changes

4. **Merging**

   When the PR passes all checks and reviews, maintainers will:

   - Merge your code into the `main` branch
   - Close related issues (if any)
   - Thank you for your contribution

## Getting Help

If you encounter problems during the contribution process:

1. Check [Issues](https://github.com/your-repo/issues) for similar problems
2. Ask questions in Issues, clearly describing your problem
3. Refer to existing code and documentation
4. Contact maintainers for help

## License

By contributing code, you agree that your contributions will be licensed under the same license as the SpikeCV project.

## Thank You Again

Thank you for contributing to the SpikeCV project! Your contributions will help more people use and improve spike camera vision algorithms.