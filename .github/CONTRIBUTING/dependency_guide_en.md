# Dependency Management Guide

This document details the dependency management specifications and best practices for the SpikeCV project.

## 1. Dependency Classification

SpikeCV's dependencies are divided into two categories:

### Core Dependencies

Base dependencies required by all or most algorithms, including:

- torch, torchvision (deep learning frameworks)
- numpy (numerical computation)
- matplotlib, scipy (scientific computing and visualization)
- scikit-learn, scikit-image (machine learning and image processing)
- tensorboardX, tqdm (training and progress display)
- other vision processing libraries

⚠️
**Important**: Modifications to core dependencies must be handled carefully!

- Core dependencies affect all algorithms; changes must not break existing functionality
- Changing core dependency versions requires thorough testing and validation
- It is recommended to explain the reason and impact of the change in detail in the PR
- Modifications to core dependencies may require special approval from maintainers
- If your algorithm only needs a specific dependency, please add it to optional dependencies instead

### Optional Dependencies

Optional dependencies grouped by algorithm type:

- `reconstruction`: dependencies for reconstruction algorithms
- `depth_estimation`: dependencies for depth estimation
- `optical_flow`: dependencies for optical flow estimation
- `detection`: dependencies for object detection
- `recognition`: dependencies for object recognition
- `tracking`: dependencies for object tracking (e.g., motmetrics, filterpy)
- `docs`: dependencies for documentation building (e.g., sphinx)
- `test`: dependencies for testing tools (e.g., pytest)

## 2. Adding New Dependencies

### Adding Core Dependencies

⚠️
**Strongly recommended**: Prefer adding dependencies to optional dependencies!

- Core dependencies affect all algorithms, adding unnecessary dependencies bloats the project
- If your algorithm only needs a package, add it to the corresponding optional dependency group
- Only consider adding to core dependencies if the package is truly required by all algorithms
- Before adding a core dependency, please discuss it in an Issue and obtain maintainer approval

If your algorithm truly requires a new core dependency (after discussion and approval):

```toml
[project]
dependencies = [
    "numpy<2.0",
    "torch",
    "your-new-package>=1.0.0",  # add new core dependency
]
```

### Adding Optional Dependencies

If your algorithm only needs dependencies for specific functionality:

```toml
[project.optional-dependencies]
# Add dependencies for a new algorithm type
your_algorithm_type = [
    "package1>=1.0.0",
    "package2>=2.0.0",
]

# Or add to an existing algorithm type
tracking = [
    "motmetrics>=1.2.0",
    "filterpy>=1.4.0",
    "your-tracking-package>=1.0.0",  # add new tracking-related dependency
]
```

### Local Dependency Configuration (Special Cases)

**Special case handling**: If your algorithm's dependencies conflict with core dependencies

Although modifying core dependencies is not recommended, if your algorithm absolutely requires a specific version of a package that conflicts with core dependencies, you can create a local dependency configuration file in your algorithm's directory.

**Applicable scenarios**:

- Your algorithm requires a specific version of a package that conflicts with core dependencies
- Temporary solution while waiting for core dependencies to be updated
- Experimental algorithm that does not affect other functionality

**Implementation**:

Create a `requirements.txt` or `pyproject.toml` file in your algorithm's directory:

```
SpikeCV/
├── SpikeCV/
│   ├── spkProc/
│   │   ├── AlgorithmType/          # algorithm type directory (e.g., tracking)
│   │   │   ├── your_algorithm/    # your algorithm directory
│   │   │   │   ├── your_algorithm.py
│   │   │   │   ├── requirements.txt      # local dependency file
│   │   │   │   └── __init__.py
```

**requirements.txt example**:

```txt
# Specific version dependencies required by your algorithm
package-name==1.5.0
another-package>=2.0.0,<3.0.0
```

**pyproject.toml example**:

```toml
[project]
dependencies = [
    "package-name==1.5.0",
    "another-package>=2.0.0,<3.0.0",
]
```

You can also use tools like `uv` to help export dependencies: `uv export --format requirements-txt > requirements.txt`.

**Usage**:

```bash
# Install local dependencies before running your algorithm
cd SpikeCV/spkProc/AlgorithmType/your_algorithm
pip install -r requirements.txt
# or
pip install -e .
```

**Important Notes**:

- Local dependencies only affect your algorithm, not others
- Clearly document how to install the local dependencies
- It is recommended to explain the conflict and solution in the PR
- **In the long term, you should discuss with maintainers how to unify dependency versions**

**Not recommended**:

- Do not dynamically modify sys.path in algorithm code to resolve dependency conflicts
- Do not force installation of specific dependency versions within the algorithm
- Do not override core dependencies in the global environment

## 3. Dependency Version Specifications

### Version Specification Methods

```toml
# Exact version
"package==1.0.0"

# Minimum version
"package>=1.0.0"

# Version range
"package>=1.0.0,<2.0"

# Exclude specific version
"package>=1.0.0,!=1.5.0"
```

### Common Version Constraints

```toml
# numpy version constraint (to avoid incompatibility with existing code)
"numpy>=1.24.0,<2.0"

# torch version recommendation
"torch>=2.0.0"

# No version specified (not recommended, may cause compatibility issues)
"package"  # ❌ not recommended
```

## 4. Best Practices for Dependency Management

### Minimize Dependencies

- Only add necessary dependencies
- Avoid duplicate dependencies (packages with similar functionality)
- Prefer using existing core dependencies

### Version Compatibility

- Test compatibility with different versions
- Follow semantic versioning conventions
- Consider compatibility with existing code

### Dependency Grouping

```toml
[project.optional-dependencies]
# Group by algorithm type
reconstruction = [
    "package1>=1.0.0",
]

# Group by function
dev = [
    "black>=23.0.0",     # code formatting
    "flake8>=6.0.0",    # code linting
    "mypy>=1.0.0",      # type checking
]
```

## 5. Verifying Dependencies

### Check for Dependency Conflicts

```bash
# Check if dependencies conflict
pip check
```

### View Dependency Tree

```bash
# View dependency relationships
pipdeptree
```

### Test Installation

```bash
# Test installing core dependencies
pip install -e .

# Test installing a specific module
pip install -e ".[your_algorithm_type]"

# Test installing all dependencies
pip install -e ".[reconstruction,depth_estimation,optical_flow,detection,recognition,tracking,docs,test]"
```

## 6. Process for Updating Dependencies

1. **Determine dependency type**
   - Decide if it is a core or optional dependency
   - Identify the algorithm type it belongs to
   - Check for version conflicts with core dependencies
2. **Add dependency to pyproject.toml**
   - Add to the appropriate `[project.dependencies]` or `[project.optional-dependencies]` section
   - Specify an appropriate version range
   - If version conflicts exist, consider using a local dependency configuration
3. **Test installation**
   ```bash
   # Test standard dependency installation
   pip install -e ".[your_algorithm_type]"

   # If using local dependencies
   cd SpikeCV/spkProc/AlgorithmType/your_algorithm
   pip install -r requirements.txt
   ```
4. **Verify functionality**
   - Ensure the dependency can be imported correctly
   - Test that the algorithm works as expected
   - Confirm that other algorithms' functionality is not affected
5. **Commit changes**
   ```bash
   git add pyproject.toml
   git commit -m "Add dependency: your-package for your-algorithm"
   ```

## 7. Frequently Asked Questions

### Q: How to choose a dependency version?

A: It is recommended to use a minimum version constraint (`>=1.0.0`) and verify compatibility during testing. For critical dependencies, a version range (`>=1.0.0,<2.0`) can be specified.

### Q: Do I need to add all dependencies to pyproject.toml?

A: Only direct dependencies need to be added. Indirect dependencies will be resolved and installed automatically by pip.

### Q: How to handle dependency conflicts?

A: If your algorithm's dependencies conflict with core dependencies, you have several options:

1. **Prefer optional dependencies**: Add your dependencies to the corresponding optional dependency group
2. **Use local dependency configuration**: Create a `requirements.txt` or `pyproject.toml` file in your algorithm's directory, affecting only your algorithm
3. **Contact maintainers**: Discuss in an Issue to find a unified solution
4. **Wait for core dependencies to be updated**: If the conflict is temporary, you can wait for core dependencies to be updated

**Recommended order**: optional dependencies → local dependency configuration → contact maintainers

### Q: Can I remove old dependencies?

A: If a dependency is no longer used by any algorithm, it can be removed. However, ensure that it does not affect other functionality.

### Q: Can I modify core dependencies?

A: **Modifying core dependencies is not recommended!** Core dependencies affect all algorithms, and changes may break existing functionality. If your algorithm requires a specific dependency, please add it to the corresponding optional dependency group. Only consider modifying core dependencies if the package is truly required by all algorithms, and discuss it in an Issue first to obtain maintainer approval.

### Q: When should I use core dependencies instead of optional dependencies?

A: Only consider adding to core dependencies in the following cases:

- The dependency is used by multiple algorithm types
- The dependency is part of the project infrastructure (e.g., logging, configuration management, etc.)
- After discussion, maintainers agree to make it a core dependency
- Otherwise, always use optional dependencies

## 8. Example: Adding Dependencies for a New Algorithm

Suppose you add a new reconstruction algorithm `YourReconstruction` that requires the `your-recon-package` dependency:

```toml
[project.optional-dependencies]
# Add to the existing reconstruction dependency group
reconstruction = [
    "your-recon-package>=1.0.0",  # new dependency
]
```

Then, describe it in the PR:

```
## Changes Made
- [ ] Added new algorithm implementation
- [ ] Added usage example file
- [ ] Updated core_operations.rst
- [ ] Updated usage_examples.rst
- [ ] Added necessary image resources
- [-] Updated pyproject.toml dependencies

## Dependency Updates
- Added `your-recon-package>=1.0.0` to the `reconstruction` optional dependency group
```