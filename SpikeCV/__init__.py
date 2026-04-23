# -*- coding: utf-8 -*- 
# @Time : 2026/4/20 19:00 
# @Author : Singa Xu
# @Email: xusigna@pku.edu.cn
# @File : SpikeCV/__init__.py
"""
SpikeCV: An open-source framework for Spiking Computer Vision.
"""

# 放在 SpikeCV/__init__.py 顶部，这能让用户使用 `import spikecv` 来导入模块，同时内部仍然使用 `SpikeCV` 作为实际的包名。
import sys
import importlib.util
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec

class AliasMetaFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if not (fullname == 'spikecv' or fullname.startswith('spikecv.')):
            return None
        real_name = fullname.replace('spikecv', 'SpikeCV', 1)
        # 如果 path 不为 None，则说明是子包查找，可以传递 path
        spec = importlib.util.find_spec(real_name, path)
        if spec:
            spec.name = fullname
            return spec
        return None

# 安装 hook：确保它比默认 finder 更早执行
sys.meta_path.insert(0, AliasMetaFinder())

