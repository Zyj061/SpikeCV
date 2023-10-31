# -*- coding: utf-8 -*- 
# @Time : 2023/3/3 14:14 
# @Author : Yajing Zheng
# @Email: yj.zheng@pku.edu.cn
# @File : encoder.py
import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)