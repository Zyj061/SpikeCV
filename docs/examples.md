# 使用例子

在本节教程中，我们将示范如何使用提供的接口结合数据集实现各种任务。所有测试例子都可在`examples`目录下找到对应脚本。

> 注意：在进行测试前，请先确保本地项目目录`SpikeCV.spkData.datasets`下有已经解压的对应数据集，并且指定数据时需具体指定到场景名称。

## 超高速运动场景纹理重构算法

### TFP重构算法

使用`recVidarReal2019`数据集中的汽车高速行驶的场景序列`car-100kmh`进行重构，对应的测试脚本为`test_tfp.py`，其具体实现过程为：

```python
import os
import torch
import sys
sys.path.append("..")

import time

from spkData.load_dat import data_parameter_dict
from spkData.load_dat import VidarSpike
from spkProc.reconstruction.tfp import TFP
from visualization.get_video import obtain_reconstruction_video
from utils import path

# 指定数据序列及任务类型
data_filename = "recVidarReal2019/classA/car-100kmh"
label_type = 'raw'

# 加载数据集属性字典
paraDict = data_parameter_dict(data_filename, label_type)

#加载脉冲数据
vidarSpikes = VidarSpike(**paraDict)

block_len = 500
spikes = vidarSpikes.get_block_spikes(begin_idx=0, block_len=block_len)
```

使用`spkProc.reconstruction.tfp.TFP`进行重构并将结果输出保存至视频文件

```python
device = torch.device('cpu')
reconstructor = TFP(paraDict.get('spike_h'), paraDict.get('spike_w'), device)

st = time.time()
recImg = reconstructor.spikes2images(spikes, half_win_length=20)
ed = time.time()
print('shape: ', recImg.shape, 'time: {:.6f}'.format(ed - st))


filename = path.split_path_into_pieces(data_filename)
if not os.path.exists('results'):
    os.makedirs('results')

result_filename = os.path.join('results', filename[-1] + '_tfp.avi')
obtain_reconstruction_video(recImg, result_filename, **paraDict)
```

重构结果：

![car-100kmh_tfp](./assets/car_reconstruction_tfp.gif)

### TFI重构算法

使用`recVidarReal2019`数据集中的汽车高速行驶的场景序列`car-100kmh`进行重构，对应的测试脚本为`test_tfi.py`，其数据读取的过程与上述TFP重构算法的数据读取相同

使用`spkProc.reconstruction.tfp.TFP`进行重构并将结果输出保存至视频文件

```python
device = torch.device('cpu')
reconstructor = TFP(paraDict.get('spike_h'), paraDict.get('spike_w'), device)

st = time.time()
recImg = reconstructor.spikes2images(spikes, half_win_length=20)
ed = time.time()
print('shape: ', recImg.shape, 'time: {:.6f}'.format(ed - st))


filename = path.split_path_into_pieces(data_filename)
if not os.path.exists('results'):
    os.makedirs('results')

result_filename = os.path.join('results', filename[-1] + '_tfp.avi')
obtain_reconstruction_video(recImg, result_filename, **paraDict)
```

重构结果：

![car-100kmh_tfi](./assets/car_reconstruction_tfi.gif)
