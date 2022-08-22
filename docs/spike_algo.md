# SpikeCV 中的核心操作

## 重构算法

`SpikeCV.spkProc.reconstruction`中的高速场景重构算法接口。

### 基于滑动窗口脉冲发放率的纹理重构算法TFP

`spkProc.reconstruction.tfp.py`中基于滑动窗口脉冲发放率的纹理重构算法`TFP`，核心思想是通过滑动窗口中各像素的脉冲发放率来对光照强度进行估计。

使用TFP算法可先通过实例化`skpProc.reconstruction.tfp.py`中的`TFP`类，其所采用的数据类型为*pytorch*的张量形式，定义时仅需传入脉冲阵列的高度`spike_h`，宽度`spike_w`，和所使用处理器`device`。

```python
from spkProc.reconstruction.tfp import TFP
import torch

reconstructor = TFP(spike_h=250, spike_w=400, torch.device('cuda'))
```

#### TFP类中的变量

- `spike_h`：脉冲阵列高度
- `spike_w`：脉冲阵列宽度
- `device`：所使用的处理器类型，`cpu`或者`cuda`

#### TFP类中的函数

- `spikes2images（spikes, half_win_length）`：将spikes整体转换为一段由TFP算法重构的图像。将传入的维度为`(timesteps, spike_h, spike_w)`的脉冲序列`spikes`转化为由窗口长度为($2 \times {\rm half\_win\_length}+1$)TFP算法重构的影像，输出的图像的维度为`(timesteps-(2 x half_win_length), spike_h, spike_w)`。
- `spikes2frame(spikes, key_ts, half_win_length)`：从spikes中获取时刻`key_ts`由TFP算法重构的图像。输入spikes的维度为`(timesteps, spike_h, spike_w)`，TFP算法的窗口长度为($2 \times {\rm half\_win\_length}+1$)，返回的图像维度为`(spike_h, spike_w)`。

### 基于脉冲间隔的纹理重构算法TFI

`spkProc.reconstruction.tfi.py`中基于脉冲间隔的纹理重构算法`TFI`，核心思想是通过各像素所处时刻相邻两次脉冲发放的间隔来推断光照强度。

使用TFI算法可先通过实例化`skpProc.reconstruction.tfi.py`中的`TFI`类，其所采用的数据类型为*pytorch*的张量形式，定义时仅需传入脉冲阵列的高度`spike_h`，宽度`spike_w`，和所使用处理器`device`。

```python
from spkProc.reconstruction.tfp import TFP
import torch

reconstructor = TFP(spike_h=250, spike_w=400, torch.device('cuda'))
```

#### TFP类中的变量

- `spike_h`：脉冲阵列高度
- `spike_w`：脉冲阵列宽度
- `device`：所使用的处理器类型，`cpu`或者`cuda`

#### TFP类中的函数

- `spikes2images（spikes, max_search_half_window=20）`：将spikes整体转换为一段由TFI算法重构的图像。将传入的维度为`(timesteps, spike_h, spike_w)`的脉冲序列`spikes`转化为TFI的重构影像，其中TFI脉冲搜索的最大距离为前向后向各`max_search_half_window`，输出的图像的维度为`(timesteps-(2 x max_search_half_window), spike_h, spike_w)`。
- `spikes2frame(spikes, key_ts, half_win_length=20)`：从spikes中获取时刻`key_ts`由TFI算法重构的图像。输入spikes的维度为`(timesteps, spike_h, spike_w)`，TFI脉冲搜索的最大距离为前向后向各`max_search_half_window`，返回的图像维度为`(spike_h, spike_w)`。
