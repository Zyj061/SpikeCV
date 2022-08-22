## 简介

### 脉冲相机

超高速脉冲相机模拟灵长类视网膜编码原理，每个像素位置直接根据光强变化独立发放脉冲。如下图(b)中显示，当光子累计达到预先设定好的阈值时就产生脉冲比特流，1表示发放脉冲，0表示无脉冲。与图(a)中展示的传统相机成像区别是，常见的数码相机按照一个较低的固定频率产生静止图片序列，无法有效记录光的高速变化过程，例如拍摄场景存在高速运动时，产生的图片会存在运动模糊。而脉冲相机目前可按照40KHZ的频率将光信号转化为脉冲比特流，记录的视觉时空变化较为完整，可应用超高速视觉场景的采集、表示、编码、检测、跟踪和识别等任务。 

![spike_camera](./docs/assets/spike_camera.png)

> 上图出自论文：Huang T, Zheng Y, Yu Z, et al. 1000× Faster Camera and Machine Vision with Ordinary Devices[J]. Engineering, 2022. 更多关于脉冲相机的细节请参考这篇文章。 

更多脉冲相机的相关文章请参见 [Publications.md](https://git.openi.org.cn/Cordium/SpikeCV/src/branch/master/Publications.md)。

若有问题可通过以下邮箱进行咨询：

* spikecv@outlook.com

## 开源许可证

SpikeCV 开源项目是在`Apache 2.0 许可证`下授权的，请参考[License.md](https://git.openi.org.cn/Cordium/SpikeCV/src/branch/master/License.md)查阅许可详情。

## 开发单位

<img src="./docs/assets/pku_logo2.png" alt="pku_logo" style="zoom: 100%;" />

**SpikeCV**是由北京大学牵头组织的开源项目。

