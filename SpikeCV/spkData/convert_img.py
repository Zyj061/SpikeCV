import numpy as np


def img_to_spike(img, gain_amp=0.5, v_th=1.0, n_timestep=100):
    '''
    脉冲模拟器：图片转脉冲

    :param img: 图片 numpy.ndarray size：h x w
    :param gain_amp: 增益系数
    :param v_th: 阈值
    :param n_timestep: 时间步
    :return: 脉冲数据 numpy.ndarray
    '''

    h, w = img.shape
    if img.max() > 1:
        img = img / 255.
    assert img.max() <= 1.0 and img.min() >= 0.0
    mem = np.zeros_like(img)
    spks = np.zeros((n_timestep, h, w))
    for t in range(n_timestep):
        mem += img * gain_amp
        spk = (mem >= v_th)
        mem = mem * (1 - spk)
        spks[t, :, :] = spk
    return spks.astype(np.float)

