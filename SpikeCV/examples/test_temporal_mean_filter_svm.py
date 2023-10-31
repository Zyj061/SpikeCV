# -*- coding: utf-8 -*-
# @Time : 2022/8/5 18:15
# @Author Homepage : https://github.com/DingJianhao
# File : test_temporal_mean_filter_svm.py

import sys
sys.path.append('../')
sys.path.append('../../')
import torch
import torchvision
import numpy as np
from SpikeCV.spkProc.recognition import svm
from SpikeCV.spkProc.filters.fir_filter import MeanFilter
import SpikeCV.spkData.data_transform as transforms
from SpikeCV.spkData import img_sim_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sb

# Perpare spiking mnist data
data_path = "G:/Dataset/mnist"
gain_amp=0.5
v_th=1.0
timesteps=10

# Use SpikeMNIST dataset
train_dataset = img_sim_dataset.SpikeMNIST(root=data_path,
                                     gain_amp=gain_amp, v_th=v_th, timesteps=timesteps,
                                     train=True,
                                     download=True,
                                     transform=transforms.ToNPYArray()
                                     )

test_dataset = img_sim_dataset.SpikeMNIST(root=data_path,
                                     gain_amp=gain_amp, v_th=v_th, timesteps=timesteps,
                                     train=False,
                                     download=True,
                                     transform=transforms.ToNPYArray()
                                     )

train_data = []
train_label = []
for spk_data, label in train_dataset:
    train_data.append(spk_data)
    train_label.append(label)
train_data = np.array(train_data)
train_label = np.array(train_label)
print('Train data size:', train_data.shape)

test_data = []
test_label = []
for spk_data, label in test_dataset:
    test_data.append(spk_data)
    test_label.append(label)
test_data = np.array(test_data)
test_label = np.array(test_label)
print('Test data size:', test_data.shape)

# Data are classified by Temporal Filtering SVM

filter_svm = svm.TemporalFilteringSVM(filter=MeanFilter(win=timesteps), dual=False)

filter_svm.fit(train_data, train_label)
pred = filter_svm.predict(test_data)
print('Accuracy:', accuracy_score(test_label, pred))

# Present the confusion matrix

cm = confusion_matrix(test_label, pred)
plt.subplots(figsize=(10, 6))
sb.heatmap(cm, annot=True, fmt='g')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SpikingMNIST Confusion Matrix")
plt.show()

