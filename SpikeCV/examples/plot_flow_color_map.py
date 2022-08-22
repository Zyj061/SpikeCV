# -*- coding: utf-8 -*- 
# @Time : 2022/7/22
# @Author : Rui Zhao
# @File : phm_encoding.py

import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt

from visualization.optical_flow_visualization import vis_color_map

color_map_list = vis_color_map(use_cv2=False)
plt.figure(figsize=[12, 4])
fontsize = 18

plt.subplot(1, 3, 1)
plt.imshow(color_map_list[0])
plt.axis('off')
plt.title('normal', fontsize=fontsize)

plt.subplot(1, 3, 2)
plt.imshow(color_map_list[1])
plt.axis('off')
plt.title('scflow', fontsize=fontsize)

plt.subplot(1, 3, 3)
plt.imshow(color_map_list[2])
plt.axis('off')
plt.title('evflow', fontsize=fontsize)

# plt.show()
plt.savefig('flow_color_map.png')