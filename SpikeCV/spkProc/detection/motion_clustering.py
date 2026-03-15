# -*- coding: utf-8 -*- 
# @Time : 2024/12/6 3:34 
# @Author : Yajing Zheng
# @Email: yj.zheng@pku.edu.cn
# @File : motion_clustering.py
from sklearn.cluster import DBSCAN, OPTICS, SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import numpy as np
import scipy.ndimage.measurements as mnts
import torch


class detect_object:

    def __init__(self, h, w):
        self.h = h
        self.w = w
        params = {'quantile': .3,
                  'eps': .4,
                  'damping': .9,
                  'preference': -200,
                  'n_neighbors': 10,
                  'min_samples': 50,
                  'xi': 0.05,
                  'min_cluster_size': 0.1,
                  'n_cluster': 2}
        self.dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'], metric='precomputed')
        self.optics = OPTICS(min_samples=params['min_samples'], xi=params['xi'],
                             min_cluster_size=params['min_cluster_size'], metric='precomputed')
        self.spectral = SpectralClustering(n_clusters=params['n_cluster'], eigen_solver='arpack',
                                           affinity='precomputed')

    def get_object(self, motion_vector, max_motion=None):
        # motion_vector = StandardScaler.transform(motion_vector)
        if max_motion is None:
            mv_idx = torch.where(torch.logical_or(motion_vector[:, :, 0] != 0, motion_vector[:, :, 1] != 0))
            if len(mv_idx[0]) < 1:
                return None, None
            fire_idx = np.zeros((2, len(mv_idx[0])), dtype=np.int)
            fire_idx[0, :] = mv_idx[0].cpu().numpy()
            fire_idx[1, :] = mv_idx[1].cpu().numpy()

        else:
            max_motion = max_motion.cpu().numpy()
            if max_motion.max() < 1:
                return None, None
            fire_idx = np.array(np.nonzero(max_motion))

        motion_vector = motion_vector.cpu().numpy()

        fire_idx = fire_idx.T
        num_events = len(fire_idx)
        fire_idx_ori = fire_idx
        spatial_vector = StandardScaler().fit_transform(fire_idx)

        motion_array = np.zeros((num_events, 2))
        motion_array[:, 0] = motion_vector[fire_idx[:, 0], fire_idx[:, 1], 0]
        motion_array[:, 1] = motion_vector[fire_idx[:, 0], fire_idx[:, 1], 1]
        motion_array = StandardScaler().fit_transform(motion_array)

        motion_dis = pairwise_distances(motion_array, metric='euclidean')
        spatial_dis = pairwise_distances(spatial_vector, metric='euclidean')
        total_dis = 0.5 * (motion_dis + spatial_dis)

        self.dbscan.fit(total_dis)
        # self.optics.fit(total_dis)
        # self.spectral.fit(total_dis)
        labels = self.dbscan.labels_.astype(np.int)
        # labels_optics = self.optics.labels_.astype(np.int)
        # labels_spetral = self.spectral.labels_.astype(np.int)

        return labels, fire_idx_ori
        # return labels_optics, fire_idx_ori
        # return labels_spetral, fire_idx_ori

    def detection_object_with_motion(self, fireID, clusterId):
        L = np.zeros((self.h, self.w), dtype=np.int)
        L[fireID[:, 0], fireID[:, 1]] = clusterId + 1

        structure = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])

        bboxSlices = mnts.find_objects(L)
        box_num = clusterId.max() + 1
        bbox = np.zeros((box_num, 4))
        for iBox in range(box_num):
            tmpBox = np.array(bboxSlices[iBox])
            begin_X = tmpBox[0].start
            end_X = tmpBox[0].stop
            begin_Y = tmpBox[1].start
            end_Y = tmpBox[1].stop

            bbox[iBox, :] = [begin_X, begin_Y, end_X, end_Y]

        # pprint(bbox)
        return bbox

