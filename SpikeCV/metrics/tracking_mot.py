# -*- coding: utf-8 -*- 
# @Time : 2022/7/15 14:24 
# @Author : Yajing Zheng
# @File : tracking_mot.py
import motmetrics as mm


class TrackingMetrics:

    def __init__(self, res_filepath, **dataDict):
        self.gt_file = dataDict.get('labeled_data_dir')
        self.gt = mm.io.loadtxt(self.gt_file, fmt="mot15-2D", min_confidence=0.5)
        model_res = mm.io.loadtxt(res_filepath, fmt="mot15-2D")

        # 根据GT和自己的结果，生成accumulator，distth是距离阈值
        self.acc = mm.utils.compare_to_groundtruth(self.gt, model_res, 'iou', distth=0.6)
        self.mh = mm.metrics.create()

        # 打印单个accumulator
        # mh模块中有内置的显示格式

    def get_results(self):
        summary = self.mh.compute_many([self.acc, self.acc.events.loc[0:1]],
                                       metrics=mm.metrics.motchallenge_metrics,
                                       names=['full', 'part'])

        strsummary = mm.io.render_summary(
            summary,
            formatters=self.mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )

        print(strsummary)
