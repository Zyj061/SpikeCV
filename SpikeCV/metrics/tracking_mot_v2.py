# -*- coding: utf-8 -*-
# @Time : 2022/7/15 14:24
# @Author : Yajing Zheng
# @File : tracking_mot.py
import os, pathlib, csv, re
import pandas as pd
import motmetrics as mm

def _coerce_single_path(x):
    """把 list/tuple/PathLike/str 统一为单个字符串路径。"""
    # PathLike -> str
    if isinstance(x, os.PathLike) or isinstance(x, pathlib.Path):
        return os.fspath(x)
    # str 直接返回
    if isinstance(x, str):
        return x
    # list/tuple：取第一个「存在的」路径；都不存在就取第一个元素转成 str
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("gt_file is an empty list/tuple")
        for cand in x:
            if isinstance(cand, (str, os.PathLike, pathlib.Path)):
                p = os.fspath(cand)
                if os.path.exists(p):
                    return p
        return os.fspath(x[0])
    raise TypeError(f"gt_file must be str/PathLike or list/tuple thereof, got: {type(x)}")

def normalize_gt_to_csv(src_path, dst_path, expected_cols=10):
    # 读二进制并规范化：去 BOM、统一换行、去控制符
    raw = open(src_path, 'rb').read()
    if raw.startswith(b'\xef\xbb\xbf'):
        raw = raw[3:]
    txt = raw.replace(b'\r\n', b'\n').replace(b'\r', b'\n').decode('utf-8', errors='ignore')
    txt = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', txt)

    rows = []
    for line in txt.split('\n'):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        # 先按逗号 CSV 解析（尊重引号）
        r = next(csv.reader([s], delimiter=',', quotechar='"', skipinitialspace=True))
        # 如果列数异常，回退为空白分隔
        if len(r) > expected_cols + 10 or len(r) == 1:
            r_ws = re.split(r'\s+', s.replace(',', ' '))
            if 3 <= len(r_ws) <= max(expected_cols, 12) + 10:
                r = r_ws
        # 裁剪/填充列数
        if len(r) < expected_cols:
            r = r + ['-1'] * (expected_cols - len(r))
        elif len(r) > expected_cols:
            r = r[:expected_cols]
        rows.append(r)

    with open(dst_path, 'w', encoding='utf-8', newline='') as g:
        csv.writer(g).writerows(rows)

def robust_load_gt(gt_file, fmt="mot15-2D"):
    path = _coerce_single_path(gt_file)  # << 关键：把 list 等转成单一路径
    expected_cols = 12 if 'mot15' in fmt.lower() else 10 
    cleaned = os.path.splitext(path)[0] + f".clean_{expected_cols}.csv"
    normalize_gt_to_csv(path, cleaned, expected_cols=expected_cols)

    # C 引擎 + 明确逗号分隔，避免 regex 分隔导致引号失效
    _ = pd.read_csv(cleaned, header=None, engine='c')  # 触发一次严格解析，若有问题可直接报错定位
    # NOTE: the load_motchallenge which handle the formatting mot15-2D, failed to 
    # create a df with 12 columns, it create a 10 column df instead, which make the gt file not match the format with dt.
    gt = mm.io.loadtxt(cleaned, fmt=fmt, min_confidence=0.5, sep=",", engine="c", skipinitialspace=True)
    return gt



class TrackingMetrics:

    def __init__(self, res_filepath, **dataDict):
        self.gt_file = dataDict.get('labeled_data_dir')
        # self.gt = mm.io.loadtxt(self.gt_file, fmt="mot15-2D", min_confidence=0.5)
        self.gt = robust_load_gt(self.gt_file, fmt="mot16")  # NOTE: due to the loadtxt bug, we need to adhere to mot16, the mod15-2D format is not supported
        model_res = mm.io.loadtxt(res_filepath, fmt="mot16") 

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
