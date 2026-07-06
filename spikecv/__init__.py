import sys

# 提供小写别名 `spikecv`，同时保留原有的 `SpikeCV` 包名
sys.modules["spikecv"] = sys.modules[__name__]
