import sys
import SpikeCV

# 将小写的 spikecv 代理给大写的 SpikeCV
sys.modules[__name__] = SpikeCV
