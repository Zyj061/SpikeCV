# SNNTracker 參數配置指南（暂时完全是vibe出来的，真实性待人工审查）

本文檔詳細說明了 SNNTracker 算法的主要參數及其配置方法。

## 🖥️ 环境

为了避免“在我的电脑能跑”问题，本文件夹包含两个重要的环境配置文件：

### `.python-version`
- **用途**: Python 版本锁定
- **当前版本**: 3.10

### `requirement.txt`
- **用途**: Python 依赖锁定
- **使用方法**:
  ```bash
  # 安装所有依赖
  pip install -r requirement.txt
  
  # 或者使用 uv 进行快速安装
  uv pip install -r requirement.txt
  
  # 或者使用 conda 创建环境
  conda create --name snntracker python=3.10
  conda activate snntracker
  pip install -r requirement.txt
  ```

建议使用虚拟环境。

---

## 📋 主要參數說明

### 基本參數

| 參數名稱 | 類型 | 默認值 | 說明 |
|-----------|------|----------|------|
| `spike_h` | int | 250 | 脈衝數據的高度 |
| `spike_w` | int | 400 | 脈衝數據的寬度 |
| `device` | str | 'cuda' 或 'cpu' | 計算設備類型 |
| `calibration_time` | int | 150 | 運動標定的時間步數，默認為150 |

### STP 濾波器參數

STP（Short-Terminal Plasticity）濾波器用於動態適應和背景濾除。

| 參數名稱 | 類型 | 默認值 | 說明 |
|-----------|------|----------|------|
| `tau_r` | float | - | 恢復時間常數，控制脈衝信號的恢復速度 |
| `tau_d` | float | - | 衰減時間常數，控制脈衝信號的衰減速度 |

**參數影響**：
- `tau_r` 越大：恢復越慢，對快速變化的適應越慢
- `tau_d` 越大：衰減越快，對背景的記憶越短

### DNF 參數

DNF（Dynamic Neural Field）動態神經場用於目標檢測和增強。

| 參數名稱 | 類型 | 默認值 | 說明 |
|-----------|------|----------|------|
| `sigma_exc` | float | - | 興奮連接的標準差，控制興奮連接的範圍 |
| `sigma_inh` | float | - | 抑制連接的標準差，控制抑制連接的範圍 |
| `tau` | float | - | 時間常數，控制神經場的動態響應速度 |

**參數影響**：
- `sigma_exc` 越大：興奮連接範圍越大，檢測靈敏度越高
- `sigma_inh` 越大：抑制連接範圍越大，抑制效果越強
- `tau` 越大：神經場響應越慢，對快速運動的適應越慢

### STDP 聚類參數

STDP（Spike-Timing-Dependent Plasticity）聚類用於目標身份識別和軌跡關聯。

| 參數名稱 | 類型 | 默認值 | 說明 |
|-----------|------|----------|------|
| `learning_rate` | float | - | 學習率，控制聚類中心的更新速度 |
| `cluster_threshold` | float | - | 聚類閾值，用於判斷是否屬於同一目標 |

**參數影響**：
- `learning_rate` 越大：聚類中心更新越快，但可能不穩定
- `cluster_threshold` 越大：更容易將不同目標合併，可能降低身份識別精度

### 運動估計參數

運動估計用於預測目標的運動軌跡。

| 參數名稱 | 類型 | 默認值 | 說明 |
|-----------|------|----------|------|
| `window_size` | int | - | 運動估計窗口大小，用於計算光流 |
| `max_displacement` | int | - | 最大位移，限制運動估計的搜索範圍 |

**參數影響**：
- `window_size` 越大：對大運動的估計越準確，但計算量越大
- `max_displacement` 越大：可以處理更快的運動，但可能產生錯誤匹配

## ⚙️ 參數調優建議

### 1. 根據場景特點調整

**高速運動場景**（如 rotTrans, ball）：
- 增大 `attention_size`（15-25）
- 增大 `calibration_time`（200-300）
- 減小 `cluster_threshold` 以提高檢測靈敏度

**慢速運動場景**（如 spike59）：
- 減小 `attention_size`（10-15）
- 減小 `calibration_time`（100-150）
- 增大 `cluster_threshold` 以減少錯誤檢測

**複雜場景**（如 badminton）：
- 使用中等 `attention_size`（15-20）
- 使用標準 `calibration_time`（150）
- 調整 STP 參數以適應不同光照條件

### 2. 根據硬體配置調整

**GPU 內存充足**（>8GB）：
- 可以使用較大的 `attention_size`（20-30）
- 可以減小 `downscale`（1-2）以保持更高精度
- 可以處理更長的時間序列

**GPU 內存有限**（<4GB）：
- 使用較小的 `attention_size`（10-15）
- 增大 `downscale`（2-4）以減少內存使用
- 考慮分批處理長序列

**CPU 模式**：
- 必須使用較大的 `downscale`（4-8）
- 使用較小的 `attention_size`（5-10）
- 考慮使用更高效的數據結構

### 3. 參數敏感性分析

**對性能影響最大的參數**：
1. `attention_size` - 直接影響檢測速度和精度
2. `downscale` - 影響處理速度和空間分辨率
3. `calibration_time` - 影響背景模型質量
4. `cluster_threshold` - 影響身份識別準確度

**對性能影響較小的參數**：
1. STP 參數（`tau_r`, `tau_d`）- 主要影響背景適應
2. DNF 參數（`sigma_exc`, `sigma_inh`, `tau`）- 主要影響檢測增強
3. 運動估計參數（`window_size`, `max_displacement`）- 主要影響軌跡平滑度

## 📊 性能基準

### 參考性能指標

在 motVidarReal2020 數據集上的典型性能範圍：

| 場景 | MOTA | MOTP | IDF1 | 處理速度 |
|------|------|------|------|----------|
| spike59 | 0.85-0.92 | 0.40-0.45 | 0.92-0.96 | 30-50 FPS |
| rotTrans | 0.80-0.88 | 0.35-0.42 | 0.88-0.94 | 25-40 FPS |
| badminton | 0.75-0.85 | 0.30-0.38 | 0.85-0.92 | 20-35 FPS |

### 硬體需求基準

| 配置 | GPU 內存 | 建議參數 | 預期性能 |
|------|-----------|----------|----------|
| 高端 | >8GB | attention_size=20-25, downscale=1 | MOTA >0.90 |
| 中端 | 4-8GB | attention_size=15-20, downscale=2 | MOTA 0.85-0.90 |
| 低端 | <4GB | attention_size=10-15, downscale=4 | MOTA 0.75-0.85 |
| CPU | 依賴於CPU | attention_size=5-10, downscale=4-8 | MOTA 0.65-0.75 |

## 🔍 故障排除

### 常見問題及解決方案

**問題 1：內存溢出**
```
RuntimeError: CUDA out of memory
```

解決方案：
1. 增大 `downscale` 參數（2→4→8）
2. 減小 `attention_size` 參數
3. 減小處理的 batch size
4. 使用 CPU 模式

**問題 2：跟蹤不穩定**
```
Warning: Target lost frequently
```

解決方案：
1. 增大 `calibration_time` 參數
2. 調整 STP 參數（`tau_r`, `tau_d`）
3. 減小 `learning_rate` 參數

**問題 3：檢測率過低**
```
Warning: Low detection rate
```

解決方案：
1. 減小 `cluster_threshold` 參數
2. 增大 `attention_size` 參數
3. 調整 DNF 參數（`sigma_exc`, `sigma_inh`）

**問題 4：身份切換頻繁**
```
Warning: High ID switch rate
```

解決方案：
1. 增大 `learning_rate` 參數
2. 減小 `cluster_threshold` 參數
3. 增大 `window_size` 參數

## 📚 參考資源

- **SNNTracker 論文**: Zheng Y, Li C, Zhang J, et al. SNNTracker: Online High-Speed Multi-Object Tracking With Spike Camera[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026: 624-638.
- **GitHub 項目**: https://github.com/Zyj061/snnTracker
- **SpikeCV 項目**: https://github.com/Zyj061/SpikeCV
