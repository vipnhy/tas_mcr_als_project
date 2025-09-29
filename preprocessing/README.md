# TAS数据预处理模块

瞬态吸收光谱(TAS)数据预处理工具集，提供完整的数据预处理功能。

## 功能特性

### 1. 基线校正 (Baseline Correction)
- **ALS基线校正**: 使用渐近最小二乘法进行自适应基线校正
- **多项式基线校正**: 基于边缘点的多项式拟合基线校正
- **滚球基线校正**: 使用滚球算法的基线校正
- **线性基线校正**: 简单的线性基线校正

### 2. 噪声过滤 (Noise Filtering)
- **高斯滤波**: 2D高斯滤波器，可分别设置时间和波长维度参数
- **中值滤波**: 保边缘的中值滤波
- **Wiener滤波**: 自适应Wiener滤波
- **FFT频域滤波**: 支持低通、高通、带通滤波
- **PCA降噪**: 基于主成分分析的降噪
- **双边滤波**: 保边缘的双边滤波

### 3. 数据平滑 (Data Smoothing)
- **Savitzky-Golay平滑**: 保形状的多项式平滑
- **移动平均**: 简单移动平均平滑
- **LOWESS平滑**: 局部加权回归平滑
- **样条插值平滑**: B样条平滑
- **Butterworth滤波**: 低通滤波平滑
- **自适应平滑**: 根据局部噪声水平调整平滑强度

### 4. 异常值检测与处理 (Outlier Detection)
- **Z-score检测**: 基于标准分数的异常值检测
- **IQR检测**: 基于四分位距的异常值检测
- **孤立森林**: 机器学习异常值检测
- **马氏距离**: 多元异常值检测
- **椭圆包络**: 协方差估计异常值检测
- **多种处理策略**: 移除、插值、裁剪、中值替换

### 5. 一体化预处理管道 (Pipeline)
- **标准管道**: 平衡的预处理流程
- **温和管道**: 保守的预处理，较少改变原始数据
- **激进管道**: 强力去噪和平滑
- **自定义管道**: 可自由组合预处理步骤

## 安装要求

```bash
pip install numpy pandas scipy scikit-learn matplotlib statsmodels
```

## 快速开始

### 1. 基本使用

```python
import numpy as np
import pandas as pd
from preprocessing import preprocess_tas_data

# 加载TAS数据
data = pd.read_csv('your_tas_data.csv', index_col=0)

# 使用标准预处理管道
processed_data = preprocess_tas_data(data, pipeline='standard')
```

### 2. 单独使用各个模块

```python
from preprocessing import BaselineCorrector, NoiseFilter, DataSmoother, OutlierDetector

# 基线校正
corrector = BaselineCorrector(method='als', lam=1e6, p=0.001)
corrected_data = corrector.correct(data)

# 噪声过滤
filter = NoiseFilter(method='gaussian', sigma=1.0)
filtered_data = filter.filter_noise(corrected_data)

# 数据平滑
smoother = DataSmoother(method='savgol', window_length=5, polyorder=2)
smoothed_data = smoother.smooth(filtered_data)

# 异常值处理
detector = OutlierDetector(method='z_score', threshold=3.0)
detector.detect_outliers(data)
clean_data = detector.process_outliers(data, strategy='interpolate')
```

### 3. 自定义预处理管道

```python
from preprocessing import TASPreprocessingPipeline

# 定义自定义预处理步骤
custom_steps = [
    {'name': 'outlier_removal', 'processor': 'outlier', 
     'params': {'method': 'iqr', 'factor': 1.5}, 'strategy': 'interpolate'},
    {'name': 'baseline_als', 'processor': 'baseline', 
     'params': {'method': 'als', 'lam': 1e7}},
    {'name': 'gaussian_filter', 'processor': 'noise', 
     'params': {'method': 'gaussian', 'sigma': 0.8}},
    {'name': 'savgol_smooth', 'processor': 'smooth', 
     'params': {'method': 'savgol', 'window_length': 7, 'polyorder': 3}}
]

# 创建和执行自定义管道
pipeline = TASPreprocessingPipeline(steps=custom_steps, verbose=True)
processed_data = pipeline.fit_transform(data)

# 获取处理摘要
summary = pipeline.get_processing_summary()
print(summary)
```

### 4. 可视化结果

```python
# 绘制预处理管道每个步骤的结果
fig = pipeline.plot_processing_pipeline(delay_index=10, figsize=(16, 12))

# 绘制处理前后对比
fig = pipeline.plot_comparison(delay_index=10, figsize=(12, 8))

# 单独模块的可视化
fig = corrector.plot_correction(data, delay_index=10)
fig = filter.plot_filtering_result(data, delay_index=10)
fig = detector.plot_outlier_detection(data)
```

### 5. 批量处理和保存结果

```python
# 批量处理多个文件
import os
from pathlib import Path

data_dir = Path('data/')
output_dir = Path('processed_data/')

for file_path in data_dir.glob('*.csv'):
    print(f"处理文件: {file_path.name}")
    
    # 加载数据
    data = pd.read_csv(file_path, index_col=0)
    
    # 预处理
    pipeline = create_standard_pipeline(verbose=False)
    processed_data = pipeline.fit_transform(data)
    
    # 保存结果
    output_file = output_dir / f"processed_{file_path.name}"
    processed_data.to_csv(output_file)
    
    # 保存处理摘要
    summary_file = output_dir / f"summary_{file_path.stem}.json"
    summary = pipeline.get_processing_summary()
    
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
```

## 预处理管道类型

### 标准管道 (`standard`)
适用于大多数TAS数据，包含：
1. Z-score异常值检测和插值处理
2. ALS基线校正
3. 高斯噪声滤波
4. Savitzky-Golay数据平滑

### 温和管道 (`gentle`)
保守的预处理，适用于高质量数据：
1. IQR异常值检测
2. 多项式基线校正
3. 轻微高斯滤波

### 激进管道 (`aggressive`)
强力降噪，适用于高噪声数据：
1. 孤立森林异常值检测
2. 强参数ALS基线校正
3. 双边滤波
4. 强参数Savitzky-Golay平滑

## 高级功能

### 1. 自适应预处理
```python
# 根据数据特征自动调整参数
smoother = DataSmoother(method='savgol')
adaptive_smoothed = smoother.adaptive_smooth(data, noise_threshold=0.01)

# 多方法异常值检测
from preprocessing.outlier_detection import multi_method_outlier_detection

outlier_mask = multi_method_outlier_detection(
    data, 
    methods=['z_score', 'iqr', 'isolation_forest'],
    consensus_threshold=0.6
)
```

### 2. 噪声水平估计
```python
# 估计数据噪声水平
noise_filter = NoiseFilter()
noise_stats = noise_filter.estimate_noise_level(data)
print(f"估计信噪比: {noise_stats['snr_estimate']:.2f}")
```

### 3. 处理质量评估
```python
# 获取详细的处理统计
pipeline = create_standard_pipeline()
processed_data = pipeline.fit_transform(data)

summary = pipeline.get_processing_summary()
improvement = summary['data_statistics']

print(f"信噪比改善: {improvement['snr_improvement']:.2f}x")
print(f"噪声降低: {improvement['noise_reduction']*100:.1f}%")
print(f"数据保真度: {improvement['data_preservation']:.3f}")
```

## 最佳实践

1. **数据质量评估**: 在预处理前先评估数据质量和噪声水平
2. **参数调优**: 根据具体数据特征调整预处理参数
3. **结果验证**: 使用可视化工具检查预处理效果
4. **管道选择**: 根据数据质量选择合适的预处理管道
5. **保存记录**: 记录预处理参数和统计信息便于重现

## 注意事项

- 预处理会改变原始数据，建议保留备份
- 不同的预处理参数可能显著影响后续分析结果
- 对于科学分析，需要在数据质量和保真度之间找到平衡
- 建议在代表性数据子集上测试不同参数组合

## 示例数据格式

TAS数据应该是以延迟时间为行索引，波长为列索引的DataFrame或2D数组：

```
             400.0  401.0  402.0  ...
delay_time                      
0.1           0.001  0.002  0.001  ...
0.2          -0.003  0.004  0.002  ...
0.5           0.005 -0.001  0.003  ...
...
```

## 联系和支持

如有问题或建议，请通过以下方式联系：
- 项目仓库: [GitHub链接]
- 邮箱: [联系邮箱]

## 许可证

本项目采用 MIT 许可证。
