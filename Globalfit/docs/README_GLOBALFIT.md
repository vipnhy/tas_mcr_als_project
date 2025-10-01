# Globalfit - 全局拟合模块使用说明

## 目录
1. [概述](#概述)
2. [模块结构](#模块结构)
3. [安装和依赖](#安装和依赖)
4. [快速开始](#快速开始)
5. [详细使用说明](#详细使用说明)
6. [API文档](#api文档)
7. [示例和教程](#示例和教程)
8. [常见问题](#常见问题)

---

## 概述

**Globalfit** 是一个用于瞬态吸收光谱(TAS)数据全局拟合分析的Python模块。该模块实现了两种主要的全局拟合方法:

- **全局寿命分析 (GLA - Global Lifetime Analysis)**: 使用多指数衰减函数拟合数据，不需要预先假设反应机理
- **全局目标分析 (GTA - Global Target Analysis)**: 基于明确的动力学模型拟合数据，参数具有物理意义

### 主要特点

✨ **无缝集成**: 可以直接使用MCR-ALS的输出结果作为输入，实现自动化分析流程

🔬 **物理意义**: GTA方法提供有明确物理意义的动力学参数

📊 **丰富的可视化**: 自动生成专业的分析图表和比较图

🎯 **灵活的模型**: 支持顺序反应、平行反应和混合反应模型

⚡ **高效优化**: 基于lmfit库实现的高效参数优化

📈 **详细报告**: 生成完整的拟合报告，包括参数不确定度

---

## 模块结构

```
Globalfit/
├── __init__.py              # 模块初始化文件
├── kinetic_models.py        # 动力学模型定义
├── model.py                 # 全局拟合核心算法
├── interface.py             # MCR-ALS输出接口
├── utils.py                 # 工具函数
├── examples/                # 示例脚本
│   └── run_global_fit_example.py
├── docs/                    # 文档
│   ├── README_GLOBALFIT.md
│   ├── TUTORIAL.md
│   └── API.md
└── tests/                   # 测试文件
    └── test_global_fit.py
```

---

## 安装和依赖

### 依赖包

Globalfit模块需要以下Python包:

```bash
pip install numpy scipy matplotlib lmfit
```

或者使用项目的requirements.txt:

```bash
pip install -r requirements.txt
```

### 推荐的Python版本

- Python 3.8 或更高版本

---

## 快速开始

### 1. 完整的自动化流程

```python
from Globalfit import MCRALSInterface, GlobalLifetimeAnalysis

# 1. 从MCR-ALS结果准备数据
interface = MCRALSInterface("results")  # MCR-ALS结果目录
data_dict = interface.prepare_for_global_fitting()

# 2. 执行全局寿命分析
gla = GlobalLifetimeAnalysis(
    data_matrix=data_dict['data_matrix'],
    time_axis=data_dict['time_axis'],
    wavelength_axis=data_dict['wavelength_axis'],
    n_components=data_dict['n_components']
)

# 3. 拟合 (使用自动估计的初始值)
results = gla.fit(tau_initial=data_dict['lifetimes_initial'])

# 4. 保存结果
interface.save_global_fit_results(results)

# 5. 可视化
from Globalfit.utils import plot_global_fit_results
plot_global_fit_results(results, data_dict['time_axis'], 
                       data_dict['wavelength_axis'])
```

### 2. 使用全局目标分析 (GTA)

```python
from Globalfit import GlobalTargetAnalysis, SequentialModel

# 创建顺序反应模型 (A → B → C)
model = SequentialModel(n_components=3)

# 创建GTA分析器
gta = GlobalTargetAnalysis(
    data_matrix=data_dict['data_matrix'],
    time_axis=data_dict['time_axis'],
    wavelength_axis=data_dict['wavelength_axis'],
    kinetic_model=model
)

# 拟合 (需要 n-1 个速率常数)
k_initial = [0.1, 0.05]  # k1, k2
results = gta.fit(k_initial=k_initial)

print(f"最优速率常数: {results['k_optimal']}")
print(f"对应寿命: {results['tau_optimal']}")
```

---

## 详细使用说明

### 步骤1: 准备数据

#### 方法A: 从MCR-ALS结果自动准备

```python
from Globalfit import MCRALSInterface

interface = MCRALSInterface("results")
data_dict = interface.prepare_for_global_fitting(
    data_file="data/TAS/TA_Average.csv",  # 可选，原始数据文件
    file_type="handle"
)
```

**自动准备的内容:**
- 加载MCR-ALS的浓度矩阵 (C) 和光谱矩阵 (S)
- 加载或重构原始数据矩阵 (D)
- 加载时间轴和波长轴
- 自动估计初始寿命和速率常数

#### 方法B: 手动准备数据

```python
import numpy as np

# 准备数据矩阵
D = np.loadtxt("your_data.csv", delimiter=',')
time_axis = np.array([...])
wavelength_axis = np.array([...])
n_components = 3

data_dict = {
    'data_matrix': D,
    'time_axis': time_axis,
    'wavelength_axis': wavelength_axis,
    'n_components': n_components,
    'lifetimes_initial': [10.0, 100.0, 1000.0],  # 估计值
    'rate_constants_initial': [0.1, 0.01, 0.001]
}
```

### 步骤2: 选择分析方法

#### 全局寿命分析 (GLA)

**适用场景:**
- 不清楚反应机理
- 需要快速获得衰减时间常数
- 用于初步探索性分析

**优点:**
- 不需要假设反应机理
- 拟合相对简单和快速
- 可以处理复杂的衰减行为

**缺点:**
- 参数缺乏直接的物理意义
- 仅是数学拟合，不能提供机理信息

```python
from Globalfit import GlobalLifetimeAnalysis

gla = GlobalLifetimeAnalysis(
    data_matrix=data_dict['data_matrix'],
    time_axis=data_dict['time_axis'],
    wavelength_axis=data_dict['wavelength_axis'],
    n_components=3
)

# 设置初始寿命和边界
tau_initial = [5.0, 50.0, 500.0]  # ps
tau_bounds = [(1.0, 20.0), (20.0, 200.0), (200.0, 2000.0)]
tau_vary = [True, True, True]  # 全部优化

results = gla.fit(
    tau_initial=tau_initial,
    tau_bounds=tau_bounds,
    tau_vary=tau_vary,
    optimization_method='leastsq'
)
```

#### 全局目标分析 (GTA)

**适用场景:**
- 已知或假设反应机理
- 需要获得有物理意义的速率常数
- 用于验证反应机理假设

**优点:**
- 参数有明确的物理意义
- 可以比较不同反应机理
- 便于与其他实验结果对比

**缺点:**
- 需要预先假设反应机理
- 如果机理错误，可能得到错误的结果

##### 顺序反应模型 (A → B → C)

```python
from Globalfit import GlobalTargetAnalysis, SequentialModel

model = SequentialModel(n_components=3)
gta = GlobalTargetAnalysis(
    data_matrix=data_dict['data_matrix'],
    time_axis=data_dict['time_axis'],
    wavelength_axis=data_dict['wavelength_axis'],
    kinetic_model=model
)

# 顺序模型需要 n-1 个速率常数
k_initial = [0.2, 0.01]  # k1: A→B, k2: B→C
results = gta.fit(k_initial=k_initial)
```

##### 平行反应模型 (A → B, A → C, A → D)

```python
from Globalfit import ParallelModel

model = ParallelModel(n_components=3)
gta = GlobalTargetAnalysis(
    data_matrix=data_dict['data_matrix'],
    time_axis=data_dict['time_axis'],
    wavelength_axis=data_dict['wavelength_axis'],
    kinetic_model=model
)

# 平行模型也需要 n-1 个速率常数
k_initial = [0.1, 0.05]  # k1: A→B, k2: A→C
results = gta.fit(k_initial=k_initial)
```

### 步骤3: 查看和保存结果

#### 查看拟合统计

```python
print(f"LOF: {results['lof']:.4f}%")
print(f"Chi-Square: {results['chi_square']:.6e}")
print(f"计算时间: {results['computation_time']:.2f} 秒")

if 'tau_optimal' in results:
    print(f"最优寿命: {results['tau_optimal']}")

if 'k_optimal' in results:
    print(f"最优速率常数: {results['k_optimal']}")
```

#### 可视化结果

```python
from Globalfit.utils import plot_global_fit_results

plot_global_fit_results(
    results,
    data_dict['time_axis'],
    data_dict['wavelength_axis'],
    save_path="global_fit_results.png",
    show_plot=True
)
```

#### 比较MCR-ALS和全局拟合

```python
from Globalfit.utils import compare_mcr_and_global_fit

mcr_results = {
    'C_mcr': data_dict['C_mcr'],
    'S_mcr': data_dict['S_mcr'],
    'mcr_lof': data_dict['mcr_lof']
}

compare_mcr_and_global_fit(
    mcr_results,
    results,
    data_dict['time_axis'],
    data_dict['wavelength_axis'],
    save_path="comparison_mcr_globalfit.png"
)
```

#### 保存结果

```python
# 使用接口保存
interface.save_global_fit_results(results, output_dir="results/global_fit")

# 或导出为文本报告
from Globalfit.utils import export_results_to_txt
export_results_to_txt(results, "global_fit_report.txt")
```

### 步骤4: 结果解释

#### GLA结果解释

- **寿命 (τ)**: 每个指数组分的衰减时间常数
  - 较短的寿命对应快速衰减过程
  - 较长的寿命对应慢速衰减过程
  
- **DAS (Decay-Associated Spectra)**: 每个指数组分关联的光谱
  - 正值: 吸收信号的贡献
  - 负值: 基态漂白或受激发射的贡献

#### GTA结果解释

- **速率常数 (k)**: 反应的速率
  - k值越大，反应越快
  - k = 1/τ (寿命的倒数)
  
- **SAS (Species-Associated Spectra)**: 每个物种的特征光谱
  - 对应于反应中间体或产物的真实吸收光谱
  - 具有明确的物理意义

- **浓度轮廓**: 各物种随时间的浓度变化
  - 反映了反应过程的动力学行为
  - 可以验证假设的反应机理是否合理

---

## API文档

### MCRALSInterface类

**主要方法:**

- `load_mcr_results()`: 加载MCR-ALS结果
- `load_original_data(data_file, file_type)`: 加载原始数据
- `estimate_lifetimes_from_mcr()`: 从MCR-ALS浓度轮廓估计寿命
- `prepare_for_global_fitting(data_file, file_type)`: 准备全局拟合所需的所有数据
- `save_global_fit_results(results, output_dir)`: 保存全局拟合结果

### GlobalLifetimeAnalysis类

**初始化参数:**
- `data_matrix`: 数据矩阵 (n_times × n_wavelengths)
- `time_axis`: 时间轴数组
- `wavelength_axis`: 波长轴数组
- `n_components`: 组分数量

**主要方法:**
- `fit(tau_initial, tau_bounds, tau_vary, optimization_method)`: 执行GLA拟合

**返回结果:**
- `tau_optimal`: 最优寿命
- `C_fit`: 拟合的浓度矩阵
- `S_fit`: 拟合的光谱矩阵 (DAS)
- `D_reconstructed`: 重构数据
- `residuals`: 残差矩阵
- `lof`: 拟合缺失度
- `chi_square`: 卡方值

### GlobalTargetAnalysis类

**初始化参数:**
- `data_matrix`: 数据矩阵
- `time_axis`: 时间轴数组
- `wavelength_axis`: 波长轴数组
- `kinetic_model`: 动力学模型对象

**主要方法:**
- `fit(k_initial, k_bounds, k_vary, optimization_method)`: 执行GTA拟合

**返回结果:**
- `k_optimal`: 最优速率常数
- `tau_optimal`: 对应寿命 (1/k)
- `C_fit`: 拟合的浓度矩阵
- `S_fit`: 拟合的光谱矩阵 (SAS)
- `D_reconstructed`: 重构数据
- `residuals`: 残差矩阵
- `lof`: 拟合缺失度
- `kinetic_model`: 使用的动力学模型名称

### 动力学模型

**SequentialModel(n_components)**: 顺序反应模型
- 反应方案: A → B → C → D → ...
- 需要 n-1 个速率常数

**ParallelModel(n_components)**: 平行反应模型
- 反应方案: A → B, A → C, A → D, ...
- 需要 n-1 个速率常数

**MixedModel(n_components, reaction_network)**: 混合模型
- 自定义反应网络
- 灵活定义任意复杂的反应方案

---

## 示例和教程

### 示例1: 基本使用流程

运行提供的示例脚本:

```bash
cd Globalfit/examples
python run_global_fit_example.py
```

该脚本将:
1. 从MCR-ALS结果加载数据
2. 执行GLA分析
3. 执行GTA分析 (顺序和平行模型)
4. 生成比较图
5. 保存所有结果

### 示例2: 在Jupyter Notebook中使用

参见 `Globalfit/examples/notebook_tutorial.ipynb` (如果提供)

### 示例3: 自定义动力学模型

```python
from Globalfit import MixedModel

# 定义自定义反应网络
# (from_idx, to_idx, k_idx)
reaction_network = [
    (0, 1, 0),  # A --k0--> B
    (1, 2, 1),  # B --k1--> C
    (0, 3, 2)   # A --k2--> D (分支反应)
]

model = MixedModel(n_components=4, reaction_network=reaction_network)
```

---

## 常见问题

### Q1: 拟合收敛失败怎么办?

**A:** 尝试以下方法:
1. 调整初始参数值，使其更接近真实值
2. 放宽参数边界
3. 更改优化方法 (如从 'leastsq' 改为 'least_squares')
4. 检查数据质量，确保没有NaN或Inf值
5. 减少组分数量

### Q2: GLA和GTA哪个更好?

**A:** 这取决于你的研究目标:
- **探索性分析**: 使用GLA快速获得衰减时间
- **机理研究**: 使用GTA验证反应机理假设
- **最佳实践**: 先用GLA了解系统，再用GTA确定机理

### Q3: 如何选择合适的动力学模型?

**A:** 考虑以下因素:
1. **文献和预期**: 查看类似体系的文献
2. **浓度轮廓形状**: 观察MCR-ALS的浓度曲线
   - 单调衰减 → 顺序模型
   - 同时出现多个峰 → 平行模型
3. **试错法**: 尝试不同模型，比较LOF值和物理合理性
4. **Occam's Razor**: 在解释数据的前提下，选择最简单的模型

### Q4: LOF值应该是多少才算好?

**A:** 一般准则:
- LOF < 5%: 优秀拟合
- LOF 5-10%: 良好拟合
- LOF 10-20%: 可接受 (取决于数据质量)
- LOF > 20%: 拟合较差，需要检查模型或数据

但要注意: LOF低不一定代表模型正确，还需要检查:
- 残差是否为随机噪声
- 参数是否有物理意义
- 拟合结果是否可重复

### Q5: 如何估计参数的不确定度?

**A:** 使用工具函数:

```python
from Globalfit.utils import estimate_uncertainty

uncertainty = estimate_uncertainty(results['fit_result'], confidence_level=0.95)
for param_name, info in uncertainty.items():
    print(f"{param_name}: {info['value']:.4f} ± {info['uncertainty']:.4f}")
```

### Q6: 拟合参数不稳定怎么办?

**A:** 可能原因:
1. **过度拟合**: 组分数量过多，减少组分
2. **初始值不佳**: 改进初始值估计
3. **参数相关性**: 某些参数可能高度相关，考虑固定其中一些
4. **数据噪声**: 检查数据质量，考虑预处理

### Q7: 能否在拟合过程中施加约束?

**A:** 是的，lmfit支持多种约束:

```python
# 固定某个参数
tau_vary = [True, False, True]  # 固定第二个寿命

# 设置参数边界
tau_bounds = [(1, 50), (50, 500), (500, 5000)]

# 通过lmfit添加更复杂的约束
# 例如: 确保寿命递增 τ1 < τ2 < τ3
# 这需要在Parameters对象中使用表达式约束
```

---

## 技术支持

如有问题或建议，请:
1. 查看本文档的常见问题部分
2. 查看示例脚本
3. 提交Issue到项目仓库

---

## 参考文献

1. Van Stokkum, I. H., et al. (2004). Global and target analysis of time-resolved spectra. *Biochimica et Biophysica Acta (BBA)-Bioenergetics*, 1657(2-3), 82-104.

2. Snellenburg, J. J., et al. (2012). Glotaran: a Java-based graphical user interface for the R package TIMP. *Journal of Statistical Software*, 49(3), 1-22.

3. Mullen, K. M., & van Stokkum, I. H. (2007). TIMP: an R package for modeling multi-way spectroscopic measurements. *Journal of Statistical Software*, 18(3), 1-46.

---

*文档版本: 1.0.0*  
*最后更新: 2024*
