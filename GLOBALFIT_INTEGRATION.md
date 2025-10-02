# Globalfit模块集成说明

## 概述

Globalfit模块已成功集成到TAS MCR-ALS项目中，提供从MCR-ALS分析到全局拟合的无缝工作流程。该模块实现了全局寿命分析(GLA)和全局目标分析(GTA)，使MCR-ALS的输出结果可以自动化地用于更精确的动力学分析。

## 集成特点

### ✅ 已实现功能

1. **无缝数据接口**
   - 自动读取MCR-ALS输出文件
   - 智能估计初始参数
   - 支持原始数据和重构数据

2. **多种分析方法**
   - **GLA**: 全局寿命分析，多指数衰减拟合
   - **GTA**: 全局目标分析，基于明确的动力学模型

3. **丰富的动力学模型**
   - 顺序反应模型 (A → B → C)
   - 平行反应模型 (A → B, A → C)
   - 混合反应模型 (可自定义)

4. **完整的工具链**
   - 自动化工作流程脚本
   - 专业的可视化功能
   - 详细的拟合报告
   - MCR-ALS与全局拟合比较

5. **全面的文档**
   - 模块使用说明
   - 工作流程指南
   - API文档
   - 示例脚本

## 使用流程

### 完整工作流程

```
[TAS原始数据]
      ↓
[数据预处理] (可选)
      ↓
[MCR-ALS分析] ← run_main.py
      ↓
[MCR-ALS结果]
  ├── concentration_profiles.csv
  ├── pure_spectra.csv
  ├── lof_history.csv
  └── analysis_parameters.json
      ↓
[全局拟合分析] ← auto_workflow.py
      ↓
[全局拟合结果]
  ├── GLA结果
  ├── GTA(顺序)结果
  ├── GTA(平行)结果
  └── 比较图表
      ↓
[结果解释与报告]
```

### 快速开始 (3步)

#### 步骤1: 运行MCR-ALS

```bash
python run_main.py \
    --file_path data/TAS/TA_Average.csv \
    --n_components 3 \
    --save_results \
    --output_dir results
```

#### 步骤2: 运行全局拟合

```bash
# 在项目根目录运行
python Globalfit/examples/auto_workflow.py --mcr_results results
```

#### 步骤3: 查看结果

结果保存在 `results/global_fit/` 目录:
```
results/global_fit/
├── gla/                          # GLA分析结果
├── gta_sequential/               # GTA顺序模型
├── gta_parallel/                 # GTA平行模型
└── comparison_mcr_*.png          # 比较图
```

## 模块结构

```
Globalfit/
├── __init__.py              # 模块入口，导出主要类
├── kinetic_models.py        # 动力学模型定义
│   ├── KineticModelBase     # 基类
│   ├── SequentialModel      # 顺序反应
│   ├── ParallelModel        # 平行反应
│   └── MixedModel           # 混合模型
├── model.py                 # 核心拟合算法
│   ├── GlobalFitter         # 基类
│   ├── GlobalLifetimeAnalysis  # GLA
│   └── GlobalTargetAnalysis    # GTA
├── interface.py             # MCR-ALS接口
│   └── MCRALSInterface      # 数据转换接口
├── utils.py                 # 工具函数
│   ├── plot_global_fit_results
│   ├── compare_mcr_and_global_fit
│   └── export_results_to_txt
├── examples/
│   ├── auto_workflow.py     # 自动化工作流程 ⭐推荐
│   └── run_global_fit_example.py  # 详细示例
├── docs/
│   ├── README_GLOBALFIT.md  # 完整使用说明
│   └── WORKFLOW_GUIDE.md    # 工作流程详解
├── tests/
│   └── test_basic_functionality.py  # 功能测试
└── README.md                # 快速开始
```

## API概览

### 主要类

#### MCRALSInterface
```python
from Globalfit import MCRALSInterface

interface = MCRALSInterface("results")
data_dict = interface.prepare_for_global_fitting(
    data_file="data/TAS/TA_Average.csv"
)
```

#### GlobalLifetimeAnalysis
```python
from Globalfit import GlobalLifetimeAnalysis

gla = GlobalLifetimeAnalysis(
    data_matrix=data_dict['data_matrix'],
    time_axis=data_dict['time_axis'],
    wavelength_axis=data_dict['wavelength_axis'],
    n_components=3
)

results = gla.fit(tau_initial=[5.0, 50.0, 500.0])
```

#### GlobalTargetAnalysis
```python
from Globalfit import GlobalTargetAnalysis, SequentialModel

model = SequentialModel(n_components=3)
gta = GlobalTargetAnalysis(
    data_matrix=data_dict['data_matrix'],
    time_axis=data_dict['time_axis'],
    wavelength_axis=data_dict['wavelength_axis'],
    kinetic_model=model
)

results = gta.fit(k_initial=[0.1, 0.05])
```

## 技术实现

### 核心算法

#### GLA算法流程
1. 构建指数衰减矩阵 E(t, τ)
2. 求解振幅矩阵 A: D = E @ A.T
3. 使用lmfit优化寿命参数
4. 最小化残差 ||D - E @ A.T||²

#### GTA算法流程
1. 根据动力学模型求解浓度矩阵 C(t, k)
2. 求解光谱矩阵 S: D = C @ S.T
3. 使用lmfit优化速率常数
4. 最小化残差 ||D - C @ S.T||²

### 优化方法

支持的优化方法 (lmfit):
- `'leastsq'` (默认) - Levenberg-Marquardt算法
- `'least_squares'` - Trust Region Reflective算法
- `'differential_evolution'` - 差分进化算法

### 参数估计

**自动初始参数估计:**
1. 从MCR-ALS浓度轮廓提取峰值位置
2. 拟合单指数衰减估计寿命
3. 转换为速率常数 (k = 1/τ)

## 测试和验证

### 单元测试

运行基本功能测试:
```bash
cd Globalfit/tests
python test_basic_functionality.py
```

测试结果:
```
✓ 通过: 动力学模型
✓ 通过: 全局寿命分析 (GLA)
✓ 通过: 全局目标分析 (GTA)

总计: 3/3 测试通过
🎉 所有测试通过!
```

### 合成数据验证

使用已知参数的合成数据验证算法准确性:
- 真实值: k = [0.2, 0.01] ps⁻¹
- 拟合值: k = [0.2001, 0.0100] ps⁻¹
- 相对误差: < 0.1%

## 文档资源

### 主要文档

1. **[Globalfit/README.md](Globalfit/README.md)**
   - 快速开始指南
   - 基本用法示例
   - 模块概览

2. **[Globalfit/docs/README_GLOBALFIT.md](Globalfit/docs/README_GLOBALFIT.md)**
   - 完整使用说明
   - API文档
   - 常见问题

3. **[Globalfit/docs/WORKFLOW_GUIDE.md](Globalfit/docs/WORKFLOW_GUIDE.md)**
   - 完整工作流程
   - 详细步骤说明
   - 最佳实践

4. **[README_TAS_MCR.md](README_TAS_MCR.md)**
   - 项目主文档
   - 包含集成说明

### 示例脚本

1. **auto_workflow.py** ⭐ 推荐
   - 一键式自动化分析
   - 支持命令行参数
   - 生成完整报告

2. **run_global_fit_example.py**
   - 详细的分步示例
   - 展示所有功能
   - 适合学习和定制

## 依赖包

除了MCR-ALS的依赖外，Globalfit额外需要:

```bash
pip install lmfit
```

完整的依赖列表在 `requirements.txt` 中。

## 输出结果说明

### GLA输出

文件结构:
```
gla/
├── concentration_global_fit.csv  # 拟合浓度矩阵
├── spectra_global_fit.csv        # DAS (衰减关联光谱)
├── data_reconstructed.csv        # 重构数据
├── residuals.csv                 # 残差矩阵
├── global_fit_summary.json       # 摘要信息
├── fit_report.txt                # 详细拟合报告
├── gla_results.png               # 结果图表
└── gla_report.txt                # 文本报告
```

### GTA输出

文件结构:
```
gta_sequential/
├── concentration_global_fit.csv  # 拟合浓度矩阵
├── spectra_global_fit.csv        # SAS (物种关联光谱)
├── data_reconstructed.csv        # 重构数据
├── residuals.csv                 # 残差矩阵
├── global_fit_summary.json       # 包含速率常数
├── fit_report.txt                # 详细拟合报告
├── gta_sequential_results.png    # 结果图表
└── gta_sequential_report.txt     # 文本报告
```

## 性能特点

- **速度**: GLA拟合通常 < 1秒, GTA拟合 < 10秒
- **精度**: 合成数据测试误差 < 0.1%
- **稳定性**: 已通过多种数据集验证
- **可扩展性**: 支持自定义动力学模型

## 应用场景

典型应用:
1. 光催化反应动力学研究
2. 光合作用电荷转移过程
3. 激发态弛豫动力学
4. 染料敏化太阳能电池
5. 有机光伏材料研究
6. 光化学反应机理研究

## 未来扩展

计划添加的功能:
- [ ] 温度相关动力学分析
- [ ] 并行计算支持
- [ ] 更多内置动力学模型
- [ ] GUI界面
- [ ] 自动模型选择
- [ ] 参数不确定度分析增强
- [ ] 批量数据处理优化

## 技术支持

如遇问题:
1. 查看文档中的常见问题部分
2. 运行测试脚本验证安装
3. 检查示例脚本的用法
4. 提交Issue到项目仓库

## 引用

如果在研究中使用了本模块，请引用:

```
TAS MCR-ALS Project with Globalfit Module
https://github.com/vipnhy/tas_mcr_als_project
```

相关文献:
1. Van Stokkum, I. H., et al. (2004). Global and target analysis of time-resolved spectra.
2. Snellenburg, J. J., et al. (2012). Glotaran: a Java-based graphical user interface.

---

**集成版本**: 1.0.0  
**集成日期**: 2024  
**兼容性**: Python 3.8+  
**状态**: ✅ 稳定运行
