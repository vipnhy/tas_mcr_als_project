# Globalfit - 全局拟合模块

瞬态吸收光谱(TAS)数据的全局拟合分析模块，实现MCR-ALS到全局拟合的无缝集成。

## 快速开始

### 安装依赖

```bash
pip install numpy scipy matplotlib lmfit
```

### 基本使用

```python
from Globalfit import MCRALSInterface, GlobalLifetimeAnalysis

# 从MCR-ALS结果自动准备数据
interface = MCRALSInterface("results")  # MCR-ALS结果目录
data_dict = interface.prepare_for_global_fitting()

# 执行全局寿命分析
gla = GlobalLifetimeAnalysis(
    data_matrix=data_dict['data_matrix'],
    time_axis=data_dict['time_axis'],
    wavelength_axis=data_dict['wavelength_axis'],
    n_components=data_dict['n_components']
)

# 拟合
results = gla.fit(tau_initial=data_dict['lifetimes_initial'])
print(f"最优寿命: {results['tau_optimal']}")
print(f"LOF: {results['lof']:.4f}%")
```

### 自动化工作流程

```bash
cd examples
python auto_workflow.py --mcr_results ../../results
```

## 主要功能

- ✨ **无缝集成**: 直接使用MCR-ALS输出结果
- 🔬 **GLA分析**: 全局寿命分析，无需假设反应机理
- 🎯 **GTA分析**: 全局目标分析，基于明确的动力学模型
- 📊 **丰富可视化**: 自动生成专业分析图表
- 📈 **详细报告**: 完整的拟合报告和统计信息

## 文档

- [完整使用说明](docs/README_GLOBALFIT.md)
- [工作流程指南](docs/WORKFLOW_GUIDE.md)
- [示例脚本](examples/)

## 模块结构

```
Globalfit/
├── __init__.py              # 模块入口
├── kinetic_models.py        # 动力学模型 (顺序/平行/混合)
├── model.py                 # GLA和GTA核心算法
├── interface.py             # MCR-ALS接口
├── utils.py                 # 工具函数
├── examples/                # 示例脚本
│   ├── auto_workflow.py     # 自动化工作流程
│   └── run_global_fit_example.py
├── docs/                    # 文档
│   ├── README_GLOBALFIT.md  # 完整说明
│   └── WORKFLOW_GUIDE.md    # 工作流程
└── README.md                # 本文件
```

## 支持的动力学模型

### 1. 顺序反应模型
```
A → B → C → D
```

### 2. 平行反应模型
```
A → B
↓
C
↓
D
```

### 3. 混合模型
```
可自定义任意复杂的反应网络
```

## 典型应用场景

- 光催化反应动力学研究
- 光合作用电荷转移过程
- 激发态弛豫动力学
- 染料敏化太阳能电池
- 有机光伏材料研究

## 示例结果

### GLA结果
- 寿命: [5.2 ps, 87 ps, 1200 ps]
- LOF: 3.5%

### GTA结果 (顺序模型)
- 速率常数: [0.19 ps⁻¹, 0.011 ps⁻¹]
- 对应寿命: [5.3 ps, 91 ps, 1230 ps]
- LOF: 4.1%

## 参考文献

1. Van Stokkum, I. H., et al. (2004). Global and target analysis of time-resolved spectra. *Biochimica et Biophysica Acta*, 1657(2-3), 82-104.

2. Snellenburg, J. J., et al. (2012). Glotaran: a Java-based graphical user interface for the R package TIMP. *Journal of Statistical Software*, 49(3), 1-22.

## 许可证

本模块是TAS MCR-ALS项目的一部分。

---

**版本**: 1.0.0  
**作者**: TAS Analysis Team  
**最后更新**: 2024
