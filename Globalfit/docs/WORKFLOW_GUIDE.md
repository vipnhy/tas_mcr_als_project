# 瞬态吸收光谱分析完整工作流程指南

本指南介绍如何使用MCR-ALS模块和Globalfit模块完成从数据预处理到全局拟合的完整分析流程。

---

## 工作流程概览

```
┌─────────────────────────────────────────────────────────────────┐
│                     瞬态吸收光谱分析流程                        │
└─────────────────────────────────────────────────────────────────┘

1. 数据预处理
   ├── 读取原始TAS数据
   ├── 波长/时间范围筛选
   ├── 基线校正
   └── 噪声处理
            ↓
2. MCR-ALS分析
   ├── 确定组分数量
   ├── 设置约束条件
   ├── 运行MCR-ALS
   └── 评估拟合质量
            ↓
3. 全局拟合分析
   ├── 加载MCR-ALS结果
   ├── 估计初始参数
   ├── 选择动力学模型
   ├── 执行全局拟合 (GLA/GTA)
   └── 比较不同方法
            ↓
4. 结果解释
   ├── 分析浓度轮廓
   ├── 解释光谱特征
   ├── 确定反应机理
   └── 撰写报告
```

---

## 第一阶段: MCR-ALS分析

### 步骤1.1: 准备配置文件

创建 `config.json`:

```json
{
  "file_path": "data/TAS/TA_Average.csv",
  "file_type": "handle",
  "wavelength_range": [420, 750],
  "delay_range": [0.1, 50],
  "n_components": 3,
  "max_iter": 200,
  "tol": 1e-7,
  "save_plots": true,
  "save_results": true,
  "output_dir": "results",
  "language": "chinese",
  "constraint_config": null,
  "penalty": 0.0,
  "init_method": "svd",
  "random_seed": 42
}
```

### 步骤1.2: 运行MCR-ALS分析

```bash
# 使用配置文件运行
python run_main.py --config config.json

# 或使用命令行参数
python run_main.py \
    --file_path data/TAS/TA_Average.csv \
    --n_components 3 \
    --wavelength_range 420 750 \
    --delay_range 0.1 50 \
    --save_plots \
    --save_results \
    --output_dir results
```

### 步骤1.3: 检查MCR-ALS结果

运行后会在 `results/` 目录生成:
- `concentration_profiles.csv` - 浓度矩阵
- `pure_spectra.csv` - 纯光谱矩阵
- `lof_history.csv` - LOF收敛历史
- `analysis_parameters.json` - 分析参数
- `mcr_als_results.png` - 结果图表

**评估标准:**
- LOF < 5%: 优秀
- 残差应为随机噪声
- 光谱形状应合理
- 浓度轮廓应符合物理规律

---

## 第二阶段: 全局拟合分析

### 方法A: 使用自动化工作流程脚本 (推荐)

这是最简单的方法，适合大多数用户。

```bash
cd Globalfit/examples

# 运行所有分析 (GLA + GTA顺序 + GTA平行)
python auto_workflow.py --mcr_results ../../results

# 如果有原始数据文件
python auto_workflow.py \
    --mcr_results ../../results \
    --data_file ../../data/TAS/TA_Average.csv

# 只运行GLA
python auto_workflow.py --mcr_results ../../results --method gla

# 只运行GTA顺序模型
python auto_workflow.py \
    --mcr_results ../../results \
    --method gta \
    --model sequential
```

**输出结果:**

```
results/
└── global_fit/
    ├── gla/                          # GLA结果
    │   ├── gla_results.png
    │   ├── gla_report.txt
    │   ├── concentration_global_fit.csv
    │   ├── spectra_global_fit.csv
    │   └── global_fit_summary.json
    ├── gta_sequential/               # GTA顺序模型结果
    │   ├── gta_sequential_results.png
    │   └── ...
    ├── gta_parallel/                 # GTA平行模型结果
    │   ├── gta_parallel_results.png
    │   └── ...
    ├── comparison_mcr_gla.png        # MCR vs GLA比较
    ├── comparison_mcr_gta_sequential.png
    └── comparison_mcr_gta_parallel.png
```

### 方法B: 使用Python脚本 (灵活性更高)

如果需要更多控制和自定义，可以编写自己的脚本:

```python
import sys
sys.path.append('..')

from Globalfit import MCRALSInterface, GlobalLifetimeAnalysis
from Globalfit import GlobalTargetAnalysis, SequentialModel

# 1. 准备数据
interface = MCRALSInterface("../../results")
data_dict = interface.prepare_for_global_fitting(
    data_file="../../data/TAS/TA_Average.csv"
)

# 2. 执行GLA
gla = GlobalLifetimeAnalysis(
    data_matrix=data_dict['data_matrix'],
    time_axis=data_dict['time_axis'],
    wavelength_axis=data_dict['wavelength_axis'],
    n_components=data_dict['n_components']
)

gla_results = gla.fit(
    tau_initial=data_dict['lifetimes_initial']
)

# 3. 执行GTA
model = SequentialModel(n_components=data_dict['n_components'])
gta = GlobalTargetAnalysis(
    data_matrix=data_dict['data_matrix'],
    time_axis=data_dict['time_axis'],
    wavelength_axis=data_dict['wavelength_axis'],
    kinetic_model=model
)

k_initial = data_dict['rate_constants_initial'][:-1]
gta_results = gta.fit(k_initial=k_initial)

# 4. 可视化和保存
from Globalfit.utils import plot_global_fit_results

plot_global_fit_results(gla_results, 
                       data_dict['time_axis'],
                       data_dict['wavelength_axis'])

interface.save_global_fit_results(gla_results)
```

---

## 第三阶段: 结果分析和解释

### 3.1 比较不同方法

打开生成的比较图，观察:

1. **浓度轮廓差异**
   - MCR-ALS vs GLA: 通常差异较小
   - GLA vs GTA: 可能有较大差异，取决于模型合理性

2. **光谱形状**
   - DAS (GLA) vs SAS (GTA): 
     - DAS可能有负值
     - SAS应该全为正值或负值

3. **LOF值比较**
   ```
   MCR-ALS:  通常最低 (但没有物理约束)
   GLA:      略高于MCR-ALS
   GTA:      取决于模型合理性
   ```

### 3.2 选择最佳模型

考虑以下因素:

| 标准 | 权重 | 评估方法 |
|------|------|----------|
| 拟合质量 | ⭐⭐⭐ | LOF < 10%, 残差为随机噪声 |
| 物理合理性 | ⭐⭐⭐⭐⭐ | 参数有明确意义，符合预期 |
| 可重复性 | ⭐⭐⭐ | 多次运行结果一致 |
| 与文献一致 | ⭐⭐⭐⭐ | 寿命、机理与已知体系相符 |

**决策树:**

```
拟合质量好? 
  ├─ 否 → 检查数据质量，调整参数，或减少组分数
  └─ 是 → 继续
        │
        参数有物理意义?
          ├─ 否 → 可能过度拟合，尝试更简单的模型
          └─ 是 → 继续
                │
                与文献/预期一致?
                  ├─ 是 → ✓ 接受该模型
                  └─ 否 → 仔细检查，可能有新发现
```

### 3.3 提取关键信息

从拟合报告中提取:

**对于GLA:**
```
寿命 (τ):
  - τ1 = 5.2 ± 0.3 ps    (快速过程)
  - τ2 = 87 ± 5 ps      (中等速度)
  - τ3 = 1200 ± 100 ps  (慢速过程)
```

**对于GTA (顺序模型):**
```
速率常数 (k):
  - k1 = 0.19 ± 0.02 ps⁻¹  (A → B)
  - k2 = 0.011 ± 0.001 ps⁻¹ (B → C)

对应寿命:
  - τ1 = 5.3 ps   (A的寿命)
  - τ2 = 91 ps    (B的寿命)
```

### 3.4 解释反应机理

基于GTA结果，可以推断:

**示例: 光催化体系**

```
     激发光
      ↓
  A* ─────→ A  (激发态 → 基态，快速内转换)
      k1≈0.2 ps⁻¹
      τ1≈5 ps

  A ──────→ B  (电荷分离)
      k2≈0.05 ps⁻¹
      τ2≈20 ps

  B ──────→ C  (电荷复合/产物形成)
      k3≈0.001 ps⁻¹
      τ3≈1000 ps
```

---

## 第四阶段: 报告撰写

### 4.1 结果部分

**图表:**
1. MCR-ALS结果 (4子图)
2. 全局拟合结果 (6子图)
3. MCR-ALS vs 全局拟合比较 (4子图)

**表格:**

| 方法 | LOF (%) | τ1 (ps) | τ2 (ps) | τ3 (ps) | 备注 |
|------|---------|---------|---------|---------|------|
| MCR-ALS | 3.2 | - | - | - | 无物理约束 |
| GLA | 3.5 | 5.2 | 87 | 1200 | 数学拟合 |
| GTA (顺序) | 4.1 | 5.3 | 91 | 1230 | A→B→C模型 |
| GTA (平行) | 6.8 | 5.1 | 85 | 1180 | 拟合较差 |

### 4.2 讨论部分

模板:

```
我们使用MCR-ALS方法初步分析了瞬态吸收光谱数据，
确定了3个主要组分。随后通过全局拟合方法进一步精确
测定了动力学参数。

全局寿命分析(GLA)表明体系存在三个衰减过程，
特征时间分别为5.2 ps、87 ps和1200 ps。

为了理解反应机理，我们使用全局目标分析(GTA)
测试了顺序反应模型(A→B→C)和平行反应模型。
结果表明，顺序反应模型能更好地描述实验数据
(LOF=4.1% vs 6.8%)，且得到的动力学参数与
已报道的类似体系一致[参考文献]。

基于GTA分析，我们提出以下反应机理:
1. 初始激发态A*通过快速内转换(τ≈5ps)转化为
   热弛豫的激发态A
2. A通过电荷分离过程(τ≈91ps)生成中间体B
3. B缓慢复合或转化为最终产物C(τ≈1230ps)

这些结果为理解[体系名称]的光物理过程提供了
详细的动力学信息。
```

---

## 常见问题诊断

### 问题1: 全局拟合LOF比MCR-ALS高很多

**可能原因:**
- 动力学模型不合适
- 组分数量不对
- 初始参数估计不准

**解决方案:**
1. 尝试不同的动力学模型
2. 调整组分数量
3. 手动提供更好的初始值
4. 检查MCR-ALS结果是否合理

### 问题2: 拟合参数不稳定

**症状:**
- 多次运行得到不同结果
- 参数不确定度很大
- 相关系数接近1

**解决方案:**
1. 减少组分数量
2. 固定某些参数
3. 使用更严格的参数边界
4. 改进初始值估计

### 问题3: 全局拟合收敛失败

**解决步骤:**
1. 检查数据是否有NaN或Inf
2. 调整初始参数，使其更接近真实值
3. 放宽参数边界
4. 尝试不同的优化方法:
   ```python
   # 默认: 'leastsq'
   # 备选: 'least_squares', 'differential_evolution'
   results = gla.fit(
       tau_initial=tau_initial,
       optimization_method='least_squares'
   )
   ```

---

## 最佳实践

1. **始终保存中间结果**: 每个阶段的结果都应保存
2. **记录分析参数**: 使用配置文件记录所有设置
3. **多次验证**: 使用不同初始值多次运行以确认结果稳定
4. **物理合理性优先**: LOF低不一定代表模型正确
5. **查阅文献**: 与类似体系的文献结果对比
6. **保持怀疑**: 对异常结果保持警惕，仔细检查

---

## 进阶应用

### 自定义动力学模型

```python
from Globalfit import MixedModel

# 定义复杂的反应网络
# 例如: A → B → C 和 A → D (分支反应)
reaction_network = [
    (0, 1, 0),  # A --k0--> B
    (1, 2, 1),  # B --k1--> C
    (0, 3, 2)   # A --k2--> D
]

model = MixedModel(
    n_components=4,
    reaction_network=reaction_network
)

gta = GlobalTargetAnalysis(
    data_matrix=data_dict['data_matrix'],
    time_axis=data_dict['time_axis'],
    wavelength_axis=data_dict['wavelength_axis'],
    kinetic_model=model
)

# 需要3个速率常数
k_initial = [0.1, 0.05, 0.02]
results = gta.fit(k_initial=k_initial)
```

### 批量处理多个数据集

```python
import os
from pathlib import Path

mcr_results_dirs = [
    "results_sample1",
    "results_sample2",
    "results_sample3"
]

all_results = {}

for results_dir in mcr_results_dirs:
    print(f"\n处理: {results_dir}")
    
    interface = MCRALSInterface(results_dir)
    data_dict = interface.prepare_for_global_fitting()
    
    gla = GlobalLifetimeAnalysis(
        data_matrix=data_dict['data_matrix'],
        time_axis=data_dict['time_axis'],
        wavelength_axis=data_dict['wavelength_axis'],
        n_components=data_dict['n_components']
    )
    
    results = gla.fit(tau_initial=data_dict['lifetimes_initial'])
    all_results[results_dir] = results
    
    # 保存结果
    interface.save_global_fit_results(results)

# 比较不同样品的寿命
for sample, results in all_results.items():
    print(f"{sample}: τ = {results['tau_optimal']}")
```

---

## 附录: 文件格式说明

### MCR-ALS输出文件

**concentration_profiles.csv:**
```
Component_1,Component_2,Component_3
0.982,0.015,0.003
0.954,0.038,0.008
...
```

**pure_spectra.csv:**
```
Component_1,Component_2,Component_3
-0.0012,0.0045,0.0023
-0.0015,0.0052,0.0028
...
```

### 全局拟合输出文件

**global_fit_summary.json:**
```json
{
  "chi_square": 1.23e-06,
  "lof": 3.45,
  "computation_time": 15.67,
  "tau_optimal": [5.23, 87.45, 1234.56],
  "k_optimal": [0.191, 0.0114],
  "kinetic_model": "SequentialModel"
}
```

---

*文档版本: 1.0.0*  
*适用于: Globalfit v1.0.0*  
*最后更新: 2024*
