# MCR-ALS 实验框架使用指南

## 概述

本实验框架是为MCR-ALS算法设计的综合性实验系统，支持多轮分析、约束测试和参数扩展性评估。框架按照要求实现了以下核心功能：

### ✅ 核心特性
- **多轮MCR-ALS分析**：支持初始值随机化5次（可配置），引入可复现/时间随机两种种子策略
- **约束LOF值记录**：追踪不同约束下的LOF值，并将关键统计导出为CSV/JSON，目标LOF < 0.2%
- **参数扩展性测试**：
  - 组分数量扩展测试（默认1→4组分，可配置）
  - 平滑度约束强度梯度测试（惩罚因子默认0.1→1.0，可通过构造函数自定义）
- **分级目录结构**：自动创建多层级目录，并在实验根目录同步保存核心汇总与配置文件
- **全面分析报告**：生成JSON、CSV、Excel（可选）等多格式分析结果，便于二次处理
- **可视化图表**：自动生成性能对比、扩展性分析、参数敏感性与收敛性图表

## 目录结构


> 💡 想要调整平滑度强度测试范围？在创建 `MCRExperimentRunner` 时传入 `smoothness_strengths=[...]` 即可，框架会自动生成新的 `smoothness_X` 配置，并在汇总文件中同步记录。
实验框架会自动创建如下分级目录结构：

```
mcr_experiments/
└── experiment_YYYYMMDD_HHMMSS/
  ├── experiment_summary.json         # 顶级目录中的核心汇总拷贝
  ├── experiment_config.json          # 本次实验的运行配置
  ├── level1_summary/                 # 一级汇总（分析汇总入口）
  │   ├── experiment_summary.json     # 主要汇总报告（原始文件）
  │   ├── experiment_results.xlsx     # Excel格式结果（需要openpyxl，可选安装）
  │   └── all_results.csv             # 全部运行记录（平铺数据）
  ├── level2_constraint_analysis/     # 约束分析
  │   ├── constraint_analysis.json
  │   └── constraint_lof_summary.csv  # 各约束LOF表现汇总
  ├── level3_component_scaling/       # 组分扩展性分析
  │   ├── component_scaling.json
  │   └── component_lof_summary.csv   # 组分数量表现汇总
    ├── level4_parameter_tuning/        # 参数调优分析
    │   └── parameter_tuning.json
    ├── level5_individual_runs/         # 单次实验详细结果
    │   ├── basic_comp1_seed42/
    │   │   ├── summary.json
    │   │   ├── concentration_matrix.csv
    │   │   ├── spectra_matrix.csv
    │   │   └── lof_history.csv
    │   └── ...
    ├── plots/                          # 可视化图表
    │   ├── constraint_performance.png
    │   ├── component_scaling.png
    │   ├── parameter_tuning.png
    │   └── convergence_analysis.png
  └── data/                           # 平铺实验数据
  └── lof_records.csv             # 各约束运行的LOF详表
```

## 快速开始

### 基本使用

```python
from mcr_experiment import run_example_experiment

# 运行完整实验示例
runner = run_example_experiment()
```

### 命令行快速实验

想直接验证默认流程，可按以下步骤操作：

1. 安装依赖：`pip install -r requirements.txt`
2. （可选）若需Excel汇总：`pip install openpyxl`
3. 进入项目根目录后执行：`python mcr_experiment.py`

脚本会调用 `run_example_experiment()`，自动生成合成数据并运行完整多轮实验，输出结果位于 `mcr_experiments/experiment_时间戳/`。

### 自定义实验

```python
from mcr_experiment import MCRExperimentRunner
from main import generate_synthetic_tas_data

# 生成测试数据
data_matrix, C_true, S_true = generate_synthetic_tas_data(
    n_times=100, n_wls=200, n_components=3
)

# 创建实验运行器，可自定义平滑度梯度和随机种子策略
runner = MCRExperimentRunner(
  output_base_dir="my_experiments",
  smoothness_strengths=[0.1, 0.2, 0.5, 1.0],  # 自定义惩罚因子范围
  base_random_seed="time"                     # 使用实时种子获得完全随机的初始化
)

# 运行实验
runner.run_multi_round_experiment(
    data_matrix=data_matrix,
    n_components_range=[1, 2, 3, 4],    # 测试组分数量
    num_random_runs=5,                  # 每个配置随机运行5次
    max_iter=200,                       # 最大迭代次数
    tolerance=1e-6,                     # 收敛容差
    target_lof=0.2                      # 目标LOF值 (%)
)
```

> ℹ️ 默认情况下，`base_random_seed` 使用固定值保证结果可复现；传入 `"time"` 可让每次实验使用不同的随机种子。`smoothness_strengths` 也可以传入任意升序列表，以扩展或缩小惩罚因子测试范围。

## 约束配置

框架支持多种约束类型和强度测试：

### 内置约束类型
- **basic**: 基本约束（仅非负性）
- **smoothness_X**: 光谱平滑度约束，X为惩罚因子(0.1, 0.2, 0.5, 1.0)
- **combined**: 组合约束（非负性 + 平滑度）

### 约束强度梯度测试
惩罚因子范围：0.1 → 0.2 → 0.5 → 1.0

```python
# 框架自动创建不同强度的约束配置
configurations = runner._create_constraint_configurations()
```

## 实验结果分析

### 一级汇总报告 (level1_summary/)

包含实验的核心统计信息：

```json
{
  "experiment_metadata": {
    "timestamp": "20250929_091253",
    "total_experiments": 120,
    "target_lof": 0.2,
    "successful_convergence": 90,
    "target_lof_achieved": 0
  },
  "overall_performance": {
    "success_rate": 75.0,
    "target_achievement_rate": 0.0,
    "average_lof": 39.32,
    "best_lof": 14.35,
    "optimal_component_count": 3
  }
}
```

- `experiment_summary.json` 同步拷贝至实验根目录，便于快速查看。
- `all_results.csv` 为全部运行的平铺数据，可直接用于Excel或Pandas分析。
- 如果安装了 `openpyxl`，还会生成 `experiment_results.xlsx`（含约束/组分多表汇总）。

### 关键性能指标

- **成功率 (success_rate)**: 收敛成功的实验比例
- **目标达成率 (target_achievement_rate)**: 达到目标LOF的实验比例  
- **最佳LOF (best_lof)**: 所有实验中的最低LOF值
- **最优组分数量 (optimal_component_count)**: 性能最佳的组分数量

### 约束性能分析 (level2_constraint_analysis/)

比较不同约束类型的性能：

- 各约束类型的成功率和LOF表现
- 最佳约束类型推荐
- 约束强度与性能的关系
- `constraint_lof_summary.csv` 提供按约束汇总的完整表格，方便筛选目标LOF < 0.2% 的配置。

### 组分扩展性分析 (level3_component_scaling/)

评估组分数量对算法性能的影响：

- 不同组分数量的LOF表现
- 计算复杂度评估
- 扩展性评级（优秀/良好/受限）
- `component_lof_summary.csv` 汇总每个组分数量的关键指标。

### 参数调优分析 (level4_parameter_tuning/)

分析惩罚因子对算法的影响：

- 参数敏感性分析
- 最优参数推荐
- 性能稳定性评估

### 平铺数据记录 (data/)

`data/lof_records.csv` 为所有实验运行的明细表，包含约束类型、组分数量、随机种子、最终LOF、是否收敛以及计算时间，可直接导入数据分析工具进行进一步筛选与可视化。

### 实验配置记录 (experiment_config.json)

实验根目录与 `level1_summary/` 下都会生成 `experiment_config.json`，记录本次运行的全部关键参数，包括：

- 组分数量范围、随机运行次数、迭代次数等基础设置
- 平滑度强度列表与约束名称
- 随机种子策略与基础随机种子值（便于复现实验）
- 输出目录信息与时间戳

该文件可用于审计实验、复现设置或与团队共享实验方案。

## 可视化图表

框架自动生成四类分析图表：

1. **约束性能对比** (`constraint_performance.png`)
   - LOF均值对比
   - 成功率对比
   - 计算时间对比
   - LOF分布箱线图

2. **组分扩展性** (`component_scaling.png`)
   - LOF vs 组分数量
   - 成功率 vs 组分数量
   - 计算时间 vs 组分数量

3. **参数调优** (`parameter_tuning.png`)
   - LOF vs 平滑度强度
   - 目标达成率 vs 平滑度强度

4. **收敛性分析** (`convergence_analysis.png`)
   - 迭代次数分布
   - LOF vs 迭代次数
   - 计算时间 vs LOF
   - 成功率饼图

## 实验示例

### 示例1：标准实验流程

```python
# 完整的标准实验
from mcr_experiment import run_example_experiment

runner = run_example_experiment()

# 实验统计
print(f"总实验次数: {len(runner.results)}")
print(f"成功收敛: {sum(1 for r in runner.results if r.converged)}")
print(f"达到目标LOF: {sum(1 for r in runner.results if r.final_lof < 0.2)}")
```

### 示例2：小规模测试

```python
# 快速测试（较少实验）
from mcr_experiment import MCRExperimentRunner
from main import generate_synthetic_tas_data

data_matrix, _, _ = generate_synthetic_tas_data(n_times=50, n_wls=80, n_components=2)
runner = MCRExperimentRunner(output_base_dir="quick_test")

runner.run_multi_round_experiment(
    data_matrix=data_matrix,
    n_components_range=[1, 2],  # 只测试1-2组分
    num_random_runs=3,          # 每个配置3次
    target_lof=0.2
)
```

### 示例3：自定义参数范围

```python
# 扩展组分数量测试
runner.run_multi_round_experiment(
    data_matrix=data_matrix,
    n_components_range=[1, 2, 3, 4, 5, 6],  # 扩展到6组分
    num_random_runs=10,                      # 更多随机运行
    max_iter=500,                            # 更高迭代限制
    target_lof=0.1                           # 更严格的目标LOF
)
```

## 结果文件说明

### JSON格式结果
- 便于程序化分析
- 包含完整的数值结果
- 支持进一步的统计分析

### Excel格式结果
- 便于人工查看和分析
- 包含多个工作表的分类汇总
- 支持数据透视和图表制作

> 💡 **可选依赖提醒**：若需要生成 `experiment_results.xlsx`，请先安装 `openpyxl`（`pip install openpyxl`）。未安装时框架会自动跳过Excel导出，仅保留JSON/CSV输出，并在控制台给出提示。

### CSV格式矩阵
- 浓度矩阵和光谱矩阵的原始数据
- LOF收敛历史记录
- 便于外部工具分析

## 性能优化建议

### 实验规模控制
- **小规模测试**: 1-2组分，2-3次随机运行
- **标准测试**: 1-4组分，5次随机运行  
- **全面测试**: 1-6组分，10次随机运行

### 计算资源考虑
- 大数据矩阵建议减少随机运行次数
- 高组分数量测试需要更多迭代次数
- 可调整tolerance参数平衡精度和速度

## 扩展和定制

### 添加新约束类型

```python
# 在MCRExperimentRunner._create_constraint_configurations()中添加
custom_config = ConstraintConfig()
# 配置自定义约束...
configurations["custom"] = custom_config
```

### 修改分析指标

```python
# 在相关分析函数中添加新的统计指标
def _create_level1_summary(self, df, target_lof):
    # 添加自定义分析...
    summary["custom_metric"] = calculate_custom_metric(df)
    return summary
```

### 自定义可视化

```python
# 添加新的绘图函数
def _plot_custom_analysis(self, df):
    # 自定义图表代码...
    plt.savefig(self.current_experiment_dir / "plots" / "custom_analysis.png")
```

## 常见问题

### Q: 为什么没有实验达到目标LOF<0.2%？
A: 这是正常的，因为：
- 合成数据的复杂性和噪声水平
- MCR-ALS算法的固有限制
- 可以调整tolerance、max_iter或使用更高质量的数据

### Q: 如何提高实验成功率？
A: 建议：
- 增加最大迭代次数(max_iter)
- 调整收敛容差(tolerance)
- 使用合适的约束配置
- 确保数据质量和预处理

### Q: 实验结果如何重现？
A: 框架使用确定性种子序列，相同配置下结果可重现

## 技术支持

如需技术支持或功能改进建议，请参考项目文档或联系开发团队。

---

*实验框架版本: 1.0*  
*最后更新: 2024年9月*