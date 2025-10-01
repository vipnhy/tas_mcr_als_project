# MCR-ALS 网格搜索实验指南

## 概述

此目录包含用于执行多轮 MCR-ALS（多变量曲线分辨-交替最小二乘）分析的批量运行脚本。该实验系统可以系统地测试不同参数组合，包括组分数量、惩罚因子和随机初始值，以评估最优配置。

## 实验设计

### 参数空间
- **组分数量**: 1-4个组分（可配置）
- **惩罚因子**: 0.1-1.0范围，可配置步长（默认0.1）
- **随机初始值**: 每个参数组合运行多次（默认5次）
- **约束配置**: 支持预设约束（default, standard, strict, relaxed）

### 输出结构
```
experiments/results/mcr_als_grid/outputs/run_YYYYMMDD_HHMMSS/
├── experiment_manifest.json    # 实验配置清单
├── summary.csv                 # 详细结果汇总
├── summary.json               # JSON格式详细结果
├── summary_aggregated.csv     # 聚合统计结果
├── summary_aggregated.json    # JSON格式聚合统计
└── [constraint]/              # 按约束分组
    └── ncomp_[N]/            # 按组分数分组
        └── pen_[P]/          # 按惩罚因子分组
            └── seed_[S]/     # 按随机种子分组
                ├── concentration_profiles.csv
                ├── pure_spectra.csv
                ├── mcr_als_results.png
                ├── data_comparison.png
                └── analysis_parameters.json
```

## 使用方法

### 基本命令
```bash
cd d:\TAS\tas_mcr_als_project
python experiments\scripts\mcr_als_grid\run_grid_search.py --data-file "data\TAS\TA_Average-crop-TCA-tzs-tzs-tzs-chirp.csv"
```

### 完整参数示例
```bash
python experiments\scripts\mcr_als_grid\run_grid_search.py \
    --data-file "data\TAS\TA_Average-crop-TCA-tzs-tzs-tzs-chirp.csv" \
    --components "1,2,3,4" \
    --penalty-min 0.1 \
    --penalty-max 1.0 \
    --penalty-step 0.1 \
    --random-runs 5 \
    --constraints "default,standard" \
    --max-iter 500 \
    --save-plots
```

### 参数说明
- `--data-file`: 输入数据文件路径
- `--components`: 组分数量配置（如 "1,2,3,4" 或 "1-4"）
- `--penalty-min/max/step`: 惩罚因子范围和步长
- `--random-runs`: 每个参数组合的随机化次数
- `--constraints`: 约束配置（用逗号分隔）
- `--max-iter`: 最大迭代次数（默认500）
- `--save-plots`: 是否保存可视化图表
- `--dry-run`: 仅生成实验计划，不执行分析

### 约束配置选项
- `default`: 默认约束（无特殊约束）
- `standard`: 标准约束配置
- `strict`: 严格约束配置
- `relaxed`: 宽松约束配置

## 输出文件说明

### 汇总文件
1. **summary.csv**: 包含每次运行的详细结果
   - constraint: 使用的约束配置
   - n_components: 组分数量
   - penalty: 惩罚因子
   - random_seed: 随机种子
   - final_lof: 最终LOF值
   - iterations: 迭代次数
   - status: 运行状态（success/failed/error）
   - output_dir: 输出目录路径

2. **summary_aggregated.csv**: 聚合统计结果
   - total_runs/successful_runs: 总运行次数和成功次数
   - avg_lof/std_lof/min_lof/max_lof: LOF统计指标
   - best_seed: 最佳LOF对应的随机种子
   - avg_iterations: 平均迭代次数

### 单次运行输出
每个参数组合的具体输出包括：
- **concentration_profiles.csv**: 浓度轮廓数据
- **pure_spectra.csv**: 纯组分光谱数据
- **mcr_als_results.png**: MCR-ALS分解结果图
- **data_comparison.png**: 数据对比图
- **analysis_parameters.json**: 分析参数记录

## 示例使用场景

### 1. 快速测试（小规模）
```bash
python experiments\scripts\mcr_als_grid\run_grid_search.py \
    --components "2,3" \
    --penalty-min 0.1 \
    --penalty-max 0.4 \
    --penalty-step 0.3 \
    --random-runs 2 \
    --constraints "default" \
    --max-iter 50
```

### 2. 完整网格搜索
```bash
python experiments\scripts\mcr_als_grid\run_grid_search.py \
    --components "1-4" \
    --penalty-min 0.1 \
    --penalty-max 1.0 \
    --penalty-step 0.1 \
    --random-runs 5 \
    --constraints "default,standard,strict,relaxed" \
    --save-plots
```

### 3. 仅生成实验计划
```bash
python experiments\scripts\mcr_als_grid\run_grid_search.py \
    --components "1-4" \
    --penalty-min 0.1 \
    --penalty-max 1.0 \
    --penalty-step 0.1 \
    --random-runs 5 \
    --constraints "default,standard" \
    --dry-run
```

## 结果分析

### 最优参数识别
1. 查看 `summary_aggregated.csv` 中的 `min_lof` 列，找到最低LOF值
2. 对应的 `constraint`, `n_components`, `penalty` 即为最优参数组合
3. `best_seed` 提供了获得最优结果的随机种子

### 稳定性评估
1. 查看 `std_lof` 列评估不同随机初始值的变异性
2. 较小的标准差表示更稳定的参数组合
3. `successful_runs`/`total_runs` 比值反映参数组合的可靠性

### 趋势分析
- 比较不同组分数量的LOF值，识别最适合的组分数
- 分析惩罚因子对收敛性和LOF值的影响
- 评估不同约束配置的效果

## 注意事项

1. **内存使用**: 大规模网格搜索可能消耗大量内存，建议分批运行
2. **运行时间**: 完整网格搜索可能需要数小时，建议先用 `--dry-run` 估算
3. **存储空间**: 每次运行生成多个文件，确保有足够磁盘空间
4. **并行化**: 当前实现为串行执行，大规模任务可考虑并行化改进

## 故障排除

### 常见错误
1. **FileNotFoundError**: 检查数据文件路径是否正确
2. **约束配置错误**: 确保约束名称在预设列表中
3. **内存不足**: 减少参数组合数量或增加虚拟内存

### 调试建议
1. 使用 `--dry-run` 验证参数配置
2. 先用小规模参数测试（少组分、少随机化）
3. 检查 `summary.csv` 中的 `status` 和 `message` 列

## 扩展功能

可通过修改 `run_grid_search.py` 添加：
- 新的约束配置
- 其他初始化方法
- 自定义评估指标
- 并行执行支持