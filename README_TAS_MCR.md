# TAS MCR-ALS 项目使用说明

## 项目概述

本项目实现了使用多元曲线分辨-交替最小二乘法 (MCR-ALS) 对瞬态吸收光谱 (TAS) 数据进行分析的完整工具包。

## 主要功能

### 1. 数据读取 (`data/data.py`)
- 支持读取 TAS 原始数据文件
- 提供波长范围和时间延迟范围的筛选功能
- 处理无穷值和缺失值
- 支持多种文件格式 (raw/handle)

### 2. MCR-ALS 分析 (`mcr/mcr_als.py`)
- 实现完整的 MCR-ALS 算法
- 支持自定义组分数量
- 包含收敛判断和迭代控制
- 提供拟合质量评估 (LOF)

### 3. 主分析程序 (`main.py` / `run_main.py`)
- 集成数据读取和 MCR-ALS 分析
- 自动可视化分析结果
- 支持实际 TAS 数据和合成数据
- 包含错误处理和回退机制

### 4. **全局拟合模块 (`Globalfit/`) ⭐ 新增**
- **全局寿命分析 (GLA)**: 多指数衰减拟合
- **全局目标分析 (GTA)**: 基于动力学模型的拟合
- **无缝集成**: 自动使用MCR-ALS结果作为输入
- **多种动力学模型**: 顺序反应、平行反应、混合模型
- **详细说明**: 参见 [Globalfit模块使用说明](Globalfit/README.md)

### 5. **高级批量工作流 (`analysis_tool/`) ⭐ 新增**
- **配置驱动**: 使用 `config_workflow.json` 等 JSON 文件配置数据输入、MCR 组分范围、预处理和全局拟合策略
- **批量执行**: `MCRBatchRunner` 自动遍历组分与初始化种子, 统一保存结果与摘要
- **多模型全局拟合**: `GlobalFitBatchRunner` 针对每个 MCR 结果尝试 GLA/GTA 多种模型并排序推荐
- **自动报告**: `AnalysisReporter` 生成 Markdown 报告, 汇总排名、路径与关键指标
- **一键入口**: `advanced_analysis.py` 将上述流程整合为单个 CLI, 支持参数覆盖与可重复随机种子

### 6. 测试模块 (`test/`)
- `test_MCR_ALS.py`: 基本 MCR-ALS 功能测试
- `test_real_data.py`: 实际 TAS 数据分析测试

## 完整分析流程 (MCR-ALS + 全局拟合)

### 全自动批量工作流 (推荐)

使用新增的 `advanced_analysis.py` 可以一键完成数据加载、MCR 批处理、全局拟合与报告生成：

```powershell
D:/TAS/tas_mcr_als_project/venv/Scripts/python.exe advanced_analysis.py --config config_workflow.json --verbose
```

- 默认在 `analysis_runs/` 下按时间戳创建工作目录
- 自动保存 `mcr/mcr_summary.json`、`global_fit/global_fit_summary.json` 以及最终 `final_report.md`
- 可通过命令行参数覆盖配置中的分析名称、光谱类型、组分范围与排序指标

执行完成后, 参阅工作目录内的 `final_report.md` 获取排序后的最佳模型与关键信息。

### 步骤1: 运行 MCR-ALS 分析
```bash
# 使用推荐的参数运行 MCR-ALS
python run_main.py ^
    --file_path data/TAS/TA_Average.csv ^
    --n_components 3 ^
    --wavelength_range 420 750 ^
    --delay_range 0.1 50 ^
    --save_plots ^
    --save_results ^
    --output_dir results
```

### 步骤2: 运行全局拟合分析
```bash
# 接着运行自动化全局拟合工作流
python Globalfit/examples/auto_workflow.py --mcr_results results
```
结果将保存在 `results/global_fit` 目录下。

---

## 单独运行 MCR-ALS

### 环境配置
```bash
# 安装依赖包 (包括全局拟合所需的lmfit)
pip install numpy matplotlib pandas scikit-learn scipy lmfit
```

### 运行主程序

#### 方法1: 直接运行 main.py
```bash
python main.py
```

#### 方法2: 使用参数化运行工具 run_main.py (推荐)

**命令行参数方式:**
```bash
# 基本用法
python run_main.py --file_path "data/TAS/TA_Average.csv" --n_components 3

# 完整参数
python run_main.py --file_path "data/TAS/TA_Average-crop-TCA-tzs-tzs-tzs-chirp.csv" --file_type handle --wavelength_range 420 750 --delay_range 0.1 50 --n_components 3 --save_plots --save_results
```

**配置文件方式:**
```bash
# 使用配置文件
python run_main.py --config config_example.json
```

**交互式输入方式:**
```bash
# 交互式参数输入
python run_main.py
```

### 运行测试
```bash
# 测试基本功能
python test/test_MCR_ALS.py

# 测试实际数据分析
python test/test_real_data.py
```

## 数据格式要求

### TAS 数据文件
- CSV 格式，第一列为时间延迟，第一行为波长
- 数据矩阵：行为时间点，列为波长点
- 支持的文件类型：
  - `"raw"`: 原始数据格式
  - `"handle"`: 处理后的数据格式

### 参数设置
- `file_path`: TAS数据文件路径
- `file_type`: 文件类型 ("handle" 为处理后数据, "raw" 为原始数据)
- `wavelength_range`: 波长范围筛选，例如 (420, 750)
- `delay_range`: 时间延迟范围筛选，例如 (0.1, 50)
- `n_components`: MCR-ALS 组分数量，建议从 2-4 开始尝试

### run_main.py 参数详解

#### 必需参数
- `--file_path`: TAS数据文件的路径

#### 可选参数
- `--file_type`: 文件类型，默认 "handle"
- `--wavelength_range`: 波长范围，默认 [400, 800]
- `--delay_range`: 时间延迟范围，默认 [0, 10]
- `--n_components`: 组分数量，默认 3
- `--max_iter`: 最大迭代次数，默认 200
- `--tol`: 收敛容差，默认 1e-7
- `--save_plots`: 保存图表到文件
- `--save_results`: 保存数值结果到文件
- `--output_dir`: 输出目录，默认 "results"
- `--language`: 界面语言，可选 "chinese" 或 "english"，默认 "chinese"
- `--config`: 从JSON配置文件读取参数

#### 配置文件格式 (JSON)
```json
{
  "file_path": "data/TAS/TA_Average.csv",
  "file_type": "handle",
  "wavelength_range": [420, 750],
  "delay_range": [0.1, 50],
  "n_components": 3,
  "language": "chinese"
}
```

## 输出结果

### 可视化图表
1. **浓度轮廓 (Concentration Profiles)**：各组分随时间的变化
2. **纯光谱 (Pure Spectra)**：各组分的特征光谱
3. **LOF 收敛曲线**：算法收敛过程
4. **残差图 (Residuals)**：拟合质量评估
5. **原始数据与重构数据对比**

### 数值结果
- `C_resolved`: 浓度矩阵 (时间 × 组分)
- `S_resolved`: 光谱矩阵 (波长 × 组分)
- `lof_`: 拟合缺失度历史
- `residuals_`: 残差矩阵

## 可视化改进

### y轴方向修正
为了符合常见的科学绘图习惯，所有热图的y轴都已调整为：
- **0值在下方**
- **最大值在上方**

这通过以下参数实现：
```python
# 设置正确的extent和origin参数
extent=[wavelength_axis.min(), wavelength_axis.max(),
       time_axis.min(), time_axis.max()],  # 注意：min在前，max在后
origin='lower'  # 确保y轴从底部开始
```

### 图表类型
1. **原始TAS数据热图**：显示实验测量的瞬态吸收信号
2. **重构数据热图**：显示MCR-ALS重构的数据
3. **残差热图**：显示原始数据与重构数据的差异

## 项目结构
```
tas_mcr_als_project/
├── main.py                    # 主分析程序
├── run_main.py               # 参数化运行工具 (推荐使用)
├── config_example.json       # 配置文件示例 (中文)
├── config_english.json       # 英文配置文件示例
├── config_detailed.json      # 详细配置文件示例
├── requirements.txt          # 依赖包列表
├── data/
│   ├── data.py              # 数据读取模块
│   └── TAS/                 # TAS 数据文件夹
├── mcr/
│   ├── mcr_als.py          # MCR-ALS 算法实现
│   ├── constraints.py      # 约束条件
│   ├── initializers.py     # 初始化方法
│   └── constraint_config.py # 约束配置
├── Globalfit/              # ⭐ 全局拟合模块
│   ├── __init__.py
│   ├── kinetic_models.py   # 动力学模型
│   ├── model.py            # GLA和GTA核心算法
│   ├── interface.py        # MCR-ALS接口
│   ├── utils.py            # 工具函数
│   ├── examples/           # 示例脚本
│   │   ├── auto_workflow.py          # 自动化工作流程
│   │   └── run_global_fit_example.py # 详细示例
│   ├── docs/               # 文档
│   │   ├── README_GLOBALFIT.md       # 完整说明
│   │   └── WORKFLOW_GUIDE.md         # 工作流程指南
│   ├── tests/              # 测试
│   │   └── test_basic_functionality.py
│   └── README.md           # 模块说明
├── test/
│   ├── test_MCR_ALS.py     # MCR-ALS 功能测试
│   └── test_real_data.py   # 实际数据测试
├── utils/
│   ├── metrics.py          # 评估指标
│   └── plotting.py         # 绘图工具
├── preprocessing/          # 数据预处理
├── experiments/            # 实验脚本
├── flask/                  # Web界面 (可选)
└── results/                # 输出结果目录 (自动创建)
```

## 在测试文件夹中导入主目录模块的方法

### 方法 1: 添加路径到 sys.path (当前使用)
```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 然后正常导入
from mcr.mcr_als import MCRALS
from data.data import read_file
```

### 方法 2: 使用相对导入
```python
# 需要将项目设为包（添加 __init__.py）
from ..mcr.mcr_als import MCRALS
from ..data.data import read_file
```

### 方法 3: 使用 PYTHONPATH 环境变量
```bash
set PYTHONPATH=%PYTHONPATH%;d:\TAS\tas_mcr_als_project
python test/test_MCR_ALS.py
```

## 注意事项

1. **数据质量检查**：运行前检查数据中的 NaN 和无穷值
2. **组分数量选择**：建议从少量组分开始，逐步增加
3. **收敛判断**：注意 LOF 值，通常应低于几个百分点
4. **结果解释**：MCR-ALS 结果可能存在旋转模糊性，需要结合化学知识解释

## 常见问题解决

### 导入错误
- 确保已添加项目根目录到 Python 路径
- 检查虚拟环境是否正确激活
- 验证依赖包是否正确安装

### 数据读取失败
- 检查文件路径是否正确
- 确认文件格式是否符合要求
- 尝试不同的 `file_type` 参数

### MCR-ALS 收敛问题
- 增加最大迭代次数
- 调整收敛容差
- 尝试不同的组分数量
- 检查数据预处理是否恰当

## 扩展功能

### run_main.py 的优势
1. **参数化运行**: 无需修改源代码即可改变分析参数
2. **批量处理**: 可以通过脚本批量处理多个文件
3. **结果保存**: 自动保存分析结果和图表
4. **配置管理**: 支持配置文件，便于参数管理和重现分析
5. **交互式模式**: 适合初学者的友好界面
6. **多语言支持**: 支持中文和英文界面，图表标签自动适配
7. **字体优化**: 自动处理中文字体显示问题

### 使用示例

#### 示例1: 快速分析单个文件
```bash
& "D:/TAS/tas_mcr_als_project/venv/Scripts/python.exe" run_main.py --file_path "data/TAS/TA_Average.csv" --n_components 3 --save_plots
```

#### 示例2: 优化波长和时间范围
```bash
& "D:/TAS/tas_mcr_als_project/venv/Scripts/python.exe" run_main.py --file_path "data/TAS/TA_Average.csv" --wavelength_range 450 700 --delay_range 1 100 --n_components 4 --save_results --output_dir "results_optimized"
```

#### 示例3: 使用配置文件进行标准化分析
```bash
# 中文界面
& "D:/TAS/tas_mcr_als_project/venv/Scripts/python.exe" run_main.py --config config_example.json

# 英文界面
& "D:/TAS/tas_mcr_als_project/venv/Scripts/python.exe" run_main.py --config config_english.json
```

#### 示例4: 指定语言和输出目录
```bash
# 英文界面，保存结果
& "D:/TAS/tas_mcr_als_project/venv/Scripts/python.exe" run_main.py --file_path "data/TAS/TA_Average.csv" --language english --save_plots --save_results --output_dir "results_en"
```

## 完整分析工作流程: MCR-ALS → 全局拟合

本项目提供了从MCR-ALS分析到全局拟合的完整自动化工作流程。

### 步骤1: 运行MCR-ALS分析

```bash
# 使用run_main.py进行MCR-ALS分析
python run_main.py --file_path "data/TAS/TA_Average.csv" \
    --n_components 3 \
    --wavelength_range 420 750 \
    --delay_range 0.1 50 \
    --save_plots \
    --save_results \
    --output_dir "results"
```

这将在 `results/` 目录下生成:
- `concentration_profiles.csv` - 浓度矩阵
- `pure_spectra.csv` - 纯光谱矩阵
- `lof_history.csv` - LOF收敛历史
- `analysis_parameters.json` - 分析参数
- `mcr_als_results.png` - 可视化结果

### 步骤2: 运行全局拟合分析

```bash
# 切换到Globalfit示例目录
cd Globalfit/examples

# 运行自动化全局拟合工作流程
python auto_workflow.py --mcr_results ../../results --data_file ../../data/TAS/TA_Average.csv
```

或者只运行特定的分析方法:

```bash
# 只运行GLA (全局寿命分析)
python auto_workflow.py --mcr_results ../../results --method gla

# 只运行GTA顺序模型 (全局目标分析)
python auto_workflow.py --mcr_results ../../results --method gta --model sequential

# 只运行GTA平行模型
python auto_workflow.py --mcr_results ../../results --method gta --model parallel
```

全局拟合结果将保存在 `results/global_fit/` 目录下 (默认仅运行 GTA 模型):
```
results/global_fit/
├── sequential_<pathway>/         # 顺序模型结果 (如 sequential_a_to_b_to_c)
├── parallel_<branches>/          # 平行模型结果 (如 parallel_a_to_c__b_to_c)
├── mixed_direct_<pathway>/       # 混合模型 - 直接
├── mixed_reversible_<pathway>/   # 混合模型 - 可逆
└── comparison_mcr_*.png          # 对比图表
```

### 方法比较

| 方法 | 说明 | 优点 | 适用场景 |
|------|------|------|----------|
| **MCR-ALS** | 无模型假设的分解 | 快速、灵活 | 初步分析，组分分离 |
| **GLA** | 多指数衰减拟合 | 简单、直观 | 获取特征寿命 |
| **GTA (顺序)** | A→B→C模型 | 物理意义明确 | 串联反应机理 |
| **GTA (平行)** | A→B,C,D模型 | 物理意义明确 | 并行衰减通道 |

### 详细文档

- [Globalfit模块完整说明](Globalfit/docs/README_GLOBALFIT.md)
- [完整工作流程指南](Globalfit/docs/WORKFLOW_GUIDE.md)
- [Globalfit快速开始](Globalfit/README.md)

---

可以进一步添加的功能：
- 更多约束条件（非负性、单峰性等）
- 不同的初始化方法
- 自动组分数量选择
- 批量数据处理
- 更多动力学模型
