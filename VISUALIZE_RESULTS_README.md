# TAS MCR-ALS 结果可视化工具

## 概述

这是一个专门为TAS（Transient Absorption Spectroscopy）分析结果设计的可视化工具，能够将MCR-ALS与全局拟合的输出结果转换为直观的图文报告。

## 功能特性

- **综合图表生成**: 自动生成包含浓度剖面、谱图、残差分析和质量指标的综合图表
- **详细文本解释**: 提供拟合质量评估、参数分析和专业建议
- **多语言支持**: 支持中文（默认）和英文报告，完全本地化的图表标签和解释
- **HTML报告**: 生成美观的HTML格式报告，便于分享和查看
- **命令行界面**: 支持命令行参数，方便集成到工作流中

## 使用方法

### 基本用法

```bash
python visualize_results.py "输出目录路径"
```

例如：
```bash
python visualize_results.py "analysis_runs/demo_vis_batch_20251002_102611/global_fit/components_4/run_02_seed_4009801342/sequential_a_to_b_to_c_to_d/attempt_02"
```

### 指定输出文件和语言

```bash
python visualize_results.py "输出目录路径" --output "自定义报告.html" --language zh
```

### 语言选项

- `--language zh` 或 `-l zh`: 中文报告（默认）
- `--language en` 或 `-l en`: 英文报告

### 查看帮助

```bash
python visualize_results.py --help
```

## 输入文件要求

工具需要以下输出文件存在于指定的目录中：

- `concentration_global_fit.csv`: 浓度剖面数据
- `spectra_global_fit.csv`: 谱数据
- `residuals.csv`: 残差数据
- `fit_report.txt`: 拟合报告
- `global_fit_summary.json`: 全局拟合摘要（可选）

## 输出文件

- `visualization_report.html`: HTML格式的详细报告
- `visualization_plots.png`: 综合图表PNG文件

## 依赖包

- pandas
- numpy
- matplotlib
- pathlib (Python标准库)
- json (Python标准库)
- argparse (Python标准库)

## 安装依赖

```bash
pip install pandas numpy matplotlib
```

## 报告内容

生成的HTML报告包含以下部分：

1. **概述**: 分析基本信息和输出目录
2. **结果解释**: 拟合质量概述、动力学参数分析、数据维度和分析建议
3. **可视化图表**: 综合图表展示
4. **拟合质量评估**: 详细的质量指标
5. **参数分析**: 参数值、不确定度和初始值表格
6. **数据摘要**: 数据维度和类型信息
7. **使用说明**: 重新生成报告的命令

## 注意事项

- 确保输出目录路径正确且包含所需的文件
- 中文字体可能显示为方块，这是matplotlib的已知问题，不影响功能
- 报告会覆盖同名文件，请注意备份重要报告

## 集成到工作流

可以将此工具集成到`advanced_analysis.py`或其他分析脚本中，作为后处理步骤：

```bash
# 生成中文可视化报告（默认）
python advanced_analysis.py --config config.json --visualize

# 生成英文可视化报告
python advanced_analysis.py --config config.json --visualize --vis-language en
```

或在代码中直接调用：

```python
from visualize_results import TASResultsVisualizer

# 中文报告（默认）
visualizer = TASResultsVisualizer(output_dir)
visualizer.generate_report()

# 英文报告
visualizer = TASResultsVisualizer(output_dir, language='en')
visualizer.generate_report()
```