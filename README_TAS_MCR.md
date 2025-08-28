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

### 3. 主分析程序 (`main.py`)
- 集成数据读取和 MCR-ALS 分析
- 自动可视化分析结果
- 支持实际 TAS 数据和合成数据
- 包含错误处理和回退机制

### 4. 测试模块 (`test/`)
- `test_MCR_ALS.py`: 基本 MCR-ALS 功能测试
- `test_real_data.py`: 实际 TAS 数据分析测试

## 使用方法

### 环境配置
```bash
# 激活虚拟环境
D:/TAS/tas_mcr_als_project/venv/Scripts/python.exe

# 安装依赖包
pip install numpy matplotlib pandas scikit-learn scipy
```

### 运行主程序
```bash
cd "d:\TAS\tas_mcr_als_project"
& "D:/TAS/tas_mcr_als_project/venv/Scripts/python.exe" main.py
```

### 运行测试
```bash
# 测试基本功能
& "D:/TAS/tas_mcr_als_project/venv/Scripts/python.exe" test/test_MCR_ALS.py

# 测试实际数据分析
& "D:/TAS/tas_mcr_als_project/venv/Scripts/python.py" test/test_real_data.py
```

## 数据格式要求

### TAS 数据文件
- CSV 格式，第一列为时间延迟，第一行为波长
- 数据矩阵：行为时间点，列为波长点
- 支持的文件类型：
  - `"raw"`: 原始数据格式
  - `"handle"`: 处理后的数据格式

### 参数设置
- `wavelength_range`: 波长范围筛选，例如 (420, 750)
- `delay_range`: 时间延迟范围筛选，例如 (0.1, 50)
- `n_components`: MCR-ALS 组分数量，建议从 2-4 开始尝试

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
├── requirements.txt           # 依赖包列表
├── data/
│   ├── data.py               # 数据读取模块
│   └── TAS/                  # TAS 数据文件夹
├── mcr/
│   ├── mcr_als.py           # MCR-ALS 算法实现
│   ├── constraints.py       # 约束条件
│   └── initializers.py      # 初始化方法
├── test/
│   ├── test_MCR_ALS.py      # 基本功能测试
│   └── test_real_data.py    # 实际数据测试
└── utils/
    ├── metrics.py           # 评估指标
    └── plotting.py          # 绘图工具
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

可以进一步添加的功能：
- 更多约束条件（非负性、单峰性等）
- 不同的初始化方法
- 自动组分数量选择
- 批量数据处理
- 结果导出功能
