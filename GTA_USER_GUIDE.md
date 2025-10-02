# GTA Analysis Module User Guide

## 概述

本模块为TAS MCR-ALS项目实现了完整的Global Target Analysis (GTA)功能，基于EfsTA的设计原理，为瞬态吸收光谱数据分析提供了强大的动力学建模和参数拟合能力。

## 主要特性

### 1. 预定义动力学模型 (8种)

基于EfsTA的设计，提供8个经过验证的动力学模型：

- **Model 1**: `A → B → C → ... → Z → 0` (最后物种衰减到基态)
- **Model 2**: `A → B → C → ... → Z` (最后物种稳定)
- **Model 3**: `A → B → C → D; B → D` (B→D分支)
- **Model 4**: `A → B → C → D → E; B → E` (B→E分支)
- **Model 5**: `A → B → C → D → E; C → E` (C→E分支)
- **Model 6**: `A → B → C → D → E → F; C → F` (C→F分支)
- **Model 7**: `A → B; A → C` (A的平行分支)
- **Model 8**: `A → B; B → C; B → D` (B的顺序+平行分支)

### 2. 自定义反应方程

支持灵活的反应方程语法：
- 箭头记号: `A->B->C->v` 或 `A→B→C→v`
- 紧凑记号: `ABCv`
- 分支反应: `A->B->C->v;B->A;A->C`
- 基态衰减: `v` 表示衰减到基态

### 3. 高级参数优化

- 使用lmfit进行参数拟合
- 支持参数边界约束
- 支持固定参数
- 多种优化算法支持
- 置信区间计算

### 4. 综合可视化

- 物种关联光谱 (SAS)
- 浓度时间演化曲线
- 2D残差分析
- 拟合质量评估
- 双语支持 (中文/英文)

## 快速开始

### 基本使用

```python
from gta_analysis import GTASolver, KineticModels, GTAVisualization

# 1. 准备数据
wavelengths = np.linspace(400, 700, 100)  # nm
delays = np.logspace(-1, 3, 50)  # ps
experimental_data = ...  # 形状: (delays x wavelengths)

# 2. 创建动力学模型
rate_constants = [1.0, 0.5]  # ps^-1
kinetic_model = KineticModels(rate_constants)
K, n_species = kinetic_model.get_kinetic_matrix(2)  # Model 2

# 3. 初始化求解器
solver = GTASolver(wavelengths, delays, experimental_data)
solver.set_kinetic_matrix(K)

# 4. 参数拟合
fit_result = solver.fit_parameters(
    initial_rates=rate_constants,
    parameter_bounds=[(0.1, 10.0), (0.01, 5.0)],
    method='leastsq'
)

# 5. 结果可视化
viz = GTAVisualization(language='zh')
fig_sas = viz.plot_species_spectra(wavelengths, solver.species_spectra)
fig_conc = viz.plot_concentration_profiles(delays, solver.concentration_profiles)
```

### 自定义反应模型

```python
from gta_analysis import ReactionParser

# 创建自定义反应方程
parser = ReactionParser()
equation = "A->B->C->D;B->D"  # 顺序反应加分支
rate_constants = [1.0, 0.5, 0.3, 0.2]

# 解析反应方程
K, n_species, info = parser.parse_reaction_equation(equation, rate_constants)

# 使用自定义模型进行分析
solver = GTASolver(wavelengths, delays, experimental_data)
solver.set_kinetic_matrix(K)
fit_result = solver.fit_parameters(initial_rates=rate_constants)
```

### 配置系统集成

```python
from gta_analysis import create_gta_config_from_template, GTAIntegration

# 创建配置
config = create_gta_config_from_template(
    model_type="sequential_stable",
    rate_constants=[1.0, 0.5, 0.1],
    parameter_bounds=[(0.1, 10.0), (0.01, 5.0), (0.001, 1.0)]
)

# 转换为globalfit格式
integration = GTAIntegration()
globalfit_config = integration.convert_to_globalfit_config(config)
```

## 详细功能说明

### 动力学模型类 (KineticModels)

#### 主要方法

- `get_kinetic_matrix(model_number)`: 获取指定模型的动力学矩阵
- `get_model_info(model_number)`: 获取模型信息
- `list_models()`: 列出所有可用模型
- `validate_model(model_number)`: 验证模型是否可用

#### 使用示例

```python
from gta_analysis import KineticModels

# 创建动力学模型
k_values = [1.0, 0.5, 0.1]  # 速率常数
model = KineticModels(k_values)

# 查看可用模型
model.list_models()

# 获取Model 3的动力学矩阵
K, n_species = model.get_kinetic_matrix(3)
print(f"动力学矩阵形状: {K.shape}")
print(f"物种数量: {n_species}")

# 获取模型信息
info = model.get_model_info(3)
print(f"模型描述: {info['description']}")
print(f"反应类型: {info['pathway']}")
```

### 反应方程解析器 (ReactionParser)

#### 支持的语法

1. **箭头记号**: `A->B->C->v`
2. **紧凑记号**: `ABCv`
3. **分支反应**: `A->B->C->v;B->A;A->C`
4. **基态衰减**: `v` 代表基态
5. **物种命名**: A-Z (最多26个物种)

#### 主要方法

- `parse_reaction_equation(equation, rate_constants)`: 解析反应方程
- `validate_equation_syntax(equation)`: 验证方程语法
- `get_equation_info(equation)`: 获取方程信息
- `suggest_rate_constants(equation)`: 建议速率常数

#### 使用示例

```python
from gta_analysis import ReactionParser

parser = ReactionParser()

# 解析简单顺序反应
equation = "A->B->C->v"
rate_constants = [1.0, 0.5, 0.1]

K, n_species, info = parser.parse_reaction_equation(equation, rate_constants)

print(f"物种: {info['species']}")
print(f"反应数: {info['n_reactions']}")
print(f"路径类型: {info['pathway_type']}")

# 解析分支反应
branching_eq = "A->B->C->D;B->D"
branching_rates = [2.0, 1.0, 0.5, 0.3]

K_branch, n_branch, info_branch = parser.parse_reaction_equation(
    branching_eq, branching_rates
)

# 验证方程语法
is_valid, message = parser.validate_equation_syntax("A->B->X")
print(f"有效: {is_valid}, 消息: {message}")
```

### GTA求解器 (GTASolver)

#### 主要功能

1. **ODE求解**: 求解物种浓度时间演化
2. **参数优化**: 使用lmfit进行非线性拟合
3. **SAS计算**: 计算物种关联光谱
4. **残差分析**: 计算和分析拟合残差

#### 主要方法

- `set_kinetic_matrix(K, initial_concentrations)`: 设置动力学矩阵
- `solve_concentration_profiles(rate_constants)`: 求解浓度演化
- `fit_parameters(initial_rates, ...)`: 参数拟合
- `calculate_confidence_intervals(fit_result)`: 计算置信区间
- `set_solver_options(ode_method, ...)`: 设置求解器选项

#### 使用示例

```python
from gta_analysis import GTASolver

# 初始化求解器
solver = GTASolver(wavelengths, delays, experimental_data)

# 设置动力学矩阵
K = np.array([[-1.0, 0.0], [1.0, -0.5]])
solver.set_kinetic_matrix(K)

# 设置求解器选项
solver.set_solver_options(
    ode_method='BDF',
    rtol=1e-8,
    atol=1e-10
)

# 参数拟合
fit_result = solver.fit_parameters(
    initial_rates=[0.8, 0.4],
    parameter_bounds=[(0.1, 10.0), (0.01, 5.0)],
    fixed_parameters=[False, False],
    method='leastsq'
)

# 检查拟合结果
print(f"拟合成功: {fit_result['success']}")
print(f"拟合速率: {fit_result['fitted_rates']}")
print(f"参数误差: {fit_result['parameter_errors']}")
print(f"约化χ²: {fit_result['reduced_chi_squared']}")

# 计算置信区间
ci = solver.calculate_confidence_intervals(fit_result)
print(f"置信区间: {ci}")

# 获取分析摘要
summary = solver.get_analysis_summary()
print(f"相对误差: {summary['relative_error_percent']:.2f}%")
```

### 可视化模块 (GTAVisualization)

#### 支持的图表类型

1. **物种关联光谱 (SAS)**
2. **浓度时间演化**
3. **2D残差热图**
4. **实验数据vs拟合数据对比**
5. **拟合质量综合评估**

#### 主要方法

- `plot_species_spectra()`: 绘制SAS光谱
- `plot_concentration_profiles()`: 绘制浓度演化
- `plot_residuals_2d()`: 绘制2D残差
- `plot_data_vs_fit_comparison()`: 绘制数据对比
- `plot_fit_quality_summary()`: 绘制拟合质量摘要
- `create_comprehensive_report()`: 创建完整报告

#### 使用示例

```python
from gta_analysis import GTAVisualization

# 初始化可视化（中文）
viz = GTAVisualization(language='zh', style='scientific')

# 绘制物种关联光谱
fig_sas = viz.plot_species_spectra(
    wavelengths=wavelengths,
    species_spectra=solver.species_spectra,
    species_names=['物种 A', '物种 B', '物种 C'],
    title="GTA分析 - 物种关联光谱",
    save_path="sas_spectra.png"
)

# 绘制浓度演化（对数时间轴）
fig_conc = viz.plot_concentration_profiles(
    delays=delays,
    concentration_profiles=solver.concentration_profiles,
    log_scale=True,
    save_path="concentration_profiles.png"
)

# 绘制残差分析
fig_res = viz.plot_residuals_2d(
    wavelengths=wavelengths,
    delays=delays,
    residuals=solver.residuals,
    save_path="residuals_2d.png"
)

# 绘制拟合质量摘要
fig_quality = viz.plot_fit_quality_summary(
    fit_result=fit_result,
    save_path="fit_quality.png"
)

# 创建完整报告
saved_files = viz.create_comprehensive_report(
    gta_solver=solver,
    fit_result=fit_result,
    output_dir="gta_analysis_report"
)
print(f"保存了 {len(saved_files)} 个图表文件")
```

### 配置系统 (GTAConfig & GTAIntegration)

#### 配置管理

```python
from gta_analysis import GTAConfig

config_manager = GTAConfig()

# 创建配置
config = config_manager.create_gta_config(
    model_type="sequential_b_to_d_branch",
    rate_constants_initial=[1.0, 0.5, 0.3, 0.2],
    parameter_bounds=[(0.1, 10.0), (0.01, 5.0), (0.01, 3.0), (0.01, 2.0)],
    fixed_parameters=[False, False, True, False],  # 固定第3个参数
    ode_solver={'method': 'BDF', 'rtol': 1e-8},
    optimization={'method': 'leastsq', 'max_nfev': 2000}
)

# 验证配置
is_valid, errors = config_manager.validate_config(config)
if not is_valid:
    print(f"配置错误: {errors}")

# 导出配置
config_manager.export_config(config, "gta_config.json")

# 导入配置
loaded_config = config_manager.import_config("gta_config.json")
```

#### 系统集成

```python
from gta_analysis import GTAIntegration

integration = GTAIntegration()

# 转换为globalfit格式
globalfit_config = integration.convert_to_globalfit_config(config)

# 生成结果目录名
result_name = integration.generate_result_naming(config, variant="fast")
print(f"结果目录: {result_name}")

# 导出结果
exported_files = integration.export_results_to_globalfit_format(
    gta_solver=solver,
    fit_result=fit_result,
    output_dir="results/" + result_name
)
```

## 高级用法

### 批量分析

```python
from gta_analysis import KineticModels, GTASolver

# 定义多个模型进行比较
models_to_test = [
    {'type': 'predefined', 'number': 1, 'rates': [1.0, 0.5]},
    {'type': 'predefined', 'number': 2, 'rates': [1.0, 0.5]},
    {'type': 'predefined', 'number': 3, 'rates': [1.0, 0.5, 0.3, 0.2]},
]

results = []

for model_config in models_to_test:
    # 创建动力学矩阵
    kinetic_model = KineticModels(model_config['rates'])
    K, n_species = kinetic_model.get_kinetic_matrix(model_config['number'])
    
    # 运行分析
    solver = GTASolver(wavelengths, delays, experimental_data)
    solver.set_kinetic_matrix(K)
    
    fit_result = solver.fit_parameters(
        initial_rates=model_config['rates'],
        method='leastsq'
    )
    
    # 存储结果
    results.append({
        'model': f"Model {model_config['number']}",
        'success': fit_result['success'],
        'reduced_chi_squared': fit_result['reduced_chi_squared'],
        'fitted_rates': fit_result['fitted_rates']
    })

# 比较结果
best_model = min(
    [r for r in results if r['success']], 
    key=lambda x: x['reduced_chi_squared']
)
print(f"最佳模型: {best_model['model']}")
```

### 自定义优化策略

```python
# 分步优化策略
def multi_step_optimization(solver, initial_rates):
    # 第一步：粗略优化
    fit_result_1 = solver.fit_parameters(
        initial_rates=initial_rates,
        method='nelder',  # 全局搜索
        parameter_bounds=[(0.01, 100.0)] * len(initial_rates)
    )
    
    # 第二步：精细优化
    fit_result_2 = solver.fit_parameters(
        initial_rates=fit_result_1['fitted_rates'],
        method='leastsq',  # 局部精化
        parameter_bounds=[(r*0.1, r*10) for r in fit_result_1['fitted_rates']]
    )
    
    return fit_result_2

# 使用自定义优化
final_result = multi_step_optimization(solver, [1.0, 0.5])
```

### 模型选择和验证

```python
from scipy import stats

def model_selection_criteria(fit_results):
    """计算模型选择标准"""
    criteria = {}
    
    for name, result in fit_results.items():
        n = result['n_data_points']
        k = result['n_parameters']
        chi2 = result['chi_squared']
        
        # AIC (Akaike Information Criterion)
        aic = n * np.log(chi2/n) + 2*k
        
        # BIC (Bayesian Information Criterion)
        bic = n * np.log(chi2/n) + k*np.log(n)
        
        criteria[name] = {
            'AIC': aic,
            'BIC': bic,
            'reduced_chi2': result['reduced_chi_squared']
        }
    
    return criteria

# 应用模型选择
criteria = model_selection_criteria({
    'Model_1': fit_result_1,
    'Model_2': fit_result_2,
    'Custom': fit_result_custom
})

# 找到最佳模型
best_aic = min(criteria.items(), key=lambda x: x[1]['AIC'])
print(f"AIC最佳模型: {best_aic[0]}")
```

## 与现有系统集成

### 与analysis_tool集成

```python
# 在analysis_tool/global_batch.py中添加GTA支持
from gta_analysis import GTAConfig, GTAIntegration

def add_gta_models_to_config():
    gta_config = GTAConfig()
    
    # 添加GTA模型到现有配置
    gta_models = {
        'gta_sequential_stable': {
            'display_name': 'GTA Sequential (Stable)',
            'model_type': 'kinetic',
            'analysis_method': 'gta',
            'model_number': 2
        },
        'gta_parallel_branching': {
            'display_name': 'GTA Parallel Branching',
            'model_type': 'kinetic', 
            'analysis_method': 'gta',
            'model_number': 7
        }
    }
    
    return gta_models
```

### 与可视化系统集成

```python
# 在visualize_results.py中添加GTA结果可视化
def visualize_gta_results(result_dir):
    # 加载GTA结果
    concentration_profiles = np.loadtxt(f"{result_dir}/concentration_profiles.csv", delimiter=',')
    species_spectra = np.loadtxt(f"{result_dir}/species_spectra.csv", delimiter=',')
    
    # 使用GTA可视化模块
    viz = GTAVisualization(language='zh')
    
    # 生成标准化报告
    viz.create_comprehensive_report(
        gta_solver=None,  # 如果有保存的solver对象
        fit_result=loaded_fit_result,
        output_dir=f"{result_dir}/gta_visualization"
    )
```

## 故障排除

### 常见问题

1. **ODE求解失败**
   - 检查动力学矩阵是否合理
   - 尝试不同的ODE方法 ('BDF', 'RK45', 'LSODA')
   - 调整容差参数 (rtol, atol)

2. **参数拟合不收敛**
   - 检查初始参数猜测是否合理
   - 设置适当的参数边界
   - 尝试不同的优化算法
   - 使用分步优化策略

3. **反应方程解析错误**
   - 确保物种按字母顺序命名 (A, B, C, ...)
   - 检查箭头语法是否正确
   - 验证分支反应的分隔符使用

4. **内存或性能问题**
   - 减少时间点或波长点数量
   - 使用并行处理进行批量分析
   - 优化数据类型 (float32 vs float64)

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查中间结果
solver = GTASolver(wavelengths, delays, data)
solver.set_kinetic_matrix(K)

# 手动检查浓度演化
test_rates = [1.0, 0.5]
concentration_profiles = solver.solve_concentration_profiles(test_rates)
print(f"浓度范围: {np.min(concentration_profiles)} - {np.max(concentration_profiles)}")

# 检查物种光谱计算
species_spectra = solver.calculate_species_spectra(concentration_profiles)
print(f"光谱形状: {species_spectra.shape}")

# 验证重构数据
reconstructed = concentration_profiles @ species_spectra
residuals = data - reconstructed
print(f"残差统计: 均值={np.mean(residuals):.2e}, 标准差={np.std(residuals):.2e}")
```

## 参考资料

1. **EfsTA论文和文档**: 了解GTA理论基础
2. **lmfit文档**: 参数优化详细信息
3. **scipy.integrate文档**: ODE求解器选项
4. **TAS分析理论**: 瞬态吸收光谱分析基础

## 更新日志

- **v1.0.0**: 初始版本，实现完整GTA功能
  - 8个预定义动力学模型
  - 自定义反应方程解析
  - 完整的参数优化和可视化
  - 与现有系统的集成支持