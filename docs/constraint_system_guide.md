# MCR-ALS 约束系统使用指南

## 概述

MCR-ALS 约束系统为多元曲线分辨交替最小二乘法提供了灵活的约束配置功能，用户可以根据具体需求选择和调整各种约束条件。

## 功能特性

- ✅ **非负性约束**：确保浓度矩阵C和光谱矩阵S中所有元素≥0
- ✅ **光谱平滑度约束**：使用二阶导数惩罚项确保光谱平滑
- ✅ **组分数量范围验证**：限制组分数量在合理范围内（默认1-4个）
- ✅ **JSON模板配置系统**：支持保存和加载约束配置
- ✅ **动态参数调整**：运行时修改约束参数
- ✅ **自定义约束添加**：用户可以定义新的约束类型

## 快速开始

### 基本使用

```python
from mcr.mcr_als import MCRALS
import numpy as np

# 使用默认约束（非负性 + 组分数量验证）
mcr = MCRALS(n_components=3)
mcr.fit(data_matrix)
```

### 启用光谱平滑约束

```python
from mcr.constraint_config import ConstraintConfig

# 创建配置并启用平滑约束
config = ConstraintConfig()
config.enable_constraint("spectral_smoothness")
config.set_constraint_parameter("spectral_smoothness", "lambda", 0.01)

mcr = MCRALS(n_components=3, constraint_config=config)
mcr.fit(data_matrix)
```

## 约束类型详解

### 1. 非负性约束 (Non-negativity)

**描述**：确保浓度和光谱矩阵中所有元素非负。
**默认状态**：启用
**适用矩阵**：C（浓度）和 S（光谱）

```python
# 默认已启用，无需额外配置
mcr = MCRALS(n_components=3)
```

### 2. 光谱平滑度约束 (Spectral Smoothness)

**描述**：使用二阶导数惩罚项确保光谱平滑，减少噪声影响。
**默认状态**：禁用
**适用矩阵**：S（光谱）

**参数说明**：
- `lambda`: 平滑度惩罚系数（默认：0.001）
- `order`: 导数阶数（默认：2）

```python
config = ConstraintConfig()
config.enable_constraint("spectral_smoothness")
config.set_constraint_parameter("spectral_smoothness", "lambda", 0.01)  # 调整平滑强度
```

### 3. 组分数量范围验证 (Component Count Range)

**描述**：限制MCR-ALS模型的组分数量在合理范围内。
**默认状态**：启用

**参数说明**：
- `min_components`: 最小组分数（默认：1）
- `max_components`: 最大组分数（默认：4）
- `default_components`: 推荐组分数（默认：3）

```python
config = ConstraintConfig()
config.set_constraint_parameter("component_count_range", "max_components", 6)
mcr = MCRALS(n_components=5, constraint_config=config)  # 现在可以使用5个组分
```

## 约束模板系统

系统提供三种预定义的约束模板：

### 1. 标准约束模板 (`standard_constraints.json`)

```json
{
  "constraints": {
    "non_negativity": {"enabled": true},
    "spectral_smoothness": {"enabled": false, "parameters": {"lambda": 0.001}},
    "component_count_range": {"enabled": true, "parameters": {"max_components": 4}}
  },
  "active_constraints": ["non_negativity", "component_count_range"]
}
```

### 2. 严格约束模板 (`strict_constraints.json`)

启用所有约束，适用于高噪声数据：

```python
mcr = MCRALS(n_components=3, 
            constraint_config="mcr/constraint_templates/strict_constraints.json")
```

### 3. 宽松约束模板 (`relaxed_constraints.json`)

较少的约束限制，适用于高质量数据：

```python
mcr = MCRALS(n_components=3,
            constraint_config="mcr/constraint_templates/relaxed_constraints.json")
```

## 自定义约束配置

### 创建自定义约束

```python
config = ConstraintConfig()

# 添加自定义约束
custom_constraint = {
    "name": "强平滑度约束",
    "description": "用于极高噪声数据的强平滑约束",
    "type": "spectral_smoothness",
    "enabled": True,
    "apply_to": ["S"],
    "parameters": {
        "lambda": 0.1,  # 更强的平滑
        "order": 2
    }
}

config.add_constraint("strong_smoothness", custom_constraint)
config.enable_constraint("strong_smoothness")

# 保存配置
config.save_to_file("my_custom_constraints.json")
```

### 动态修改参数

```python
mcr = MCRALS(n_components=3, constraint_config=config)

# 运行时调整参数
mcr.set_constraint_parameter("spectral_smoothness", "lambda", 0.05)
mcr.fit(data_matrix)
```

## 实际应用示例

### 处理高噪声数据

```python
# 对于噪声较大的TAS数据，推荐使用光谱平滑约束
config = ConstraintConfig()
config.enable_constraint("spectral_smoothness")
config.set_constraint_parameter("spectral_smoothness", "lambda", 0.02)  # 较强平滑

mcr = MCRALS(n_components=3, max_iter=100, constraint_config=config)
mcr.fit(noisy_data)
```

### 处理复杂体系

```python
# 对于复杂的多组分体系，可能需要更多组分
config = ConstraintConfig()
config.set_constraint_parameter("component_count_range", "max_components", 6)
config.enable_constraint("spectral_smoothness")  # 同时启用平滑约束

mcr = MCRALS(n_components=5, constraint_config=config)
mcr.fit(complex_data)
```

## 最佳实践

1. **开始时使用默认约束**：对于大多数数据，默认的非负性约束和组分数量验证足够。

2. **根据数据质量调整**：
   - 高噪声数据：启用光谱平滑约束，增加λ值
   - 高质量数据：可以禁用一些约束以获得更好的拟合

3. **组分数量选择**：
   - 从2-3个组分开始
   - 根据LOF和物理意义调整
   - 避免使用过多组分导致过拟合

4. **参数调优**：
   - λ值通常在1e-4到1e-1之间
   - 较大的λ值产生更平滑的光谱
   - 通过交叉验证选择最佳参数

## 故障排除

### 常见问题

1. **"组分数量超出范围"错误**
   ```python
   # 解决方法：修改组分数量范围或减少组分数
   config = ConstraintConfig()
   config.set_constraint_parameter("component_count_range", "max_components", 6)
   ```

2. **收敛缓慢或不收敛**
   ```python
   # 尝试调整约束参数或增加迭代次数
   mcr = MCRALS(n_components=3, max_iter=200, tol=1e-5)
   ```

3. **光谱过度平滑**
   ```python
   # 减小平滑度参数
   config.set_constraint_parameter("spectral_smoothness", "lambda", 1e-4)
   ```

## 扩展开发

如需添加新的约束类型，请参考 `mcr/constraints.py` 中的实现，并更新 `apply_constraint_from_config` 函数。

## 版本信息

- 版本：1.0
- 兼容的MCR-ALS版本：所有版本
- 依赖：NumPy, SciPy

## 更多资源

- 完整示例：`examples/constraint_examples.py`
- 测试代码：`test/test_constraints.py`
- 约束模板：`mcr/constraint_templates/`