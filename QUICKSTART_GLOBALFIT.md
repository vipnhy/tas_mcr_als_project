# Globalfit模块快速入门指南

## 5分钟快速上手

本指南将带你快速完成从MCR-ALS到全局拟合的完整分析流程。

---

## 前提条件

确保已安装所有依赖:

```bash
pip install numpy scipy matplotlib lmfit
```

---

## 完整示例

### 步骤1: 运行MCR-ALS分析 (1-2分钟)

```bash
# 在项目根目录运行
python run_main.py \
    --file_path data/TAS/TA_Average.csv \
    --n_components 3 \
    --wavelength_range 420 750 \
    --delay_range 0.1 50 \
    --save_plots \
    --save_results \
    --output_dir results
```

**输出**: `results/` 目录下的MCR-ALS分析结果

### 步骤2: 运行全局拟合 (1分钟)

```bash
# 在项目根目录运行
python Globalfit/examples/auto_workflow.py --mcr_results results
```

**输出**: `results/global_fit/` 目录下的全局拟合结果

### 步骤3: 查看结果

打开以下图片查看结果:

1. **MCR-ALS结果**: `results/mcr_als_results.png`
2. **GLA结果**: `results/global_fit/gla/gla_results.png`
3. **GTA结果**: `results/global_fit/sequential_a_to_b_to_c/sequential_a_to_b_to_c_results.png`
4. **比较图**: `results/global_fit/comparison_mcr_gla.png`

---

## 结果解读

### MCR-ALS vs GLA vs GTA

| 方法 | 输出 | 物理意义 | 适用场景 |
|------|------|----------|----------|
| **MCR-ALS** | 浓度轮廓 + 光谱 | 一般 | 组分分离 |
| **GLA** | 寿命 τ + DAS | 较少 | 快速获取时间常数 |
| **GTA** | 速率常数 k + SAS | 明确 | 确定反应机理 |

### 典型输出示例

```
方法比较:
  MCR-ALS LOF:         3.2%
  GLA LOF:             3.5%
  GTA(顺序) LOF:       4.1%
  GTA(平行) LOF:       6.8%

GLA最优寿命:
  τ1 = 5.2 ps
  τ2 = 87 ps
  τ3 = 1200 ps

GTA最优速率常数:
  k1 = 0.19 ps⁻¹  (对应 τ1 = 5.3 ps)
  k2 = 0.011 ps⁻¹ (对应 τ2 = 91 ps)
```

**解释**:
- LOF越低，拟合越好
- GTA的LOF略高于MCR-ALS是正常的(因为有物理约束)
- 如果某个模型的LOF明显偏高，说明模型不合适

---

## 高级用法

### 只运行GLA

```bash
python Globalfit/examples/auto_workflow.py --mcr_results results --method gla
```

### 只运行GTA顺序模型

```bash
python Globalfit/examples/auto_workflow.py --mcr_results results --method gta --model sequential
```

### 提供原始数据文件

```bash
python Globalfit/examples/auto_workflow.py \
    --mcr_results results \
    --data_file data/TAS/TA_Average.csv
```

---

## Python脚本使用

如果你想在自己的脚本中使用Globalfit:

```python
import sys
sys.path.append('..')

from Globalfit import MCRALSInterface, GlobalLifetimeAnalysis

# 1. 准备数据
interface = MCRALSInterface("results")
data_dict = interface.prepare_for_global_fitting()

# 2. 执行GLA
gla = GlobalLifetimeAnalysis(
    data_matrix=data_dict['data_matrix'],
    time_axis=data_dict['time_axis'],
    wavelength_axis=data_dict['wavelength_axis'],
    n_components=data_dict['n_components']
)

results = gla.fit(tau_initial=data_dict['lifetimes_initial'])

# 3. 查看结果
print(f"最优寿命: {results['tau_optimal']}")
print(f"LOF: {results['lof']:.4f}%")

# 4. 保存结果
interface.save_global_fit_results(results)

# 5. 可视化
from Globalfit.utils import plot_global_fit_results
plot_global_fit_results(results, 
                       data_dict['time_axis'],
                       data_dict['wavelength_axis'])
```

---

## 验证安装

运行测试确保一切正常:

```bash
cd Globalfit/tests
python test_basic_functionality.py
```

期望输出:
```
✓ 通过: 动力学模型
✓ 通过: 全局寿命分析 (GLA)
✓ 通过: 全局目标分析 (GTA)

总计: 3/3 测试通过
🎉 所有测试通过!
```

---

## 常见问题

### Q: 拟合失败怎么办?

A: 检查以下几点:
1. MCR-ALS结果目录是否正确
2. 必需的文件是否存在(concentration_profiles.csv, pure_spectra.csv)
3. 时间轴和波长轴数据是否正常

### Q: LOF很高是什么原因?

A: 可能的原因:
1. 动力学模型不合适 → 尝试其他模型
2. 组分数量不对 → 重新运行MCR-ALS
3. 初始参数估计不准 → 手动调整初始值

### Q: 如何选择最佳模型?

A: 考虑三个因素:
1. **LOF值**: 越低越好(但不是唯一标准)
2. **物理意义**: 参数是否合理
3. **文献对比**: 是否与已知体系一致

---

## 下一步

- 阅读 [完整使用说明](Globalfit/docs/README_GLOBALFIT.md)
- 查看 [工作流程指南](Globalfit/docs/WORKFLOW_GUIDE.md)
- 了解 [集成说明](GLOBALFIT_INTEGRATION.md)
- 运行 [详细示例](Globalfit/examples/run_global_fit_example.py)

---

## 获取帮助

```bash
# 查看auto_workflow.py的帮助
python auto_workflow.py --help

# 查看run_main.py的帮助
python run_main.py --help
```

---

**需要更多帮助?**

- 查看文档中的常见问题部分
- 查看示例脚本
- 提交Issue到项目仓库

---

祝你分析顺利! 🎉
