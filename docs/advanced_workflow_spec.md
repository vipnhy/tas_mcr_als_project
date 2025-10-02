# 高级瞬态吸收分析工作流设计草案

## 1. 目标

构建一个可配置、可重复的瞬态吸收光谱(TAS)分析工作流，实现以下能力：

- 在单个入口脚本中串联 **多轮 MCR-ALS** 与 **多模型全局拟合**。
- 支持用户指定 **组分数范围、光谱类型、时间/波长窗口**、非负性约束、预处理流程、初始化策略、拟合轮数等。
- 自动遍历不同的 **组分数量 × 初始化种子** 组合，保存全部中间结果，并为后续全局拟合提供结构化输入。
- 针对每个 MCR 输出构建 **5 个（默认）不同的动力学模型**（顺序、平行、混合等），并允许根据 MCR 推断、随机或用户自定义初值进行多次拟合。
- 对所有全局拟合结果按用户定义的指标（默认 LOF）排序，生成汇总报告，输出至独立的分析目录中。

## 2. 配置概览

新增统一配置文件（JSON/YAML，默认 `config_workflow.json`），主要分为以下四大区段：

```jsonc
{
  "analysis": {
    "name": "demo_vis_run",
    "output_root": "analysis_runs",
    "working_dir_mode": "timestamp"  // timestamp | overwrite | custom
  },
  "input": {
    "file_path": "data/TAS/TA_Average-crop-TCA-tzs-tzs-tzs-chirp.csv",
    "file_type": "handle",
    "spectral_type": "VIS",          // UV | VIS | NIR
    "wavelength_override": null,       // [min, max] 覆盖 spectral_type 的默认范围
    "delay_range": [0.1, 50.0],
    "lifetime_window_ps": [0.1, 200.0] // 用于初始化和约束
  },
  "mcr": {
    "component_range": [2, 4],          // 或显式列表 [2,3,4]
    "max_iter": 250,
    "tol": 1e-7,
    "enforce_nonneg": true,             // 仅对浓度非负，可选对光谱
    "spectral_nonneg": false,
    "preprocessing": {
      "enabled": false,
      "steps": []                      // 复用 preprocessing.pipeline 定义
    },
    "initialization": {
      "mode": "random",               // random | svd | user
      "runs_per_component": 3,
      "user_seeds": []
    },
    "random_seed": 12345,
    "save_intermediate": true
  },
  "global_fit": {
    "models": [
      {"type": "gla"},
      {"type": "gta", "model": "sequential"},
      {"type": "gta", "model": "parallel"},
      {"type": "gta", "model": "mixed", "network": [[0,1,0],[1,2,1],[0,2,2]]},
      {"type": "gta", "model": "mixed", "network": [[0,1,0],[1,2,1],[1,3,2]]}
    ],
    "attempts_per_mcr": 5,
    "init_strategy": "mcr",            // mcr | random | user
    "user_initials": {
      "gla": [],
      "gta": []
    },
    "random_seed": 9876,
    "sort_metric": "lof",              // lof | chi_square | computation_time
    "sort_order": "asc",
    "top_n": null                       // 若设置则只保留前 N 项
  }
}
```

### 2.1 光谱类型默认波长范围

| 光谱类型 | 默认波长范围 (nm) |
|----------|-------------------|
| UV       | [200, 400]        |
| VIS      | [400, 800]        |
| NIR      | [800, 1700]       |

若 `wavelength_override` 非空，则优先使用该范围并与数据实际范围求交集。

### 2.2 约束策略

- `enforce_nonneg = true` 时，对浓度矩阵 C 启用非负性；如 `spectral_nonneg = true` 则同时约束光谱 S。
- 提供 `constraints` 字段以允许加载自定义 JSON 模板。

## 3. MCR 批处理设计

### 3.1 数据准备

1. 读取输入文件，基于 `spectral_type/wavelength_override` 裁剪波长，`delay_range` 裁剪时间；记录轴信息。
2. 若启用预处理，构造 `TASPreprocessingPipeline` 并执行 `fit_transform`。
3. 将裁剪/预处理后的矩阵缓存，供多次 MCR 复用。

### 3.2 批量运行逻辑

外层遍历组分数 `k`，内层遍历初始化运行次数 `r`：

```
for k in component_range:
  seeds = pick_seeds(k, r)
  for run_idx, seed in enumerate(seeds):
    mcr = MCRALS(n_components=k, random_state=seed, ...)
    mcr.fit(data)
    save_run_outputs(k, run_idx, seed, mcr)
```

- 每次运行输出目录：`<working_dir>/mcr/components_{k}/run_{idx}_seed_{seed}/`
- 输出内容：
  - `concentration_profiles.csv`
  - `pure_spectra.csv`
  - `lof_history.csv`
  - `analysis_parameters.json`（包含 seed、约束、预处理摘要、输入窗口等）
  - 可选图表 (`mcr_als_results.png`, `data_comparison.png`)
- 汇总文件：`
  <working_dir>/mcr/mcr_summary.json
`
  包含所有 runs 的指标 (LOF、迭代次数、R²、seed、组件数等)。

## 4. 全局拟合批处理设计

### 4.1 输入准备

- 从 `mcr_summary.json` 读取所有有效运行信息。
- 为每个 MCR 结果构造 `MCRALSInterface`，指向对应目录，并提供原始数据文件路径（或使用 MCR 重构数据）。

### 4.2 模型族

默认生成 5 个动力学模型：

1. GLA（多指数衰减）
2. GTA Sequential（A→B→C→...）
3. GTA Parallel（A→B, A→C, ...）
4. GTA Mixed-1：顺序 + 直接通道（A→B→C, A→C）
5. GTA Mixed-2：顺序 + B→C 辅助通道（可覆盖更多并联路径）。

`network` 定义采用 `(from, to, k_index)` 三元组列表。

### 4.3 初始化策略

- `mcr`：基于 `MCRALSInterface.estimate_lifetimes_from_mcr()` 自动生成初值。
- `random`：在 `lifetime_window_ps` 或 `rate` 范围内均匀/对数随机采样。
- `user`：使用配置中提供的列表；不足时回退到 `mcr`。

每个 MCR 结果 × 模型组合执行 `attempts_per_mcr` 次拟合，记录 seed/初值。输出结构：

```
<working_dir>/global_fit/
  mcr_components_{k}_run_{idx}/
    model_{name}/
      attempt_{j}/
        global_fit_summary.json
        concentration_global_fit.csv
        spectra_global_fit.csv
        ...
```

### 4.4 汇总与排序

- 生成 `global_fit/global_fit_summary.json`，包含所有结果及指标。
- 按 `sort_metric`（默认 `lof`）排序，输出前 `top_n`（若设置）到 `global_fit/global_fit_ranking.csv`。
- 同时生成 Markdown 报告 `global_fit/global_fit_report.md`，包含：
  - 总览表（MCR组合、模型、LOF、Chi²、耗时）
  - 最佳模型详述（含参数、寿命、速率常数）
  - 链接到对应结果文件。

## 5. 入口脚本与 CLI

新增 `advanced_analysis.py`：

- `python advanced_analysis.py --config config_workflow.json`
- 可选覆盖项：`--analysis-name`, `--spectral-type`, `--component-range 2 5`, `--no-preprocessing`, `--sort-metric chi_square` 等。
- 脚本流程：
  1. 解析配置并创建工作目录
  2. 执行 MCR 批量分析
  3. 执行全局拟合批量分析
  4. 汇总结果并生成最终报告
  5. 打印输出路径和排行榜摘要

## 6. 报表与再利用

- 生成 `final_report.md`，记录：
  - 用户输入配置摘要
  - MCR 批量结果统计（分组分数、初值、LOF 分布）
  - 全局拟合排序表
  - 推荐最佳模型及理由
- 更新顶层 `ANALYSIS_REPORT.md` 和 `README_TAS_MCR.md`，介绍新工具使用方法与示例。

## 7. 后续扩展接口（预留）

- 支持多输入文件批处理（列表/通配符）
- 拓展更多动力学网络模板（可与 Domain JSON 结合）
- 引入 GUI 或 notebook 接口
- 与 Flask 服务集成，实现远程执行

---

> 本设计草案用于指导实现阶段的模块拆分与接口定义，后续如需调整，请同步更新本文档。
