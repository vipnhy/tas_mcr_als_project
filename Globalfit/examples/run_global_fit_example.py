"""
run_global_fit_example.py - 全局拟合示例脚本

该脚本展示了如何使用全局拟合模块对MCR-ALS结果进行进一步分析。

使用方法:
    python run_global_fit_example.py

或者在Jupyter Notebook中运行。
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from Globalfit import (
    MCRALSInterface,
    GlobalLifetimeAnalysis,
    GlobalTargetAnalysis,
    SequentialModel,
    ParallelModel
)
from Globalfit.utils import (
    plot_global_fit_results,
    compare_mcr_and_global_fit,
    export_results_to_txt
)


def main():
    """主函数"""
    
    print("=" * 70)
    print("瞬态吸收光谱全局拟合分析示例")
    print("=" * 70)
    
    # ========== 步骤1: 从MCR-ALS结果准备数据 ==========
    print("\n步骤1: 加载MCR-ALS结果并准备数据")
    print("-" * 70)
    
    # 指定MCR-ALS结果目录
    mcr_results_dir = "results"  # 修改为你的MCR-ALS结果目录
    
    # 创建接口对象
    interface = MCRALSInterface(mcr_results_dir)
    
    # 准备数据 (可以选择提供原始数据文件)
    data_file = None  # 如果有原始数据，在这里指定路径
    # data_file = "data/TAS/TA_Average.csv"
    
    try:
        data_dict = interface.prepare_for_global_fitting(
            data_file=data_file,
            file_type="handle"
        )
    except Exception as e:
        print(f"错误: 无法准备数据 - {e}")
        print("请确保MCR-ALS结果目录存在且包含必要的文件。")
        return
    
    # ========== 步骤2: 全局寿命分析 (GLA) ==========
    print("\n步骤2: 执行全局寿命分析 (GLA)")
    print("-" * 70)
    
    # 创建GLA分析器
    gla = GlobalLifetimeAnalysis(
        data_matrix=data_dict['data_matrix'],
        time_axis=data_dict['time_axis'],
        wavelength_axis=data_dict['wavelength_axis'],
        n_components=data_dict['n_components']
    )
    
    # 使用从MCR-ALS估计的寿命作为初始值
    tau_initial = data_dict['lifetimes_initial']
    
    # 执行拟合
    gla_results = gla.fit(
        tau_initial=tau_initial,
        optimization_method='leastsq'
    )
    
    # 可视化GLA结果
    print("\n生成GLA结果图表...")
    plot_global_fit_results(
        gla_results,
        data_dict['time_axis'],
        data_dict['wavelength_axis'],
        save_path=os.path.join(mcr_results_dir, "global_fit", "gla_results.png"),
        show_plot=False
    )
    
    # 保存GLA结果
    interface.save_global_fit_results(gla_results, 
                                     output_dir=os.path.join(mcr_results_dir, "global_fit", "gla"))
    
    # ========== 步骤3: 全局目标分析 (GTA) - 顺序模型 ==========
    print("\n步骤3: 执行全局目标分析 (GTA) - 顺序反应模型")
    print("-" * 70)
    
    # 创建顺序反应模型 (A → B → C → ...)
    sequential_model = SequentialModel(n_components=data_dict['n_components'])
    
    # 创建GTA分析器
    gta_seq = GlobalTargetAnalysis(
        data_matrix=data_dict['data_matrix'],
        time_axis=data_dict['time_axis'],
        wavelength_axis=data_dict['wavelength_axis'],
        kinetic_model=sequential_model
    )
    
    # 使用从MCR-ALS估计的速率常数作为初始值
    k_initial = data_dict['rate_constants_initial'][:-1]  # 顺序模型需要 n-1 个速率常数
    
    # 执行拟合
    gta_seq_results = gta_seq.fit(
        k_initial=k_initial,
        optimization_method='leastsq'
    )
    
    # 可视化GTA结果
    print("\n生成GTA(顺序模型)结果图表...")
    plot_global_fit_results(
        gta_seq_results,
        data_dict['time_axis'],
        data_dict['wavelength_axis'],
        save_path=os.path.join(mcr_results_dir, "global_fit", "gta_sequential_results.png"),
        show_plot=False
    )
    
    # 保存GTA结果
    interface.save_global_fit_results(gta_seq_results,
                                     output_dir=os.path.join(mcr_results_dir, "global_fit", "gta_sequential"))
    
    # ========== 步骤4: 全局目标分析 (GTA) - 平行模型 ==========
    print("\n步骤4: 执行全局目标分析 (GTA) - 平行反应模型")
    print("-" * 70)
    
    # 创建平行反应模型 (A → B, A → C, A → D, ...)
    parallel_model = ParallelModel(n_components=data_dict['n_components'])
    
    # 创建GTA分析器
    gta_par = GlobalTargetAnalysis(
        data_matrix=data_dict['data_matrix'],
        time_axis=data_dict['time_axis'],
        wavelength_axis=data_dict['wavelength_axis'],
        kinetic_model=parallel_model
    )
    
    # 执行拟合
    gta_par_results = gta_par.fit(
        k_initial=k_initial,
        optimization_method='leastsq'
    )
    
    # 可视化GTA结果
    print("\n生成GTA(平行模型)结果图表...")
    plot_global_fit_results(
        gta_par_results,
        data_dict['time_axis'],
        data_dict['wavelength_axis'],
        save_path=os.path.join(mcr_results_dir, "global_fit", "gta_parallel_results.png"),
        show_plot=False
    )
    
    # 保存GTA结果
    interface.save_global_fit_results(gta_par_results,
                                     output_dir=os.path.join(mcr_results_dir, "global_fit", "gta_parallel"))
    
    # ========== 步骤5: 比较不同方法的结果 ==========
    print("\n步骤5: 比较MCR-ALS和全局拟合结果")
    print("-" * 70)
    
    # 准备MCR-ALS结果字典
    mcr_results = {
        'C_mcr': data_dict['C_mcr'],
        'S_mcr': data_dict['S_mcr'],
        'mcr_lof': data_dict['mcr_lof']
    }
    
    # 比较MCR-ALS和GLA
    print("\n生成MCR-ALS vs GLA比较图...")
    compare_mcr_and_global_fit(
        mcr_results,
        gla_results,
        data_dict['time_axis'],
        data_dict['wavelength_axis'],
        save_path=os.path.join(mcr_results_dir, "global_fit", "comparison_mcr_gla.png"),
        show_plot=False
    )
    
    # 比较MCR-ALS和GTA(顺序)
    print("生成MCR-ALS vs GTA(顺序)比较图...")
    compare_mcr_and_global_fit(
        mcr_results,
        gta_seq_results,
        data_dict['time_axis'],
        data_dict['wavelength_axis'],
        save_path=os.path.join(mcr_results_dir, "global_fit", "comparison_mcr_gta_seq.png"),
        show_plot=False
    )
    
    # ========== 步骤6: 生成综合报告 ==========
    print("\n步骤6: 生成综合分析报告")
    print("-" * 70)
    
    # 导出GLA报告
    export_results_to_txt(
        gla_results,
        os.path.join(mcr_results_dir, "global_fit", "gla_report.txt")
    )
    
    # 导出GTA(顺序)报告
    export_results_to_txt(
        gta_seq_results,
        os.path.join(mcr_results_dir, "global_fit", "gta_sequential_report.txt")
    )
    
    # 导出GTA(平行)报告
    export_results_to_txt(
        gta_par_results,
        os.path.join(mcr_results_dir, "global_fit", "gta_parallel_report.txt")
    )
    
    # ========== 总结 ==========
    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)
    print("\n方法比较:")
    print(f"  MCR-ALS LOF:         {data_dict['mcr_lof']:.4f}%")
    print(f"  GLA LOF:             {gla_results['lof']:.4f}%")
    print(f"  GTA(顺序) LOF:       {gta_seq_results['lof']:.4f}%")
    print(f"  GTA(平行) LOF:       {gta_par_results['lof']:.4f}%")
    
    print("\n所有结果已保存到:", os.path.join(mcr_results_dir, "global_fit"))
    print("\n建议:")
    print("  1. 查看比较图以了解不同方法的差异")
    print("  2. 根据LOF值和物理意义选择最合适的模型")
    print("  3. 检查拟合参数的不确定度")
    print("  4. 验证残差是否为随机噪声")


if __name__ == "__main__":
    main()
