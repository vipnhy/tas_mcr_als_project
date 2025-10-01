"""
auto_workflow.py - MCR-ALS到全局拟合的自动化工作流程

该脚本提供了一个简化的命令行接口，实现从MCR-ALS结果
自动进行全局拟合分析的完整工作流程。

使用方法:
    python auto_workflow.py --mcr_results results --data_file data/TAS/TA_Average.csv
    
或者:
    python auto_workflow.py --mcr_results results --method gla
    python auto_workflow.py --mcr_results results --method gta --model sequential
"""

import sys
import os
import argparse
from pathlib import Path

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


def run_gla(data_dict, output_dir):
    """运行全局寿命分析"""
    print("\n" + "=" * 70)
    print("执行全局寿命分析 (GLA)")
    print("=" * 70)
    
    gla = GlobalLifetimeAnalysis(
        data_matrix=data_dict['data_matrix'],
        time_axis=data_dict['time_axis'],
        wavelength_axis=data_dict['wavelength_axis'],
        n_components=data_dict['n_components']
    )
    
    results = gla.fit(
        tau_initial=data_dict['lifetimes_initial'],
        optimization_method='leastsq'
    )
    
    # 保存结果
    gla_dir = os.path.join(output_dir, "gla")
    os.makedirs(gla_dir, exist_ok=True)
    
    # 可视化
    plot_global_fit_results(
        results,
        data_dict['time_axis'],
        data_dict['wavelength_axis'],
        save_path=os.path.join(gla_dir, "gla_results.png"),
        show_plot=False
    )
    
    # 导出报告
    export_results_to_txt(results, os.path.join(gla_dir, "gla_report.txt"))
    
    # 保存数据
    interface = MCRALSInterface(os.path.dirname(output_dir))
    interface.time_axis = data_dict['time_axis']
    interface.wavelength_axis = data_dict['wavelength_axis']
    interface.save_global_fit_results(results, output_dir=gla_dir)
    
    print(f"\nGLA结果已保存到: {gla_dir}")
    
    return results


def run_gta(data_dict, output_dir, model_type='sequential'):
    """运行全局目标分析"""
    print("\n" + "=" * 70)
    print(f"执行全局目标分析 (GTA) - {model_type.upper()}模型")
    print("=" * 70)
    
    # 创建动力学模型
    if model_type.lower() == 'sequential':
        model = SequentialModel(n_components=data_dict['n_components'])
        model_name = "sequential"
    elif model_type.lower() == 'parallel':
        model = ParallelModel(n_components=data_dict['n_components'])
        model_name = "parallel"
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    gta = GlobalTargetAnalysis(
        data_matrix=data_dict['data_matrix'],
        time_axis=data_dict['time_axis'],
        wavelength_axis=data_dict['wavelength_axis'],
        kinetic_model=model
    )
    
    # 准备速率常数初始值 (需要 n-1 个)
    k_initial = data_dict['rate_constants_initial'][:-1]
    
    results = gta.fit(
        k_initial=k_initial,
        optimization_method='leastsq'
    )
    
    # 保存结果
    gta_dir = os.path.join(output_dir, f"gta_{model_name}")
    os.makedirs(gta_dir, exist_ok=True)
    
    # 可视化
    plot_global_fit_results(
        results,
        data_dict['time_axis'],
        data_dict['wavelength_axis'],
        save_path=os.path.join(gta_dir, f"gta_{model_name}_results.png"),
        show_plot=False
    )
    
    # 导出报告
    export_results_to_txt(results, os.path.join(gta_dir, f"gta_{model_name}_report.txt"))
    
    # 保存数据
    interface = MCRALSInterface(os.path.dirname(output_dir))
    interface.time_axis = data_dict['time_axis']
    interface.wavelength_axis = data_dict['wavelength_axis']
    interface.save_global_fit_results(results, output_dir=gta_dir)
    
    print(f"\nGTA结果已保存到: {gta_dir}")
    
    return results


def create_comparison(mcr_results, global_results, data_dict, output_dir, method_name):
    """创建MCR-ALS与全局拟合的比较图"""
    compare_mcr_and_global_fit(
        mcr_results,
        global_results,
        data_dict['time_axis'],
        data_dict['wavelength_axis'],
        save_path=os.path.join(output_dir, f"comparison_mcr_{method_name}.png"),
        show_plot=False
    )
    print(f"比较图已保存: comparison_mcr_{method_name}.png")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="MCR-ALS到全局拟合的自动化工作流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行GLA和两种GTA模型 (默认)
  python auto_workflow.py --mcr_results results
  
  # 只运行GLA
  python auto_workflow.py --mcr_results results --method gla
  
  # 运行GTA顺序模型
  python auto_workflow.py --mcr_results results --method gta --model sequential
  
  # 提供原始数据文件
  python auto_workflow.py --mcr_results results --data_file data/TAS/TA_Average.csv
        """
    )
    
    parser.add_argument('--mcr_results', type=str, required=True,
                       help='MCR-ALS结果目录路径')
    parser.add_argument('--data_file', type=str, default=None,
                       help='原始数据文件路径 (可选)')
    parser.add_argument('--file_type', type=str, default='handle',
                       choices=['handle', 'raw'],
                       help='数据文件类型 (默认: handle)')
    parser.add_argument('--method', type=str, default='all',
                       choices=['all', 'gla', 'gta'],
                       help='分析方法 (默认: all)')
    parser.add_argument('--model', type=str, default='both',
                       choices=['both', 'sequential', 'parallel'],
                       help='GTA动力学模型 (默认: both)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录 (默认: mcr_results/global_fit)')
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        output_dir = os.path.join(args.mcr_results, "global_fit")
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("瞬态吸收光谱自动化分析工作流程")
    print("MCR-ALS → 全局拟合")
    print("=" * 70)
    
    # 步骤1: 准备数据
    print("\n步骤1: 从MCR-ALS结果准备数据")
    print("-" * 70)
    
    interface = MCRALSInterface(args.mcr_results)
    
    try:
        data_dict = interface.prepare_for_global_fitting(
            data_file=args.data_file,
            file_type=args.file_type
        )
    except Exception as e:
        print(f"\n错误: 无法准备数据")
        print(f"详细信息: {e}")
        print("\n请确保:")
        print("  1. MCR-ALS结果目录存在且包含必要文件:")
        print("     - concentration_profiles.csv")
        print("     - pure_spectra.csv")
        print("  2. 如果提供了数据文件，请确保路径正确")
        return 1
    
    # 准备MCR-ALS结果字典 (用于比较)
    mcr_results = {
        'C_mcr': data_dict['C_mcr'],
        'S_mcr': data_dict['S_mcr'],
        'mcr_lof': data_dict.get('mcr_lof', None)
    }
    
    # 步骤2: 执行分析
    all_results = {}
    
    if args.method in ['all', 'gla']:
        try:
            gla_results = run_gla(data_dict, output_dir)
            all_results['gla'] = gla_results
            create_comparison(mcr_results, gla_results, data_dict, output_dir, 'gla')
        except Exception as e:
            print(f"\nGLA分析失败: {e}")
    
    if args.method in ['all', 'gta']:
        models_to_run = []
        if args.model == 'both':
            models_to_run = ['sequential', 'parallel']
        else:
            models_to_run = [args.model]
        
        for model_type in models_to_run:
            try:
                gta_results = run_gta(data_dict, output_dir, model_type)
                all_results[f'gta_{model_type}'] = gta_results
                create_comparison(mcr_results, gta_results, data_dict, 
                                output_dir, f'gta_{model_type}')
            except Exception as e:
                print(f"\nGTA ({model_type})分析失败: {e}")
    
    # 步骤3: 生成总结
    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)
    
    print("\n拟合质量比较:")
    print("-" * 70)
    if mcr_results['mcr_lof'] is not None:
        print(f"MCR-ALS LOF:              {mcr_results['mcr_lof']:.4f}%")
    
    for method_name, results in all_results.items():
        print(f"{method_name.upper()} LOF:              {results['lof']:.4f}%")
        if 'tau_optimal' in results:
            tau_str = ', '.join([f'{t:.2f}' for t in results['tau_optimal']])
            print(f"  寿命: [{tau_str}]")
    
    print("\n所有结果已保存到:", output_dir)
    print("\n生成的文件:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")
    
    print("\n建议:")
    print("  1. 查看各方法的拟合图表和比较图")
    print("  2. 根据LOF值和物理意义选择最合适的模型")
    print("  3. 阅读详细拟合报告了解参数不确定度")
    print("  4. 检查残差图以验证拟合质量")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
