#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCR-ALS实验框架演示脚本
展示实验框架的核心功能和使用方法
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcr_experiment import MCRExperimentRunner
from main import generate_synthetic_tas_data


def demo_basic_experiment():
    """演示基本实验功能"""
    print("=" * 50)
    print("MCR-ALS实验框架演示")
    print("=" * 50)
    
    print("📊 生成演示数据...")
    # 生成合成TAS数据
    data_matrix, C_true, S_true = generate_synthetic_tas_data(
        n_times=60, n_wls=100, n_components=3
    )
    print(f"   数据矩阵形状: {data_matrix.shape}")
    print(f"   真实组分数量: {C_true.shape[1]}")
    
    print("\n🔬 创建实验运行器...")
    runner = MCRExperimentRunner(output_base_dir="demo_experiments")
    print(f"   输出目录: {runner.current_experiment_dir}")
    
    print("\n⚡ 开始实验 (简化版本)...")
    print("   配置: 1-3组分, 每个配置3次随机运行")
    
    # 运行简化实验
    runner.run_multi_round_experiment(
        data_matrix=data_matrix,
        n_components_range=[1, 2, 3],  # 测试1-3个组分
        num_random_runs=3,             # 每个配置运行3次
        max_iter=150,                  # 较少迭代次数以加快演示
        tolerance=1e-6,
        target_lof=0.2                 # 目标LOF < 0.2%
    )
    
    return runner


def display_results(runner):
    """展示实验结果"""
    print("\n" + "=" * 50)
    print("实验结果汇总")
    print("=" * 50)
    
    # 基本统计
    total_experiments = len(runner.results)
    successful_runs = sum(1 for r in runner.results if r.converged)
    target_achieved = sum(1 for r in runner.results if r.final_lof < 0.2)
    
    print(f"📈 实验统计:")
    print(f"   总实验次数: {total_experiments}")
    print(f"   成功收敛: {successful_runs} ({successful_runs/total_experiments*100:.1f}%)")
    print(f"   达到目标LOF(<0.2%): {target_achieved} ({target_achieved/total_experiments*100:.1f}%)")
    
    # 性能分析
    if runner.results:
        lof_values = [r.final_lof for r in runner.results if r.converged]
        if lof_values:
            print(f"\n📊 LOF性能:")
            print(f"   最佳LOF: {min(lof_values):.4f}%")
            print(f"   平均LOF: {sum(lof_values)/len(lof_values):.4f}%")
            print(f"   最差LOF: {max(lof_values):.4f}%")
            
            # 最佳配置
            best_result = min(runner.results, key=lambda x: x.final_lof)
            print(f"\n🏆 最佳配置:")
            print(f"   实验ID: {best_result.experiment_id}")
            print(f"   LOF值: {best_result.final_lof:.4f}%")
            print(f"   约束类型: {best_result.constraint_type}")
            print(f"   组分数量: {best_result.n_components}")
            print(f"   迭代次数: {best_result.iterations_to_converge}")
            print(f"   计算时间: {best_result.computation_time:.3f}秒")


def show_file_structure(runner):
    """展示生成的文件结构"""
    print("\n" + "=" * 50)
    print("生成的文件结构")
    print("=" * 50)
    
    experiment_dir = runner.current_experiment_dir
    
    print(f"📁 实验根目录: {experiment_dir.name}/")
    
    # 遍历目录结构
    for level_dir in sorted(experiment_dir.iterdir()):
        if level_dir.is_dir():
            print(f"├── 📁 {level_dir.name}/")
            
            # 显示目录中的主要文件
            files = list(level_dir.iterdir())
            for i, file_path in enumerate(sorted(files)[:3]):  # 只显示前3个文件
                prefix = "│   ├──" if i < min(2, len(files)-1) else "│   └──"
                if file_path.is_file():
                    size_kb = file_path.stat().st_size / 1024
                    print(f"{prefix} 📄 {file_path.name} ({size_kb:.1f}KB)")
                elif file_path.is_dir():
                    print(f"{prefix} 📁 {file_path.name}/")
            
            if len(files) > 3:
                print(f"│   └── ... 还有{len(files)-3}个文件")


def show_summary_report(runner):
    """展示汇总报告内容"""
    print("\n" + "=" * 50)
    print("Level 1 汇总报告预览")
    print("=" * 50)
    
    summary_file = runner.current_experiment_dir / "level1_summary" / "experiment_summary.json"
    
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        print("📋 实验元数据:")
        metadata = summary['experiment_metadata']
        for key, value in metadata.items():
            print(f"   {key}: {value}")
        
        print("\n📈 整体性能:")
        performance = summary['overall_performance']
        for key, value in performance.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        
        print("\n🎯 最佳配置:")
        best_config = summary['best_configurations']['best_overall_lof']
        for key, value in best_config.items():
            print(f"   {key}: {value}")
    else:
        print("⚠️  汇总报告文件未找到")


def demo_constraint_analysis():
    """演示约束分析功能"""
    print("\n" + "=" * 50)
    print("约束分析演示")
    print("=" * 50)
    
    print("🔍 支持的约束类型:")
    print("   • basic: 基本约束 (仅非负性)")
    print("   • smoothness_0.1: 弱平滑度约束 (λ=0.1)")
    print("   • smoothness_0.2: 中等平滑度约束 (λ=0.2)")
    print("   • smoothness_0.5: 强平滑度约束 (λ=0.5)")
    print("   • smoothness_1.0: 很强平滑度约束 (λ=1.0)")
    print("   • combined: 组合约束 (非负性 + 平滑度)")
    
    print("\n📊 约束强度梯度测试:")
    print("   惩罚因子范围: 0.1 → 0.2 → 0.5 → 1.0")
    print("   测试目标: 找到最优约束强度平衡性能和稳定性")


def main():
    """主演示函数"""
    try:
        # 运行基本实验演示
        runner = demo_basic_experiment()
        
        # 显示实验结果
        display_results(runner)
        
        # 显示文件结构
        show_file_structure(runner)
        
        # 显示汇总报告
        show_summary_report(runner)
        
        # 显示约束分析说明
        demo_constraint_analysis()
        
        print("\n" + "=" * 50)
        print("演示完成!")
        print("=" * 50)
        print(f"📁 完整结果保存在: {runner.current_experiment_dir}")
        print("📊 可查看生成的图表文件 (plots/)")
        print("📋 可分析JSON和Excel格式的详细报告")
        print("\n💡 提示: 运行完整实验请使用 mcr_experiment.py")
        
        return runner
        
    except KeyboardInterrupt:
        print("\n演示被用户中断")
        return None
    except Exception as e:
        print(f"\n演示过程中发生错误: {e}")
        return None


if __name__ == "__main__":
    runner = main()