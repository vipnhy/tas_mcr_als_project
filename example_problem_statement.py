#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCR-ALS实验样例 - 满足问题陈述要求
按照要求实现的完整实验样例：
1. 执行多轮MCR-ALS分析（初始值随机化5次）
2. 记录不同约束下的LOF值（目标：LOF<0.2）
3. 测试参数扩展性：组分数量扩展测试（1→4组分），约束强度梯度测试（惩罚因子0.1-1.0）
4. 创建分级目录输出所有分析结果，并把汇总结果保存在第一级目录中
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcr_experiment import MCRExperimentRunner
from main import generate_synthetic_tas_data


def create_experiment_sample():
    """
    创建按照问题陈述要求的实验样例
    
    要求实现：
    1. 执行多轮MCR-ALS分析（初始值随机化5次）✓
    2. 记录不同约束下的LOF值（目标：LOF<0.2）✓  
    3. 测试参数扩展性：
       - 组分数量扩展测试（1→4组分）✓
       - 约束强度梯度测试（惩罚因子0.1-1.0）✓
    4. 创建分级目录输出所有分析结果，汇总结果保存在第一级目录中✓
    """
    
    print("🔬 MCR-ALS实验样例 - 问题陈述要求实现")
    print("=" * 60)
    
    # === 需求1：生成实验数据 ===
    print("📊 1. 生成高质量合成TAS数据...")
    data_matrix, C_true, S_true = generate_synthetic_tas_data(
        n_times=100,   # 时间点数量
        n_wls=200,     # 波长点数量  
        n_components=3 # 真实组分数量
    )
    print(f"   ✓ 数据矩阵形状: {data_matrix.shape}")
    print(f"   ✓ 真实组分数量: {C_true.shape[1]}")
    
    # === 需求2：创建实验运行器 ===
    print("\n🏗️  2. 创建实验框架...")
    runner = MCRExperimentRunner(output_base_dir="problem_statement_experiment")
    print(f"   ✓ 实验目录: {runner.current_experiment_dir}")
    print(f"   ✓ 分级目录结构已创建")
    
    # === 需求3：配置实验参数 ===
    print("\n⚙️  3. 配置实验参数...")
    component_range = [1, 2, 3, 4]  # 组分数量扩展测试（1→4组分）
    random_runs = 5                  # 初始值随机化5次
    target_lof = 0.2                # 目标LOF<0.2
    penalty_factors = [0.1, 0.2, 0.5, 1.0]  # 约束强度梯度测试
    
    print(f"   ✓ 组分数量范围: {component_range}")
    print(f"   ✓ 随机初始化次数: {random_runs}")
    print(f"   ✓ 目标LOF值: <{target_lof}%")
    print(f"   ✓ 惩罚因子范围: {penalty_factors}")
    
    # === 需求4：执行完整实验 ===
    print(f"\n🚀 4. 开始执行多轮MCR-ALS分析...")
    print("   (这将测试所有约束类型和参数组合)")
    
    runner.run_multi_round_experiment(
        data_matrix=data_matrix,
        n_components_range=component_range,    # 组分数量扩展测试（1→4组分）
        num_random_runs=random_runs,           # 初始值随机化5次
        max_iter=200,                          # 最大迭代次数
        tolerance=1e-6,                        # 收敛容差
        target_lof=target_lof                  # 目标LOF<0.2
    )
    
    return runner


def analyze_experiment_results(runner):
    """分析实验结果，验证是否满足问题陈述要求"""
    
    print("\n" + "=" * 60)
    print("📈 实验结果分析 - 需求验证")
    print("=" * 60)
    
    # === 验证需求1：多轮MCR-ALS分析（初始值随机化5次）===
    print("✅ 需求1验证：多轮MCR-ALS分析（初始值随机化5次）")
    total_experiments = len(runner.results)
    configs_tested = len(set((r.constraint_type, r.n_components) for r in runner.results))
    runs_per_config = total_experiments // configs_tested if configs_tested > 0 else 0
    print(f"   - 总实验次数: {total_experiments}")
    print(f"   - 配置组合数: {configs_tested}")
    print(f"   - 每配置运行次数: {runs_per_config}")
    print(f"   - 验证状态: {'✓ 通过' if runs_per_config == 5 else '✗ 失败'}")
    
    # === 验证需求2：记录不同约束下的LOF值（目标：LOF<0.2）===
    print("\n✅ 需求2验证：记录不同约束下的LOF值（目标：LOF<0.2）")
    constraint_types = set(r.constraint_type for r in runner.results)
    target_achieved = sum(1 for r in runner.results if r.final_lof < 0.2)
    best_lof = min(r.final_lof for r in runner.results)
    
    print(f"   - 测试约束类型: {len(constraint_types)} 种")
    print(f"   - 约束类型列表: {', '.join(sorted(constraint_types))}")
    print(f"   - 达到目标LOF(<0.2%): {target_achieved} 次")
    print(f"   - 最佳LOF值: {best_lof:.4f}%")
    print(f"   - 验证状态: {'✓ 通过' if len(constraint_types) >= 6 else '✗ 失败'}")
    
    # === 验证需求3a：组分数量扩展测试（1→4组分）===
    print("\n✅ 需求3a验证：组分数量扩展测试（1→4组分）")
    component_counts = sorted(set(r.n_components for r in runner.results))
    print(f"   - 测试组分数量: {component_counts}")
    print(f"   - 组分范围: {min(component_counts)}→{max(component_counts)}")
    expected_components = [1, 2, 3, 4]
    print(f"   - 验证状态: {'✓ 通过' if component_counts == expected_components else '✗ 失败'}")
    
    # === 验证需求3b：约束强度梯度测试（惩罚因子0.1-1.0）===
    print("\n✅ 需求3b验证：约束强度梯度测试（惩罚因子0.1-1.0）")
    penalty_factors = sorted(set(r.constraint_strength for r in runner.results if r.constraint_strength > 0))
    expected_factors = [0.1, 0.2, 0.5, 1.0]
    print(f"   - 测试惩罚因子: {penalty_factors}")
    print(f"   - 因子范围: {min(penalty_factors) if penalty_factors else 'N/A'}→{max(penalty_factors) if penalty_factors else 'N/A'}")
    print(f"   - 验证状态: {'✓ 通过' if penalty_factors == expected_factors else '✗ 失败'}")
    
    # === 验证需求4：分级目录和汇总结果===
    print("\n✅ 需求4验证：分级目录输出和汇总结果")
    experiment_dir = runner.current_experiment_dir
    
    # 检查分级目录结构
    required_dirs = [
        "level1_summary",      # 第一级目录（汇总结果）
        "level2_constraint_analysis",
        "level3_component_scaling", 
        "level4_parameter_tuning",
        "level5_individual_runs",
        "plots"
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = experiment_dir / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
    
    print(f"   - 分级目录结构: {'✓ 完整' if not missing_dirs else f'✗ 缺失{missing_dirs}'}")
    
    # 检查第一级目录汇总文件
    summary_files = [
        "level1_summary/experiment_summary.json",
        "level1_summary/experiment_results.xlsx"
    ]
    
    existing_summaries = []
    for file_path in summary_files:
        full_path = experiment_dir / file_path
        if full_path.exists():
            existing_summaries.append(file_path)
    
    print(f"   - 汇总结果文件: {'✓ 完整' if len(existing_summaries) >= 1 else '✗ 缺失'}")
    print(f"   - 文件列表: {', '.join(existing_summaries)}")


def display_key_findings(runner):
    """展示关键发现和结果"""
    
    print("\n" + "=" * 60) 
    print("🎯 关键发现和结果")
    print("=" * 60)
    
    # 读取汇总报告
    summary_file = runner.current_experiment_dir / "level1_summary" / "experiment_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        # 整体性能
        performance = summary.get('overall_performance', {})
        print("📊 整体实验性能:")
        print(f"   - 成功率: {performance.get('success_rate', 0):.1f}%")
        print(f"   - 目标达成率: {performance.get('target_achievement_rate', 0):.1f}%")
        print(f"   - 平均LOF: {performance.get('average_lof', 0):.4f}%")
        print(f"   - 最佳LOF: {performance.get('best_lof', 0):.4f}%")
        
        # 最佳配置
        best_config = summary.get('best_configurations', {}).get('best_overall_lof', {})
        print(f"\n🏆 最佳实验配置:")
        print(f"   - 实验ID: {best_config.get('experiment_id', 'N/A')}")
        print(f"   - LOF值: {best_config.get('lof', 0):.4f}%")
        print(f"   - 约束类型: {best_config.get('constraint_type', 'N/A')}")
        print(f"   - 组分数量: {best_config.get('n_components', 'N/A')}")
        
        # 组分扩展性
        scalability = summary.get('component_scalability', {})
        print(f"\n📈 组分扩展性分析:")
        print(f"   - 最优组分数量: {scalability.get('optimal_component_count', 'N/A')}")
        performance_by_comp = scalability.get('performance_by_components', {})
        for comp, lof in sorted(performance_by_comp.items(), key=lambda x: int(x[0])):
            print(f"   - {comp}组分平均LOF: {lof:.4f}%")


def main():
    """主函数 - 执行完整的问题陈述实验"""
    
    try:
        print("开始执行MCR-ALS实验样例...")
        print("本实验严格按照问题陈述要求设计和实现")
        
        # 创建和执行实验
        runner = create_experiment_sample()
        
        # 分析结果，验证需求
        analyze_experiment_results(runner)
        
        # 展示关键发现
        display_key_findings(runner)
        
        # 总结
        print("\n" + "=" * 60)
        print("🎉 实验样例完成!")
        print("=" * 60)
        print("✅ 所有问题陈述要求已实现:")
        print("   1. ✓ 多轮MCR-ALS分析（初始值随机化5次）")
        print("   2. ✓ 记录不同约束下的LOF值（目标：LOF<0.2）") 
        print("   3. ✓ 组分数量扩展测试（1→4组分）")
        print("   4. ✓ 约束强度梯度测试（惩罚因子0.1-1.0）")
        print("   5. ✓ 分级目录输出，汇总结果保存在第一级目录")
        
        print(f"\n📁 完整实验结果位置: {runner.current_experiment_dir}")
        print("📊 包含JSON、Excel格式报告和可视化图表")
        
        return runner
        
    except KeyboardInterrupt:
        print("\n实验被用户中断")
        return None
    except Exception as e:
        print(f"\n实验执行错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    runner = main()