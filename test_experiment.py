#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCR-ALS实验框架测试脚本
小规模测试以验证功能正确性
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from mcr_experiment import MCRExperimentRunner
from main import generate_synthetic_tas_data


def run_small_test():
    """运行小规模测试"""
    print("=== MCR-ALS实验框架测试 ===")
    
    # 生成小规模合成数据
    print("生成测试数据...")
    data_matrix, C_true, S_true = generate_synthetic_tas_data(n_times=50, n_wls=80, n_components=2)
    print(f"数据矩阵形状: {data_matrix.shape}")
    
    # 创建实验运行器
    runner = MCRExperimentRunner(output_base_dir="test_experiments")
    
    # 运行小规模实验
    print("开始小规模实验...")
    runner.run_multi_round_experiment(
        data_matrix=data_matrix,
        n_components_range=[1, 2],  # 只测试1-2个组分
        num_random_runs=2,  # 每个配置只运行2次
        max_iter=100,  # 减少最大迭代次数
        tolerance=1e-6,
        target_lof=0.2
    )
    
    print(f"测试完成! 结果保存在: {runner.current_experiment_dir}")
    
    # 显示测试结果
    print("\n=== 测试结果汇总 ===")
    print(f"总实验次数: {len(runner.results)}")
    successful = [r for r in runner.results if r.converged]
    print(f"成功收敛: {len(successful)}")
    
    if successful:
        lof_values = [r.final_lof for r in successful]
        print(f"LOF范围: {min(lof_values):.4f}% - {max(lof_values):.4f}%")
        target_achieved = [r for r in successful if r.final_lof < 0.2]
        print(f"达到目标LOF(<0.2%): {len(target_achieved)}")
    
    return runner


if __name__ == "__main__":
    runner = run_small_test()