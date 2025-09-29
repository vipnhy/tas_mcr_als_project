#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试约束功能的集成测试
"""
import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mcr.mcr_als import MCRALS
from mcr.constraint_config import ConstraintConfig
from main import generate_synthetic_tas_data


def test_default_constraints():
    """测试默认约束配置"""
    print("=== 测试默认约束配置 ===")
    
    # 生成合成数据
    n_comps = 3
    D, C_true, S_true = generate_synthetic_tas_data(n_components=n_comps)
    
    # 使用默认约束配置
    mcr_solver = MCRALS(n_components=n_comps, max_iter=50)
    mcr_solver.fit(D)
    
    # 检查约束信息
    constraint_info = mcr_solver.get_constraint_info()
    print("激活的约束:")
    for name, config in constraint_info.items():
        print(f"  - {config['name']}: {config['description']}")
    
    print(f"收敛后的LOF: {mcr_solver.lof_[-1]:.4f}%")
    print()


def test_spectral_smoothness():
    """测试光谱平滑度约束"""
    print("=== 测试光谱平滑度约束 ===")
    
    # 生成带噪声的合成数据
    n_comps = 2
    D, C_true, S_true = generate_synthetic_tas_data(n_components=n_comps)
    
    # 添加噪声到数据
    noise_level = 0.02
    D_noisy = D + noise_level * np.random.randn(*D.shape)
    
    # 创建启用光谱平滑约束的配置
    config = ConstraintConfig()
    config.enable_constraint("spectral_smoothness")
    config.set_constraint_parameter("spectral_smoothness", "lambda", 1e-2)
    
    # 无平滑约束
    mcr_no_smooth = MCRALS(n_components=n_comps, max_iter=50)
    mcr_no_smooth.fit(D_noisy)
    
    # 有平滑约束
    mcr_smooth = MCRALS(n_components=n_comps, max_iter=50, constraint_config=config)
    mcr_smooth.fit(D_noisy)
    
    print(f"无平滑约束的LOF: {mcr_no_smooth.lof_[-1]:.4f}%")
    print(f"有平滑约束的LOF: {mcr_smooth.lof_[-1]:.4f}%")
    print()
    
    return mcr_no_smooth, mcr_smooth, S_true


def test_component_count_validation():
    """测试组分数量验证约束"""
    print("=== 测试组分数量验证约束 ===")
    
    # 测试超出范围的组分数量
    try:
        mcr_solver = MCRALS(n_components=5)  # 超出默认最大值4
        print("错误：应该抛出组分数量超出范围的异常")
    except ValueError as e:
        print(f"正确：捕获到预期异常 - {e}")
    
    try:
        mcr_solver = MCRALS(n_components=0)  # 低于最小值1
        print("错误：应该抛出组分数量超出范围的异常")
    except ValueError as e:
        print(f"正确：捕获到预期异常 - {e}")
    
    # 测试在范围内的组分数量
    try:
        mcr_solver = MCRALS(n_components=3)  # 在范围内
        print("正确：组分数量3在允许范围内")
    except ValueError as e:
        print(f"错误：不应该抛出异常 - {e}")
    
    print()


def test_custom_constraint_template():
    """测试自定义约束模板"""
    print("=== 测试自定义约束模板 ===")
    
    # 使用严格约束模板
    template_path = "mcr/constraint_templates/strict_constraints.json"
    
    n_comps = 2
    D, C_true, S_true = generate_synthetic_tas_data(n_components=n_comps)
    
    mcr_solver = MCRALS(n_components=n_comps, max_iter=50, constraint_config=template_path)
    mcr_solver.fit(D)
    
    constraint_info = mcr_solver.get_constraint_info()
    print("严格约束模板激活的约束:")
    for name, config in constraint_info.items():
        print(f"  - {config['name']}")
        if 'parameters' in config and config['parameters']:
            print(f"    参数: {config['parameters']}")
    
    print(f"收敛后的LOF: {mcr_solver.lof_[-1]:.4f}%")
    print()


def test_constraint_parameter_modification():
    """测试约束参数动态修改"""
    print("=== 测试约束参数动态修改 ===")
    
    n_comps = 2
    D, C_true, S_true = generate_synthetic_tas_data(n_components=n_comps)
    
    config = ConstraintConfig()
    config.enable_constraint("spectral_smoothness")
    
    mcr_solver = MCRALS(n_components=n_comps, max_iter=50, constraint_config=config)
    
    # 修改平滑度参数
    original_lambda = 1e-3
    new_lambda = 5e-2
    
    mcr_solver.set_constraint_parameter("spectral_smoothness", "lambda", new_lambda)
    
    # 验证参数已修改
    constraint_info = mcr_solver.get_constraint_info()
    actual_lambda = constraint_info["spectral_smoothness"]["parameters"]["lambda"]
    
    if actual_lambda == new_lambda:
        print(f"正确：平滑度参数已修改为 {new_lambda}")
    else:
        print(f"错误：参数修改失败，期望 {new_lambda}，实际 {actual_lambda}")
    
    mcr_solver.fit(D)
    print(f"修改参数后的LOF: {mcr_solver.lof_[-1]:.4f}%")
    print()


def visualize_smoothness_comparison(mcr_no_smooth, mcr_smooth, S_true):
    """可视化平滑约束的效果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 真实光谱
    axes[0].set_title("真实光谱")
    axes[0].plot(S_true)
    axes[0].set_xlabel("波长索引")
    axes[0].set_ylabel("强度")
    
    # 无平滑约束的结果
    axes[1].set_title("无平滑约束")
    axes[1].plot(mcr_no_smooth.S_opt_)
    axes[1].set_xlabel("波长索引")
    axes[1].set_ylabel("强度")
    
    # 有平滑约束的结果
    axes[2].set_title("有平滑约束")
    axes[2].plot(mcr_smooth.S_opt_)
    axes[2].set_xlabel("波长索引")
    axes[2].set_ylabel("强度")
    
    plt.tight_layout()
    plt.savefig('/tmp/smoothness_comparison.png', dpi=150, bbox_inches='tight')
    print("光谱平滑度比较图已保存到 /tmp/smoothness_comparison.png")


if __name__ == '__main__':
    print("MCR-ALS 约束功能集成测试\n")
    
    # 运行所有测试
    test_default_constraints()
    test_component_count_validation()
    test_custom_constraint_template()
    test_constraint_parameter_modification()
    
    # 测试光谱平滑度约束并可视化
    mcr_no_smooth, mcr_smooth, S_true = test_spectral_smoothness()
    
    # 生成对比图
    visualize_smoothness_comparison(mcr_no_smooth, mcr_smooth, S_true)
    
    print("所有测试完成!")