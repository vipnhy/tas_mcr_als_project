#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCR-ALS 约束系统使用示例
演示如何使用新的约束配置功能
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


def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 生成合成数据
    D, C_true, S_true = generate_synthetic_tas_data(n_components=3)
    
    # 使用默认约束（非负性约束 + 组分数量验证）
    mcr = MCRALS(n_components=3, max_iter=50)
    mcr.fit(D)
    
    print(f"收敛后的LOF: {mcr.lof_[-1]:.4f}%")
    print("激活的约束:")
    for name, config in mcr.get_constraint_info().items():
        print(f"  - {config['name']}")
    print()


def example_spectral_smoothness():
    """光谱平滑度约束示例"""
    print("=== 光谱平滑度约束示例 ===")
    
    # 生成带噪声的数据
    D, C_true, S_true = generate_synthetic_tas_data(n_components=2)
    D_noisy = D + 0.05 * np.random.randn(*D.shape)
    
    # 创建启用光谱平滑约束的配置
    config = ConstraintConfig()
    config.enable_constraint("spectral_smoothness")
    config.set_constraint_parameter("spectral_smoothness", "lambda", 0.01)
    
    mcr = MCRALS(n_components=2, max_iter=50, constraint_config=config)
    mcr.fit(D_noisy)
    
    print(f"使用平滑约束后的LOF: {mcr.lof_[-1]:.4f}%")
    print()


def example_custom_template():
    """使用自定义约束模板"""
    print("=== 使用约束模板示例 ===")
    
    D, C_true, S_true = generate_synthetic_tas_data(n_components=2)
    
    # 使用严格约束模板
    mcr_strict = MCRALS(n_components=2, max_iter=50, 
                       constraint_config="mcr/constraint_templates/strict_constraints.json")
    mcr_strict.fit(D)
    
    # 使用宽松约束模板  
    mcr_relaxed = MCRALS(n_components=2, max_iter=50,
                        constraint_config="mcr/constraint_templates/relaxed_constraints.json")
    mcr_relaxed.fit(D)
    
    print(f"严格约束模板LOF: {mcr_strict.lof_[-1]:.4f}%")
    print(f"宽松约束模板LOF: {mcr_relaxed.lof_[-1]:.4f}%")
    print()


def example_dynamic_parameters():
    """动态修改约束参数"""
    print("=== 动态修改约束参数示例 ===")
    
    D, C_true, S_true = generate_synthetic_tas_data(n_components=2)
    
    config = ConstraintConfig()
    config.enable_constraint("spectral_smoothness")
    
    mcr = MCRALS(n_components=2, max_iter=50, constraint_config=config)
    
    # 测试不同的平滑度参数
    smoothness_values = [1e-4, 1e-3, 1e-2]
    
    for lambda_val in smoothness_values:
        mcr.set_constraint_parameter("spectral_smoothness", "lambda", lambda_val)
        mcr.fit(D)
        print(f"平滑度参数 λ={lambda_val}: LOF = {mcr.lof_[-1]:.4f}%")
    print()


def example_create_custom_config():
    """创建自定义约束配置"""
    print("=== 创建自定义约束配置示例 ===")
    
    # 创建新的约束配置
    config = ConstraintConfig()
    
    # 添加自定义约束
    custom_constraint = {
        "name": "强平滑度约束",
        "description": "用于高噪声数据的强平滑约束",
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
    
    # 保存自定义配置
    config.save_to_file("/tmp/custom_constraints.json")
    print("自定义约束配置已保存到 /tmp/custom_constraints.json")
    
    # 使用自定义配置
    D, _, _ = generate_synthetic_tas_data(n_components=2)
    mcr = MCRALS(n_components=2, max_iter=50, constraint_config=config)
    mcr.fit(D)
    
    print(f"使用自定义约束配置的LOF: {mcr.lof_[-1]:.4f}%")
    print()


def example_constraint_validation():
    """约束验证示例"""
    print("=== 约束验证示例 ===")
    
    # 测试组分数量验证
    try:
        # 尝试创建超出范围的组分数量
        mcr = MCRALS(n_components=6)  # 默认最大为4
    except ValueError as e:
        print(f"约束验证成功: {e}")
    
    # 修改组分数量范围
    config = ConstraintConfig()
    config.set_constraint_parameter("component_count_range", "max_components", 6)
    
    try:
        # 现在可以使用6个组分
        mcr = MCRALS(n_components=6, constraint_config=config)
        print("成功创建6个组分的MCR-ALS模型（修改约束后）")
    except ValueError as e:
        print(f"创建失败: {e}")
    print()


if __name__ == '__main__':
    print("MCR-ALS 约束系统使用示例\n")
    
    # 设置随机种子以获得可重现的结果
    np.random.seed(42)
    
    # 运行所有示例
    example_basic_usage()
    example_spectral_smoothness()
    example_custom_template()
    example_dynamic_parameters()
    example_create_custom_config()
    example_constraint_validation()
    
    print("示例运行完成！")
    print("\n可用的约束模板文件:")
    template_dir = "mcr/constraint_templates"
    if os.path.exists(template_dir):
        for file in os.listdir(template_dir):
            if file.endswith('.json'):
                print(f"  - {os.path.join(template_dir, file)}")
    
    print("\n约束系统功能总结:")
    print("✓ 非负性约束（默认启用）")
    print("✓ 光谱平滑度约束（可选）")
    print("✓ 组分数量范围验证（默认启用）")
    print("✓ JSON模板配置系统")
    print("✓ 动态参数修改")
    print("✓ 自定义约束添加")