#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_run_main.py - 测试 run_main.py 功能

这个脚本用于测试 run_main.py 的各种功能
"""

import sys
import os
import tempfile
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_main import TASAnalyzer, load_config


def test_analyzer_basic():
    """测试 TASAnalyzer 基本功能"""
    print("测试 TASAnalyzer 基本功能...")
    
    # 测试基本初始化（中文）
    analyzer_cn = TASAnalyzer(
        file_path="data/TAS/TA_Average-crop-TCA-tzs-tzs-tzs-chirp.csv",
        file_type="handle",
        wavelength_range=(420, 750),
        delay_range=(0.1, 50),
        n_components=3,
        language='chinese'
    )
    
    print(f"中文版 - 文件路径: {analyzer_cn.file_path}")
    print(f"中文版 - 语言设置: {analyzer_cn.language}")
    
    # 测试英文版本
    analyzer_en = TASAnalyzer(
        file_path="data/TAS/TA_Average-crop-TCA-tzs-tzs-tzs-chirp.csv",
        file_type="handle",
        wavelength_range=(420, 750),
        delay_range=(0.1, 50),
        n_components=3,
        language='english'
    )
    
    print(f"英文版 - 文件路径: {analyzer_en.file_path}")
    print(f"英文版 - 语言设置: {analyzer_en.language}")
    print(f"英文版标签示例 - 浓度轮廓: {analyzer_en.labels['concentration_profiles']}")
    print(f"中文版标签示例 - 浓度轮廓: {analyzer_cn.labels['concentration_profiles']}")
    
    print("✓ TASAnalyzer 初始化测试通过\n")


def test_config_loading():
    """测试配置文件加载功能"""
    print("测试配置文件加载功能...")
    
    # 创建临时配置文件
    config_data = {
        "file_path": "test_data.csv",
        "file_type": "handle",
        "wavelength_range": [400, 800],
        "delay_range": [0, 100],
        "n_components": 2,
        "language": "english"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_config_file = f.name
    
    try:
        # 测试配置加载
        loaded_config = load_config(temp_config_file)
        
        if loaded_config:
            print("✓ 配置文件加载成功")
            for key, value in loaded_config.items():
                print(f"  {key}: {value}")
        else:
            print("✗ 配置文件加载失败")
    
    finally:
        # 清理临时文件
        os.unlink(temp_config_file)
    
    print()


def test_data_loading():
    """测试数据加载功能（如果数据文件存在）"""
    print("测试数据加载功能...")
    
    test_file = "data/TAS/TA_Average-crop-TCA-tzs-tzs-tzs-chirp.csv"
    
    if os.path.exists(test_file):
        analyzer = TASAnalyzer(
            file_path=test_file,
            file_type="handle",
            wavelength_range=(420, 750),
            delay_range=(0.1, 50),
            n_components=3
        )
        
        try:
            success = analyzer.load_data()
            if success:
                print("✓ 数据加载成功")
                print(f"  数据形状: {analyzer.D.shape}")
                print(f"  时间范围: {analyzer.time_axis.min():.2f} - {analyzer.time_axis.max():.2f}")
                print(f"  波长范围: {analyzer.wavelength_axis.min():.1f} - {analyzer.wavelength_axis.max():.1f}")
            else:
                print("✗ 数据加载失败")
        except Exception as e:
            print(f"✗ 数据加载异常: {e}")
    else:
        print(f"⚠ 测试数据文件不存在: {test_file}")
    
    print()


def test_config_files():
    """测试项目中的配置文件"""
    print("测试项目配置文件...")
    
    config_files = [
        "config_example.json",
        "config_english.json", 
        "config_detailed.json"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            config = load_config(config_file)
            if config:
                print(f"✓ {config_file} 加载成功")
                if 'language' in config:
                    print(f"  语言设置: {config['language']}")
            else:
                print(f"✗ {config_file} 加载失败")
        else:
            print(f"⚠ {config_file} 文件不存在")
    
    print()


def main():
    """主测试函数"""
    print("=" * 50)
    print("         run_main.py 功能测试")
    print("=" * 50)
    print()
    
    # 检查当前工作目录
    print(f"当前工作目录: {os.getcwd()}")
    print()
    
    # 运行各项测试
    test_analyzer_basic()
    test_config_loading()
    test_config_files()
    test_data_loading()
    
    print("=" * 50)
    print("         测试完成")
    print("=" * 50)
    
    print("\n要运行完整的分析测试，请使用以下命令:")
    print("# 中文版本")
    print("python run_main.py --config config_example.json")
    print("# 英文版本") 
    print("python run_main.py --config config_english.json")
    print("# 交互式")
    print("python run_main.py")


if __name__ == '__main__':
    main()
