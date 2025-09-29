#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TAS预处理模块测试脚本

验证预处理模块的基本功能
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_imports():
    """测试模块导入"""
    try:
        from preprocessing import (
            BaselineCorrector, NoiseFilter, DataSmoother, OutlierDetector,
            TASPreprocessingPipeline, preprocess_tas_data
        )
        print("✓ 所有模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        return False

def create_test_data():
    """创建测试数据"""
    wavelengths = np.linspace(400, 600, 50)
    delays = np.linspace(0.1, 10, 20)
    
    # 简单的高斯峰 + 噪声
    data = np.zeros((len(delays), len(wavelengths)))
    
    for i, delay in enumerate(delays):
        amp = np.exp(-delay/5.0)  # 指数衰减
        peak = amp * np.exp(-((wavelengths - 500) / 50)**2)
        noise = np.random.normal(0, 0.001, len(wavelengths))
        data[i, :] = peak + noise + 0.0001  # 加基线偏移
    
    df = pd.DataFrame(data, index=delays, columns=wavelengths)
    return df

def test_basic_functionality():
    """测试基本功能"""
    print("\n测试基本功能...")
    
    try:
        # 创建测试数据
        data = create_test_data()
        print(f"✓ 创建测试数据: {data.shape}")
        
        # 测试预处理管道
        from preprocessing import preprocess_tas_data
        
        processed_data = preprocess_tas_data(data, pipeline='standard', verbose=False)
        print(f"✓ 标准预处理管道完成: {processed_data.shape}")
        
        # 测试单独模块
        from preprocessing import BaselineCorrector, NoiseFilter
        
        # 基线校正
        corrector = BaselineCorrector(method='als', lam=1e5)
        corrected = corrector.correct(data)
        print(f"✓ 基线校正完成: {corrected.shape}")
        
        # 噪声过滤
        filter = NoiseFilter(method='gaussian', sigma=0.5)
        filtered = filter.filter_noise(data)
        print(f"✓ 噪声过滤完成: {filtered.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline():
    """测试预处理管道"""
    print("\n测试预处理管道...")
    
    try:
        from preprocessing import TASPreprocessingPipeline, create_standard_pipeline
        
        data = create_test_data()
        
        # 测试标准管道
        pipeline = create_standard_pipeline(verbose=False)
        processed = pipeline.fit_transform(data)
        print(f"✓ 标准管道测试通过: {processed.shape}")
        
        # 获取处理摘要
        summary = pipeline.get_processing_summary()
        print(f"✓ 处理摘要生成成功: {len(summary)} 项")
        
        # 测试自定义管道
        custom_steps = [
            {'name': 'baseline', 'processor': 'baseline', 
             'params': {'method': 'polynomial', 'degree': 2}},
            {'name': 'smooth', 'processor': 'smooth', 
             'params': {'method': 'savgol', 'window_length': 3, 'polyorder': 1}}
        ]
        
        custom_pipeline = TASPreprocessingPipeline(steps=custom_steps, verbose=False)
        custom_processed = custom_pipeline.fit_transform(data)
        print(f"✓ 自定义管道测试通过: {custom_processed.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 管道测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """运行所有测试"""
    print("开始TAS预处理模块测试...")
    print("=" * 50)
    
    tests = [
        ("导入测试", test_imports),
        ("基本功能测试", test_basic_functionality), 
        ("管道测试", test_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name}发生异常: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    
    all_passed = True
    for test_name, success in results:
        status = "通过" if success else "失败"
        symbol = "✓" if success else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not success:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 所有测试通过！预处理模块工作正常。")
    else:
        print("❌ 部分测试失败，请检查错误信息。")
    
    return all_passed

if __name__ == '__main__':
    run_all_tests()
