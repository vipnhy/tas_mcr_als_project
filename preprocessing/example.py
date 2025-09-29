#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TAS数据预处理示例脚本

演示如何使用预处理模块处理瞬态吸收光谱数据
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from preprocessing import (
    BaselineCorrector, NoiseFilter, DataSmoother, OutlierDetector,
    TASPreprocessingPipeline, create_standard_pipeline, create_gentle_pipeline,
    create_aggressive_pipeline, preprocess_tas_data
)


def create_sample_tas_data():
    """创建示例TAS数据"""
    # 参数设置
    wavelengths = np.linspace(400, 800, 200)  # 400-800nm, 200个点
    delays = np.logspace(-1, 2, 50)  # 0.1-100ps, 50个点
    
    # 创建基础信号
    data = np.zeros((len(delays), len(wavelengths)))
    
    # 添加几个高斯峰作为光谱特征
    peak1_center = 450  # nm
    peak1_width = 30
    peak2_center = 650  # nm  
    peak2_width = 50
    
    for i, delay in enumerate(delays):
        # 峰1: 衰减指数
        amp1 = 0.01 * np.exp(-delay/5.0)
        peak1 = amp1 * np.exp(-((wavelengths - peak1_center) / peak1_width)**2)
        
        # 峰2: 增长后衰减
        amp2 = 0.005 * (1 - np.exp(-delay/2.0)) * np.exp(-delay/20.0)
        peak2 = amp2 * np.exp(-((wavelengths - peak2_center) / peak2_width)**2)
        
        data[i, :] = peak1 - peak2  # 吸收和漂白
    
    # 添加基线漂移
    baseline = np.linspace(0.001, -0.002, len(wavelengths))
    data += baseline[np.newaxis, :]
    
    # 添加噪声
    noise = np.random.normal(0, 0.0005, data.shape)
    data += noise
    
    # 添加一些异常值
    outlier_indices = np.random.choice(data.size, size=int(0.001 * data.size), replace=False)
    flat_indices = np.unravel_index(outlier_indices, data.shape)
    data[flat_indices] += np.random.normal(0, 0.01, len(outlier_indices))
    
    # 创建DataFrame
    df = pd.DataFrame(data, index=delays, columns=wavelengths)
    df.index.name = 'delay_time'
    df.columns.name = 'wavelength'
    
    return df


def example_individual_modules():
    """演示各个预处理模块的单独使用"""
    print("=" * 60)
    print("演示各个预处理模块的单独使用")
    print("=" * 60)
    
    # 创建示例数据
    data = create_sample_tas_data()
    print(f"创建示例数据: {data.shape}")
    
    # 1. 异常值检测
    print("\n1. 异常值检测")
    detector = OutlierDetector(method='z_score', threshold=3.0)
    outlier_mask = detector.detect_outliers(data)
    stats = detector.get_outlier_statistics()
    print(f"   检测到异常值: {stats['outlier_count']} ({stats['outlier_percentage']:.2f}%)")
    
    # 处理异常值
    clean_data = detector.process_outliers(data, strategy='interpolate')
    print("   使用插值法处理异常值")
    
    # 2. 基线校正
    print("\n2. 基线校正")
    corrector = BaselineCorrector(method='als', lam=1e6, p=0.001)
    corrected_data = corrector.correct(clean_data)
    correction_stats = corrector.get_correction_stats()
    print(f"   基线校正完成 (方法: {correction_stats['method']})")
    
    # 3. 噪声过滤
    print("\n3. 噪声过滤")
    noise_filter = NoiseFilter(method='gaussian', sigma=1.0)
    
    # 首先估计噪声水平
    noise_stats = noise_filter.estimate_noise_level(corrected_data)
    print(f"   估计信噪比: {noise_stats['snr_estimate']:.2f}")
    
    filtered_data = noise_filter.filter_noise(corrected_data)
    filter_stats = noise_filter.get_filtering_stats()
    print(f"   噪声降低: {filter_stats['noise_reduction']['noise_reduction_ratio']*100:.1f}%")
    
    # 4. 数据平滑
    print("\n4. 数据平滑")
    smoother = DataSmoother(method='savgol', window_length=5, polyorder=2)
    smoothed_data = smoother.smooth(filtered_data)
    print("   数据平滑完成")
    
    return data, smoothed_data


def example_preprocessing_pipeline():
    """演示预处理管道的使用"""
    print("\n" + "=" * 60)
    print("演示预处理管道的使用")
    print("=" * 60)
    
    # 创建示例数据
    data = create_sample_tas_data()
    print(f"创建示例数据: {data.shape}")
    
    # 1. 标准管道
    print("\n1. 标准预处理管道")
    standard_pipeline = create_standard_pipeline(verbose=False)
    standard_processed = standard_pipeline.fit_transform(data)
    
    summary = standard_pipeline.get_processing_summary()
    print(f"   处理步骤: {summary['total_steps']}")
    print(f"   总耗时: {summary['total_processing_time']:.3f}s")
    
    improvement = summary['data_statistics']
    print(f"   信噪比改善: {improvement['snr_improvement']:.2f}x")
    print(f"   数据保真度: {improvement['data_preservation']:.3f}")
    
    # 2. 温和管道
    print("\n2. 温和预处理管道")
    gentle_processed = preprocess_tas_data(data, pipeline='gentle', verbose=False)
    print("   温和管道处理完成")
    
    # 3. 激进管道  
    print("\n3. 激进预处理管道")
    aggressive_processed = preprocess_tas_data(data, pipeline='aggressive', verbose=False)
    print("   激进管道处理完成")
    
    # 4. 自定义管道
    print("\n4. 自定义预处理管道")
    custom_steps = [
        {'name': 'outlier_detection', 'processor': 'outlier', 
         'params': {'method': 'iqr', 'factor': 1.5}, 'strategy': 'interpolate'},
        {'name': 'baseline_correction', 'processor': 'baseline', 
         'params': {'method': 'polynomial', 'degree': 3}},
        {'name': 'noise_filtering', 'processor': 'noise', 
         'params': {'method': 'bilateral', 'd': 5, 'sigma_color': 50, 'sigma_space': 50}}
    ]
    
    custom_pipeline = TASPreprocessingPipeline(steps=custom_steps, verbose=False)
    custom_processed = custom_pipeline.fit_transform(data)
    print("   自定义管道处理完成")
    
    return data, standard_processed, gentle_processed, aggressive_processed, custom_processed


def example_visualization():
    """演示可视化功能"""
    print("\n" + "=" * 60)
    print("演示可视化功能")
    print("=" * 60)
    
    # 创建数据和处理
    data = create_sample_tas_data()
    pipeline = create_standard_pipeline(verbose=False)
    processed_data = pipeline.fit_transform(data)
    
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 预处理管道步骤图
    print("1. 生成预处理管道步骤图...")
    fig1 = pipeline.plot_processing_pipeline(delay_index=10, figsize=(16, 12))
    fig1.suptitle('TAS数据预处理管道各步骤结果', fontsize=16)
    
    # 2. 处理前后对比图
    print("2. 生成处理前后对比图...")
    fig2 = pipeline.plot_comparison(delay_index=10, figsize=(12, 8))
    fig2.suptitle('TAS数据预处理前后对比', fontsize=14)
    
    # 3. 2D热图对比
    print("3. 生成2D热图对比...")
    fig3, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 原始数据热图
    im1 = axes[0].imshow(data.values, aspect='auto', origin='lower',
                        extent=[data.columns[0], data.columns[-1], 
                               data.index[0], data.index[-1]])
    axes[0].set_title('原始TAS数据')
    axes[0].set_xlabel('波长 (nm)')
    axes[0].set_ylabel('延迟时间 (ps)')
    plt.colorbar(im1, ax=axes[0])
    
    # 处理后数据热图
    im2 = axes[1].imshow(processed_data.values, aspect='auto', origin='lower',
                        extent=[processed_data.columns[0], processed_data.columns[-1], 
                               processed_data.index[0], processed_data.index[-1]])
    axes[1].set_title('预处理后TAS数据')
    axes[1].set_xlabel('波长 (nm)')
    axes[1].set_ylabel('延迟时间 (ps)')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    
    # 4. 单独模块可视化
    print("4. 生成单独模块可视化...")
    
    # 基线校正
    corrector = BaselineCorrector(method='als', lam=1e6, p=0.001)
    corrector.correct(data)
    fig4 = corrector.plot_correction(data, delay_index=10)
    fig4.suptitle('基线校正结果', fontsize=14)
    
    # 噪声过滤
    noise_filter = NoiseFilter(method='gaussian', sigma=1.0)
    noise_filter.filter_noise(data)
    fig5 = noise_filter.plot_filtering_result(data, delay_index=10)
    fig5.suptitle('噪声过滤结果', fontsize=14)
    
    plt.show()
    print("可视化完成！")


def example_batch_processing():
    """演示批量处理"""
    print("\n" + "=" * 60)
    print("演示批量处理")
    print("=" * 60)
    
    # 创建多个示例数据文件
    output_dir = Path('preprocessing_example_output')
    output_dir.mkdir(exist_ok=True)
    
    print("1. 创建示例数据文件...")
    sample_files = []
    for i in range(3):
        # 创建稍有不同的数据
        np.random.seed(42 + i)  # 不同的随机种子
        data = create_sample_tas_data()
        
        filename = output_dir / f'sample_tas_data_{i+1}.csv'
        data.to_csv(filename)
        sample_files.append(filename)
        print(f"   创建文件: {filename.name}")
    
    # 批量处理
    print("\n2. 批量预处理...")
    for i, file_path in enumerate(sample_files):
        print(f"   处理文件 {i+1}/3: {file_path.name}")
        
        # 加载数据
        data = pd.read_csv(file_path, index_col=0)
        
        # 预处理
        pipeline = create_standard_pipeline(verbose=False)
        processed_data = pipeline.fit_transform(data)
        
        # 保存结果
        processed_filename = output_dir / f'processed_{file_path.name}'
        processed_data.to_csv(processed_filename)
        
        # 保存处理摘要
        summary = pipeline.get_processing_summary()
        summary_filename = output_dir / f'summary_{file_path.stem}.json'
        
        import json
        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"      -> 保存处理结果: {processed_filename.name}")
        print(f"      -> 保存摘要: {summary_filename.name}")
    
    print(f"\n批量处理完成！结果保存在: {output_dir}")
    return output_dir


def example_advanced_features():
    """演示高级功能"""
    print("\n" + "=" * 60)
    print("演示高级功能")
    print("=" * 60)
    
    data = create_sample_tas_data()
    
    # 1. 多方法异常值检测
    print("1. 多方法异常值检测...")
    from preprocessing.outlier_detection import multi_method_outlier_detection
    
    outlier_mask = multi_method_outlier_detection(
        data, 
        methods=['z_score', 'iqr'],
        consensus_threshold=0.5,
        z_score_params={'threshold': 3.0},
        iqr_params={'factor': 1.5}
    )
    
    outlier_count = np.sum(outlier_mask)
    outlier_percentage = (outlier_count / outlier_mask.size) * 100
    print(f"   多方法一致检测异常值: {outlier_count} ({outlier_percentage:.2f}%)")
    
    # 2. 自适应平滑
    print("\n2. 自适应平滑...")
    smoother = DataSmoother(method='savgol')
    adaptive_smoothed = smoother.adaptive_smooth(data, noise_threshold=0.001)
    print("   自适应平滑完成")
    
    # 3. 渐进式处理
    print("\n3. 渐进式处理...")
    from preprocessing.noise_filtering import denoise_tas_data
    from preprocessing.data_smoother import progressive_smooth
    
    # 多方法降噪
    multi_denoised = denoise_tas_data(
        data, 
        methods=['gaussian', 'median'],
        gaussian_params={'sigma': 0.8},
        median_params={'size': 3}
    )
    
    # 渐进式平滑
    progressive_smoothed = progressive_smooth(
        multi_denoised,
        methods=['savgol', 'moving_average'],
        savgol_params={'window_length': 5, 'polyorder': 2},
        moving_average_params={'window_size': 3}
    )
    
    print("   渐进式处理完成")
    
    # 4. 详细统计分析
    print("\n4. 详细统计分析...")
    pipeline = create_standard_pipeline(verbose=False)
    processed_data = pipeline.fit_transform(data)
    
    summary = pipeline.get_processing_summary()
    
    print("   处理摘要:")
    print(f"      总步骤数: {summary['total_steps']}")
    print(f"      总处理时间: {summary['total_processing_time']:.3f}s")
    
    for step in summary['steps_summary']:
        print(f"      {step['name']}: {step['time']:.3f}s")
        
    if 'data_statistics' in summary:
        stats = summary['data_statistics']
        print(f"      信噪比改善: {stats['snr_improvement']:.2f}x")
        print(f"      噪声降低: {stats['noise_reduction']*100:.1f}%")
        print(f"      数据保真度: {stats['data_preservation']:.3f}")


def main():
    """主函数 - 运行所有示例"""
    print("TAS数据预处理模块示例演示")
    print("=" * 60)
    
    try:
        # 运行各个示例
        example_individual_modules()
        example_preprocessing_pipeline()
        example_advanced_features()
        
        # 批量处理示例
        output_dir = example_batch_processing()
        
        # 可视化示例 (最后运行，因为会显示图形)
        example_visualization()
        
        print("\n" + "=" * 60)
        print("所有示例演示完成！")
        print(f"示例输出文件保存在: {output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"示例运行中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
