#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TAS数据预处理模块

提供瞬态吸收光谱数据的预处理功能，包括：
- 基线校正
- 噪声过滤
- 数据平滑
- 异常值检测和处理
"""

try:
    from .baseline_correction import BaselineCorrector, correct_baseline
    from .noise_filtering import NoiseFilter, filter_noise, denoise_tas_data
    from .data_smoother import DataSmoother, smooth_data, progressive_smooth
    from .outlier_detection import OutlierDetector, detect_and_process_outliers, multi_method_outlier_detection
    from .spectral_cropper import SpectralCropper, crop_wavelengths
    from .chirp_correction import ChirpCorrector, correct_chirp
    from .pipeline import (TASPreprocessingPipeline, create_standard_pipeline, 
                          create_gentle_pipeline, create_aggressive_pipeline, 
                          create_chirp_corrected_pipeline, create_comprehensive_pipeline,
                          preprocess_tas_data)
    
    __all__ = [
        # 核心类
        'BaselineCorrector',
        'NoiseFilter', 
        'DataSmoother',
        'OutlierDetector',
    'SpectralCropper',
        'ChirpCorrector',
        'TASPreprocessingPipeline',
        
        # 便捷函数
        'correct_baseline',
        'filter_noise',
        'denoise_tas_data', 
        'smooth_data',
        'progressive_smooth',
        'detect_and_process_outliers',
        'multi_method_outlier_detection',
    'crop_wavelengths',
        'correct_chirp',
        'preprocess_tas_data',
        
        # 管道创建函数
        'create_standard_pipeline',
        'create_gentle_pipeline',
        'create_aggressive_pipeline',
        'create_chirp_corrected_pipeline',
        'create_comprehensive_pipeline'
    ]
    
except ImportError as e:
    print(f"警告：预处理模块导入失败: {e}")
    print("请确保已安装所需依赖: numpy, pandas, scipy, scikit-learn, matplotlib")
    
    # 提供基础导入
    __all__ = []

__version__ = '1.0.0'
