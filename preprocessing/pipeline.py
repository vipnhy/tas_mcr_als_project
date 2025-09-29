#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TAS数据预处理管道

整合所有预处理功能的一体化处理管道
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import logging
from datetime import datetime

from .baseline_correction import BaselineCorrector
from .noise_filtering import NoiseFilter
from .data_smoother import DataSmoother
from .outlier_detection import OutlierDetector
from .chirp_correction import ChirpCorrector
from .spectral_cropper import SpectralCropper


class TASPreprocessingPipeline:
    """TAS数据预处理管道"""
    
    def __init__(self, steps: Optional[List[Dict[str, Any]]] = None, verbose: bool = True):
        """
        初始化预处理管道
        
        Args:
            steps: 预处理步骤列表，每个步骤包含 {'name': str, 'processor': str, 'params': dict}
            verbose: 是否显示详细信息
            
        示例:
            steps = [
                {'name': 'outlier_detection', 'processor': 'outlier', 'params': {'method': 'z_score', 'threshold': 3.0}},
                {'name': 'baseline_correction', 'processor': 'baseline', 'params': {'method': 'als', 'lam': 1e6}},
                {'name': 'noise_filtering', 'processor': 'noise', 'params': {'method': 'gaussian', 'sigma': 1.0}},
                {'name': 'smoothing', 'processor': 'smooth', 'params': {'method': 'savgol', 'window_length': 5}}
            ]
        """
        if steps is None:
            # 默认预处理步骤
            steps = [
                {'name': 'outlier_detection', 'processor': 'outlier', 
                 'params': {'method': 'z_score', 'threshold': 3.0}},
                {'name': 'baseline_correction', 'processor': 'baseline', 
                 'params': {'method': 'als', 'lam': 1e6, 'p': 0.001}},
                {'name': 'noise_filtering', 'processor': 'noise', 
                 'params': {'method': 'gaussian', 'sigma': 1.0}},
                {'name': 'data_smoothing', 'processor': 'smooth', 
                 'params': {'method': 'savgol', 'window_length': 5, 'polyorder': 2}}
            ]
        
        self.steps = steps
        self.verbose = verbose
        self.processors = {}
        self.processing_history = []
        self.original_data = None
        self.processed_data = None
        
        # 设置日志
        if verbose:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 初始化处理器
        self._initialize_processors()
    
    def _initialize_processors(self):
        """初始化处理器"""
        for step in self.steps:
            processor_type = step['processor']
            params = step.get('params', {})
            
            if processor_type == 'outlier':
                self.processors[step['name']] = OutlierDetector(**params)
            elif processor_type == 'baseline':
                self.processors[step['name']] = BaselineCorrector(**params)
            elif processor_type == 'noise':
                self.processors[step['name']] = NoiseFilter(**params)
            elif processor_type == 'smooth':
                self.processors[step['name']] = DataSmoother(**params)
            elif processor_type == 'chirp':
                self.processors[step['name']] = ChirpCorrector(**params)
            elif processor_type in {'spectral_crop', 'crop'}:
                self.processors[step['name']] = SpectralCropper(**params)
            else:
                raise ValueError(f"不支持的处理器类型: {processor_type}")
    
    def fit(self, data: Union[np.ndarray, pd.DataFrame], 
            wavelengths: Optional[np.ndarray] = None) -> 'TASPreprocessingPipeline':
        """
        拟合预处理管道
        
        Args:
            data: TAS数据
            wavelengths: 波长数组
            
        Returns:
            自身实例
        """
        self.original_data = data.copy() if hasattr(data, 'copy') else data
        self.wavelengths = wavelengths
        
        if self.verbose:
            self.logger.info("开始预处理管道拟合...")
        
        # 预处理步骤不需要特殊的拟合过程，直接返回
        return self
    
    def transform(self, data: Union[np.ndarray, pd.DataFrame], 
                  wavelengths: Optional[np.ndarray] = None) -> Union[np.ndarray, pd.DataFrame]:
        """
        执行预处理变换
        
        Args:
            data: 输入数据
            wavelengths: 波长数组
            
        Returns:
            预处理后的数据
        """
        current_data = data
        processing_stats = []

        if isinstance(current_data, pd.DataFrame):
            current_time_axis = current_data.index.values.astype(float)
            current_wavelengths = current_data.columns.values.astype(float)
        else:
            current_time_axis = np.arange(current_data.shape[0], dtype=float)
            if wavelengths is not None:
                current_wavelengths = np.asarray(wavelengths, dtype=float)
            else:
                current_wavelengths = np.arange(current_data.shape[1], dtype=float)
        
        if self.verbose:
            self.logger.info(f"开始执行 {len(self.steps)} 个预处理步骤...")
        
        for i, step in enumerate(self.steps):
            step_name = step['name']
            processor = self.processors[step_name]
            
            if self.verbose:
                self.logger.info(f"步骤 {i+1}/{len(self.steps)}: {step_name}")
            
            start_time = datetime.now()
            
            try:
                # 执行处理步骤
                if isinstance(processor, OutlierDetector):
                    processor.detect_outliers(current_data)
                    current_data = processor.process_outliers(current_data, 
                                                            strategy=step.get('strategy', 'interpolate'))
                    step_stats = processor.get_outlier_statistics()
                elif isinstance(processor, SpectralCropper):
                    current_data, current_wavelengths = processor.crop(current_data, current_wavelengths)
                    step_stats = processor.get_crop_stats()
                    
                elif isinstance(processor, BaselineCorrector):
                    current_data = processor.correct(current_data, current_wavelengths)
                    step_stats = processor.get_correction_stats()
                    
                elif isinstance(processor, NoiseFilter):
                    current_data = processor.filter_noise(current_data)
                    step_stats = processor.get_filtering_stats()
                    
                elif isinstance(processor, DataSmoother):
                    current_data = processor.smooth(current_data)
                    step_stats = processor.get_smoothing_stats()
                
                elif isinstance(processor, ChirpCorrector):
                    time_delays = current_time_axis
                    current_data = processor.correct_chirp(current_data, time_delays)
                    step_stats = processor.get_correction_stats()
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                # 记录处理历史
                history_entry = {
                    'step_name': step_name,
                    'processor_type': step['processor'],
                    'parameters': step.get('params', {}),
                    'processing_time': processing_time,
                    'stats': step_stats
                }
                processing_stats.append(history_entry)
                
                if self.verbose:
                    self.logger.info(f"  完成 {step_name} (耗时: {processing_time:.2f}s)")
                
            except Exception as e:
                self.logger.error(f"步骤 {step_name} 执行失败: {str(e)}")
                raise
        
        self.processed_data = current_data
        self.wavelengths = current_wavelengths
        self.time_axis = current_time_axis
        self.processing_history = processing_stats
        
        if self.verbose:
            self.logger.info("预处理管道执行完成!")
        
        return current_data
    
    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame], 
                     wavelengths: Optional[np.ndarray] = None) -> Union[np.ndarray, pd.DataFrame]:
        """
        拟合并执行预处理变换
        
        Args:
            data: 输入数据
            wavelengths: 波长数组
            
        Returns:
            预处理后的数据
        """
        return self.fit(data, wavelengths).transform(data, wavelengths)
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """
        获取预处理摘要
        
        Returns:
            预处理摘要字典
        """
        if not self.processing_history:
            return {}
        
        total_time = sum(step['processing_time'] for step in self.processing_history)
        
        summary = {
            'total_steps': len(self.processing_history),
            'total_processing_time': total_time,
            'steps_summary': [],
            'data_statistics': {}
        }
        
        for step in self.processing_history:
            step_summary = {
                'name': step['step_name'],
                'type': step['processor_type'],
                'time': step['processing_time'],
                'key_stats': self._extract_key_stats(step['stats'])
            }
            summary['steps_summary'].append(step_summary)
        
        # 数据统计
        if self.original_data is not None and self.processed_data is not None:
            summary['data_statistics'] = self._calculate_data_improvement()
        
        return summary
    
    def _extract_key_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """提取关键统计信息"""
        key_stats = {}
        
        if 'outlier_count' in stats:
            key_stats['outliers_detected'] = stats['outlier_count']
            key_stats['outlier_percentage'] = stats['outlier_percentage']
        
        if 'noise_reduction' in stats:
            key_stats['noise_reduction_ratio'] = stats['noise_reduction'].get('noise_reduction_ratio', 0)
        
        if 'correction_improvement' in stats:
            key_stats['noise_reduction_baseline'] = stats['correction_improvement'].get('noise_reduction', 0)
        
        return key_stats
    
    def _calculate_data_improvement(self) -> Dict[str, Any]:
        """计算数据改进指标"""
        if isinstance(self.original_data, pd.DataFrame) and isinstance(self.processed_data, pd.DataFrame):
            aligned_orig, aligned_proc = self.original_data.align(
                self.processed_data, join='inner', axis=None
            )

            if aligned_orig.size and aligned_proc.size:
                orig_array = aligned_orig.values
                proc_array = aligned_proc.values
            else:
                orig_array = self.original_data.values
                proc_array = self.processed_data.values
        else:
            orig_array = np.asarray(self.original_data)
            proc_array = np.asarray(self.processed_data)

        orig_array = np.asarray(orig_array, dtype=float)
        proc_array = np.asarray(proc_array, dtype=float)

        if orig_array.size == 0 or proc_array.size == 0:
            return {
                'original_std': np.nan,
                'processed_std': np.nan,
                'original_snr': np.nan,
                'processed_snr': np.nan,
                'snr_improvement': 1.0,
                'noise_reduction': 0.0,
                'data_preservation': np.nan
            }

        if orig_array.shape != proc_array.shape:
            if orig_array.ndim >= 2 and proc_array.ndim >= 2:
                min_rows = min(orig_array.shape[0], proc_array.shape[0])
                min_cols = min(orig_array.shape[1], proc_array.shape[1])
                if min_rows == 0 or min_cols == 0:
                    return {
                        'original_std': np.nan,
                        'processed_std': np.nan,
                        'original_snr': np.nan,
                        'processed_snr': np.nan,
                        'snr_improvement': 1.0,
                        'noise_reduction': 0.0,
                        'data_preservation': np.nan
                    }
                orig_array = orig_array[:min_rows, :min_cols]
                proc_array = proc_array[:min_rows, :min_cols]
            else:
                min_len = min(orig_array.size, proc_array.size)
                if min_len == 0:
                    return {
                        'original_std': np.nan,
                        'processed_std': np.nan,
                        'original_snr': np.nan,
                        'processed_snr': np.nan,
                        'snr_improvement': 1.0,
                        'noise_reduction': 0.0,
                        'data_preservation': np.nan
                    }
                orig_array = orig_array.flatten()[:min_len]
                proc_array = proc_array.flatten()[:min_len]
        
        # 计算信噪比改善
        orig_std = np.std(orig_array)
        proc_std = np.std(proc_array)
        
        # 估算噪声水平 (使用高频成分)
        if len(orig_array.shape) == 2:
            orig_noise = np.std(np.diff(orig_array, n=2, axis=1)) / np.sqrt(6)
            proc_noise = np.std(np.diff(proc_array, n=2, axis=1)) / np.sqrt(6)
        else:
            orig_noise = np.std(np.diff(orig_array, n=2)) / np.sqrt(6)
            proc_noise = np.std(np.diff(proc_array, n=2)) / np.sqrt(6)
        
        orig_snr = orig_std / orig_noise if orig_noise > 0 else float('inf')
        proc_snr = proc_std / proc_noise if proc_noise > 0 else float('inf')
        
        orig_flat = orig_array.flatten()
        proc_flat = proc_array.flatten()
        finite_mask = np.isfinite(orig_flat) & np.isfinite(proc_flat)

        if finite_mask.sum() > 1:
            correlation = np.corrcoef(orig_flat[finite_mask], proc_flat[finite_mask])[0, 1]
        else:
            correlation = np.nan

        improvement = {
            'original_std': orig_std,
            'processed_std': proc_std,
            'original_snr': orig_snr,
            'processed_snr': proc_snr,
            'snr_improvement': proc_snr / orig_snr if orig_snr > 0 else 1.0,
            'noise_reduction': 1 - (proc_noise / orig_noise) if orig_noise > 0 else 0.0,
            'data_preservation': correlation
        }
        
        return improvement
    
    def plot_processing_pipeline(self, delay_index: int = 0,
                               wavelengths: Optional[np.ndarray] = None,
                               figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        绘制预处理管道的每个步骤结果
        
        Args:
            delay_index: 要显示的延迟时间索引
            wavelengths: 波长数组
            figsize: 图像大小
            
        Returns:
            matplotlib图像对象
        """
        if self.original_data is None or self.processed_data is None:
            raise ValueError("请先执行预处理管道")
        
        # 准备数据
        if isinstance(self.original_data, pd.DataFrame):
            wavelengths = self.original_data.columns.values if wavelengths is None else wavelengths
            delays = self.original_data.index.values
        else:
            wavelengths = np.arange(self.original_data.shape[1]) if wavelengths is None else wavelengths
            delays = np.arange(self.original_data.shape[0])
        
        # 重新执行管道以获得中间结果
        current_data = self.original_data
        intermediate_results = [current_data]
        step_names = ['原始数据']
        
        for step in self.steps:
            processor = self.processors[step['name']]
            
            if isinstance(processor, OutlierDetector):
                processor.detect_outliers(current_data)
                current_data = processor.process_outliers(current_data, 
                                                        strategy=step.get('strategy', 'interpolate'))
            elif isinstance(processor, BaselineCorrector):
                current_data = processor.correct(current_data, wavelengths)
            elif isinstance(processor, NoiseFilter):
                current_data = processor.filter_noise(current_data)
            elif isinstance(processor, DataSmoother):
                current_data = processor.smooth(current_data)
            
            intermediate_results.append(current_data)
            step_names.append(step['name'])
        
        # 绘制结果
        n_steps = len(intermediate_results)
        fig, axes = plt.subplots(n_steps, 1, figsize=figsize)
        
        if n_steps == 1:
            axes = [axes]
        
        for i, (data, name) in enumerate(zip(intermediate_results, step_names)):
            if isinstance(data, pd.DataFrame):
                y_data = data.iloc[delay_index].values
            else:
                y_data = data[delay_index, :]
            
            axes[i].plot(wavelengths, y_data, linewidth=2)
            axes[i].set_title(f'{name} - 延迟时间: {delays[delay_index]:.2f} ps')
            axes[i].set_ylabel('ΔOD')
            axes[i].grid(True, alpha=0.3)
            
            if i == n_steps - 1:  # 最后一个图添加x轴标签
                axes[i].set_xlabel('波长 (nm)')
        
        plt.tight_layout()
        return fig
    
    def plot_comparison(self, delay_index: int = 0,
                       wavelengths: Optional[np.ndarray] = None,
                       figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        绘制处理前后对比图
        
        Args:
            delay_index: 要显示的延迟时间索引
            wavelengths: 波长数组
            figsize: 图像大小
            
        Returns:
            matplotlib图像对象
        """
        if self.original_data is None or self.processed_data is None:
            raise ValueError("请先执行预处理管道")
        
        if isinstance(self.original_data, pd.DataFrame):
            wavelengths = self.original_data.columns.values if wavelengths is None else wavelengths
            original = self.original_data.iloc[delay_index].values
            processed = self.processed_data.iloc[delay_index].values
            delays = self.original_data.index.values
        else:
            wavelengths = np.arange(self.original_data.shape[1]) if wavelengths is None else wavelengths
            original = self.original_data[delay_index, :]
            processed = self.processed_data[delay_index, :]
            delays = np.arange(self.original_data.shape[0])
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # 对比图
        axes[0].plot(wavelengths, original, 'b-', label='原始数据', alpha=0.6, linewidth=1)
        axes[0].plot(wavelengths, processed, 'r-', label='预处理后数据', linewidth=2)
        axes[0].set_xlabel('波长 (nm)')
        axes[0].set_ylabel('吸光度变化 (ΔOD)')
        axes[0].set_title(f'预处理对比 - 延迟时间: {delays[delay_index]:.2f} ps')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 差异图
        difference = original - processed
        axes[1].plot(wavelengths, difference, 'g-', label='差异 (原始 - 预处理)', alpha=0.7)
        axes[1].set_xlabel('波长 (nm)')
        axes[1].set_ylabel('吸光度变化 (ΔOD)')
        axes[1].set_title('预处理前后差异')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_results(self, output_dir: str, filename_prefix: str = 'preprocessed') -> Dict[str, str]:
        """
        保存预处理结果
        
        Args:
            output_dir: 输出目录
            filename_prefix: 文件名前缀
            
        Returns:
            保存的文件路径字典
        """
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        
        # 保存预处理后的数据
        if self.processed_data is not None:
            if isinstance(self.processed_data, pd.DataFrame):
                data_file = os.path.join(output_dir, f'{filename_prefix}_data.csv')
                self.processed_data.to_csv(data_file)
            else:
                data_file = os.path.join(output_dir, f'{filename_prefix}_data.npy')
                np.save(data_file, self.processed_data)
            saved_files['data'] = data_file
        
        # 保存处理摘要
        summary = self.get_processing_summary()
        summary_file = os.path.join(output_dir, f'{filename_prefix}_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        saved_files['summary'] = summary_file
        
        return saved_files


# 预设的预处理管道
def create_standard_pipeline(verbose: bool = True, crop_params: Optional[Dict[str, Any]] = None) -> TASPreprocessingPipeline:
    """
    创建标准TAS数据预处理管道
    
    Args:
        verbose: 是否显示详细信息
        
    Returns:
        预处理管道实例
    """
    crop_config = {
        'auto_detect': True,
        'noise_threshold': 5e-4,
        'relative_threshold': 0.02,
        'margin': 1,
        'min_valid_span': 10,
        'verbose': verbose
    }
    if crop_params:
        crop_config.update(crop_params)

    steps = [
        {'name': 'spectral_crop', 'processor': 'spectral_crop', 'params': crop_config},
        {'name': 'outlier_detection', 'processor': 'outlier', 
         'params': {'method': 'z_score', 'threshold': 3.0}, 'strategy': 'interpolate'},
        {'name': 'baseline_correction', 'processor': 'baseline', 
         'params': {'method': 'als', 'lam': 1e6, 'p': 0.001}},
        {'name': 'noise_filtering', 'processor': 'noise', 
         'params': {'method': 'gaussian', 'sigma': 1.0}},
        {'name': 'data_smoothing', 'processor': 'smooth', 
         'params': {'method': 'savgol', 'window_length': 5, 'polyorder': 2}}
    ]
    
    return TASPreprocessingPipeline(steps=steps, verbose=verbose)


def create_gentle_pipeline(verbose: bool = True, crop_params: Optional[Dict[str, Any]] = None) -> TASPreprocessingPipeline:
    """
    创建温和的预处理管道 (较少改变原始数据)
    
    Args:
        verbose: 是否显示详细信息
        
    Returns:
        预处理管道实例
    """
    crop_config = {
        'auto_detect': True,
        'noise_threshold': 7.5e-4,
        'relative_threshold': 0.03,
        'margin': 0,
        'min_valid_span': 8,
        'verbose': verbose
    }
    if crop_params:
        crop_config.update(crop_params)

    steps = [
        {'name': 'spectral_crop', 'processor': 'spectral_crop', 'params': crop_config},
        {'name': 'outlier_detection', 'processor': 'outlier', 
         'params': {'method': 'iqr', 'factor': 2.0}, 'strategy': 'interpolate'},
        {'name': 'baseline_correction', 'processor': 'baseline', 
         'params': {'method': 'polynomial', 'degree': 2}},
        {'name': 'noise_filtering', 'processor': 'noise', 
         'params': {'method': 'gaussian', 'sigma': 0.5}}
    ]
    
    return TASPreprocessingPipeline(steps=steps, verbose=verbose)


def create_aggressive_pipeline(verbose: bool = True, crop_params: Optional[Dict[str, Any]] = None) -> TASPreprocessingPipeline:
    """
    创建激进的预处理管道 (强力去噪和平滑)
    
    Args:
        verbose: 是否显示详细信息
        
    Returns:
        预处理管道实例
    """
    crop_config = {
        'auto_detect': True,
        'noise_threshold': 4e-4,
        'relative_threshold': 0.015,
        'margin': 2,
        'min_valid_span': 12,
        'verbose': verbose
    }
    if crop_params:
        crop_config.update(crop_params)

    steps = [
        {'name': 'spectral_crop', 'processor': 'spectral_crop', 'params': crop_config},
        {'name': 'outlier_detection', 'processor': 'outlier', 
         'params': {'method': 'isolation_forest', 'contamination': 0.1}, 'strategy': 'interpolate'},
        {'name': 'baseline_correction', 'processor': 'baseline', 
         'params': {'method': 'als', 'lam': 1e8, 'p': 0.0001}},
        {'name': 'noise_filtering', 'processor': 'noise', 
         'params': {'method': 'bilateral', 'd': 9, 'sigma_color': 75, 'sigma_space': 75}},
        {'name': 'data_smoothing', 'processor': 'smooth', 
         'params': {'method': 'savgol', 'window_length': 9, 'polyorder': 3}}
    ]
    
    return TASPreprocessingPipeline(steps=steps, verbose=verbose)


def create_chirp_corrected_pipeline(
    chirp_method: str = 'cross_correlation',
    verbose: bool = True,
    crop_params: Optional[Dict[str, Any]] = None
) -> TASPreprocessingPipeline:
    """
    创建包含啁啾校正的TAS数据预处理管道
    
    Args:
        chirp_method: 啁啾校正方法 ('cross_correlation', 'solvent_response', 'polynomial')
        verbose: 是否显示详细信息
        
    Returns:
        预处理管道实例
    """
    crop_config = {
        'auto_detect': True,
        'noise_threshold': 5e-4,
        'relative_threshold': 0.02,
        'margin': 1,
        'min_valid_span': 10,
        'verbose': verbose
    }
    if crop_params:
        crop_config.update(crop_params)

    steps = [
        {'name': 'spectral_crop', 'processor': 'spectral_crop', 'params': crop_config},
        {'name': 'chirp_correction', 'processor': 'chirp', 
         'params': {'method': chirp_method}},
        {'name': 'outlier_detection', 'processor': 'outlier', 
         'params': {'method': 'z_score', 'threshold': 3.0}, 'strategy': 'interpolate'},
        {'name': 'baseline_correction', 'processor': 'baseline', 
         'params': {'method': 'als', 'lam': 1e6, 'p': 0.001}},
        {'name': 'noise_filtering', 'processor': 'noise', 
         'params': {'method': 'gaussian', 'sigma': 1.0}},
        {'name': 'data_smoothing', 'processor': 'smooth', 
         'params': {'method': 'savgol', 'window_length': 5, 'polyorder': 2}}
    ]
    
    return TASPreprocessingPipeline(steps=steps, verbose=verbose)


def create_comprehensive_pipeline(
    chirp_method: str = 'cross_correlation',
    verbose: bool = True,
    crop_params: Optional[Dict[str, Any]] = None
) -> TASPreprocessingPipeline:
    """
    创建完整的TAS数据预处理管道，包含啁啾校正和所有预处理步骤
    
    Args:
        chirp_method: 啁啾校正方法 ('cross_correlation', 'solvent_response', 'polynomial')
        verbose: 是否显示详细信息
        
    Returns:
        预处理管道实例
    """
    crop_config = {
        'auto_detect': True,
        'noise_threshold': 5e-4,
        'relative_threshold': 0.02,
        'margin': 1,
        'min_valid_span': 12,
        'verbose': verbose
    }
    if crop_params:
        crop_config.update(crop_params)

    steps = [
        # 裁剪掉噪声主导的波长
        {'name': 'spectral_crop', 'processor': 'spectral_crop', 'params': crop_config},
        # 然后进行啁啾校正
        {'name': 'chirp_correction', 'processor': 'chirp', 
         'params': {'method': chirp_method, 'solvent_wavelengths': [400, 450], 'polynomial_order': 3}},
        # 然后进行异常值检测和处理
        {'name': 'outlier_detection', 'processor': 'outlier', 
         'params': {'method': 'z_score', 'threshold': 3.0}, 'strategy': 'interpolate'},
        # 基线校正
        {'name': 'baseline_correction', 'processor': 'baseline', 
         'params': {'method': 'als', 'lam': 1e6, 'p': 0.001}},
        # 噪声过滤
        {'name': 'noise_filtering', 'processor': 'noise', 
         'params': {'method': 'gaussian', 'sigma': 1.0}},
        # 数据平滑
        {'name': 'data_smoothing', 'processor': 'smooth', 
         'params': {'method': 'savgol', 'window_length': 5, 'polyorder': 2}}
    ]
    
    return TASPreprocessingPipeline(steps=steps, verbose=verbose)


# 便捷函数
def preprocess_tas_data(data: Union[np.ndarray, pd.DataFrame], 
                       pipeline: str = 'standard',
                       wavelengths: Optional[np.ndarray] = None,
                       verbose: bool = True,
                       **kwargs) -> Union[np.ndarray, pd.DataFrame]:
    """
    便捷的TAS数据预处理函数
    
    Args:
        data: TAS数据
        pipeline: 预处理管道类型 ('standard', 'gentle', 'aggressive', 'chirp_corrected', 'comprehensive', 'custom')
        wavelengths: 波长数组
        verbose: 是否显示详细信息
        **kwargs: 自定义管道参数 (当pipeline='custom'时使用) 或啁啾校正方法参数
        
    Returns:
        预处理后的数据
    """
    crop_params = kwargs.get('crop_params')

    if pipeline == 'standard':
        processor = create_standard_pipeline(verbose, crop_params=crop_params)
    elif pipeline == 'gentle':
        processor = create_gentle_pipeline(verbose, crop_params=crop_params)
    elif pipeline == 'aggressive':
        processor = create_aggressive_pipeline(verbose, crop_params=crop_params)
    elif pipeline == 'chirp_corrected':
        chirp_method = kwargs.get('chirp_method', 'cross_correlation')
        processor = create_chirp_corrected_pipeline(chirp_method, verbose, crop_params=crop_params)
    elif pipeline == 'comprehensive':
        chirp_method = kwargs.get('chirp_method', 'cross_correlation')
        processor = create_comprehensive_pipeline(chirp_method, verbose, crop_params=crop_params)
    elif pipeline == 'custom':
        steps = kwargs.get('steps')
        if steps is None:
            raise ValueError("自定义管道需要提供 'steps' 参数")
        processor = TASPreprocessingPipeline(steps=steps, verbose=verbose)
    else:
        raise ValueError(f"不支持的管道类型: {pipeline}")
    
    return processor.fit_transform(data, wavelengths)
