#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基线校正模块

提供多种基线校正方法用于TAS数据预处理
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from typing import Optional, Union, Tuple
import matplotlib.pyplot as plt


class BaselineCorrector:
    """TAS数据基线校正器"""
    
    def __init__(self, method='als', **kwargs):
        """
        初始化基线校正器
        
        Args:
            method: 基线校正方法 ('als', 'polynomial', 'rolling_ball', 'linear')
            **kwargs: 方法特定参数
        """
        self.method = method
        self.params = kwargs
        self.baseline = None
        self.corrected_data = None
        
    def fit_baseline(self, data: Union[np.ndarray, pd.DataFrame], 
                     wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
        """
        拟合基线
        
        Args:
            data: TAS数据矩阵 (delays × wavelengths)
            wavelengths: 波长数组
            
        Returns:
            基线数据
        """
        if isinstance(data, pd.DataFrame):
            data_array = data.values
            if wavelengths is None:
                wavelengths = data.columns.values
        else:
            data_array = data
            
        if self.method == 'als':
            self.baseline = self._als_baseline(data_array)
        elif self.method == 'polynomial':
            self.baseline = self._polynomial_baseline(data_array, wavelengths)
        elif self.method == 'rolling_ball':
            self.baseline = self._rolling_ball_baseline(data_array)
        elif self.method == 'linear':
            self.baseline = self._linear_baseline(data_array, wavelengths)
        else:
            raise ValueError(f"不支持的基线校正方法: {self.method}")
            
        return self.baseline
    
    def correct(self, data: Union[np.ndarray, pd.DataFrame], 
                wavelengths: Optional[np.ndarray] = None,
                fit_baseline: bool = True) -> Union[np.ndarray, pd.DataFrame]:
        """
        执行基线校正
        
        Args:
            data: 输入数据
            wavelengths: 波长数组
            fit_baseline: 是否重新拟合基线
            
        Returns:
            基线校正后的数据
        """
        is_dataframe = isinstance(data, pd.DataFrame)
        
        if fit_baseline or self.baseline is None:
            self.fit_baseline(data, wavelengths)
        
        if is_dataframe:
            self.corrected_data = data - pd.DataFrame(self.baseline, 
                                                     index=data.index, 
                                                     columns=data.columns)
        else:
            self.corrected_data = data - self.baseline
            
        return self.corrected_data
    
    def _als_baseline(self, data: np.ndarray, lam: float = None, p: float = None) -> np.ndarray:
        """
        渐近最小二乘法(ALS)基线校正
        
        Args:
            data: 输入数据
            lam: 平滑参数
            p: 不对称参数
            
        Returns:
            基线数据
        """
        # 设置默认参数
        if lam is None:
            lam = self.params.get('lam', 1e6)
        if p is None:
            p = self.params.get('p', 0.001)
            
        niter = self.params.get('niter', 10)
        
        baseline = np.zeros_like(data)
        
        for i in range(data.shape[0]):  # 对每个时间点
            y = data[i, :]
            L = len(y)
            D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
            w = np.ones(L)
            
            for _ in range(niter):
                W = sparse.spdiags(w, 0, L, L)
                Z = W + lam * D.dot(D.transpose())
                z = spsolve(Z, w*y)
                w = p * (y > z) + (1-p) * (y < z)
                
            baseline[i, :] = z
            
        return baseline
    
    def _polynomial_baseline(self, data: np.ndarray, 
                           wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
        """
        多项式基线校正
        
        Args:
            data: 输入数据
            wavelengths: 波长数组
            
        Returns:
            基线数据
        """
        degree = self.params.get('degree', 3)
        
        if wavelengths is None:
            wavelengths = np.arange(data.shape[1])
            
        baseline = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            # 使用数据的边缘点拟合多项式
            edge_points = self.params.get('edge_points', 50)
            
            # 选择边缘数据点
            left_idx = slice(0, edge_points)
            right_idx = slice(-edge_points, None)
            
            edge_wavelengths = np.concatenate([wavelengths[left_idx], wavelengths[right_idx]])
            edge_data = np.concatenate([data[i, left_idx], data[i, right_idx]])
            
            # 拟合多项式
            coeffs = np.polyfit(edge_wavelengths, edge_data, degree)
            baseline[i, :] = np.polyval(coeffs, wavelengths)
            
        return baseline
    
    def _rolling_ball_baseline(self, data: np.ndarray) -> np.ndarray:
        """
        滚球基线校正
        
        Args:
            data: 输入数据
            
        Returns:
            基线数据
        """
        radius = self.params.get('radius', 50)
        
        baseline = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            y = data[i, :]
            
            # 简化的滚球算法实现
            # 使用移动最小值近似
            from scipy.ndimage import minimum_filter
            baseline[i, :] = minimum_filter(y, size=radius, mode='reflect')
            
        return baseline
    
    def _linear_baseline(self, data: np.ndarray, 
                        wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
        """
        线性基线校正
        
        Args:
            data: 输入数据
            wavelengths: 波长数组
            
        Returns:
            基线数据
        """
        if wavelengths is None:
            wavelengths = np.arange(data.shape[1])
            
        baseline = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            y = data[i, :]
            
            # 使用首尾点进行线性拟合
            start_points = self.params.get('start_points', 10)
            end_points = self.params.get('end_points', 10)
            
            x1, y1 = wavelengths[:start_points].mean(), y[:start_points].mean()
            x2, y2 = wavelengths[-end_points:].mean(), y[-end_points:].mean()
            
            # 计算线性基线
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            
            baseline[i, :] = slope * wavelengths + intercept
            
        return baseline
    
    def plot_correction(self, data: Union[np.ndarray, pd.DataFrame], 
                       delay_index: int = 0, 
                       wavelengths: Optional[np.ndarray] = None,
                       figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        绘制基线校正结果
        
        Args:
            data: 原始数据
            delay_index: 要显示的延迟时间索引
            wavelengths: 波长数组
            figsize: 图像大小
            
        Returns:
            matplotlib图像对象
        """
        if isinstance(data, pd.DataFrame):
            wavelengths = data.columns.values if wavelengths is None else wavelengths
            original = data.iloc[delay_index].values
            delays = data.index.values
        else:
            wavelengths = np.arange(data.shape[1]) if wavelengths is None else wavelengths
            original = data[delay_index, :]
            delays = np.arange(data.shape[0])
            
        if self.baseline is None:
            self.fit_baseline(data, wavelengths)
            
        baseline = self.baseline[delay_index, :]
        corrected = original - baseline
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # 原始数据和基线
        axes[0].plot(wavelengths, original, 'b-', label='原始数据', alpha=0.7)
        axes[0].plot(wavelengths, baseline, 'r-', label='基线', linewidth=2)
        axes[0].set_xlabel('波长 (nm)')
        axes[0].set_ylabel('吸光度变化 (ΔOD)')
        axes[0].set_title(f'基线校正 - 延迟时间: {delays[delay_index]:.2f} ps')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 校正后数据
        axes[1].plot(wavelengths, corrected, 'g-', label='校正后数据', linewidth=2)
        axes[1].set_xlabel('波长 (nm)')
        axes[1].set_ylabel('吸光度变化 (ΔOD)')
        axes[1].set_title('基线校正后数据')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_correction_stats(self) -> dict:
        """
        获取基线校正统计信息
        
        Returns:
            统计信息字典
        """
        if self.baseline is None or self.corrected_data is None:
            return {}
            
        stats = {
            'method': self.method,
            'parameters': self.params,
            'baseline_mean': np.mean(self.baseline),
            'baseline_std': np.std(self.baseline),
            'baseline_range': [np.min(self.baseline), np.max(self.baseline)],
            'correction_improvement': {
                'original_std': np.std(self.corrected_data + self.baseline),
                'corrected_std': np.std(self.corrected_data),
                'noise_reduction': 1 - np.std(self.corrected_data) / np.std(self.corrected_data + self.baseline)
            }
        }
        
        return stats


# 便捷函数
def correct_baseline(data: Union[np.ndarray, pd.DataFrame], 
                    method: str = 'als',
                    wavelengths: Optional[np.ndarray] = None,
                    **kwargs) -> Union[np.ndarray, pd.DataFrame]:
    """
    便捷的基线校正函数
    
    Args:
        data: TAS数据
        method: 校正方法
        wavelengths: 波长数组
        **kwargs: 方法特定参数
        
    Returns:
        基线校正后的数据
    """
    corrector = BaselineCorrector(method=method, **kwargs)
    return corrector.correct(data, wavelengths)
