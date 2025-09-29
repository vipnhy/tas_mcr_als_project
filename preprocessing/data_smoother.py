#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据平滑模块

提供多种数据平滑方法用于TAS数据预处理
"""

import numpy as np
import pandas as pd
from scipy import signal, interpolate
from scipy.signal import savgol_filter, butter, filtfilt
from typing import Optional, Union, Tuple
import matplotlib.pyplot as plt


class DataSmoother:
    """TAS数据平滑器"""
    
    def __init__(self, method='savgol', **kwargs):
        """
        初始化数据平滑器
        
        Args:
            method: 平滑方法 ('savgol', 'moving_average', 'lowess', 'spline', 'butterworth')
            **kwargs: 方法特定参数
        """
        self.method = method
        self.params = kwargs
        self.smoothed_data = None
        
    def smooth(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        执行数据平滑
        
        Args:
            data: 输入TAS数据
            
        Returns:
            平滑后的数据
        """
        is_dataframe = isinstance(data, pd.DataFrame)
        
        if is_dataframe:
            data_array = data.values
        else:
            data_array = data
            
        if self.method == 'savgol':
            smoothed_array = self._savgol_smooth(data_array)
        elif self.method == 'moving_average':
            smoothed_array = self._moving_average_smooth(data_array)
        elif self.method == 'lowess':
            smoothed_array = self._lowess_smooth(data_array)
        elif self.method == 'spline':
            smoothed_array = self._spline_smooth(data_array)
        elif self.method == 'butterworth':
            smoothed_array = self._butterworth_smooth(data_array)
        else:
            raise ValueError(f"不支持的平滑方法: {self.method}")
        
        if is_dataframe:
            self.smoothed_data = pd.DataFrame(smoothed_array, 
                                            index=data.index, 
                                            columns=data.columns)
        else:
            self.smoothed_data = smoothed_array
            
        return self.smoothed_data
    
    def _savgol_smooth(self, data: np.ndarray) -> np.ndarray:
        """
        Savitzky-Golay平滑
        
        Args:
            data: 输入数据
            
        Returns:
            平滑后数据
        """
        window_length = self.params.get('window_length', 5)
        polyorder = self.params.get('polyorder', 2)
        axis = self.params.get('axis', -1)
        
        # 确保窗口长度为奇数且大于polyorder
        if window_length % 2 == 0:
            window_length += 1
        if window_length <= polyorder:
            window_length = polyorder + 2
            
        return savgol_filter(data, window_length, polyorder, axis=axis)
    
    def _moving_average_smooth(self, data: np.ndarray) -> np.ndarray:
        """
        移动平均平滑
        
        Args:
            data: 输入数据
            
        Returns:
            平滑后数据
        """
        window_size = self.params.get('window_size', 5)
        mode = self.params.get('mode', 'same')
        
        # 创建均匀权重核
        kernel = np.ones(window_size) / window_size
        
        smoothed = np.zeros_like(data)
        
        if len(data.shape) == 2:
            axis = self.params.get('axis', 1)  # 默认沿波长轴平滑
            
            if axis == 1:  # 沿波长轴
                for i in range(data.shape[0]):
                    smoothed[i, :] = signal.convolve(data[i, :], kernel, mode=mode)
            else:  # 沿时间轴
                for j in range(data.shape[1]):
                    smoothed[:, j] = signal.convolve(data[:, j], kernel, mode=mode)
        else:
            smoothed = signal.convolve(data, kernel, mode=mode)
            
        return smoothed
    
    def _lowess_smooth(self, data: np.ndarray) -> np.ndarray:
        """
        LOWESS (Locally Weighted Scatterplot Smoothing) 平滑
        
        Args:
            data: 输入数据
            
        Returns:
            平滑后数据
        """
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
        except ImportError:
            raise ImportError("LOWESS平滑需要安装statsmodels: pip install statsmodels")
        
        frac = self.params.get('frac', 0.1)  # 局部回归点的比例
        it = self.params.get('it', 3)  # 迭代次数
        
        smoothed = np.zeros_like(data)
        
        if len(data.shape) == 2:
            for i in range(data.shape[0]):
                x = np.arange(data.shape[1])
                smoothed_result = lowess(data[i, :], x, frac=frac, it=it)
                smoothed[i, :] = smoothed_result[:, 1]
        else:
            x = np.arange(len(data))
            smoothed_result = lowess(data, x, frac=frac, it=it)
            smoothed = smoothed_result[:, 1]
            
        return smoothed
    
    def _spline_smooth(self, data: np.ndarray) -> np.ndarray:
        """
        样条插值平滑
        
        Args:
            data: 输入数据
            
        Returns:
            平滑后数据
        """
        smoothing_factor = self.params.get('s', None)  # 平滑因子
        degree = self.params.get('k', 3)  # 样条度数
        
        smoothed = np.zeros_like(data)
        
        if len(data.shape) == 2:
            for i in range(data.shape[0]):
                x = np.arange(data.shape[1])
                # 去除NaN值
                mask = ~np.isnan(data[i, :])
                if np.sum(mask) > degree + 1:
                    spline = interpolate.UnivariateSpline(x[mask], data[i, mask][mask], 
                                                        s=smoothing_factor, k=degree)
                    smoothed[i, :] = spline(x)
                else:
                    smoothed[i, :] = data[i, :]
        else:
            x = np.arange(len(data))
            mask = ~np.isnan(data)
            if np.sum(mask) > degree + 1:
                spline = interpolate.UnivariateSpline(x[mask], data[mask], 
                                                    s=smoothing_factor, k=degree)
                smoothed = spline(x)
            else:
                smoothed = data
                
        return smoothed
    
    def _butterworth_smooth(self, data: np.ndarray) -> np.ndarray:
        """
        Butterworth低通滤波平滑
        
        Args:
            data: 输入数据
            
        Returns:
            平滑后数据
        """
        cutoff = self.params.get('cutoff', 0.1)  # 截止频率（相对于奈奎斯特频率）
        order = self.params.get('order', 5)  # 滤波器阶数
        
        # 设计Butterworth滤波器
        b, a = butter(order, cutoff, btype='low', analog=False)
        
        smoothed = np.zeros_like(data)
        
        if len(data.shape) == 2:
            axis = self.params.get('axis', 1)  # 默认沿波长轴平滑
            
            if axis == 1:  # 沿波长轴
                for i in range(data.shape[0]):
                    smoothed[i, :] = filtfilt(b, a, data[i, :])
            else:  # 沿时间轴
                for j in range(data.shape[1]):
                    smoothed[:, j] = filtfilt(b, a, data[:, j])
        else:
            smoothed = filtfilt(b, a, data)
            
        return smoothed
    
    def adaptive_smooth(self, data: Union[np.ndarray, pd.DataFrame],
                       noise_threshold: float = 0.01) -> Union[np.ndarray, pd.DataFrame]:
        """
        自适应平滑 - 根据局部噪声水平调整平滑强度
        
        Args:
            data: 输入数据
            noise_threshold: 噪声阈值
            
        Returns:
            自适应平滑后的数据
        """
        is_dataframe = isinstance(data, pd.DataFrame)
        
        if is_dataframe:
            data_array = data.values
        else:
            data_array = data
            
        # 估计局部噪声水平
        noise_map = self._estimate_local_noise(data_array)
        
        # 根据噪声水平选择平滑参数
        smoothed = np.zeros_like(data_array)
        
        if len(data_array.shape) == 2:
            for i in range(data_array.shape[0]):
                local_noise = noise_map[i]
                
                if local_noise > noise_threshold:
                    # 高噪声区域：强平滑
                    window_length = max(7, int(local_noise / noise_threshold * 5))
                    if window_length % 2 == 0:
                        window_length += 1
                    smoothed[i, :] = savgol_filter(data_array[i, :], window_length, 2)
                else:
                    # 低噪声区域：轻微平滑或不平滑
                    smoothed[i, :] = savgol_filter(data_array[i, :], 3, 1)
        else:
            # 1D数据的自适应平滑
            window_length = max(5, int(np.std(data_array) / noise_threshold))
            if window_length % 2 == 0:
                window_length += 1
            smoothed = savgol_filter(data_array, min(window_length, len(data_array)//2), 2)
        
        if is_dataframe:
            self.smoothed_data = pd.DataFrame(smoothed, 
                                            index=data.index, 
                                            columns=data.columns)
        else:
            self.smoothed_data = smoothed
            
        return self.smoothed_data
    
    def _estimate_local_noise(self, data: np.ndarray, window_size: int = 10) -> np.ndarray:
        """
        估计局部噪声水平
        
        Args:
            data: 输入数据
            window_size: 估计窗口大小
            
        Returns:
            噪声水平数组
        """
        if len(data.shape) == 2:
            noise_map = np.zeros(data.shape[0])
            
            for i in range(data.shape[0]):
                # 使用二阶差分估计噪声
                diff2 = np.diff(data[i, :], n=2)
                
                # 滑动窗口计算局部标准差
                local_stds = []
                for j in range(0, len(diff2), window_size):
                    window_data = diff2[j:j+window_size]
                    if len(window_data) > 0:
                        local_stds.append(np.std(window_data))
                
                noise_map[i] = np.mean(local_stds) / np.sqrt(6)
        else:
            diff2 = np.diff(data, n=2)
            noise_map = np.std(diff2) / np.sqrt(6)
            
        return noise_map
    
    def plot_smoothing_result(self, original_data: Union[np.ndarray, pd.DataFrame],
                            delay_index: int = 0,
                            wavelengths: Optional[np.ndarray] = None,
                            figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        绘制数据平滑结果
        
        Args:
            original_data: 原始数据
            delay_index: 要显示的延迟时间索引
            wavelengths: 波长数组
            figsize: 图像大小
            
        Returns:
            matplotlib图像对象
        """
        if isinstance(original_data, pd.DataFrame):
            wavelengths = original_data.columns.values if wavelengths is None else wavelengths
            original = original_data.iloc[delay_index].values
            delays = original_data.index.values
        else:
            wavelengths = np.arange(original_data.shape[1]) if wavelengths is None else wavelengths
            original = original_data[delay_index, :]
            delays = np.arange(original_data.shape[0])
        
        if self.smoothed_data is None:
            raise ValueError("请先执行数据平滑")
        
        if isinstance(self.smoothed_data, pd.DataFrame):
            smoothed = self.smoothed_data.iloc[delay_index].values
        else:
            smoothed = self.smoothed_data[delay_index, :]
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # 对比图
        axes[0].plot(wavelengths, original, 'b-', label='原始数据', alpha=0.6, linewidth=1)
        axes[0].plot(wavelengths, smoothed, 'r-', label='平滑后数据', linewidth=2)
        axes[0].set_xlabel('波长 (nm)')
        axes[0].set_ylabel('吸光度变化 (ΔOD)')
        axes[0].set_title(f'数据平滑 ({self.method}) - 延迟时间: {delays[delay_index]:.2f} ps')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 差异图
        difference = original - smoothed
        axes[1].plot(wavelengths, difference, 'g-', label='差异 (原始 - 平滑)', alpha=0.7)
        axes[1].set_xlabel('波长 (nm)')
        axes[1].set_ylabel('吸光度变化 (ΔOD)')
        axes[1].set_title('平滑前后差异')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_smoothing_stats(self) -> dict:
        """
        获取数据平滑统计信息
        
        Returns:
            统计信息字典
        """
        if self.smoothed_data is None:
            return {}
        
        if isinstance(self.smoothed_data, pd.DataFrame):
            smoothed_array = self.smoothed_data.values
        else:
            smoothed_array = self.smoothed_data
            
        stats = {
            'method': self.method,
            'parameters': self.params,
            'smoothing_effect': {
                'smoothed_std': np.std(smoothed_array),
                'data_range': [np.min(smoothed_array), np.max(smoothed_array)]
            }
        }
        
        return stats


# 便捷函数
def smooth_data(data: Union[np.ndarray, pd.DataFrame], 
               method: str = 'savgol',
               **kwargs) -> Union[np.ndarray, pd.DataFrame]:
    """
    便捷的数据平滑函数
    
    Args:
        data: TAS数据
        method: 平滑方法
        **kwargs: 方法特定参数
        
    Returns:
        平滑后的数据
    """
    smoother = DataSmoother(method=method, **kwargs)
    return smoother.smooth(data)


def progressive_smooth(data: Union[np.ndarray, pd.DataFrame],
                      methods: list = ['savgol', 'moving_average'],
                      **kwargs) -> Union[np.ndarray, pd.DataFrame]:
    """
    渐进式平滑 - 使用多种方法逐步平滑
    
    Args:
        data: TAS数据
        methods: 平滑方法列表
        **kwargs: 方法参数
        
    Returns:
        平滑后的数据
    """
    current_data = data
    
    for method in methods:
        method_params = kwargs.get(f'{method}_params', {})
        smoother = DataSmoother(method=method, **method_params)
        current_data = smoother.smooth(current_data)
    
    return current_data
