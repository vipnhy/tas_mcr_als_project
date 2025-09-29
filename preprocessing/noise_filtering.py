#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
噪声过滤模块

提供多种噪声过滤方法用于TAS数据预处理
"""

import numpy as np
import pandas as pd
from scipy import signal, ndimage
from scipy.fft import fft, ifft, fftfreq
from sklearn.decomposition import PCA
from typing import Optional, Union, Tuple
import matplotlib.pyplot as plt


class NoiseFilter:
    """TAS数据噪声过滤器"""
    
    def __init__(self, method='gaussian', **kwargs):
        """
        初始化噪声过滤器
        
        Args:
            method: 过滤方法 ('gaussian', 'median', 'wiener', 'fft', 'pca', 'bilateral')
            **kwargs: 方法特定参数
        """
        self.method = method
        self.params = kwargs
        self.filtered_data = None
        self.noise_profile = None
        
    def filter_noise(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        执行噪声过滤
        
        Args:
            data: 输入TAS数据
            
        Returns:
            过滤后的数据
        """
        is_dataframe = isinstance(data, pd.DataFrame)
        
        if is_dataframe:
            data_array = data.values
        else:
            data_array = data
            
        if self.method == 'gaussian':
            filtered_array = self._gaussian_filter(data_array)
        elif self.method == 'median':
            filtered_array = self._median_filter(data_array)
        elif self.method == 'wiener':
            filtered_array = self._wiener_filter(data_array)
        elif self.method == 'fft':
            filtered_array = self._fft_filter(data_array)
        elif self.method == 'pca':
            filtered_array = self._pca_filter(data_array)
        elif self.method == 'bilateral':
            filtered_array = self._bilateral_filter(data_array)
        else:
            raise ValueError(f"不支持的噪声过滤方法: {self.method}")
        
        # 计算噪声轮廓
        self.noise_profile = data_array - filtered_array
        
        if is_dataframe:
            self.filtered_data = pd.DataFrame(filtered_array, 
                                            index=data.index, 
                                            columns=data.columns)
        else:
            self.filtered_data = filtered_array
            
        return self.filtered_data
    
    def _gaussian_filter(self, data: np.ndarray) -> np.ndarray:
        """
        高斯滤波
        
        Args:
            data: 输入数据
            
        Returns:
            过滤后数据
        """
        sigma = self.params.get('sigma', 1.0)
        mode = self.params.get('mode', 'reflect')
        
        # 对每个维度应用高斯滤波
        if len(data.shape) == 2:
            # 2D数据，分别对时间和波长维度滤波
            sigma_time = self.params.get('sigma_time', sigma)
            sigma_wavelength = self.params.get('sigma_wavelength', sigma)
            
            filtered = ndimage.gaussian_filter(data, 
                                             sigma=(sigma_time, sigma_wavelength),
                                             mode=mode)
        else:
            filtered = ndimage.gaussian_filter(data, sigma=sigma, mode=mode)
            
        return filtered
    
    def _median_filter(self, data: np.ndarray) -> np.ndarray:
        """
        中值滤波
        
        Args:
            data: 输入数据
            
        Returns:
            过滤后数据
        """
        size = self.params.get('size', 3)
        mode = self.params.get('mode', 'reflect')
        
        if len(data.shape) == 2:
            # 2D数据
            size_time = self.params.get('size_time', size)
            size_wavelength = self.params.get('size_wavelength', size)
            
            filtered = ndimage.median_filter(data, 
                                           size=(size_time, size_wavelength),
                                           mode=mode)
        else:
            filtered = ndimage.median_filter(data, size=size, mode=mode)
            
        return filtered
    
    def _wiener_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Wiener滤波
        
        Args:
            data: 输入数据
            
        Returns:
            过滤后数据
        """
        noise = self.params.get('noise', None)
        mysize = self.params.get('mysize', None)
        
        filtered = np.zeros_like(data)
        
        if len(data.shape) == 2:
            # 对每一行（时间序列）应用Wiener滤波
            for i in range(data.shape[0]):
                filtered[i, :] = signal.wiener(data[i, :], mysize=mysize, noise=noise)
        else:
            filtered = signal.wiener(data, mysize=mysize, noise=noise)
            
        return filtered
    
    def _fft_filter(self, data: np.ndarray) -> np.ndarray:
        """
        FFT频域滤波
        
        Args:
            data: 输入数据
            
        Returns:
            过滤后数据
        """
        cutoff_freq = self.params.get('cutoff_freq', 0.1)  # 截止频率（相对于奈奎斯特频率）
        filter_type = self.params.get('filter_type', 'lowpass')  # 'lowpass', 'highpass', 'bandpass'
        
        filtered = np.zeros_like(data)
        
        if len(data.shape) == 2:
            for i in range(data.shape[0]):
                filtered[i, :] = self._fft_filter_1d(data[i, :], cutoff_freq, filter_type)
        else:
            filtered = self._fft_filter_1d(data, cutoff_freq, filter_type)
            
        return filtered
    
    def _fft_filter_1d(self, signal_data: np.ndarray, cutoff_freq: float, filter_type: str) -> np.ndarray:
        """
        1D FFT滤波
        """
        # FFT变换
        fft_data = fft(signal_data)
        freqs = fftfreq(len(signal_data))
        
        # 创建滤波器
        if filter_type == 'lowpass':
            mask = np.abs(freqs) <= cutoff_freq
        elif filter_type == 'highpass':
            mask = np.abs(freqs) >= cutoff_freq
        elif filter_type == 'bandpass':
            high_freq = self.params.get('high_freq', 0.2)
            mask = (np.abs(freqs) >= cutoff_freq) & (np.abs(freqs) <= high_freq)
        else:
            raise ValueError(f"不支持的滤波器类型: {filter_type}")
        
        # 应用滤波器
        fft_filtered = fft_data.copy()
        fft_filtered[~mask] = 0
        
        # 逆FFT变换
        filtered_signal = np.real(ifft(fft_filtered))
        
        return filtered_signal
    
    def _pca_filter(self, data: np.ndarray) -> np.ndarray:
        """
        PCA降噪
        
        Args:
            data: 输入数据
            
        Returns:
            过滤后数据
        """
        n_components = self.params.get('n_components', None)
        variance_threshold = self.params.get('variance_threshold', 0.99)
        
        # 重塑数据用于PCA
        original_shape = data.shape
        if len(data.shape) == 2:
            data_reshaped = data.reshape(-1, data.shape[-1])
        else:
            data_reshaped = data.reshape(-1, 1)
        
        # 执行PCA
        pca = PCA()
        data_pca = pca.fit_transform(data_reshaped)
        
        # 确定保留的主成分数量
        if n_components is None:
            # 根据方差阈值确定
            cumsum_ratio = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum_ratio >= variance_threshold) + 1
        
        # 重构数据
        data_pca_filtered = data_pca[:, :n_components]
        data_reconstructed = pca.inverse_transform(
            np.column_stack([data_pca_filtered, 
                           np.zeros((data_pca_filtered.shape[0], 
                                   data_pca.shape[1] - n_components))])
        )
        
        # 恢复原始形状
        filtered = data_reconstructed.reshape(original_shape)
        
        return filtered
    
    def _bilateral_filter(self, data: np.ndarray) -> np.ndarray:
        """
        双边滤波（保边缘降噪）
        
        Args:
            data: 输入数据
            
        Returns:
            过滤后数据
        """
        d = self.params.get('d', 9)  # 滤波直径
        sigma_color = self.params.get('sigma_color', 75)  # 颜色空间标准差
        sigma_space = self.params.get('sigma_space', 75)  # 坐标空间标准差
        
        # 简化的双边滤波实现
        filtered = np.zeros_like(data)
        
        if len(data.shape) == 2:
            for i in range(data.shape[0]):
                filtered[i, :] = self._bilateral_filter_1d(data[i, :], d, sigma_color, sigma_space)
        else:
            filtered = self._bilateral_filter_1d(data, d, sigma_color, sigma_space)
            
        return filtered
    
    def _bilateral_filter_1d(self, signal_data: np.ndarray, d: int, 
                           sigma_color: float, sigma_space: float) -> np.ndarray:
        """
        1D双边滤波简化实现
        """
        filtered = np.zeros_like(signal_data)
        pad_width = d // 2
        padded_signal = np.pad(signal_data, pad_width, mode='reflect')
        
        for i in range(len(signal_data)):
            center_idx = i + pad_width
            center_val = padded_signal[center_idx]
            
            # 获取邻域
            neighborhood = padded_signal[center_idx - pad_width:center_idx + pad_width + 1]
            positions = np.arange(-pad_width, pad_width + 1)
            
            # 计算权重
            space_weights = np.exp(-(positions**2) / (2 * sigma_space**2))
            color_weights = np.exp(-((neighborhood - center_val)**2) / (2 * sigma_color**2))
            weights = space_weights * color_weights
            
            # 归一化权重
            weights /= np.sum(weights)
            
            # 计算滤波值
            filtered[i] = np.sum(neighborhood * weights)
        
        return filtered
    
    def estimate_noise_level(self, data: Union[np.ndarray, pd.DataFrame]) -> dict:
        """
        估计数据中的噪声水平
        
        Args:
            data: 输入数据
            
        Returns:
            噪声统计信息
        """
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data
        
        # 使用高频成分估计噪声
        # 计算二阶差分
        if len(data_array.shape) == 2:
            diff2_time = np.diff(data_array, n=2, axis=0)
            diff2_wavelength = np.diff(data_array, n=2, axis=1)
            
            noise_std_time = np.std(diff2_time) / np.sqrt(6)
            noise_std_wavelength = np.std(diff2_wavelength) / np.sqrt(6)
            
            noise_stats = {
                'overall_noise_std': np.std(data_array),
                'noise_std_time': noise_std_time,
                'noise_std_wavelength': noise_std_wavelength,
                'snr_estimate': np.std(data_array) / max(noise_std_time, noise_std_wavelength),
                'data_range': [np.min(data_array), np.max(data_array)],
                'noise_to_signal_ratio': max(noise_std_time, noise_std_wavelength) / np.std(data_array)
            }
        else:
            diff2 = np.diff(data_array, n=2)
            noise_std = np.std(diff2) / np.sqrt(6)
            
            noise_stats = {
                'overall_noise_std': np.std(data_array),
                'estimated_noise_std': noise_std,
                'snr_estimate': np.std(data_array) / noise_std,
                'data_range': [np.min(data_array), np.max(data_array)],
                'noise_to_signal_ratio': noise_std / np.std(data_array)
            }
        
        return noise_stats
    
    def plot_filtering_result(self, original_data: Union[np.ndarray, pd.DataFrame],
                            delay_index: int = 0,
                            wavelengths: Optional[np.ndarray] = None,
                            figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        绘制噪声过滤结果
        
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
        
        if self.filtered_data is None:
            raise ValueError("请先执行噪声过滤")
        
        if isinstance(self.filtered_data, pd.DataFrame):
            filtered = self.filtered_data.iloc[delay_index].values
        else:
            filtered = self.filtered_data[delay_index, :]
            
        noise = self.noise_profile[delay_index, :]
        
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # 原始数据
        axes[0].plot(wavelengths, original, 'b-', label='原始数据', alpha=0.7)
        axes[0].set_ylabel('吸光度变化 (ΔOD)')
        axes[0].set_title(f'噪声过滤 ({self.method}) - 延迟时间: {delays[delay_index]:.2f} ps')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 过滤后数据
        axes[1].plot(wavelengths, filtered, 'g-', label='过滤后数据', linewidth=2)
        axes[1].set_ylabel('吸光度变化 (ΔOD)')
        axes[1].set_title('过滤后数据')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 噪声成分
        axes[2].plot(wavelengths, noise, 'r-', label='噪声成分', alpha=0.7)
        axes[2].set_xlabel('波长 (nm)')
        axes[2].set_ylabel('吸光度变化 (ΔOD)')
        axes[2].set_title('提取的噪声成分')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_filtering_stats(self) -> dict:
        """
        获取噪声过滤统计信息
        
        Returns:
            统计信息字典
        """
        if self.filtered_data is None or self.noise_profile is None:
            return {}
        
        if isinstance(self.filtered_data, pd.DataFrame):
            filtered_array = self.filtered_data.values
        else:
            filtered_array = self.filtered_data
            
        stats = {
            'method': self.method,
            'parameters': self.params,
            'noise_reduction': {
                'original_std': np.std(filtered_array + self.noise_profile),
                'filtered_std': np.std(filtered_array),
                'noise_std': np.std(self.noise_profile),
                'noise_reduction_ratio': 1 - np.std(filtered_array) / np.std(filtered_array + self.noise_profile)
            },
            'signal_preservation': {
                'signal_correlation': np.corrcoef(
                    (filtered_array + self.noise_profile).flatten(),
                    filtered_array.flatten()
                )[0, 1],
                'amplitude_preservation': np.std(filtered_array) / np.std(filtered_array + self.noise_profile)
            }
        }
        
        return stats


# 便捷函数
def filter_noise(data: Union[np.ndarray, pd.DataFrame], 
                method: str = 'gaussian',
                **kwargs) -> Union[np.ndarray, pd.DataFrame]:
    """
    便捷的噪声过滤函数
    
    Args:
        data: TAS数据
        method: 过滤方法
        **kwargs: 方法特定参数
        
    Returns:
        过滤后的数据
    """
    noise_filter = NoiseFilter(method=method, **kwargs)
    return noise_filter.filter_noise(data)


def denoise_tas_data(data: Union[np.ndarray, pd.DataFrame],
                    methods: list = ['gaussian', 'median'],
                    **kwargs) -> Union[np.ndarray, pd.DataFrame]:
    """
    使用多种方法组合降噪
    
    Args:
        data: TAS数据
        methods: 降噪方法列表
        **kwargs: 方法参数
        
    Returns:
        降噪后的数据
    """
    current_data = data
    
    for method in methods:
        method_params = kwargs.get(f'{method}_params', {})
        noise_filter = NoiseFilter(method=method, **method_params)
        current_data = noise_filter.filter_noise(current_data)
    
    return current_data
