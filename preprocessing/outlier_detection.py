#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
异常值检测和处理模块

提供多种异常值检测和处理方法用于TAS数据预处理
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from typing import Optional, Union, Tuple, List
import matplotlib.pyplot as plt


class OutlierDetector:
    """TAS数据异常值检测器"""
    
    def __init__(self, method='z_score', **kwargs):
        """
        初始化异常值检测器
        
        Args:
            method: 检测方法 ('z_score', 'iqr', 'isolation_forest', 'mahalanobis', 'elliptic_envelope')
            **kwargs: 方法特定参数
        """
        self.method = method
        self.params = kwargs
        self.outlier_mask = None
        self.processed_data = None
        self.outlier_indices = None
        
    def detect_outliers(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        检测异常值
        
        Args:
            data: 输入TAS数据
            
        Returns:
            异常值掩码 (True表示异常值)
        """
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data
            
        if self.method == 'z_score':
            self.outlier_mask = self._z_score_detection(data_array)
        elif self.method == 'iqr':
            self.outlier_mask = self._iqr_detection(data_array)
        elif self.method == 'isolation_forest':
            self.outlier_mask = self._isolation_forest_detection(data_array)
        elif self.method == 'mahalanobis':
            self.outlier_mask = self._mahalanobis_detection(data_array)
        elif self.method == 'elliptic_envelope':
            self.outlier_mask = self._elliptic_envelope_detection(data_array)
        else:
            raise ValueError(f"不支持的异常值检测方法: {self.method}")
        
        # 获取异常值索引
        self.outlier_indices = np.where(self.outlier_mask)
        
        return self.outlier_mask
    
    def process_outliers(self, data: Union[np.ndarray, pd.DataFrame], 
                        strategy: str = 'interpolate') -> Union[np.ndarray, pd.DataFrame]:
        """
        处理异常值
        
        Args:
            data: 输入数据
            strategy: 处理策略 ('remove', 'interpolate', 'clip', 'median_replace')
            
        Returns:
            处理后的数据
        """
        is_dataframe = isinstance(data, pd.DataFrame)
        
        if is_dataframe:
            data_array = data.values
        else:
            data_array = data
        
        if self.outlier_mask is None:
            self.detect_outliers(data)
        
        processed_array = data_array.copy()
        
        if strategy == 'remove':
            processed_array = self._remove_outliers(processed_array)
        elif strategy == 'interpolate':
            processed_array = self._interpolate_outliers(processed_array)
        elif strategy == 'clip':
            processed_array = self._clip_outliers(processed_array)
        elif strategy == 'median_replace':
            processed_array = self._median_replace_outliers(processed_array)
        else:
            raise ValueError(f"不支持的异常值处理策略: {strategy}")
        
        if is_dataframe:
            if strategy == 'remove':
                # 对于移除策略，需要重新构建DataFrame
                valid_rows = ~np.any(self.outlier_mask, axis=1)
                self.processed_data = data.loc[data.index[valid_rows]]
            else:
                self.processed_data = pd.DataFrame(processed_array,
                                                 index=data.index,
                                                 columns=data.columns)
        else:
            self.processed_data = processed_array
            
        return self.processed_data
    
    def _z_score_detection(self, data: np.ndarray, threshold: float = None) -> np.ndarray:
        """
        Z-score异常值检测
        
        Args:
            data: 输入数据
            threshold: Z-score阈值
            
        Returns:
            异常值掩码
        """
        if threshold is None:
            threshold = self.params.get('threshold', 3.0)
        
        if len(data.shape) == 2:
            z_scores = np.abs(stats.zscore(data, axis=None))
        else:
            z_scores = np.abs(stats.zscore(data))
        
        return z_scores > threshold
    
    def _iqr_detection(self, data: np.ndarray, factor: float = None) -> np.ndarray:
        """
        四分位距(IQR)异常值检测
        
        Args:
            data: 输入数据
            factor: IQR倍数因子
            
        Returns:
            异常值掩码
        """
        if factor is None:
            factor = self.params.get('factor', 1.5)
        
        # 计算四分位数
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        # 计算异常值边界
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        return (data < lower_bound) | (data > upper_bound)
    
    def _isolation_forest_detection(self, data: np.ndarray) -> np.ndarray:
        """
        孤立森林异常值检测
        
        Args:
            data: 输入数据
            
        Returns:
            异常值掩码
        """
        contamination = self.params.get('contamination', 0.1)
        n_estimators = self.params.get('n_estimators', 100)
        random_state = self.params.get('random_state', 42)
        
        # 重塑数据用于sklearn
        original_shape = data.shape
        if len(data.shape) == 2:
            data_reshaped = data.reshape(-1, data.shape[-1])
        else:
            data_reshaped = data.reshape(-1, 1)
        
        # 执行孤立森林检测
        iso_forest = IsolationForest(contamination=contamination,
                                   n_estimators=n_estimators,
                                   random_state=random_state)
        
        outlier_labels = iso_forest.fit_predict(data_reshaped)
        
        # 转换为掩码 (-1表示异常值)
        outlier_mask = (outlier_labels == -1).reshape(original_shape)
        
        return outlier_mask
    
    def _mahalanobis_detection(self, data: np.ndarray, threshold: float = None) -> np.ndarray:
        """
        马氏距离异常值检测
        
        Args:
            data: 输入数据
            threshold: 马氏距离阈值
            
        Returns:
            异常值掩码
        """
        if threshold is None:
            threshold = self.params.get('threshold', 3.0)
        
        if len(data.shape) == 1:
            # 1D数据，使用标准化距离
            mean = np.mean(data)
            std = np.std(data)
            distances = np.abs(data - mean) / std
        else:
            # 2D数据，计算马氏距离
            # 重塑数据
            data_reshaped = data.reshape(-1, data.shape[-1])
            
            # 计算协方差矩阵
            try:
                cov_matrix = np.cov(data_reshaped.T)
                inv_cov_matrix = np.linalg.inv(cov_matrix)
                mean_vector = np.mean(data_reshaped, axis=0)
                
                # 计算马氏距离
                distances = np.array([
                    mahalanobis(point, mean_vector, inv_cov_matrix)
                    for point in data_reshaped
                ])
                
                distances = distances.reshape(data.shape[:-1])
                
            except np.linalg.LinAlgError:
                # 协方差矩阵奇异，回退到欧几里得距离
                mean_vector = np.mean(data_reshaped, axis=0)
                distances = np.linalg.norm(data_reshaped - mean_vector, axis=1)
                distances = distances.reshape(data.shape[:-1])
        
        return distances > threshold
    
    def _elliptic_envelope_detection(self, data: np.ndarray) -> np.ndarray:
        """
        椭圆包络异常值检测
        
        Args:
            data: 输入数据
            
        Returns:
            异常值掩码
        """
        contamination = self.params.get('contamination', 0.1)
        support_fraction = self.params.get('support_fraction', None)
        random_state = self.params.get('random_state', 42)
        
        # 重塑数据用于sklearn
        original_shape = data.shape
        if len(data.shape) == 2:
            data_reshaped = data.reshape(-1, data.shape[-1])
        else:
            data_reshaped = data.reshape(-1, 1)
        
        # 执行椭圆包络检测
        elliptic = EllipticEnvelope(contamination=contamination,
                                  support_fraction=support_fraction,
                                  random_state=random_state)
        
        outlier_labels = elliptic.fit_predict(data_reshaped)
        
        # 转换为掩码 (-1表示异常值)
        outlier_mask = (outlier_labels == -1).reshape(original_shape)
        
        return outlier_mask
    
    def _remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """
        移除异常值
        """
        if len(data.shape) == 2:
            # 移除包含异常值的行
            valid_rows = ~np.any(self.outlier_mask, axis=1)
            return data[valid_rows, :]
        else:
            # 1D数据，移除异常值点
            return data[~self.outlier_mask]
    
    def _interpolate_outliers(self, data: np.ndarray) -> np.ndarray:
        """
        插值替换异常值
        """
        processed = data.copy()
        
        if len(data.shape) == 2:
            for i in range(data.shape[0]):
                row_mask = self.outlier_mask[i, :]
                if np.any(row_mask):
                    # 对异常值位置进行插值
                    valid_indices = np.where(~row_mask)[0]
                    outlier_indices = np.where(row_mask)[0]
                    
                    if len(valid_indices) > 1:
                        processed[i, outlier_indices] = np.interp(
                            outlier_indices, valid_indices, data[i, valid_indices]
                        )
                    else:
                        # 如果有效点太少，使用行均值
                        processed[i, outlier_indices] = np.mean(data[i, ~row_mask])
        else:
            # 1D数据插值
            valid_indices = np.where(~self.outlier_mask)[0]
            outlier_indices = np.where(self.outlier_mask)[0]
            
            if len(valid_indices) > 1:
                processed[outlier_indices] = np.interp(
                    outlier_indices, valid_indices, data[valid_indices]
                )
            else:
                processed[outlier_indices] = np.mean(data[~self.outlier_mask])
        
        return processed
    
    def _clip_outliers(self, data: np.ndarray) -> np.ndarray:
        """
        裁剪异常值到合理范围
        """
        processed = data.copy()
        
        # 计算合理范围
        if len(data.shape) == 2:
            q1 = np.percentile(data, 25, axis=None)
            q3 = np.percentile(data, 75, axis=None)
        else:
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
        
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # 裁剪到合理范围
        processed[self.outlier_mask] = np.clip(
            data[self.outlier_mask], lower_bound, upper_bound
        )
        
        return processed
    
    def _median_replace_outliers(self, data: np.ndarray) -> np.ndarray:
        """
        用中值替换异常值
        """
        processed = data.copy()
        
        if len(data.shape) == 2:
            # 2D数据，使用全局中值或局部中值
            use_local_median = self.params.get('use_local_median', False)
            
            if use_local_median:
                # 使用局部中值
                window_size = self.params.get('window_size', 5)
                for i in range(data.shape[0]):
                    row_mask = self.outlier_mask[i, :]
                    if np.any(row_mask):
                        for j in np.where(row_mask)[0]:
                            # 计算局部窗口
                            start = max(0, j - window_size // 2)
                            end = min(data.shape[1], j + window_size // 2 + 1)
                            local_data = data[i, start:end]
                            local_mask = self.outlier_mask[i, start:end]
                            
                            # 使用局部中值
                            if np.any(~local_mask):
                                processed[i, j] = np.median(local_data[~local_mask])
                            else:
                                processed[i, j] = np.median(data[i, :])
            else:
                # 使用全局中值
                median_value = np.median(data[~self.outlier_mask])
                processed[self.outlier_mask] = median_value
        else:
            # 1D数据
            median_value = np.median(data[~self.outlier_mask])
            processed[self.outlier_mask] = median_value
        
        return processed
    
    def get_outlier_statistics(self) -> dict:
        """
        获取异常值统计信息
        
        Returns:
            异常值统计信息字典
        """
        if self.outlier_mask is None:
            return {}
        
        total_points = self.outlier_mask.size
        outlier_count = np.sum(self.outlier_mask)
        outlier_percentage = (outlier_count / total_points) * 100
        
        stats = {
            'method': self.method,
            'parameters': self.params,
            'total_points': total_points,
            'outlier_count': outlier_count,
            'outlier_percentage': outlier_percentage,
            'outlier_indices': self.outlier_indices
        }
        
        if len(self.outlier_mask.shape) == 2:
            # 2D数据的额外统计
            outliers_per_row = np.sum(self.outlier_mask, axis=1)
            outliers_per_col = np.sum(self.outlier_mask, axis=0)
            
            stats.update({
                'outliers_per_time_point': outliers_per_row.tolist(),
                'outliers_per_wavelength': outliers_per_col.tolist(),
                'max_outliers_per_row': np.max(outliers_per_row),
                'max_outliers_per_col': np.max(outliers_per_col)
            })
        
        return stats
    
    def plot_outlier_detection(self, data: Union[np.ndarray, pd.DataFrame],
                             wavelengths: Optional[np.ndarray] = None,
                             delays: Optional[np.ndarray] = None,
                             figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        绘制异常值检测结果
        
        Args:
            data: 原始数据
            wavelengths: 波长数组
            delays: 延迟时间数组
            figsize: 图像大小
            
        Returns:
            matplotlib图像对象
        """
        if isinstance(data, pd.DataFrame):
            wavelengths = data.columns.values if wavelengths is None else wavelengths
            delays = data.index.values if delays is None else delays
            data_array = data.values
        else:
            data_array = data
            wavelengths = np.arange(data.shape[1]) if wavelengths is None else wavelengths
            delays = np.arange(data.shape[0]) if delays is None else delays
        
        if self.outlier_mask is None:
            raise ValueError("请先执行异常值检测")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 原始数据热图
        im1 = axes[0, 0].imshow(data_array, aspect='auto', origin='lower',
                               extent=[wavelengths[0], wavelengths[-1], 
                                      delays[0], delays[-1]])
        axes[0, 0].set_title('原始数据')
        axes[0, 0].set_xlabel('波长 (nm)')
        axes[0, 0].set_ylabel('延迟时间 (ps)')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 异常值掩码
        im2 = axes[0, 1].imshow(self.outlier_mask.astype(int), aspect='auto', origin='lower',
                               extent=[wavelengths[0], wavelengths[-1], 
                                      delays[0], delays[-1]],
                               cmap='Reds')
        axes[0, 1].set_title('异常值分布 (红色=异常值)')
        axes[0, 1].set_xlabel('波长 (nm)')
        axes[0, 1].set_ylabel('延迟时间 (ps)')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 每个时间点的异常值数量
        outliers_per_time = np.sum(self.outlier_mask, axis=1)
        axes[1, 0].plot(delays, outliers_per_time, 'ro-')
        axes[1, 0].set_xlabel('延迟时间 (ps)')
        axes[1, 0].set_ylabel('异常值数量')
        axes[1, 0].set_title('每个时间点的异常值数量')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 每个波长的异常值数量
        outliers_per_wavelength = np.sum(self.outlier_mask, axis=0)
        axes[1, 1].plot(wavelengths, outliers_per_wavelength, 'bo-')
        axes[1, 1].set_xlabel('波长 (nm)')
        axes[1, 1].set_ylabel('异常值数量')
        axes[1, 1].set_title('每个波长的异常值数量')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_processing_result(self, original_data: Union[np.ndarray, pd.DataFrame],
                             delay_index: int = 0,
                             wavelengths: Optional[np.ndarray] = None,
                             figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        绘制异常值处理结果
        """
        if isinstance(original_data, pd.DataFrame):
            wavelengths = original_data.columns.values if wavelengths is None else wavelengths
            original = original_data.iloc[delay_index].values
            delays = original_data.index.values
        else:
            wavelengths = np.arange(original_data.shape[1]) if wavelengths is None else wavelengths
            original = original_data[delay_index, :]
            delays = np.arange(original_data.shape[0])
        
        if self.processed_data is None:
            raise ValueError("请先执行异常值处理")
        
        if isinstance(self.processed_data, pd.DataFrame):
            processed = self.processed_data.iloc[delay_index].values
        else:
            processed = self.processed_data[delay_index, :]
        
        outlier_mask_1d = self.outlier_mask[delay_index, :]
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # 对比图
        axes[0].plot(wavelengths, original, 'b-', label='原始数据', alpha=0.6)
        axes[0].plot(wavelengths, processed, 'g-', label='处理后数据', linewidth=2)
        
        # 标记异常值
        if np.any(outlier_mask_1d):
            axes[0].scatter(wavelengths[outlier_mask_1d], original[outlier_mask_1d], 
                           color='red', s=50, label='检测到的异常值', zorder=5)
        
        axes[0].set_xlabel('波长 (nm)')
        axes[0].set_ylabel('吸光度变化 (ΔOD)')
        axes[0].set_title(f'异常值处理 ({self.method}) - 延迟时间: {delays[delay_index]:.2f} ps')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 处理效果
        difference = original - processed
        axes[1].plot(wavelengths, difference, 'r-', label='处理前后差异', alpha=0.7)
        axes[1].set_xlabel('波长 (nm)')
        axes[1].set_ylabel('吸光度变化 (ΔOD)')
        axes[1].set_title('处理前后差异')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# 便捷函数
def detect_and_process_outliers(data: Union[np.ndarray, pd.DataFrame],
                              detection_method: str = 'z_score',
                              processing_strategy: str = 'interpolate',
                              **kwargs) -> Union[np.ndarray, pd.DataFrame]:
    """
    便捷的异常值检测和处理函数
    
    Args:
        data: TAS数据
        detection_method: 检测方法
        processing_strategy: 处理策略
        **kwargs: 方法特定参数
        
    Returns:
        处理后的数据
    """
    detector = OutlierDetector(method=detection_method, **kwargs)
    detector.detect_outliers(data)
    return detector.process_outliers(data, strategy=processing_strategy)


def multi_method_outlier_detection(data: Union[np.ndarray, pd.DataFrame],
                                 methods: List[str] = ['z_score', 'iqr'],
                                 consensus_threshold: float = 0.5,
                                 **kwargs) -> np.ndarray:
    """
    多方法异常值检测 - 使用多种方法的一致性结果
    
    Args:
        data: TAS数据
        methods: 检测方法列表
        consensus_threshold: 一致性阈值 (0-1)
        **kwargs: 方法参数
        
    Returns:
        异常值掩码
    """
    if isinstance(data, pd.DataFrame):
        data_array = data.values
    else:
        data_array = data
    
    outlier_masks = []
    
    for method in methods:
        method_params = kwargs.get(f'{method}_params', {})
        detector = OutlierDetector(method=method, **method_params)
        mask = detector.detect_outliers(data)
        outlier_masks.append(mask.astype(float))
    
    # 计算一致性
    consensus_mask = np.mean(outlier_masks, axis=0)
    final_mask = consensus_mask >= consensus_threshold
    
    return final_mask
