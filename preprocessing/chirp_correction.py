#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
啁啾校正模块

用于瞬态吸收光谱数据的啁啾校正处理
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any, Union
import warnings

class ChirpCorrector:
    """啁啾校正器"""
    
    def __init__(self, method='cross_correlation', verbose=True, 
                 solvent_wavelengths=None, polynomial_order=3, **kwargs):
        """
        初始化啁啾校正器
        
        Args:
            method: 校正方法 ('polynomial', 'spline', 'cross_correlation', 'solvent_response')
            verbose: 是否显示详细信息
            solvent_wavelengths: 溶剂响应波长范围 (用于 solvent_response 方法)
            polynomial_order: 多项式阶数 (用于 polynomial 方法)
            **kwargs: 其他参数
        """
        self.method = method
        self.verbose = verbose
        self.solvent_wavelengths = solvent_wavelengths or [400, 450]
        self.polynomial_order = polynomial_order
        self.chirp_function = None
        self.correction_params = {}
        self.corrected_data = None
        
    def detect_chirp_from_solvent(self, data: pd.DataFrame, 
                                solvent_wavelengths: Optional[list] = None) -> np.ndarray:
        """
        基于溶剂响应检测啁啾
        
        Args:
            data: TAS数据 (index=time, columns=wavelength)
            solvent_wavelengths: 溶剂响应波长范围
            
        Returns:
            啁啾时间偏移数组
        """
        wavelengths = data.columns.values
        delays = data.index.values
        
        # 如果未指定溶剂响应波长，使用实例变量或自动选择蓝端区域
        if solvent_wavelengths is None:
            solvent_wavelengths = self.solvent_wavelengths
            if solvent_wavelengths is None:
                # 通常溶剂响应在短波长区域最明显
                wl_start = wavelengths[0]
                wl_end = wavelengths[0] + (wavelengths[-1] - wavelengths[0]) * 0.3
                solvent_wavelengths = [wl_start, wl_end]
        
        if self.verbose:
            print(f"🔍 使用溶剂响应检测啁啾，波长范围: {solvent_wavelengths[0]:.1f}-{solvent_wavelengths[1]:.1f} nm")
        
        # 选择溶剂响应区域
        mask = (wavelengths >= solvent_wavelengths[0]) & (wavelengths <= solvent_wavelengths[1])
        solvent_data = data.iloc[:, mask]
        
        time_zeros = []
        
        for i, wl in enumerate(solvent_data.columns):
            spectrum = solvent_data.iloc[:, i].values
            
            # 寻找最早的显著信号变化点
            try:
                # 方法1：寻找最大梯度点
                gradient = np.gradient(spectrum)
                abs_gradient = np.abs(gradient)
                
                # 排除前几个点避免噪声影响
                start_idx = max(1, len(delays) // 20)
                max_grad_idx = start_idx + np.argmax(abs_gradient[start_idx:])
                
                # 方法2：寻找信号幅度的一定百分比点
                max_signal = np.max(np.abs(spectrum))
                threshold = max_signal * 0.1  # 10%阈值
                
                for j in range(len(spectrum)):
                    if np.abs(spectrum[j]) >= threshold:
                        threshold_idx = j
                        break
                else:
                    threshold_idx = max_grad_idx
                
                # 选择更保守的估计
                time_zero_idx = min(max_grad_idx, threshold_idx)
                time_zero = delays[time_zero_idx]
                
            except:
                time_zero = 0.0
            
            time_zeros.append(time_zero)
        
        # 插值到所有波长
        valid_wls = solvent_data.columns.values
        time_zeros = np.array(time_zeros)
        
        # 使用样条插值扩展到所有波长
        try:
            if len(valid_wls) > 3:
                spline = UnivariateSpline(valid_wls, time_zeros, s=0.1, k=min(3, len(valid_wls)-1))
                chirp_correction = spline(wavelengths)
            else:
                # 如果数据点太少，使用线性插值
                interp_func = interp1d(valid_wls, time_zeros, 
                                     kind='linear', fill_value='extrapolate')
                chirp_correction = interp_func(wavelengths)
        except:
            # 如果插值失败，使用多项式拟合
            if self.verbose:
                print("⚠️ 样条插值失败，改用多项式拟合")
            chirp_correction = self._fit_polynomial_chirp(valid_wls, time_zeros, wavelengths)
        
        return chirp_correction
    
    def detect_chirp_cross_correlation(self, data: pd.DataFrame, 
                                     reference_wavelength: Optional[float] = None) -> np.ndarray:
        """
        基于互相关检测啁啾
        
        Args:
            data: TAS数据
            reference_wavelength: 参考波长
            
        Returns:
            啁啾时间偏移数组
        """
        wavelengths = data.columns.values
        delays = data.index.values
        
        if reference_wavelength is None:
            # 选择中间波长作为参考
            reference_wavelength = wavelengths[len(wavelengths) // 2]
        
        if self.verbose:
            print(f"🔍 使用互相关检测啁啾，参考波长: {reference_wavelength:.1f} nm")
        
        # 找到参考波长索引
        ref_idx = np.argmin(np.abs(wavelengths - reference_wavelength))
        ref_spectrum = data.iloc[:, ref_idx].values
        
        time_shifts = []
        
        for i, wl in enumerate(wavelengths):
            spectrum = data.iloc[:, i].values
            
            # 计算互相关
            correlation = np.correlate(ref_spectrum, spectrum, mode='full')
            
            # 找到最大相关性对应的时间偏移
            max_corr_idx = np.argmax(correlation)
            shift_samples = max_corr_idx - (len(ref_spectrum) - 1)
            
            # 转换为时间偏移
            if len(delays) > 1:
                dt = delays[1] - delays[0] if len(delays) > 1 else 1.0
                time_shift = shift_samples * dt
            else:
                time_shift = 0.0
            
            time_shifts.append(time_shift)
        
        return np.array(time_shifts)
    
    def _fit_polynomial_chirp(self, wl_data: np.ndarray, time_data: np.ndarray, 
                            target_wavelengths: np.ndarray, degree: int = None) -> np.ndarray:
        """拟合多项式啁啾函数"""
        if degree is None:
            degree = self.polynomial_order
            
        try:
            coeffs = np.polyfit(wl_data, time_data, degree)
            poly_func = np.poly1d(coeffs)
            return poly_func(target_wavelengths)
        except:
            # 如果拟合失败，返回零偏移
            if self.verbose:
                print("⚠️ 多项式拟合失败，返回零偏移")
            return np.zeros_like(target_wavelengths)
    
    def fit_chirp(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        拟合啁啾函数
        
        Args:
            data: TAS数据
            **kwargs: 方法特定参数
            
        Returns:
            拟合结果字典
        """
        wavelengths = data.columns.values
        
        if self.method == 'solvent_response':
            chirp_offsets = self.detect_chirp_from_solvent(data, 
                kwargs.get('solvent_wavelengths', None))
        elif self.method == 'cross_correlation':
            chirp_offsets = self.detect_chirp_cross_correlation(data,
                kwargs.get('reference_wavelength', None))
        elif self.method == 'polynomial':
            # 用户提供的校正点
            calib_points = kwargs.get('calibration_points', None)
            if calib_points is None:
                if self.verbose:
                    print("⚠️ 多项式方法需要校正点，改用溶剂响应方法")
                chirp_offsets = self.detect_chirp_from_solvent(data)
            else:
                wl_calib, time_calib = zip(*calib_points)
                chirp_offsets = self._fit_polynomial_chirp(
                    np.array(wl_calib), np.array(time_calib), wavelengths,
                    kwargs.get('degree', 3))
        else:
            raise ValueError(f"未知的啁啾校正方法: {self.method}")
        
        # 创建插值函数
        self.chirp_function = interp1d(wavelengths, chirp_offsets, 
                                     kind='linear', fill_value='extrapolate')
        
        self.correction_params = {
            'wavelengths': wavelengths,
            'time_offsets': chirp_offsets,
            'method': self.method,
            'fit_quality': self._evaluate_fit_quality(chirp_offsets)
        }
        
        if self.verbose:
            print(f"✅ 啁啾拟合完成，时间偏移范围: {chirp_offsets.min():.3f} - {chirp_offsets.max():.3f} ps")
        
        return self.correction_params
    
    def _evaluate_fit_quality(self, offsets: np.ndarray) -> Dict[str, float]:
        """评估拟合质量"""
        return {
            'offset_range': offsets.max() - offsets.min(),
            'offset_std': np.std(offsets),
            'smoothness': np.mean(np.abs(np.diff(offsets, 2))) if len(offsets) > 2 else 0
        }
    
    def apply_correction(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        应用啁啾校正
        
        Args:
            data: 待校正的TAS数据
            
        Returns:
            校正后的数据
        """
        if self.chirp_function is None:
            raise ValueError("请先调用 fit_chirp() 拟合啁啾函数")
        
        wavelengths = data.columns.values
        delays = data.index.values
        corrected_data = data.copy()
        
        if self.verbose:
            print("🔧 应用啁啾校正...")
        
        # 对每个波长应用时间偏移校正
        for i, wl in enumerate(wavelengths):
            time_offset = self.chirp_function(wl)
            
            if abs(time_offset) > 1e-10:  # 只有偏移量足够大时才校正
                # 获取该波长的光谱
                spectrum = data.iloc[:, i].values
                
                # 创建校正后的时间轴
                corrected_delays = delays - time_offset
                
                # 插值到原始时间网格
                try:
                    interp_func = interp1d(corrected_delays, spectrum, 
                                         kind='linear', fill_value=0.0, bounds_error=False)
                    corrected_spectrum = interp_func(delays)
                    corrected_data.iloc[:, i] = corrected_spectrum
                except:
                    if self.verbose:
                        print(f"⚠️ 波长 {wl:.1f} nm 校正失败，保持原数据")
        
        self.corrected_data = corrected_data
        
        if self.verbose:
            print("✅ 啁啾校正完成")
        
        return corrected_data
    
    def plot_chirp_function(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (10, 6)):
        """绘制啁啾函数"""
        if self.correction_params is None:
            raise ValueError("请先拟合啁啾函数")
        
        wavelengths = self.correction_params['wavelengths']
        time_offsets = self.correction_params['time_offsets']
        
        plt.figure(figsize=figsize)
        plt.plot(wavelengths, time_offsets, 'b-', linewidth=2, label='啁啾函数')
        plt.scatter(wavelengths[::10], time_offsets[::10], c='red', s=30, zorder=5)
        
        plt.xlabel('波长 (nm)')
        plt.ylabel('时间偏移 (ps)')
        plt.title(f'啁啾校正函数 ({self.method})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 添加统计信息
        fit_quality = self.correction_params['fit_quality']
        textstr = f'偏移范围: {fit_quality["offset_range"]:.3f} ps\n'
        textstr += f'标准差: {fit_quality["offset_std"]:.3f} ps'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✅ 啁啾函数图保存至: {save_path}")
        
        return plt.gcf()
    
    def get_correction_stats(self) -> dict:
        """获取啁啾校正统计信息"""
        if self.correction_params is None:
            return {}
            
        stats = {
            'method': self.method,
            'parameters': self.correction_params,
            'fit_quality': self.correction_params['fit_quality']
        }
        
        return stats

    def correct_chirp(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        time_delays: np.ndarray = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        执行啁啾校正
        
        Args:
            data: 输入数据 (numpy array, shape: wavelengths x time_delays)
            time_delays: 时间延迟数组
            
        Returns:
            啁啾校正后的数据
        """
        if self.verbose:
            print(f"开始执行 {self.method} 啁啾校正...")
        
        is_dataframe = isinstance(data, pd.DataFrame)

        if is_dataframe:
            df = data.copy()
            if time_delays is None:
                time_delays = df.index.values
            wavelengths = df.columns.values
        else:
            array = np.asarray(data)
            if time_delays is None:
                time_delays = np.arange(array.shape[1])
            wavelengths = np.arange(array.shape[0])
            df = pd.DataFrame(array.T, index=time_delays, columns=wavelengths)
        
        try:
            # 拟合啁啾函数
            self.fit_chirp(df)
            
            # 应用校正
            corrected_df = self.apply_correction(df)
            
            if is_dataframe:
                self.corrected_data = corrected_df
                result = corrected_df
            else:
                # 转回原始格式 (波长 x 时间)
                self.corrected_data = corrected_df.T.values
                result = self.corrected_data
            
            if self.verbose:
                print(f"✅ {self.method} 啁啾校正完成")
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"❌ {self.method} 啁啾校正失败: {str(e)}")
            return data  # 返回原始数据


# 便捷函数
def correct_chirp(data: pd.DataFrame, 
                 method: str = 'solvent_response',
                 **kwargs) -> pd.DataFrame:
    """
    便捷的啁啾校正函数
    
    Args:
        data: TAS数据
        method: 校正方法
        **kwargs: 方法特定参数
        
    Returns:
        校正后的数据
    """
    corrector = ChirpCorrector(method=method, **kwargs)
    corrector.fit_chirp(data, **kwargs)
    return corrector.apply_correction(data)
