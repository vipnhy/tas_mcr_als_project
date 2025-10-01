"""
model.py - 全局拟合核心算法

该模块实现了全局寿命分析(GLA)和全局目标分析(GTA)的核心算法，
参考了EfsTA项目的实现方法。
"""

import numpy as np
from lmfit import Parameters, minimize, fit_report
from typing import List, Dict, Tuple, Optional, Union, Callable
import time

from .kinetic_models import (
    KineticModelBase, 
    SequentialModel, 
    ParallelModel,
    create_exponential_decay_model
)


class GlobalFitter:
    """
    全局拟合器基类
    
    该类提供了全局拟合的基本框架，包括参数优化、
    残差计算和结果分析功能。
    """
    
    def __init__(self, data_matrix: np.ndarray, 
                 time_axis: np.ndarray,
                 wavelength_axis: np.ndarray):
        """
        初始化全局拟合器
        
        参数:
        - data_matrix: 实验数据矩阵 D (n_times × n_wavelengths)
        - time_axis: 时间轴数组 (n_times,)
        - wavelength_axis: 波长轴数组 (n_wavelengths,)
        """
        self.D = data_matrix
        self.time_axis = time_axis
        self.wavelength_axis = wavelength_axis
        self.n_times, self.n_wavelengths = data_matrix.shape
        
        # 拟合结果
        self.fit_result = None
        self.C_fit = None  # 拟合得到的浓度矩阵
        self.S_fit = None  # 拟合得到的光谱矩阵
        self.D_reconstructed = None  # 重构的数据矩阵
        self.residuals = None  # 残差矩阵
        self.chi_square = None  # 卡方值
        
    def calculate_residuals(self, D_model: np.ndarray) -> np.ndarray:
        """
        计算残差
        
        参数:
        - D_model: 模型计算的数据矩阵
        
        返回:
        - residuals: 残差矩阵 (扁平化为一维)
        """
        residuals = self.D - D_model
        return residuals.ravel()
    
    def calculate_chi_square(self, residuals: np.ndarray) -> float:
        """
        计算卡方值
        
        参数:
        - residuals: 残差数组
        
        返回:
        - chi_square: 归一化卡方值
        """
        chi_square = np.sum(residuals**2) / len(residuals)
        return chi_square
    
    def calculate_lof(self, residuals: np.ndarray) -> float:
        """
        计算拟合缺失度 (Lack of Fit)
        
        参数:
        - residuals: 残差数组
        
        返回:
        - lof: LOF百分比
        """
        data_norm = np.linalg.norm(self.D)
        residual_norm = np.linalg.norm(residuals)
        lof = 100 * residual_norm / data_norm
        return lof
    
    def get_fit_report(self) -> str:
        """
        获取拟合报告
        
        返回:
        - report: 格式化的拟合报告字符串
        """
        if self.fit_result is None:
            return "尚未进行拟合"
        
        report = fit_report(self.fit_result)
        
        # 添加额外信息
        if self.residuals is not None:
            lof = self.calculate_lof(self.residuals)
            report += f"\n\nLack of Fit (LOF): {lof:.4f}%"
        
        if self.chi_square is not None:
            report += f"\nChi-Square: {self.chi_square:.6e}"
        
        return report


class GlobalLifetimeAnalysis(GlobalFitter):
    """
    全局寿命分析 (GLA)
    
    该方法使用多指数衰减模型拟合数据:
    D(t, λ) = Σ A_i(λ) * exp(-t / τ_i)
    
    优点: 不需要预先假设反应机理
    缺点: 缺乏物理意义，仅是数学拟合
    """
    
    def __init__(self, data_matrix: np.ndarray,
                 time_axis: np.ndarray,
                 wavelength_axis: np.ndarray,
                 n_components: int):
        """
        初始化GLA分析器
        
        参数:
        - data_matrix: 实验数据矩阵 D (n_times × n_wavelengths)
        - time_axis: 时间轴数组 (n_times,)
        - wavelength_axis: 波长轴数组 (n_wavelengths,)
        - n_components: 指数衰减组分数量
        """
        super().__init__(data_matrix, time_axis, wavelength_axis)
        self.n_components = n_components
        
    def fit(self, tau_initial: List[float],
            tau_bounds: Optional[List[Tuple[float, float]]] = None,
            tau_vary: Optional[List[bool]] = None,
            optimization_method: str = 'leastsq') -> Dict:
        """
        执行GLA拟合
        
        参数:
        - tau_initial: 寿命初始值列表 [τ1, τ2, ...]
        - tau_bounds: 寿命边界列表 [(min1, max1), (min2, max2), ...]
        - tau_vary: 是否优化每个寿命参数的列表 [True, False, ...]
        - optimization_method: 优化方法 ('leastsq', 'least_squares', 'differential_evolution')
        
        返回:
        - results: 包含拟合结果的字典
        """
        if len(tau_initial) != self.n_components:
            raise ValueError(f"寿命初始值数量 ({len(tau_initial)}) "
                           f"与组分数量 ({self.n_components}) 不匹配")
        
        # 设置默认值
        if tau_bounds is None:
            tau_bounds = [(tau * 0.1, tau * 10) for tau in tau_initial]
        if tau_vary is None:
            tau_vary = [True] * self.n_components
        
        # 创建参数对象
        params = Parameters()
        for i in range(self.n_components):
            params.add(f'tau_{i}', 
                      value=tau_initial[i],
                      min=tau_bounds[i][0],
                      max=tau_bounds[i][1],
                      vary=tau_vary[i])
        
        print(f"\n开始GLA拟合，组分数量: {self.n_components}")
        print(f"优化方法: {optimization_method}")
        start_time = time.time()
        
        # 执行优化
        self.fit_result = minimize(
            self._objective_function_gla,
            params,
            method=optimization_method,
            nan_policy='propagate'
        )
        
        elapsed_time = time.time() - start_time
        print(f"拟合完成，耗时: {elapsed_time:.2f} 秒")
        
        # 提取最优参数
        tau_optimal = [self.fit_result.params[f'tau_{i}'].value 
                       for i in range(self.n_components)]
        
        # 计算最终的浓度和光谱矩阵
        self.C_fit, self.S_fit = self._calculate_C_and_S(tau_optimal)
        self.D_reconstructed = self.C_fit @ self.S_fit.T
        self.residuals = self.calculate_residuals(self.D_reconstructed)
        self.chi_square = self.calculate_chi_square(self.residuals)
        
        # 整理结果
        results = {
            'tau_optimal': tau_optimal,
            'C_fit': self.C_fit,
            'S_fit': self.S_fit,
            'D_reconstructed': self.D_reconstructed,
            'residuals': self.residuals.reshape(self.D.shape),
            'chi_square': self.chi_square,
            'lof': self.calculate_lof(self.residuals),
            'fit_result': self.fit_result,
            'computation_time': elapsed_time
        }
        
        print(f"最优寿命: {tau_optimal}")
        print(f"LOF: {results['lof']:.4f}%")
        
        return results
    
    def _objective_function_gla(self, params: Parameters) -> np.ndarray:
        """
        GLA的目标函数 (用于优化)
        
        参数:
        - params: lmfit参数对象
        
        返回:
        - residuals: 残差数组 (扁平化)
        """
        # 提取寿命参数
        tau_values = [params[f'tau_{i}'].value for i in range(self.n_components)]
        
        # 计算浓度和光谱矩阵
        C, S = self._calculate_C_and_S(tau_values)
        
        # 重构数据矩阵
        D_model = C @ S.T
        
        # 计算残差
        residuals = self.calculate_residuals(D_model)
        
        return residuals
    
    def _calculate_C_and_S(self, tau_values: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据寿命计算浓度矩阵C和光谱矩阵S
        
        参数:
        - tau_values: 寿命值列表
        
        返回:
        - C: 浓度矩阵 (n_times × n_components)
        - S: 光谱矩阵 (n_wavelengths × n_components)
        """
        # 构建指数衰减矩阵 E(t, τ)
        E = np.zeros((self.n_times, self.n_components))
        for i in range(self.n_components):
            E[:, i] = np.exp(-self.time_axis / tau_values[i])
        
        # 求解振幅矩阵 A: D = E @ A.T
        # A = (E.T @ E)^-1 @ E.T @ D
        # A的形状: (n_wavelengths × n_components)
        try:
            # 使用伪逆求解
            A = np.linalg.lstsq(E, self.D, rcond=None)[0].T
        except np.linalg.LinAlgError:
            # 如果失败，使用正则化
            lambda_reg = 1e-6 * np.trace(E.T @ E) / self.n_components
            EtE_reg = E.T @ E + lambda_reg * np.eye(self.n_components)
            A = (np.linalg.solve(EtE_reg, E.T @ self.D)).T
        
        # 在GLA中，C = E，S = A
        C = E
        S = A
        
        return C, S


class GlobalTargetAnalysis(GlobalFitter):
    """
    全局目标分析 (GTA)
    
    该方法基于明确的动力学模型拟合数据:
    1. 根据动力学模型求解浓度矩阵 C(t, k)
    2. D(t, λ) = C(t, k) @ S(λ).T
    
    优点: 有明确的物理意义，参数可解释
    缺点: 需要预先假设反应机理
    """
    
    def __init__(self, data_matrix: np.ndarray,
                 time_axis: np.ndarray,
                 wavelength_axis: np.ndarray,
                 kinetic_model: KineticModelBase):
        """
        初始化GTA分析器
        
        参数:
        - data_matrix: 实验数据矩阵 D (n_times × n_wavelengths)
        - time_axis: 时间轴数组 (n_times,)
        - wavelength_axis: 波长轴数组 (n_wavelengths,)
        - kinetic_model: 动力学模型对象
        """
        super().__init__(data_matrix, time_axis, wavelength_axis)
        self.kinetic_model = kinetic_model
        self.n_components = kinetic_model.n_components
        
    def fit(self, k_initial: List[float],
            k_bounds: Optional[List[Tuple[float, float]]] = None,
            k_vary: Optional[List[bool]] = None,
            optimization_method: str = 'leastsq') -> Dict:
        """
        执行GTA拟合
        
        参数:
        - k_initial: 速率常数初始值列表 [k1, k2, ...]
        - k_bounds: 速率常数边界列表 [(min1, max1), (min2, max2), ...]
        - k_vary: 是否优化每个速率常数的列表 [True, False, ...]
        - optimization_method: 优化方法
        
        返回:
        - results: 包含拟合结果的字典
        """
        n_rate_constants = self.kinetic_model.n_rate_constants
        
        if len(k_initial) != n_rate_constants:
            raise ValueError(f"速率常数初始值数量 ({len(k_initial)}) "
                           f"与模型要求 ({n_rate_constants}) 不匹配")
        
        # 设置默认值
        if k_bounds is None:
            k_bounds = [(k * 0.01, k * 100) for k in k_initial]
        if k_vary is None:
            k_vary = [True] * n_rate_constants
        
        # 创建参数对象
        params = Parameters()
        for i in range(n_rate_constants):
            params.add(f'k_{i}', 
                      value=k_initial[i],
                      min=k_bounds[i][0],
                      max=k_bounds[i][1],
                      vary=k_vary[i])
        
        print(f"\n开始GTA拟合，组分数量: {self.n_components}")
        print(f"速率常数数量: {n_rate_constants}")
        print(f"优化方法: {optimization_method}")
        start_time = time.time()
        
        # 执行优化
        self.fit_result = minimize(
            self._objective_function_gta,
            params,
            method=optimization_method,
            nan_policy='propagate'
        )
        
        elapsed_time = time.time() - start_time
        print(f"拟合完成，耗时: {elapsed_time:.2f} 秒")
        
        # 提取最优参数
        k_optimal = [self.fit_result.params[f'k_{i}'].value 
                     for i in range(n_rate_constants)]
        
        # 计算最终的浓度和光谱矩阵
        self.C_fit, self.S_fit = self._calculate_C_and_S(k_optimal)
        self.D_reconstructed = self.C_fit @ self.S_fit.T
        self.residuals = self.calculate_residuals(self.D_reconstructed)
        self.chi_square = self.calculate_chi_square(self.residuals)
        
        # 计算寿命 (τ = 1/k)
        tau_optimal = [1.0/k if k > 0 else np.inf for k in k_optimal]
        
        # 整理结果
        results = {
            'k_optimal': k_optimal,
            'tau_optimal': tau_optimal,
            'C_fit': self.C_fit,
            'S_fit': self.S_fit,
            'D_reconstructed': self.D_reconstructed,
            'residuals': self.residuals.reshape(self.D.shape),
            'chi_square': self.chi_square,
            'lof': self.calculate_lof(self.residuals),
            'fit_result': self.fit_result,
            'computation_time': elapsed_time,
            'kinetic_model': self.kinetic_model.__class__.__name__
        }
        
        print(f"最优速率常数: {k_optimal}")
        print(f"对应寿命: {tau_optimal}")
        print(f"LOF: {results['lof']:.4f}%")
        
        return results
    
    def _objective_function_gta(self, params: Parameters) -> np.ndarray:
        """
        GTA的目标函数 (用于优化)
        
        参数:
        - params: lmfit参数对象
        
        返回:
        - residuals: 残差数组 (扁平化)
        """
        # 提取速率常数
        n_k = self.kinetic_model.n_rate_constants
        k_values = [params[f'k_{i}'].value for i in range(n_k)]
        
        # 计算浓度和光谱矩阵
        C, S = self._calculate_C_and_S(k_values)
        
        # 重构数据矩阵
        D_model = C @ S.T
        
        # 计算残差
        residuals = self.calculate_residuals(D_model)
        
        return residuals
    
    def _calculate_C_and_S(self, k_values: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据速率常数计算浓度矩阵C和光谱矩阵S
        
        参数:
        - k_values: 速率常数列表
        
        返回:
        - C: 浓度矩阵 (n_times × n_components)
        - S: 光谱矩阵 (n_wavelengths × n_components)
        """
        # 求解动力学微分方程得到浓度矩阵
        try:
            C = self.kinetic_model.solve(self.time_axis, k_values)
        except Exception as e:
            # 如果求解失败，返回零矩阵
            print(f"警告: 动力学方程求解失败 - {e}")
            C = np.zeros((self.n_times, self.n_components))
            C[:, 0] = 1.0  # 设置初始值
        
        # 求解光谱矩阵: D = C @ S.T
        # S = (C.T @ C)^-1 @ C.T @ D
        try:
            S = np.linalg.lstsq(C, self.D, rcond=None)[0].T
        except np.linalg.LinAlgError:
            # 如果失败，使用正则化
            lambda_reg = 1e-6 * np.trace(C.T @ C) / self.n_components
            CtC_reg = C.T @ C + lambda_reg * np.eye(self.n_components)
            S = (np.linalg.solve(CtC_reg, C.T @ self.D)).T
        
        return C, S
