"""
interface.py - MCR-ALS输出接口

该模块提供了从MCR-ALS结果自动化转换到全局拟合输入的接口，
实现无缝集成的工作流程。
"""

import numpy as np
import json
import os
from typing import Dict, Tuple, Optional, List
from pathlib import Path


class MCRALSInterface:
    """
    MCR-ALS与全局拟合的接口类
    
    该类负责:
    1. 读取MCR-ALS的输出结果
    2. 将结果转换为全局拟合所需的格式
    3. 提供初始参数估计
    """
    
    def __init__(self, results_dir: str):
        """
        初始化接口
        
        参数:
        - results_dir: MCR-ALS结果目录路径
        """
        self.results_dir = Path(results_dir)
        
        # MCR-ALS结果
        self.C_mcr = None  # 浓度矩阵
        self.S_mcr = None  # 光谱矩阵
        self.time_axis = None  # 时间轴
        self.wavelength_axis = None  # 波长轴
        self.lof_history = None  # LOF历史
        self.parameters = None  # 分析参数
        
        # 原始数据
        self.D_original = None  # 原始数据矩阵
        
    def load_mcr_results(self) -> bool:
        """
        从文件加载MCR-ALS结果
        
        返回:
        - success: 是否成功加载
        """
        try:
            # 加载浓度矩阵
            c_file = self.results_dir / "concentration_profiles.csv"
            if c_file.exists():
                self.C_mcr = np.loadtxt(c_file, delimiter=',', skiprows=1)
                print(f"已加载浓度矩阵: {self.C_mcr.shape}")
            else:
                print(f"警告: 未找到浓度矩阵文件 {c_file}")
                return False
            
            # 加载光谱矩阵
            s_file = self.results_dir / "pure_spectra.csv"
            if s_file.exists():
                self.S_mcr = np.loadtxt(s_file, delimiter=',', skiprows=1)
                print(f"已加载光谱矩阵: {self.S_mcr.shape}")
            else:
                print(f"警告: 未找到光谱矩阵文件 {s_file}")
                return False
            
            # 加载LOF历史
            lof_file = self.results_dir / "lof_history.csv"
            if lof_file.exists():
                self.lof_history = np.loadtxt(lof_file, delimiter=',', skiprows=1)
                print(f"已加载LOF历史: 最终LOF = {self.lof_history[-1]:.4f}%")
            
            # 加载参数信息
            param_file = self.results_dir / "analysis_parameters.json"
            if param_file.exists():
                with open(param_file, 'r') as f:
                    self.parameters = json.load(f)
                print(f"已加载分析参数")
            
            return True
            
        except Exception as e:
            print(f"加载MCR-ALS结果失败: {e}")
            return False
    
    def load_original_data(self, data_file: str, 
                          file_type: str = "handle") -> bool:
        """
        加载原始TAS数据
        
        参数:
        - data_file: 数据文件路径
        - file_type: 文件类型 ("raw" 或 "handle")
        
        返回:
        - success: 是否成功加载
        """
        try:
            # 导入数据读取模块
            import sys
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, project_root)
            
            from data.data import read_file
            
            # 读取数据
            self.D_original, self.time_axis, self.wavelength_axis = read_file(
                data_file, 
                file_type=file_type
            )
            
            print(f"已加载原始数据: {self.D_original.shape}")
            print(f"时间范围: {self.time_axis[0]:.2f} - {self.time_axis[-1]:.2f}")
            print(f"波长范围: {self.wavelength_axis[0]:.2f} - {self.wavelength_axis[-1]:.2f}")
            
            return True
            
        except Exception as e:
            print(f"加载原始数据失败: {e}")
            return False
    
    def estimate_lifetimes_from_mcr(self) -> List[float]:
        """
        从MCR-ALS浓度轮廓估计初始寿命
        
        该方法通过拟合指数衰减来估计每个组分的特征寿命，
        用作全局拟合的初始值。
        
        返回:
        - lifetimes: 估计的寿命列表
        """
        if self.C_mcr is None or self.time_axis is None:
            raise ValueError("请先加载MCR-ALS结果和时间轴")
        
        from scipy.optimize import curve_fit
        
        n_components = self.C_mcr.shape[1]
        lifetimes = []
        
        print("\n从MCR-ALS浓度轮廓估计初始寿命:")
        
        for i in range(n_components):
            # 获取该组分的浓度轮廓
            c_profile = self.C_mcr[:, i]
            
            # 找到峰值位置
            peak_idx = np.argmax(c_profile)
            peak_value = c_profile[peak_idx]
            
            # 使用峰值后的数据拟合指数衰减
            if peak_idx < len(c_profile) - 1:
                t_fit = self.time_axis[peak_idx:]
                c_fit = c_profile[peak_idx:]
                
                # 归一化
                c_fit_norm = c_fit / peak_value
                
                # 定义指数衰减函数
                def exp_decay(t, tau):
                    return np.exp(-(t - t_fit[0]) / tau)
                
                try:
                    # 初始猜测：半衰期
                    half_idx = peak_idx + np.argmin(np.abs(c_fit - peak_value/2))
                    tau_guess = self.time_axis[half_idx] - self.time_axis[peak_idx]
                    tau_guess = max(tau_guess, 0.1)  # 确保正值
                    
                    # 拟合
                    popt, _ = curve_fit(exp_decay, t_fit, c_fit_norm, 
                                       p0=[tau_guess],
                                       bounds=(0.01, 1e6),
                                       maxfev=1000)
                    tau = popt[0]
                except:
                    # 如果拟合失败，使用简单估计
                    tau = (self.time_axis[-1] - self.time_axis[peak_idx]) / 2
                
            else:
                # 如果峰值在最后，使用总时间的一半
                tau = self.time_axis[-1] / 2
            
            lifetimes.append(tau)
            print(f"  组分 {i+1}: τ ≈ {tau:.4f}")
        
        return lifetimes
    
    def estimate_rate_constants_from_lifetimes(self, 
                                               lifetimes: List[float]) -> List[float]:
        """
        从寿命估计速率常数
        
        参数:
        - lifetimes: 寿命列表
        
        返回:
        - rate_constants: 速率常数列表 (k = 1/τ)
        """
        rate_constants = [1.0/tau if tau > 0 else 1e-6 for tau in lifetimes]
        
        print("\n从寿命估计速率常数:")
        for i, (tau, k) in enumerate(zip(lifetimes, rate_constants)):
            print(f"  组分 {i+1}: τ = {tau:.4f} → k = {k:.4e}")
        
        return rate_constants
    
    def prepare_for_global_fitting(self, 
                                   data_file: Optional[str] = None,
                                   file_type: str = "handle") -> Dict:
        """
        准备全局拟合所需的所有数据
        
        参数:
        - data_file: 原始数据文件路径 (可选)
        - file_type: 文件类型
        
        返回:
        - data_dict: 包含所有必要数据的字典
        """
        # 加载MCR-ALS结果
        if not self.load_mcr_results():
            raise RuntimeError("无法加载MCR-ALS结果")
        
        # 加载原始数据
        if data_file is not None:
            if not self.load_original_data(data_file, file_type):
                print("警告: 无法加载原始数据，将使用MCR-ALS重构的数据")
                self.D_original = self.C_mcr @ self.S_mcr.T
        else:
            # 使用MCR-ALS重构的数据
            print("未提供原始数据文件，使用MCR-ALS重构的数据")
            self.D_original = self.C_mcr @ self.S_mcr.T
        
        # 如果没有时间轴和波长轴，创建索引数组
        if self.time_axis is None:
            self.time_axis = np.arange(self.C_mcr.shape[0])
            print("警告: 未找到时间轴，使用索引数组")
        
        if self.wavelength_axis is None:
            self.wavelength_axis = np.arange(self.S_mcr.shape[0])
            print("警告: 未找到波长轴，使用索引数组")
        
        # 估计初始参数
        lifetimes_initial = self.estimate_lifetimes_from_mcr()
        rate_constants_initial = self.estimate_rate_constants_from_lifetimes(lifetimes_initial)
        
        # 组织数据
        data_dict = {
            'data_matrix': self.D_original,
            'time_axis': self.time_axis,
            'wavelength_axis': self.wavelength_axis,
            'n_components': self.C_mcr.shape[1],
            'C_mcr': self.C_mcr,
            'S_mcr': self.S_mcr,
            'lifetimes_initial': lifetimes_initial,
            'rate_constants_initial': rate_constants_initial,
            'mcr_lof': self.lof_history[-1] if self.lof_history is not None else None,
            'mcr_parameters': self.parameters
        }
        
        print("\n数据准备完成!")
        print(f"  数据矩阵形状: {data_dict['data_matrix'].shape}")
        print(f"  组分数量: {data_dict['n_components']}")
        print(f"  初始寿命: {lifetimes_initial}")
        
        return data_dict
    
    def save_global_fit_results(self, results: Dict, 
                               output_dir: Optional[str] = None) -> None:
        """
        保存全局拟合结果
        
        参数:
        - results: 全局拟合结果字典
        - output_dir: 输出目录 (如果为None，使用results_dir)
        """
        if output_dir is None:
            output_dir = self.results_dir / "global_fit"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n保存全局拟合结果到: {output_dir}")
        
        # 保存浓度矩阵
        if 'C_fit' in results:
            np.savetxt(output_dir / "concentration_global_fit.csv",
                      results['C_fit'], delimiter=',',
                      header=','.join([f'Component_{i+1}' 
                                      for i in range(results['C_fit'].shape[1])]))
            print("  已保存: concentration_global_fit.csv")
        
        # 保存光谱矩阵
        if 'S_fit' in results:
            np.savetxt(output_dir / "spectra_global_fit.csv",
                      results['S_fit'], delimiter=',',
                      header=','.join([f'Component_{i+1}' 
                                      for i in range(results['S_fit'].shape[1])]))
            print("  已保存: spectra_global_fit.csv")
        
        # 保存重构数据
        if 'D_reconstructed' in results:
            np.savetxt(output_dir / "data_reconstructed.csv",
                      results['D_reconstructed'], delimiter=',')
            print("  已保存: data_reconstructed.csv")
        
        # 保存残差
        if 'residuals' in results:
            np.savetxt(output_dir / "residuals.csv",
                      results['residuals'], delimiter=',')
            print("  已保存: residuals.csv")
        
        # 保存参数和统计信息
        summary = {
            'chi_square': float(results.get('chi_square', 0)),
            'lof': float(results.get('lof', 0)),
            'computation_time': float(results.get('computation_time', 0))
        }
        
        if 'tau_optimal' in results:
            summary['tau_optimal'] = [float(t) for t in results['tau_optimal']]
        
        if 'k_optimal' in results:
            summary['k_optimal'] = [float(k) for k in results['k_optimal']]
        
        if 'kinetic_model' in results:
            summary['kinetic_model'] = results['kinetic_model']
        
        with open(output_dir / "global_fit_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print("  已保存: global_fit_summary.json")
        
        # 保存拟合报告
        if 'fit_result' in results:
            from lmfit import fit_report
            report = fit_report(results['fit_result'])
            with open(output_dir / "fit_report.txt", 'w') as f:
                f.write(report)
            print("  已保存: fit_report.txt")
        
        print("全局拟合结果保存完成!")
