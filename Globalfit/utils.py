"""
utils.py - 工具函数

该模块提供全局拟合相关的工具函数，包括可视化、
数据处理和结果分析等。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from pathlib import Path


def plot_global_fit_results(results: Dict,
                            time_axis: np.ndarray,
                            wavelength_axis: np.ndarray,
                            save_path: Optional[str] = None,
                            show_plot: bool = True) -> None:
    """
    可视化全局拟合结果
    
    参数:
    - results: 全局拟合结果字典
    - time_axis: 时间轴数组
    - wavelength_axis: 波长轴数组
    - save_path: 保存路径 (可选)
    - show_plot: 是否显示图表
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    C_fit = results['C_fit']
    S_fit = results['S_fit']
    D_reconstructed = results['D_reconstructed']
    residuals = results['residuals']
    n_components = C_fit.shape[1]
    
    # 1. 浓度轮廓
    ax = axes[0, 0]
    for i in range(n_components):
        ax.plot(time_axis, C_fit[:, i], label=f'Component {i+1}', linewidth=2)
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Concentration (a.u.)', fontsize=12)
    ax.set_title('Concentration Profiles (Global Fit)', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 纯光谱
    ax = axes[0, 1]
    for i in range(n_components):
        ax.plot(wavelength_axis, S_fit[:, i], label=f'Component {i+1}', linewidth=2)
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Absorption (a.u.)', fontsize=12)
    ax.set_title('Species-Associated Spectra (Global Fit)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 重构数据
    ax = axes[0, 2]
    im = ax.imshow(D_reconstructed, aspect='auto', cmap='jet', origin='lower',
                   extent=[wavelength_axis[0], wavelength_axis[-1],
                          time_axis[0], time_axis[-1]])
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Time (ps)', fontsize=12)
    ax.set_title('Reconstructed Data (C @ S.T)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    plt.colorbar(im, ax=ax, label='ΔA')
    
    # 4. 残差分布
    ax = axes[1, 0]
    residuals_flat = residuals.ravel()
    ax.hist(residuals_flat, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Residual Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.grid(True, alpha=0.3)
    
    # 5. 残差图
    ax = axes[1, 1]
    vmax = np.max(np.abs(residuals))
    im = ax.imshow(residuals, aspect='auto', cmap='coolwarm', origin='lower',
                   vmin=-vmax, vmax=vmax,
                   extent=[wavelength_axis[0], wavelength_axis[-1],
                          time_axis[0], time_axis[-1]])
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Time (ps)', fontsize=12)
    ax.set_title('Residuals (Data - Fit)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    plt.colorbar(im, ax=ax, label='ΔA')
    
    # 6. 拟合统计
    ax = axes[1, 2]
    ax.axis('off')
    
    # 显示统计信息
    stats_text = "Fitting Statistics:\n\n"
    stats_text += f"LOF: {results['lof']:.4f}%\n"
    stats_text += f"Chi-Square: {results['chi_square']:.6e}\n"
    stats_text += f"Computation Time: {results['computation_time']:.2f} s\n\n"
    
    if 'tau_optimal' in results:
        stats_text += "Optimal Lifetimes (τ):\n"
        for i, tau in enumerate(results['tau_optimal']):
            stats_text += f"  τ{i+1} = {tau:.4f}\n"
    
    if 'k_optimal' in results:
        stats_text += "\nOptimal Rate Constants (k):\n"
        for i, k in enumerate(results['k_optimal']):
            stats_text += f"  k{i+1} = {k:.4e}\n"
    
    if 'kinetic_model' in results:
        stats_text += f"\nKinetic Model: {results['kinetic_model']}\n"
    
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def compare_mcr_and_global_fit(mcr_results: Dict,
                               global_results: Dict,
                               time_axis: np.ndarray,
                               wavelength_axis: np.ndarray,
                               save_path: Optional[str] = None,
                               show_plot: bool = True) -> None:
    """
    比较MCR-ALS和全局拟合的结果
    
    参数:
    - mcr_results: MCR-ALS结果字典 (包含C_mcr, S_mcr)
    - global_results: 全局拟合结果字典
    - time_axis: 时间轴数组
    - wavelength_axis: 波长轴数组
    - save_path: 保存路径 (可选)
    - show_plot: 是否显示图表
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    C_mcr = mcr_results['C_mcr']
    S_mcr = mcr_results['S_mcr']
    C_global = global_results['C_fit']
    S_global = global_results['S_fit']
    n_components = C_mcr.shape[1]
    
    # 1. 浓度轮廓对比
    ax = axes[0, 0]
    for i in range(n_components):
        ax.plot(time_axis, C_mcr[:, i], '--', label=f'MCR-ALS Comp {i+1}', 
                linewidth=2, alpha=0.7)
        ax.plot(time_axis, C_global[:, i], '-', label=f'Global Fit Comp {i+1}', 
                linewidth=2)
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Concentration (a.u.)', fontsize=12)
    ax.set_title('Concentration Profiles Comparison', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. 光谱对比
    ax = axes[0, 1]
    for i in range(n_components):
        ax.plot(wavelength_axis, S_mcr[:, i], '--', label=f'MCR-ALS Comp {i+1}', 
                linewidth=2, alpha=0.7)
        ax.plot(wavelength_axis, S_global[:, i], '-', label=f'Global Fit Comp {i+1}', 
                linewidth=2)
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Absorption (a.u.)', fontsize=12)
    ax.set_title('Spectra Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. 浓度轮廓差异
    ax = axes[1, 0]
    for i in range(n_components):
        # 归一化以便比较
        c_mcr_norm = C_mcr[:, i] / np.max(np.abs(C_mcr[:, i]))
        c_global_norm = C_global[:, i] / np.max(np.abs(C_global[:, i]))
        diff = c_global_norm - c_mcr_norm
        ax.plot(time_axis, diff, label=f'Component {i+1}', linewidth=2)
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Normalized Difference', fontsize=12)
    ax.set_title('Concentration Profile Differences', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. LOF比较
    ax = axes[1, 1]
    ax.axis('off')
    
    comparison_text = "Method Comparison:\n\n"
    comparison_text += "MCR-ALS:\n"
    comparison_text += f"  LOF: {mcr_results.get('mcr_lof', 'N/A')}\n\n"
    comparison_text += "Global Fit:\n"
    comparison_text += f"  LOF: {global_results['lof']:.4f}%\n"
    comparison_text += f"  Chi-Square: {global_results['chi_square']:.6e}\n"
    
    if 'tau_optimal' in global_results:
        comparison_text += f"\n  Lifetimes: {global_results['tau_optimal']}\n"
    
    if 'kinetic_model' in global_results:
        comparison_text += f"  Model: {global_results['kinetic_model']}\n"
    
    ax.text(0.1, 0.5, comparison_text, fontsize=12, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"比较图已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def calculate_correlation_matrix(C: np.ndarray) -> np.ndarray:
    """
    计算浓度矩阵的相关系数矩阵
    
    参数:
    - C: 浓度矩阵 (n_times × n_components)
    
    返回:
    - corr_matrix: 相关系数矩阵 (n_components × n_components)
    """
    corr_matrix = np.corrcoef(C.T)
    return corr_matrix


def estimate_uncertainty(fit_result, confidence_level: float = 0.95) -> Dict:
    """
    估计拟合参数的不确定度
    
    参数:
    - fit_result: lmfit拟合结果对象
    - confidence_level: 置信水平
    
    返回:
    - uncertainty_dict: 包含参数不确定度的字典
    """
    from scipy.stats import t
    
    uncertainty_dict = {}
    n_data = fit_result.ndata
    n_params = fit_result.nvarys
    dof = n_data - n_params  # 自由度
    
    # t分布的临界值
    t_value = t.ppf((1 + confidence_level) / 2, dof)
    
    for name, param in fit_result.params.items():
        if param.vary:
            stderr = param.stderr if param.stderr is not None else 0
            uncertainty = t_value * stderr
            uncertainty_dict[name] = {
                'value': param.value,
                'stderr': stderr,
                'uncertainty': uncertainty,
                'relative_uncertainty': uncertainty / param.value if param.value != 0 else np.inf
            }
    
    return uncertainty_dict


def export_results_to_txt(results: Dict, 
                          output_file: str,
                          include_matrices: bool = False) -> None:
    """
    将结果导出为文本文件
    
    参数:
    - results: 结果字典
    - output_file: 输出文件路径
    - include_matrices: 是否包含完整矩阵 (可能很大)
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("全局拟合结果报告\n")
        f.write("=" * 60 + "\n\n")
        
        # 统计信息
        f.write("拟合统计:\n")
        f.write(f"  LOF: {results['lof']:.4f}%\n")
        f.write(f"  Chi-Square: {results['chi_square']:.6e}\n")
        f.write(f"  计算时间: {results['computation_time']:.2f} 秒\n\n")
        
        # 寿命参数
        if 'tau_optimal' in results:
            f.write("最优寿命 (τ):\n")
            for i, tau in enumerate(results['tau_optimal']):
                f.write(f"  τ{i+1} = {tau:.6f}\n")
            f.write("\n")
        
        # 速率常数
        if 'k_optimal' in results:
            f.write("最优速率常数 (k):\n")
            for i, k in enumerate(results['k_optimal']):
                f.write(f"  k{i+1} = {k:.6e}\n")
            f.write("\n")
        
        # 动力学模型
        if 'kinetic_model' in results:
            f.write(f"动力学模型: {results['kinetic_model']}\n\n")
        
        # 矩阵形状
        f.write("数据形状:\n")
        f.write(f"  浓度矩阵 C: {results['C_fit'].shape}\n")
        f.write(f"  光谱矩阵 S: {results['S_fit'].shape}\n")
        f.write(f"  重构数据 D: {results['D_reconstructed'].shape}\n\n")
        
        # 完整拟合报告
        if 'fit_result' in results:
            from lmfit import fit_report
            f.write("=" * 60 + "\n")
            f.write("详细拟合报告:\n")
            f.write("=" * 60 + "\n")
            f.write(fit_report(results['fit_result']))
            f.write("\n")
    
    print(f"结果已导出到: {output_file}")
