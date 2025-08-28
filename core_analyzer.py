#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core_analyzer.py - TAS MCR-ALS 核心分析接口

这个模块提供了统一的TAS MCR-ALS分析接口，可以被本地脚本、Flask应用等不同前端调用。
将核心算法与界面分离，便于维护和扩展。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import json
from datetime import datetime
import uuid

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from mcr.mcr_als import MCRALS
from data.data import read_file


class TASCoreAnalyzer:
    """
    TAS MCR-ALS 核心分析器
    
    提供统一的分析接口，支持多种前端调用方式
    """
    
    def __init__(self, language='chinese'):
        """
        初始化分析器
        
        参数:
        - language: 界面语言 ('chinese' 或 'english')
        """
        self.language = language
        self.setup_fonts()
        self.labels = self.get_labels()
        
        # 分析状态
        self.is_analyzed = False
        self.analysis_id = None
        
        # 数据存储
        self.D = None
        self.time_axis = None
        self.wavelength_axis = None
        
        # 分析结果
        self.mcr_solver = None
        self.C_resolved = None
        self.S_resolved = None
        
        # 分析参数
        self.analysis_params = {}
    
    def setup_fonts(self):
        """设置matplotlib字体"""
        if self.language == 'chinese':
            chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'DejaVu Sans']
            font_found = False
            for font in chinese_fonts:
                try:
                    rcParams['font.sans-serif'] = [font]
                    rcParams['axes.unicode_minus'] = False
                    test_font = fm.FontProperties(family=font)
                    if test_font.get_name() in [f.name for f in fm.fontManager.ttflist]:
                        font_found = True
                        break
                except:
                    continue
            
            if not font_found:
                rcParams['font.sans-serif'] = ['DejaVu Sans']
        else:
            rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
            rcParams['axes.unicode_minus'] = False
    
    def get_labels(self):
        """获取标签字典"""
        if self.language == 'chinese':
            return {
                'concentration_profiles': '浓度轮廓',
                'pure_spectra': '纯光谱',
                'lof_convergence': 'LOF收敛曲线',
                'residuals': '残差图',
                'original_data': '原始TAS数据',
                'reconstructed_data': '重构数据',
                'residuals_comparison': '残差对比',
                'time_delay': '时间延迟 (ps)',
                'wavelength': '波长 (nm)',
                'concentration': '浓度 (a.u.)',
                'absorption': 'ΔA (a.u.)',
                'iteration': '迭代次数',
                'lof_percent': 'LOF (%)',
                'wavelength_index': '波长索引',
                'time_index': '时间索引',
                'component': '组分',
                'delta_a': 'ΔA'
            }
        else:
            return {
                'concentration_profiles': 'Concentration Profiles',
                'pure_spectra': 'Pure Spectra',
                'lof_convergence': 'LOF Convergence',
                'residuals': 'Residuals',
                'original_data': 'Original TAS Data',
                'reconstructed_data': 'Reconstructed Data',
                'residuals_comparison': 'Residuals Comparison',
                'time_delay': 'Time Delay (ps)',
                'wavelength': 'Wavelength (nm)',
                'concentration': 'Concentration (a.u.)',
                'absorption': 'ΔA (a.u.)',
                'iteration': 'Iteration',
                'lof_percent': 'LOF (%)',
                'wavelength_index': 'Wavelength Index',
                'time_index': 'Time Index',
                'component': 'Component',
                'delta_a': 'ΔA'
            }
    
    def load_data(self, file_path, file_type="handle", wavelength_range=(400, 800), 
                  delay_range=(0, 10)):
        """
        加载TAS数据
        
        参数:
        - file_path: 数据文件路径
        - file_type: 文件类型 ("handle" 或 "raw")
        - wavelength_range: 波长范围 (min, max)
        - delay_range: 时间延迟范围 (min, max)
        
        返回:
        - dict: 包含状态和信息的字典
        """
        try:
            df = read_file(
                file_path, 
                file_type=file_type, 
                inf_handle=True,
                wavelength_range=wavelength_range, 
                delay_range=delay_range
            )
            
            if df is None:
                return {
                    'success': False,
                    'message': '数据加载失败，请检查文件路径和格式'
                }
            
            # 转换为矩阵格式
            self.D = df.values
            self.time_axis = df.index.values
            self.wavelength_axis = df.columns.values
            
            # 保存加载参数
            self.analysis_params = {
                'file_path': file_path,
                'file_type': file_type,
                'wavelength_range': wavelength_range,
                'delay_range': delay_range,
                'data_shape': self.D.shape,
                'time_range': [float(self.time_axis.min()), float(self.time_axis.max())],
                'wavelength_range_actual': [float(self.wavelength_axis.min()), float(self.wavelength_axis.max())]
            }
            
            # 数据质量检查
            warnings = []
            if np.any(np.isnan(self.D)):
                warnings.append("数据中包含NaN值")
            if np.any(np.isinf(self.D)):
                warnings.append("数据中包含无穷值")
            
            return {
                'success': True,
                'message': '数据加载成功',
                'data_info': {
                    'shape': self.D.shape,
                    'time_range': self.analysis_params['time_range'],
                    'wavelength_range': self.analysis_params['wavelength_range_actual']
                },
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'数据加载错误: {str(e)}'
            }
    
    def run_analysis(self, n_components=3, max_iter=200, tol=1e-7):
        """
        运行MCR-ALS分析
        
        参数:
        - n_components: 组分数量
        - max_iter: 最大迭代次数
        - tol: 收敛容差
        
        返回:
        - dict: 包含分析结果的字典
        """
        if self.D is None:
            return {
                'success': False,
                'message': '请先加载数据'
            }
        
        try:
            # 生成分析ID
            self.analysis_id = str(uuid.uuid4())[:8]
            
            # 更新分析参数
            self.analysis_params.update({
                'n_components': n_components,
                'max_iter': max_iter,
                'tol': tol,
                'analysis_id': self.analysis_id,
                'timestamp': datetime.now().isoformat()
            })
            
            # 运行MCR-ALS
            self.mcr_solver = MCRALS(
                n_components=n_components, 
                max_iter=max_iter, 
                tol=tol
            )
            self.mcr_solver.fit(self.D)
            
            # 获取结果
            self.C_resolved = self.mcr_solver.C_opt_
            self.S_resolved = self.mcr_solver.S_opt_
            self.is_analyzed = True
            
            # 更新最终结果参数
            self.analysis_params.update({
                'iterations': len(self.mcr_solver.lof_),
                'final_lof': float(self.mcr_solver.lof_[-1]),
                'converged': len(self.mcr_solver.lof_) < max_iter
            })
            
            return {
                'success': True,
                'message': '分析完成',
                'results': {
                    'analysis_id': self.analysis_id,
                    'iterations': self.analysis_params['iterations'],
                    'final_lof': self.analysis_params['final_lof'],
                    'converged': self.analysis_params['converged'],
                    'concentration_shape': self.C_resolved.shape,
                    'spectra_shape': self.S_resolved.shape
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'分析错误: {str(e)}'
            }
    
    def generate_plots(self, output_dir="results", save_format='png', dpi=300):
        """
        生成分析结果图表
        
        参数:
        - output_dir: 输出目录
        - save_format: 保存格式 ('png', 'svg', 'pdf')
        - dpi: 图片分辨率
        
        返回:
        - dict: 包含生成的图表文件路径
        """
        if not self.is_analyzed:
            return {
                'success': False,
                'message': '请先运行分析'
            }
        
        try:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            plot_files = {}
            
            # 1. 主要结果图表 (2x2)
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 浓度轮廓
            axes[0, 0].set_title(self.labels['concentration_profiles'])
            for i in range(self.analysis_params['n_components']):
                component_label = f"{self.labels['component']} {i+1}"
                axes[0, 0].plot(self.time_axis, self.C_resolved[:, i], 
                               label=component_label, linewidth=2)
            axes[0, 0].set_xlabel(self.labels['time_delay'])
            axes[0, 0].set_ylabel(self.labels['concentration'])
            axes[0, 0].set_xscale('log')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 纯光谱
            axes[0, 1].set_title(self.labels['pure_spectra'])
            for i in range(self.analysis_params['n_components']):
                component_label = f"{self.labels['component']} {i+1}"
                axes[0, 1].plot(self.wavelength_axis, self.S_resolved[:, i], 
                               label=component_label, linewidth=2)
            axes[0, 1].set_xlabel(self.labels['wavelength'])
            axes[0, 1].set_ylabel(self.labels['absorption'])
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # LOF收敛曲线
            axes[1, 0].set_title(self.labels['lof_convergence'])
            axes[1, 0].plot(self.mcr_solver.lof_, 'b-', linewidth=2)
            axes[1, 0].set_xlabel(self.labels['iteration'])
            axes[1, 0].set_ylabel(self.labels['lof_percent'])
            axes[1, 0].grid(True, alpha=0.3)
            
            # 残差图
            im = axes[1, 1].imshow(self.mcr_solver.residuals_, aspect='auto', cmap='coolwarm', 
                                   vmin=-np.max(np.abs(self.mcr_solver.residuals_)), 
                                   vmax=np.max(np.abs(self.mcr_solver.residuals_)),
                                   origin='lower')
            axes[1, 1].set_title(self.labels['residuals'])
            axes[1, 1].set_xlabel(self.labels['wavelength_index'])
            axes[1, 1].set_ylabel(self.labels['time_index'])
            fig.colorbar(im, ax=axes[1, 1])
            
            plt.tight_layout()
            main_plot_file = os.path.join(output_dir, f"mcr_results_{self.analysis_id}.{save_format}")
            plt.savefig(main_plot_file, dpi=dpi, bbox_inches='tight')
            plt.close()
            plot_files['main_results'] = main_plot_file
            
            # 2. 数据对比图表 (1x3)
            fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            
            # 原始数据
            im1 = ax1.imshow(self.D, aspect='auto', cmap='coolwarm', 
                            extent=[self.wavelength_axis.min(), self.wavelength_axis.max(),
                                   self.time_axis.min(), self.time_axis.max()],
                            origin='lower')
            ax1.set_title(self.labels['original_data'])
            ax1.set_xlabel(self.labels['wavelength'])
            ax1.set_ylabel(self.labels['time_delay'])
            fig2.colorbar(im1, ax=ax1, label=self.labels['delta_a'])
            
            # 重构数据
            D_reconstructed = self.C_resolved @ self.S_resolved.T
            im2 = ax2.imshow(D_reconstructed, aspect='auto', cmap='coolwarm',
                            extent=[self.wavelength_axis.min(), self.wavelength_axis.max(),
                                   self.time_axis.min(), self.time_axis.max()],
                            vmin=im1.get_clim()[0], vmax=im1.get_clim()[1],
                            origin='lower')
            ax2.set_title(self.labels['reconstructed_data'])
            ax2.set_xlabel(self.labels['wavelength'])
            ax2.set_ylabel(self.labels['time_delay'])
            fig2.colorbar(im2, ax=ax2, label=self.labels['delta_a'])
            
            # 残差对比
            residuals = self.D - D_reconstructed
            im3 = ax3.imshow(residuals, aspect='auto', cmap='coolwarm',
                            extent=[self.wavelength_axis.min(), self.wavelength_axis.max(),
                                   self.time_axis.min(), self.time_axis.max()],
                            vmin=-np.max(np.abs(residuals)), vmax=np.max(np.abs(residuals)),
                            origin='lower')
            ax3.set_title(self.labels['residuals_comparison'])
            ax3.set_xlabel(self.labels['wavelength'])
            ax3.set_ylabel(self.labels['time_delay'])
            fig2.colorbar(im3, ax=ax3, label=self.labels['delta_a'])
            
            plt.tight_layout()
            comparison_plot_file = os.path.join(output_dir, f"data_comparison_{self.analysis_id}.{save_format}")
            plt.savefig(comparison_plot_file, dpi=dpi, bbox_inches='tight')
            plt.close()
            plot_files['data_comparison'] = comparison_plot_file
            
            return {
                'success': True,
                'message': '图表生成成功',
                'plot_files': plot_files
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'图表生成错误: {str(e)}'
            }
    
    def save_results(self, output_dir="results"):
        """
        保存数值结果
        
        参数:
        - output_dir: 输出目录
        
        返回:
        - dict: 包含保存的文件路径
        """
        if not self.is_analyzed:
            return {
                'success': False,
                'message': '请先运行分析'
            }
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            result_files = {}
            
            # 保存浓度矩阵
            conc_file = os.path.join(output_dir, f"concentration_{self.analysis_id}.csv")
            conc_df = pd.DataFrame(
                self.C_resolved, 
                index=self.time_axis,
                columns=[f'Component_{i+1}' for i in range(self.analysis_params['n_components'])]
            )
            conc_df.to_csv(conc_file)
            result_files['concentration'] = conc_file
            
            # 保存光谱矩阵
            spectra_file = os.path.join(output_dir, f"spectra_{self.analysis_id}.csv")
            spectra_df = pd.DataFrame(
                self.S_resolved, 
                index=self.wavelength_axis,
                columns=[f'Component_{i+1}' for i in range(self.analysis_params['n_components'])]
            )
            spectra_df.to_csv(spectra_file)
            result_files['spectra'] = spectra_file
            
            # 保存LOF历史
            lof_file = os.path.join(output_dir, f"lof_history_{self.analysis_id}.csv")
            lof_df = pd.DataFrame({'LOF_percent': self.mcr_solver.lof_})
            lof_df.to_csv(lof_file, index_label='Iteration')
            result_files['lof_history'] = lof_file
            
            # 保存分析参数
            params_file = os.path.join(output_dir, f"parameters_{self.analysis_id}.json")
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_params, f, indent=2, ensure_ascii=False)
            result_files['parameters'] = params_file
            
            return {
                'success': True,
                'message': '结果保存成功',
                'result_files': result_files
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'结果保存错误: {str(e)}'
            }
    
    def get_analysis_summary(self):
        """
        获取分析摘要
        
        返回:
        - dict: 分析摘要信息
        """
        if not self.is_analyzed:
            return {
                'success': False,
                'message': '请先运行分析'
            }
        
        return {
            'success': True,
            'summary': {
                'analysis_id': self.analysis_id,
                'data_info': {
                    'shape': self.analysis_params['data_shape'],
                    'time_points': len(self.time_axis),
                    'wavelength_points': len(self.wavelength_axis),
                    'time_range': self.analysis_params['time_range'],
                    'wavelength_range': self.analysis_params['wavelength_range_actual']
                },
                'analysis_params': {
                    'n_components': self.analysis_params['n_components'],
                    'max_iter': self.analysis_params['max_iter'],
                    'tol': self.analysis_params['tol']
                },
                'results': {
                    'iterations': self.analysis_params['iterations'],
                    'final_lof': self.analysis_params['final_lof'],
                    'converged': self.analysis_params['converged']
                }
            }
        }


def analyze_tas_data(file_path, file_type="handle", wavelength_range=(400, 800), 
                    delay_range=(0, 10), n_components=3, max_iter=200, tol=1e-7,
                    language='chinese', output_dir="results", save_plots=True, 
                    save_results=True):
    """
    一站式TAS数据分析函数
    
    这是一个便捷函数，封装了完整的分析流程，适合脚本调用
    
    参数:
    - file_path: 数据文件路径
    - file_type: 文件类型
    - wavelength_range: 波长范围
    - delay_range: 时间延迟范围
    - n_components: 组分数量
    - max_iter: 最大迭代次数
    - tol: 收敛容差
    - language: 界面语言
    - output_dir: 输出目录
    - save_plots: 是否保存图表
    - save_results: 是否保存数值结果
    
    返回:
    - dict: 完整的分析结果
    """
    # 创建分析器
    analyzer = TASCoreAnalyzer(language=language)
    
    # 执行分析流程
    results = {
        'analysis_successful': False,
        'analyzer': analyzer
    }
    
    # 1. 加载数据
    load_result = analyzer.load_data(file_path, file_type, wavelength_range, delay_range)
    results['data_loading'] = load_result
    
    if not load_result['success']:
        return results
    
    # 2. 运行分析
    analysis_result = analyzer.run_analysis(n_components, max_iter, tol)
    results['analysis'] = analysis_result
    
    if not analysis_result['success']:
        return results
    
    # 3. 生成图表
    if save_plots:
        plot_result = analyzer.generate_plots(output_dir)
        results['plots'] = plot_result
    
    # 4. 保存结果
    if save_results:
        save_result = analyzer.save_results(output_dir)
        results['saved_files'] = save_result
    
    # 5. 获取摘要
    summary_result = analyzer.get_analysis_summary()
    results['summary'] = summary_result
    
    results['analysis_successful'] = True
    return results
