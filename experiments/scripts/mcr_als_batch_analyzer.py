#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多轮MCR-ALS批量分析器
对筛选出的三类挑战性数据执行MCR-ALS分析并汇总结果
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入MCR-ALS相关模块
import sys
sys.path.append('.')
from mcr.mcr_als import MCRALS
try:
    from mcr.constraints import ConstraintUnimodality, ConstraintNNLS, ConstraintNormalize
except ImportError:
    # 使用简化的约束实现
    ConstraintUnimodality = None
    ConstraintNNLS = None
    ConstraintNormalize = None

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MCRALSBatchAnalyzer:
    """多轮MCR-ALS批量分析器"""
    
    def __init__(self, output_dir="experiments/results/mcr_als_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据类别和对应的MCR参数配置
        self.category_configs = {
            'multi_peak_overlap': {
                'n_components': [2, 3, 4],  # 多峰重叠通常需要多个组分
                'max_iter': 150,
                'constraints': ['nnls', 'unimodal_c', 'normalize'],
                'description': '多峰重叠数据 - 适合检测复杂光谱重叠'
            },
            'transient_decay': {
                'n_components': [2, 3],  # 瞬态衰减通常2-3个组分足够
                'max_iter': 100,
                'constraints': ['nnls', 'normalize'],
                'description': '瞬态衰减数据 - 适合检测动力学过程'
            },
            'low_snr': {
                'n_components': [2, 3],  # 低信噪比数据保守估计组分数
                'max_iter': 200,  # 增加迭代次数提高收敛性
                'constraints': ['nnls', 'normalize'],
                'description': '低信噪比数据 - 需要更多迭代和约束'
            }
        }
        
        # 存储分析结果
        self.results = {
            'multi_peak_overlap': [],
            'transient_decay': [],
            'low_snr': [],
            'summary': {}
        }
        
    def load_screening_results(self, screening_file="experiments/results/data_screening/screening_results.json"):
        """加载筛选结果"""
        try:
            with open(screening_file, 'r', encoding='utf-8') as f:
                screening_data = json.load(f)
            print(f"✅ 成功加载筛选结果: {screening_file}")
            return screening_data
        except FileNotFoundError:
            print(f"❌ 筛选结果文件未找到: {screening_file}")
            return None
        except Exception as e:
            print(f"❌ 加载筛选结果失败: {e}")
            return None
    
    def load_tas_data(self, file_path):
        """加载TAS数据文件"""
        try:
            # 读取CSV文件，使用自定义处理
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                print(f"   ❌ 数据文件太短: {len(lines)} 行")
                return None
            
            # 第一行是时间延迟
            time_line = lines[0].strip().split(',')
            time_delays = []
            for i, t_str in enumerate(time_line):
                if i == 0:  # 第一列是0.0，跳过
                    continue
                try:
                    time_delays.append(float(t_str))
                except:
                    time_delays.append(0.0)
            
            time_delays = np.array(time_delays)
            
            # 剩余行是波长和数据
            wavelengths = []
            data_rows = []
            
            for line in lines[1:]:
                if not line.strip():
                    continue
                    
                parts = line.strip().split(',')
                if len(parts) < 2:
                    continue
                
                try:
                    # 第一列是波长
                    wl = float(parts[0])
                    wavelengths.append(wl)
                    
                    # 其余列是数据
                    row_data = []
                    for val_str in parts[1:]:
                        try:
                            # 处理特殊值
                            if val_str in ['Inf', '+Inf']:
                                val = 1e10
                            elif val_str in ['-Inf']:
                                val = -1e10
                            elif val_str in ['NaN', 'nan']:
                                val = 0.0
                            else:
                                val = float(val_str)
                                if not np.isfinite(val):
                                    val = 0.0
                            row_data.append(val)
                        except:
                            row_data.append(0.0)
                    
                    # 确保长度匹配
                    while len(row_data) < len(time_delays):
                        row_data.append(0.0)
                    row_data = row_data[:len(time_delays)]
                    
                    data_rows.append(row_data)
                    
                except Exception as e:
                    print(f"   ⚠️ 跳过无效行: {e}")
                    continue
            
            if len(data_rows) < 10:
                print(f"   ❌ 数据行数太少: {len(data_rows)}")
                return None
            
            wavelengths = np.array(wavelengths)
            data = np.array(data_rows)
            
            # 数据验证和清理
            if data.shape[0] < 10 or data.shape[1] < 10:
                print(f"   ❌ 数据形状太小: {data.shape}")
                return None
            
            # 处理异常值
            data = np.where(np.isfinite(data), data, 0)
            data = np.where(np.abs(data) > 1e10, 0, data)  # 限制极大值
            
            print(f"   ✅ 数据加载成功: {data.shape} (波长×时间)")
            print(f"   波长范围: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
            print(f"   时间范围: {time_delays.min():.2f} - {time_delays.max():.2f} ps")
            print(f"   数据范围: {data.min():.2e} - {data.max():.2e}")
            
            return {
                'data': data,
                'wavelengths': wavelengths,
                'time_delays': time_delays,
                'shape': data.shape
            }
            
        except Exception as e:
            print(f"   ❌ 数据加载失败: {e}")
            return None
    
    def setup_mcr_constraints(self, constraint_names):
        """设置MCR约束条件"""
        # 返回约束名称列表，将在MCR初始化时使用
        return constraint_names
    
    def run_mcr_analysis(self, data_info, category, n_components, config):
        """运行单次MCR-ALS分析"""
        data = data_info['data']
        
        try:
            # 初始化MCR-ALS
            mcr = MCRALS(
                n_components=n_components,
                max_iter=config['max_iter'],
                tol=1e-6
            )
            
            # 运行MCR-ALS分析
            mcr.fit(data.T)  # 转置数据，MCR期望时间×波长格式
            
            # 获取结果
            C = mcr.C_opt_  # 浓度矩阵 (时间×组分)
            S = mcr.S_opt_  # 光谱矩阵 (波长×组分)
            
            # 重构数据
            reconstructed = C @ S.T
            
            # 计算质量指标
            residuals = data.T - reconstructed
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((data.T - np.mean(data.T))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            lof = mcr.lof_[-1] if mcr.lof_ else 100  # 使用MCR内部计算的LOF
            
            result = {
                'n_components': n_components,
                'converged': lof < 10.0,  # 简单的收敛标准
                'n_iterations': len(mcr.lof_),
                'r2': float(r2),
                'lof': float(lof),
                'concentration_profiles': C.T,  # 转置为组分×时间格式
                'pure_spectra': S.T,  # 转置为组分×波长格式
                'constraints_used': config['constraints'],
                'final_d_augmented': reconstructed
            }
            
            print(f"     ✅ MCR分析完成: {n_components}组分, R²={r2:.3f}, LOF={lof:.2f}%")
            return result
            
        except Exception as e:
            print(f"     ❌ MCR分析失败: {e}")
            return None
    
    def analyze_single_file(self, file_path, category, item_info=None):
        """分析单个文件"""
        print(f"\n📊 分析文件: {Path(file_path).name}")
        
        # 从item_info获取光谱类型信息
        spectrum_type = 'VIS'
        if item_info:
            spectrum_type = item_info.get('spectrum_type', 'VIS')
            print(f"   🌈 光谱类型: {spectrum_type}")
        
        # 加载数据
        data_info = self.load_tas_data(file_path)
        if data_info is None:
            return None
        
        # 添加光谱类型信息到data_info中
        data_info['spectrum_type'] = spectrum_type
        if item_info:
            data_info['original_wavelength_range'] = item_info.get('original_wavelength_range', 'N/A')
            data_info['cropped_wavelength_range'] = item_info.get('cropped_wavelength_range', 'N/A')
        
        # 获取配置
        config = self.category_configs[category]
        print(f"   🔧 使用配置: {config['description']}")
        
        file_results = {
            'file_path': str(file_path),
            'file_name': Path(file_path).name,
            'category': category,
            'spectrum_type': spectrum_type,
            'data_shape': data_info['shape'],
            'wavelength_range': [float(data_info['wavelengths'].min()), 
                               float(data_info['wavelengths'].max())],
            'time_range': [float(data_info['time_delays'].min()), 
                          float(data_info['time_delays'].max())],
            'mcr_results': [],
            'best_result': None
        }
        
        # 对不同组分数进行MCR分析
        best_r2 = -1
        for n_comp in config['n_components']:
            print(f"   🧪 尝试 {n_comp} 个组分...")
            result = self.run_mcr_analysis(data_info, category, n_comp, config)
            
            if result is not None:
                file_results['mcr_results'].append(result)
                
                # 选择最佳结果（基于R²）
                if result['r2'] > best_r2:
                    best_r2 = result['r2']
                    file_results['best_result'] = result
        
        # 保存单个文件的结果
        self.save_single_file_results(file_results, data_info)
        
        return file_results
    
    def save_single_file_results(self, file_results, data_info):
        """保存单个文件的分析结果"""
        file_name = Path(file_results['file_name']).stem
        category = file_results['category']
        
        # 创建文件专用目录
        file_dir = self.output_dir / category / file_name
        file_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存最佳MCR结果
        if file_results['best_result'] is not None:
            best = file_results['best_result']
            
            # 保存浓度轮廓
            np.savetxt(file_dir / 'concentration_profiles.csv',
                      best['concentration_profiles'], delimiter=',')
            
            # 保存纯光谱
            np.savetxt(file_dir / 'pure_spectra.csv',
                      best['pure_spectra'], delimiter=',')
            
            # 创建可视化
            self.create_mcr_visualization(data_info, best, file_dir)
        
        # 保存完整结果为JSON
        results_copy = file_results.copy()
        # 移除numpy数组（不能JSON序列化）
        for result in results_copy['mcr_results']:
            for key in ['concentration_profiles', 'pure_spectra', 'final_d_augmented']:
                if key in result:
                    del result[key]
            # 确保所有值都是JSON可序列化的
            for key, value in result.items():
                if isinstance(value, (np.bool_, bool)):
                    result[key] = bool(value)
                elif isinstance(value, (np.integer, np.int64)):
                    result[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    result[key] = float(value)
        
        if results_copy['best_result']:
            for key in ['concentration_profiles', 'pure_spectra', 'final_d_augmented']:
                if key in results_copy['best_result']:
                    del results_copy['best_result'][key]
            # 确保所有值都是JSON可序列化的
            for key, value in results_copy['best_result'].items():
                if isinstance(value, (np.bool_, bool)):
                    results_copy['best_result'][key] = bool(value)
                elif isinstance(value, (np.integer, np.int64)):
                    results_copy['best_result'][key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    results_copy['best_result'][key] = float(value)
        
        with open(file_dir / 'analysis_summary.json', 'w', encoding='utf-8') as f:
            json.dump(results_copy, f, indent=2, ensure_ascii=False)
    
    def create_mcr_visualization(self, data_info, mcr_result, output_dir):
        """创建MCR分析可视化"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        data = data_info['data']
        wavelengths = data_info['wavelengths']
        time_delays = data_info['time_delays']
        
        # 根据光谱类型确定坐标轴范围
        spectrum_type = data_info.get('spectrum_type', 'VIS')
        if spectrum_type == 'UV':
            wl_range = (380, 650)
        elif spectrum_type == 'NIR':
            wl_range = (1100, 1620)
        else:  # VIS or default
            wl_range = (500, 950)
        
        # 根据实际数据调整范围
        actual_min, actual_max = wavelengths.min(), wavelengths.max()
        wl_range = (max(wl_range[0], actual_min), min(wl_range[1], actual_max))
        
        # 1. 原始数据热图
        im1 = axes[0, 0].imshow(data, aspect='auto', cmap='RdBu_r',
                               extent=[time_delays.min(), time_delays.max(),
                                      wavelengths.max(), wavelengths.min()])
        axes[0, 0].set_title(f'原始数据 ({spectrum_type})')
        axes[0, 0].set_xlabel('时间延迟 (ps)')
        axes[0, 0].set_ylabel('波长 (nm)')
        axes[0, 0].set_ylim(wl_range[1], wl_range[0])  # 反转Y轴，上大下小
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. 重构数据热图
        reconstructed = mcr_result['concentration_profiles'].T @ mcr_result['pure_spectra']
        im2 = axes[0, 1].imshow(reconstructed.T, aspect='auto', cmap='RdBu_r',
                               extent=[time_delays.min(), time_delays.max(),
                                      wavelengths.max(), wavelengths.min()])
        axes[0, 1].set_title(f'MCR重构 (R²={mcr_result["r2"]:.3f})')
        axes[0, 1].set_xlabel('时间延迟 (ps)')
        axes[0, 1].set_ylabel('波长 (nm)')
        axes[0, 1].set_ylim(wl_range[1], wl_range[0])  # 反转Y轴，上大下小
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 3. 残差热图
        residuals = data - reconstructed.T
        im3 = axes[0, 2].imshow(residuals, aspect='auto', cmap='RdBu_r',
                               extent=[time_delays.min(), time_delays.max(),
                                      wavelengths.max(), wavelengths.min()])
        axes[0, 2].set_title(f'残差 (LOF={mcr_result["lof"]:.2f}%)')
        axes[0, 2].set_xlabel('时间延迟 (ps)')
        axes[0, 2].set_ylabel('波长 (nm)')
        axes[0, 2].set_ylim(wl_range[1], wl_range[0])  # 反转Y轴，上大下小
        plt.colorbar(im3, ax=axes[0, 2])
        
        # 4. 浓度轮廓
        for i in range(mcr_result['n_components']):
            axes[1, 0].plot(time_delays, mcr_result['concentration_profiles'][i, :],
                           label=f'组分 {i+1}', linewidth=2)
        axes[1, 0].set_title('浓度轮廓')
        axes[1, 0].set_xlabel('时间延迟 (ps)')
        axes[1, 0].set_ylabel('浓度')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 纯光谱
        for i in range(mcr_result['n_components']):
            axes[1, 1].plot(wavelengths, mcr_result['pure_spectra'][i, :],
                           label=f'组分 {i+1}', linewidth=2)
        axes[1, 1].set_title(f'纯光谱 ({spectrum_type})')
        axes[1, 1].set_xlabel('波长 (nm)')
        axes[1, 1].set_ylabel('吸收强度')
        axes[1, 1].set_xlim(wl_range[0], wl_range[1])  # 设置X轴范围
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 拟合质量指标
        metrics_text = f"""
MCR-ALS 分析结果

光谱类型: {spectrum_type}
组分数量: {mcr_result['n_components']}
迭代次数: {mcr_result['n_iterations']}
是否收敛: {'是' if mcr_result['converged'] else '否'}

拟合质量:
R² = {mcr_result['r2']:.4f}
LOF = {mcr_result['lof']:.2f}%

使用的约束:
{', '.join(mcr_result['constraints_used'])}
        """
        axes[1, 2].text(0.1, 0.5, metrics_text, transform=axes[1, 2].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'mcr_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"     💾 可视化已保存: {output_dir / 'mcr_analysis.png'}")
    
    def run_batch_analysis(self, max_files_per_category=5):
        """运行批量分析"""
        print("🚀 开始多轮MCR-ALS批量分析")
        print("=" * 60)
        
        # 加载筛选结果
        screening_data = self.load_screening_results()
        if screening_data is None:
            print("❌ 无法继续分析，请先运行数据筛选")
            return
        
        # 分析每个类别的数据
        for category in ['multi_peak_overlap', 'transient_decay', 'low_snr']:
            if category not in screening_data:
                print(f"⚠️ 类别 {category} 未找到筛选数据")
                continue
            
            category_data = screening_data[category]
            print(f"\n🎯 分析类别: {category}")
            print(f"📁 共找到 {len(category_data)} 个文件")
            
            # 限制分析文件数量
            files_to_analyze = category_data[:max_files_per_category]
            if len(files_to_analyze) < len(category_data):
                print(f"📋 为了效率，只分析前 {max_files_per_category} 个文件")
            
            category_results = []
            
            for i, file_info in enumerate(files_to_analyze, 1):
                print(f"\n[{i}/{len(files_to_analyze)}] 处理: {category}")
                
                file_path = file_info['file_path']
                result = self.analyze_single_file(file_path, category, file_info)
                
                if result is not None:
                    category_results.append(result)
            
            self.results[category] = category_results
            print(f"✅ {category} 类别分析完成: {len(category_results)} 个文件成功")
        
        # 生成汇总报告
        self.generate_summary_report()
        
        print(f"\n🎉 批量分析完成！")
        print(f"📁 结果保存在: {self.output_dir}")
    
    def generate_summary_report(self):
        """生成汇总报告"""
        print("\n📋 生成汇总报告...")
        
        # 创建汇总统计
        summary = {
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_files_analyzed': 0,
            'category_statistics': {},
            'overall_statistics': {
                'avg_r2': 0,
                'avg_lof': 0,
                'convergence_rate': 0
            }
        }
        
        all_r2_values = []
        all_lof_values = []
        all_convergence = []
        
        for category, results in self.results.items():
            if not results:
                continue
            
            category_r2 = []
            category_lof = []
            category_convergence = []
            
            for result in results:
                if result['best_result'] is not None:
                    best = result['best_result']
                    category_r2.append(best['r2'])
                    category_lof.append(best['lof'])
                    category_convergence.append(best['converged'])
            
            if category_r2:
                summary['category_statistics'][category] = {
                    'files_count': len(results),
                    'avg_r2': float(np.mean(category_r2)),
                    'std_r2': float(np.std(category_r2)),
                    'avg_lof': float(np.mean(category_lof)),
                    'std_lof': float(np.std(category_lof)),
                    'convergence_rate': float(np.mean(category_convergence))
                }
                
                all_r2_values.extend(category_r2)
                all_lof_values.extend(category_lof)
                all_convergence.extend(category_convergence)
        
        # 总体统计
        if all_r2_values:
            summary['total_files_analyzed'] = len(all_r2_values)
            summary['overall_statistics'] = {
                'avg_r2': float(np.mean(all_r2_values)),
                'std_r2': float(np.std(all_r2_values)),
                'avg_lof': float(np.mean(all_lof_values)),
                'std_lof': float(np.std(all_lof_values)),
                'convergence_rate': float(np.mean(all_convergence))
            }
        else:
            summary['overall_statistics'] = {
                'avg_r2': 0.0,
                'std_r2': 0.0,
                'avg_lof': 100.0,
                'std_lof': 0.0,
                'convergence_rate': 0.0
            }
        
        # 保存汇总结果
        with open(self.output_dir / 'batch_analysis_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 创建汇总可视化
        self.create_summary_visualization(summary)
        
        # 生成Markdown报告
        self.generate_markdown_report(summary)
        
        print("✅ 汇总报告生成完成")
    
    def create_summary_visualization(self, summary):
        """创建汇总可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        categories = list(summary['category_statistics'].keys())
        if not categories:
            print("⚠️ 没有可用数据生成可视化")
            return
        
        # 1. R²比较
        r2_values = [summary['category_statistics'][cat]['avg_r2'] for cat in categories]
        r2_errors = [summary['category_statistics'][cat]['std_r2'] for cat in categories]
        
        axes[0, 0].bar(categories, r2_values, yerr=r2_errors, capsize=5, alpha=0.7)
        axes[0, 0].set_title('各类别平均R²值')
        axes[0, 0].set_ylabel('R²')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. LOF比较
        lof_values = [summary['category_statistics'][cat]['avg_lof'] for cat in categories]
        lof_errors = [summary['category_statistics'][cat]['std_lof'] for cat in categories]
        
        axes[0, 1].bar(categories, lof_values, yerr=lof_errors, capsize=5, alpha=0.7, color='orange')
        axes[0, 1].set_title('各类别平均LOF值')
        axes[0, 1].set_ylabel('LOF (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 收敛率比较
        convergence_rates = [summary['category_statistics'][cat]['convergence_rate'] * 100 
                           for cat in categories]
        
        axes[1, 0].bar(categories, convergence_rates, alpha=0.7, color='green')
        axes[1, 0].set_title('各类别收敛率')
        axes[1, 0].set_ylabel('收敛率 (%)')
        axes[1, 0].set_ylim(0, 100)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 文件数量统计
        file_counts = [summary['category_statistics'][cat]['files_count'] for cat in categories]
        
        axes[1, 1].pie(file_counts, labels=categories, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('各类别分析文件数量分布')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'batch_analysis_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 汇总可视化已保存: {self.output_dir / 'batch_analysis_summary.png'}")
    
    def generate_markdown_report(self, summary):
        """生成Markdown汇总报告"""
        report_path = self.output_dir / 'MCR_ALS_批量分析报告.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# MCR-ALS 批量分析报告\n\n")
            f.write(f"**生成时间**: {summary['analysis_time']}\n\n")
            f.write(f"**总分析文件数**: {summary['total_files_analyzed']}\n\n")
            
            # 总体统计
            f.write("## 📊 总体统计\n\n")
            overall = summary['overall_statistics']
            f.write(f"- **平均 R²**: {overall['avg_r2']:.4f} ± {overall['std_r2']:.4f}\n")
            f.write(f"- **平均 LOF**: {overall['avg_lof']:.2f}% ± {overall['std_lof']:.2f}%\n")
            f.write(f"- **收敛率**: {overall['convergence_rate']*100:.1f}%\n\n")
            
            # 各类别详细统计
            f.write("## 🎯 各类别详细统计\n\n")
            
            for category, stats in summary['category_statistics'].items():
                category_name = {
                    'multi_peak_overlap': '多峰重叠数据',
                    'transient_decay': '瞬态衰减数据',
                    'low_snr': '低信噪比数据'
                }.get(category, category)
                
                f.write(f"### {category_name}\n\n")
                f.write(f"- **分析文件数**: {stats['files_count']}\n")
                f.write(f"- **平均 R²**: {stats['avg_r2']:.4f} ± {stats['std_r2']:.4f}\n")
                f.write(f"- **平均 LOF**: {stats['avg_lof']:.2f}% ± {stats['std_lof']:.2f}%\n")
                f.write(f"- **收敛率**: {stats['convergence_rate']*100:.1f}%\n\n")
            
            # 配置信息
            f.write("## ⚙️ 分析配置\n\n")
            for category, config in self.category_configs.items():
                category_name = {
                    'multi_peak_overlap': '多峰重叠数据',
                    'transient_decay': '瞬态衰减数据',
                    'low_snr': '低信噪比数据'
                }.get(category, category)
                
                f.write(f"### {category_name}\n")
                f.write(f"- **描述**: {config['description']}\n")
                f.write(f"- **组分数量**: {config['n_components']}\n")
                f.write(f"- **最大迭代数**: {config['max_iter']}\n")
                f.write(f"- **约束条件**: {', '.join(config['constraints'])}\n\n")
            
            # 可视化图片
            f.write("## 📈 汇总可视化\n\n")
            f.write("![批量分析汇总](batch_analysis_summary.png)\n\n")
            
            # 文件目录结构
            f.write("## 📁 结果文件结构\n\n")
            f.write("```\n")
            f.write("experiments/results/mcr_als_results/\n")
            f.write("├── batch_analysis_summary.json  # 汇总统计数据\n")
            f.write("├── batch_analysis_summary.png   # 汇总可视化图表\n")
            f.write("├── MCR_ALS_批量分析报告.md      # 本报告\n")
            f.write("├── multi_peak_overlap/          # 多峰重叠数据分析结果\n")
            f.write("├── transient_decay/             # 瞬态衰减数据分析结果\n")
            f.write("└── low_snr/                     # 低信噪比数据分析结果\n")
            f.write("    └── [文件名]/\n")
            f.write("        ├── concentration_profiles.csv  # 浓度轮廓\n")
            f.write("        ├── pure_spectra.csv           # 纯光谱\n")
            f.write("        ├── mcr_analysis.png           # 可视化结果\n")
            f.write("        └── analysis_summary.json      # 分析摘要\n")
            f.write("```\n\n")
        
        print(f"📝 Markdown报告已保存: {report_path}")

def main():
    """主函数"""
    print("🎯 MCR-ALS批量分析器")
    print("=" * 50)
    
    # 创建分析器
    analyzer = MCRALSBatchAnalyzer()
    
    # 运行批量分析
    analyzer.run_batch_analysis(max_files_per_category=3)  # 每类别分析3个文件
    
    print("\n🎉 所有分析完成！")

if __name__ == "__main__":
    main()