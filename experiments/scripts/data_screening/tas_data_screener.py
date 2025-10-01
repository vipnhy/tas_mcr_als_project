#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TAS数据筛选器 - 从data/TAS目录筛选三类挑战性数据
1. 多峰重叠数据（模拟复杂体系）
2. 瞬态信号衰减数据（时间分辨验证）
3. 低信噪比数据（SNR=5:1）
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import signal
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TASDataScreener:
    """TAS数据筛选器"""
    
    def __init__(self, data_root="data/TAS", output_root="experiments/results/data_screening"):
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # 筛选标准
        self.criteria = {
            'multi_peak_overlap': {
                'description': '多峰重叠数据 - 光谱维度多个重叠峰',
                'min_peaks': 3,
                'overlap_threshold': 0.6,
                'peak_prominence_factor': 0.15
            },
            'transient_decay': {
                'description': '瞬态信号衰减数据 - 时间维度明显衰减',
                'min_decay_ratio': 2.0,  # 初始/最终信号比值
                'exponential_fit_r2': 0.7,
                'time_constant_range': (0.1, 100)  # ps
            },
            'low_snr': {
                'description': '低信噪比数据 - SNR ≤ 5:1',
                'snr_threshold': 5.0,
                'noise_estimation_regions': 4  # 边缘区域数量
            }
        }
        
        self.results = {
            'multi_peak_overlap': [],
            'transient_decay': [],
            'low_snr': [],
            'failed_files': []
        }
    
    def find_all_data_files(self):
        """递归查找所有数据文件"""
        data_files = []
        
        # 支持的文件格式
        patterns = ['*.csv', '*.txt', '*.dat']
        
        for pattern in patterns:
            files = list(self.data_root.rglob(pattern))
            data_files.extend(files)
        
        # 过滤掉明显的非数据文件
        filtered_files = []
        for file in data_files:
            # 跳过太小的文件（可能是配置文件）
            if file.stat().st_size < 1000:  # 小于1KB
                continue
            # 跳过明显的非数据文件名
            if any(skip in file.name.lower() for skip in ['readme', 'config', 'log']):
                continue
            filtered_files.append(file)
        
        print(f"找到 {len(filtered_files)} 个潜在数据文件")
        return filtered_files
    
    def load_tas_data(self, file_path):
        """加载TAS数据文件"""
        try:
            # 尝试不同的加载方式
            for sep in [',', '\t', ' ']:
                try:
                    # 尝试第一行为header
                    df = pd.read_csv(file_path, sep=sep, index_col=0)
                    
                    # 检查是否为数值数据
                    if df.select_dtypes(include=[np.number]).shape[1] < 3:
                        continue
                    
                    # 基本验证
                    if df.shape[0] < 10 or df.shape[1] < 10:
                        continue
                    
                    # 转换为数值类型
                    df = df.select_dtypes(include=[np.number])
                    
                    # 检查索引是否像波长（通常300-800nm）
                    if df.index.dtype in [np.float64, np.int64]:
                        wavelengths = df.index.values
                        if 200 <= wavelengths.min() <= 1000 and 300 <= wavelengths.max() <= 1200:
                            # 检查列是否像时间延迟
                            try:
                                time_delays = df.columns.astype(float)
                                if len(time_delays) > 5:
                                    return {
                                        'data': df.values,
                                        'wavelengths': wavelengths,
                                        'time_delays': time_delays.values,
                                        'shape': df.shape,
                                        'file_path': str(file_path)
                                    }
                            except:
                                continue
                    
                except Exception:
                    continue
            
            return None
            
        except Exception as e:
            print(f"加载失败 {file_path.name}: {e}")
            return None
    
    def analyze_multi_peak_overlap(self, data_info):
        """分析多峰重叠特征"""
        data = data_info['data']
        wavelengths = data_info['wavelengths']
        
        overlap_scores = []
        
        # 分析多个时间点的光谱
        time_points_to_check = min(10, data.shape[1])
        step = max(1, data.shape[1] // time_points_to_check)
        
        for i in range(0, data.shape[1], step):
            spectrum = data[:, i]
            
            # 平滑光谱以减少噪声影响
            if len(spectrum) > 20:
                from scipy.ndimage import gaussian_filter1d
                smoothed = gaussian_filter1d(spectrum, sigma=2)
            else:
                smoothed = spectrum
            
            # 找峰（正峰和负峰）
            abs_spectrum = np.abs(smoothed)
            
            # 动态调整峰识别参数
            prominence = np.std(abs_spectrum) * self.criteria['multi_peak_overlap']['peak_prominence_factor']
            min_distance = len(abs_spectrum) // 30
            
            peaks, properties = signal.find_peaks(
                abs_spectrum,
                prominence=prominence,
                distance=min_distance
            )
            
            if len(peaks) >= self.criteria['multi_peak_overlap']['min_peaks']:
                # 计算峰宽和间距
                try:
                    peak_widths = signal.peak_widths(abs_spectrum, peaks, rel_height=0.5)[0]
                    
                    if len(peaks) > 1:
                        peak_distances = np.diff(peaks)
                        avg_width = np.mean(peak_widths)
                        avg_distance = np.mean(peak_distances)
                        
                        # 重叠度 = 平均峰宽 / 平均峰间距
                        overlap_ratio = avg_width / avg_distance if avg_distance > 0 else 0
                        
                        if overlap_ratio > self.criteria['multi_peak_overlap']['overlap_threshold']:
                            overlap_scores.append({
                                'time_index': i,
                                'num_peaks': len(peaks),
                                'overlap_ratio': overlap_ratio,
                                'peak_positions': peaks,
                                'peak_intensities': abs_spectrum[peaks]
                            })
                
                except Exception:
                    continue
        
        if overlap_scores:
            avg_overlap = np.mean([s['overlap_ratio'] for s in overlap_scores])
            max_peaks = max([s['num_peaks'] for s in overlap_scores])
            
            return {
                'is_multi_peak': True,
                'score': avg_overlap,
                'max_peaks': max_peaks,
                'details': overlap_scores[:3]  # 保存前3个详细信息
            }
        
        return {'is_multi_peak': False, 'score': 0}
    
    def analyze_transient_decay(self, data_info):
        """分析瞬态信号衰减特征"""
        data = data_info['data']
        time_delays = data_info['time_delays']
        
        # 只分析正时间延迟
        positive_time_mask = time_delays > 0
        if np.sum(positive_time_mask) < 5:
            return {'is_transient': False, 'score': 0}
        
        pos_times = time_delays[positive_time_mask]
        pos_data = data[:, positive_time_mask]
        
        decay_results = []
        
        # 分析多个波长的衰减行为
        wavelength_step = max(1, pos_data.shape[0] // 20)
        
        for i in range(0, pos_data.shape[0], wavelength_step):
            kinetic = pos_data[i, :]
            
            # 检查是否有衰减趋势
            if len(kinetic) < 5:
                continue
            
            initial_signal = np.mean(kinetic[:3])  # 前3个点
            final_signal = np.mean(kinetic[-3:])   # 后3个点
            
            if abs(initial_signal) < 1e-10:  # 避免除零
                continue
            
            decay_ratio = abs(initial_signal / final_signal) if abs(final_signal) > 1e-10 else float('inf')
            
            if decay_ratio >= self.criteria['transient_decay']['min_decay_ratio']:
                try:
                    # 指数衰减拟合
                    def exp_decay(t, a, tau, c):
                        return a * np.exp(-t / tau) + c
                    
                    # 初始猜测
                    a_guess = initial_signal - final_signal
                    tau_guess = pos_times[len(pos_times)//2]
                    c_guess = final_signal
                    
                    popt, pcov = curve_fit(
                        exp_decay,
                        pos_times,
                        kinetic,
                        p0=[a_guess, tau_guess, c_guess],
                        bounds=([-np.inf, 0.01, -np.inf], [np.inf, 1000, np.inf]),
                        maxfev=1000
                    )
                    
                    # 计算拟合质量
                    fitted = exp_decay(pos_times, *popt)
                    ss_res = np.sum((kinetic - fitted) ** 2)
                    ss_tot = np.sum((kinetic - np.mean(kinetic)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    tau = abs(popt[1])
                    
                    if (r2 >= self.criteria['transient_decay']['exponential_fit_r2'] and
                        self.criteria['transient_decay']['time_constant_range'][0] <= tau <= 
                        self.criteria['transient_decay']['time_constant_range'][1]):
                        
                        decay_results.append({
                            'wavelength_index': i,
                            'time_constant': tau,
                            'r2': r2,
                            'decay_ratio': decay_ratio,
                            'amplitude': popt[0]
                        })
                
                except Exception:
                    continue
        
        if decay_results:
            avg_r2 = np.mean([r['r2'] for r in decay_results])
            avg_tau = np.mean([r['time_constant'] for r in decay_results])
            
            return {
                'is_transient': True,
                'score': avg_r2,
                'avg_time_constant': avg_tau,
                'num_decay_components': len(decay_results),
                'details': decay_results[:3]
            }
        
        return {'is_transient': False, 'score': 0}
    
    def analyze_low_snr(self, data_info):
        """分析信噪比"""
        data = data_info['data']
        
        # 估计噪声水平（使用数据边缘）
        h, w = data.shape
        edge_size = min(5, h//10, w//10)
        
        if edge_size == 0:
            return {'is_low_snr': False, 'snr': float('inf')}
        
        # 从四个角落估计噪声
        noise_regions = [
            data[:edge_size, :edge_size],      # 左上
            data[:edge_size, -edge_size:],     # 右上
            data[-edge_size:, :edge_size],     # 左下
            data[-edge_size:, -edge_size:]     # 右下
        ]
        
        noise_stds = [np.std(region) for region in noise_regions if region.size > 0]
        
        if not noise_stds:
            return {'is_low_snr': False, 'snr': float('inf')}
        
        noise_level = np.mean(noise_stds)
        
        # 估计信号强度（中心区域的最大绝对值）
        center_h_start = h//4
        center_h_end = 3*h//4
        center_w_start = w//4
        center_w_end = 3*w//4
        
        center_data = data[center_h_start:center_h_end, center_w_start:center_w_end]
        signal_strength = np.max(np.abs(center_data)) if center_data.size > 0 else 0
        
        # 计算SNR
        snr = signal_strength / noise_level if noise_level > 0 else float('inf')
        
        return {
            'is_low_snr': snr <= self.criteria['low_snr']['snr_threshold'],
            'snr': snr,
            'noise_level': noise_level,
            'signal_strength': signal_strength
        }
    
    def create_visualization(self, data_info, analysis_results, category, output_path):
        """创建数据可视化图"""
        data = data_info['data']
        wavelengths = data_info['wavelengths']
        time_delays = data_info['time_delays']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{category} - {Path(data_info["file_path"]).name}', fontsize=16, fontweight='bold')
        
        # 1. 热图
        ax1 = axes[0, 0]
        im1 = ax1.imshow(data, aspect='auto', cmap='RdBu_r', 
                        extent=[time_delays.min(), time_delays.max(), 
                               wavelengths.max(), wavelengths.min()])
        ax1.set_xlabel('时间延迟 (ps)')
        ax1.set_ylabel('波长 (nm)')
        ax1.set_title('2D光谱热图')
        plt.colorbar(im1, ax=ax1, label='ΔA')
        
        # 2. 选择性时间切片光谱
        ax2 = axes[0, 1]
        time_indices = [0, len(time_delays)//4, len(time_delays)//2, -1]
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, (idx, color) in enumerate(zip(time_indices, colors)):
            if idx < len(time_delays):
                spectrum = data[:, idx]
                label = f't = {time_delays[idx]:.2f} ps'
                ax2.plot(wavelengths, spectrum, color=color, label=label, linewidth=2)
        
        ax2.set_xlabel('波长 (nm)')
        ax2.set_ylabel('ΔA')
        ax2.set_title('不同时间的光谱切片')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 选择性波长动力学
        ax3 = axes[1, 0]
        wl_indices = [len(wavelengths)//4, len(wavelengths)//2, 3*len(wavelengths)//4]
        
        for i, idx in enumerate(wl_indices):
            if idx < len(wavelengths):
                kinetic = data[idx, :]
                label = f'λ = {wavelengths[idx]:.0f} nm'
                ax3.plot(time_delays, kinetic, label=label, linewidth=2, marker='o', markersize=3)
        
        ax3.set_xlabel('时间延迟 (ps)')
        ax3.set_ylabel('ΔA')
        ax3.set_title('不同波长的动力学曲线')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 特征分析图
        ax4 = axes[1, 1]
        
        if category == 'multi_peak_overlap':
            # 显示峰识别结果
            if analysis_results['is_multi_peak']:
                detail = analysis_results['details'][0]
                spectrum = data[:, detail['time_index']]
                ax4.plot(wavelengths, np.abs(spectrum), 'b-', linewidth=2, label='光谱')
                
                # 标记识别的峰
                peak_positions = detail['peak_positions']
                peak_wavelengths = wavelengths[peak_positions]
                peak_intensities = detail['peak_intensities']
                
                ax4.plot(peak_wavelengths, peak_intensities, 'ro', markersize=8, label=f'峰 (n={len(peak_positions)})')
                ax4.set_title(f'多峰识别\n重叠度: {detail["overlap_ratio"]:.3f}')
            else:
                ax4.plot(wavelengths, np.abs(data[:, 0]), 'b-', linewidth=2)
                ax4.set_title('未检测到多峰重叠')
        
        elif category == 'transient_decay':
            # 显示衰减拟合
            if analysis_results['is_transient']:
                detail = analysis_results['details'][0]
                wl_idx = detail['wavelength_index']
                kinetic = data[wl_idx, time_delays > 0]
                pos_times = time_delays[time_delays > 0]
                
                ax4.plot(pos_times, kinetic, 'bo', markersize=4, label='实验数据')
                
                # 拟合曲线
                def exp_decay(t, a, tau, c):
                    return a * np.exp(-t / tau) + c
                
                tau = detail['time_constant']
                # 简单估计参数用于显示
                a_est = kinetic[0] - kinetic[-1]
                c_est = kinetic[-1]
                fitted = exp_decay(pos_times, a_est, tau, c_est)
                
                ax4.plot(pos_times, fitted, 'r-', linewidth=2, 
                        label=f'拟合 (τ={tau:.2f}ps, R²={detail["r2"]:.3f})')
                
                ax4.set_title(f'瞬态衰减拟合\nλ = {wavelengths[wl_idx]:.0f} nm')
                ax4.set_xlabel('时间延迟 (ps)')
            else:
                ax4.plot(time_delays, data[len(wavelengths)//2, :], 'b-', linewidth=2)
                ax4.set_title('未检测到明显衰减')
        
        elif category == 'low_snr':
            # 显示信噪比分析
            snr = analysis_results['snr']
            
            # 显示数据分布直方图
            ax4.hist(data.flatten(), bins=50, alpha=0.7, density=True, color='skyblue')
            ax4.axvline(0, color='red', linestyle='--', alpha=0.7, label='零基线')
            ax4.set_title(f'数据分布\nSNR = {snr:.2f}')
            ax4.set_xlabel('信号强度')
            ax4.set_ylabel('密度')
        
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def screen_all_files(self):
        """筛选所有文件"""
        print("开始筛选TAS数据文件...")
        print("=" * 60)
        
        # 找到所有数据文件
        data_files = self.find_all_data_files()
        
        if not data_files:
            print("未找到数据文件！")
            return
        
        total_files = len(data_files)
        processed = 0
        
        for file_path in data_files:
            processed += 1
            print(f"\n[{processed}/{total_files}] 分析: {file_path.relative_to(self.data_root)}")
            
            # 加载数据
            data_info = self.load_tas_data(file_path)
            if data_info is None:
                self.results['failed_files'].append(str(file_path))
                print(f"  ❌ 加载失败")
                continue
            
            print(f"  📊 数据形状: {data_info['shape']} (波长×时间)")
            
            # 分析三类特征
            multi_peak = self.analyze_multi_peak_overlap(data_info)
            transient = self.analyze_transient_decay(data_info)
            low_snr = self.analyze_low_snr(data_info)
            
            # 记录结果
            file_result = {
                'file_path': str(file_path),
                'relative_path': str(file_path.relative_to(self.data_root)),
                'shape': data_info['shape'],
                'multi_peak': multi_peak,
                'transient': transient,
                'low_snr': low_snr
            }
            
            # 分类并创建可视化
            categories_found = []
            
            if multi_peak['is_multi_peak']:
                self.results['multi_peak_overlap'].append(file_result)
                categories_found.append('multi_peak_overlap')
                print(f"  ✅ 多峰重叠: 评分={multi_peak['score']:.3f}, 峰数={multi_peak['max_peaks']}")
            
            if transient['is_transient']:
                self.results['transient_decay'].append(file_result)
                categories_found.append('transient_decay')
                print(f"  ✅ 瞬态衰减: R²={transient['score']:.3f}, τ平均={transient['avg_time_constant']:.2f}ps")
            
            if low_snr['is_low_snr']:
                self.results['low_snr'].append(file_result)
                categories_found.append('low_snr')
                print(f"  ✅ 低信噪比: SNR={low_snr['snr']:.2f}")
            
            # 为每个符合条件的类别创建可视化
            for category in categories_found:
                self.create_category_visualization(data_info, file_result, category)
            
            if not categories_found:
                print(f"  ➖ 常规数据 (SNR={low_snr['snr']:.2f})")
        
        # 生成汇总报告
        self.generate_summary_report()
        self.save_results()
    
    def create_category_visualization(self, data_info, file_result, category):
        """为特定类别创建可视化"""
        # 创建类别输出目录
        category_dir = self.output_root / category
        category_dir.mkdir(exist_ok=True)
        
        # 生成安全的文件名
        safe_name = Path(file_result['relative_path']).stem.replace(' ', '_').replace('/', '_')
        output_file = category_dir / f"{safe_name}.png"
        
        # 创建可视化
        analysis_result = file_result[category.split('_')[0] if category != 'multi_peak_overlap' else 'multi_peak']
        
        try:
            self.create_visualization(data_info, analysis_result, 
                                    self.get_category_chinese_name(category), 
                                    output_file)
            print(f"    💾 可视化保存: {output_file.relative_to(self.output_root)}")
        except Exception as e:
            print(f"    ❌ 可视化失败: {e}")
    
    def get_category_chinese_name(self, category):
        """获取类别中文名称"""
        names = {
            'multi_peak_overlap': '多峰重叠数据',
            'transient_decay': '瞬态信号衰减数据',
            'low_snr': '低信噪比数据'
        }
        return names.get(category, category)
    
    def generate_summary_report(self):
        """生成汇总报告"""
        report_file = self.output_root / "TAS数据筛选报告.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# TAS数据筛选报告\n\n")
            f.write(f"筛选时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据源目录: {self.data_root}\n\n")
            
            # 统计信息
            f.write("## 📊 筛选统计\n\n")
            f.write(f"- **总文件数**: {sum(len(v) if isinstance(v, list) else 0 for v in self.results.values())}\n")
            f.write(f"- **多峰重叠数据**: {len(self.results['multi_peak_overlap'])} 个\n")
            f.write(f"- **瞬态衰减数据**: {len(self.results['transient_decay'])} 个\n")
            f.write(f"- **低信噪比数据**: {len(self.results['low_snr'])} 个\n")
            f.write(f"- **加载失败**: {len(self.results['failed_files'])} 个\n\n")
            
            # 详细列表
            for category, chinese_name in [
                ('multi_peak_overlap', '多峰重叠数据'),
                ('transient_decay', '瞬态信号衰减数据'),
                ('low_snr', '低信噪比数据')
            ]:
                f.write(f"## 🎯 {chinese_name}\n\n")
                
                if self.results[category]:
                    f.write("| 序号 | 文件路径 | 数据形状 | 特征评分 | 可视化 |\n")
                    f.write("|------|----------|----------|----------|--------|\n")
                    
                    for i, item in enumerate(self.results[category], 1):
                        rel_path = item['relative_path']
                        shape = f"{item['shape'][0]}×{item['shape'][1]}"
                        
                        # 获取特征评分
                        if category == 'multi_peak_overlap':
                            score = f"重叠度={item['multi_peak']['score']:.3f}"
                        elif category == 'transient_decay':
                            score = f"R²={item['transient']['score']:.3f}"
                        else:  # low_snr
                            score = f"SNR={item['low_snr']['snr']:.2f}"
                        
                        # 可视化文件
                        safe_name = Path(rel_path).stem.replace(' ', '_').replace('/', '_')
                        viz_file = f"{category}/{safe_name}.png"
                        
                        f.write(f"| {i} | `{rel_path}` | {shape} | {score} | ![viz]({viz_file}) |\n")
                    
                    f.write("\n")
                else:
                    f.write("暂未找到符合条件的数据文件。\n\n")
            
            # 筛选标准
            f.write("## ⚙️ 筛选标准\n\n")
            for category, criteria in self.criteria.items():
                chinese_name = self.get_category_chinese_name(category)
                f.write(f"### {chinese_name}\n")
                f.write(f"- **描述**: {criteria['description']}\n")
                
                if category == 'multi_peak_overlap':
                    f.write(f"- **最少峰数**: {criteria['min_peaks']}\n")
                    f.write(f"- **重叠阈值**: {criteria['overlap_threshold']}\n")
                elif category == 'transient_decay':
                    f.write(f"- **最小衰减比**: {criteria['min_decay_ratio']}\n")
                    f.write(f"- **拟合R²阈值**: {criteria['exponential_fit_r2']}\n")
                    f.write(f"- **时间常数范围**: {criteria['time_constant_range']} ps\n")
                elif category == 'low_snr':
                    f.write(f"- **SNR阈值**: ≤ {criteria['snr_threshold']}\n")
                
                f.write("\n")
            
            # 使用建议
            f.write("## 💡 使用建议\n\n")
            f.write("### 多峰重叠数据\n")
            f.write("- 适合测试MCR-ALS在复杂光谱重叠情况下的分辨能力\n")
            f.write("- 建议使用更严格的约束条件和更多组分\n\n")
            
            f.write("### 瞬态信号衰减数据\n")
            f.write("- 适合验证时间分辨MCR-ALS的动力学分析能力\n")
            f.write("- 关注衰减时间常数的物理合理性\n\n")
            
            f.write("### 低信噪比数据\n")
            f.write("- 适合测试算法在噪声环境下的鲁棒性\n")
            f.write("- 建议增加随机初始化次数和预处理步骤\n\n")
        
        print(f"\n📋 汇总报告已生成: {report_file}")
    
    def save_results(self):
        """保存筛选结果"""
        # 处理numpy类型以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_clean = convert_numpy(self.results)
        
        output_file = self.output_root / "screening_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_clean, f, indent=2, ensure_ascii=False)
        
        print(f"📄 详细结果已保存: {output_file}")

def main():
    """主函数"""
    print("🔍 TAS挑战性数据筛选器")
    print("=" * 50)
    
    screener = TASDataScreener()
    screener.screen_all_files()
    
    # 输出最终统计
    print("\n" + "=" * 60)
    print("🎉 筛选完成！")
    print(f"📁 结果保存在: {screener.output_root}")
    print(f"🎯 多峰重叠数据: {len(screener.results['multi_peak_overlap'])} 个")
    print(f"⚡ 瞬态衰减数据: {len(screener.results['transient_decay'])} 个")
    print(f"📉 低信噪比数据: {len(screener.results['low_snr'])} 个")
    
    # 显示输出目录结构
    print(f"\n📂 输出目录结构:")
    for item in screener.output_root.iterdir():
        if item.is_dir():
            file_count = len(list(item.glob('*.png')))
            print(f"  📁 {item.name}/ ({file_count} 个可视化文件)")
        else:
            print(f"  📄 {item.name}")

if __name__ == "__main__":
    main()