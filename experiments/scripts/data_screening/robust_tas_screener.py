#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TAS数据筛选器 - 针对实际数据格式优化版本
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy import signal
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RobustTASScreener:
    """针对实际TAS数据格式优化的筛选器"""
    
    def __init__(self, output_root="experiments/results/data_screening"):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # 扫描data/TAS目录中的所有数据文件
        self.known_files = self._scan_tas_directory()
        
        # 初始化结果存储
        self.results = {
            'multi_peak_overlap': [],
            'transient_decay': [],
            'low_snr': [],
            'analysis_summary': {}
        }
        
        # 还可以检查results目录中的合成数据（用于测试）
        if Path("results/synthetic_tas_dataset.csv").exists():
            self.known_files.append("results/synthetic_tas_dataset.csv")
            
        # 新生成的挑战性数据（用于测试）
        challenging_files = [
            self.output_root / "challenging_multi_peak_overlap.csv",
            self.output_root / "challenging_transient_decay.csv", 
            self.output_root / "challenging_low_snr.csv",
            self.output_root / "obvious_transient_decay.csv"
        ]
        
        for file in challenging_files:
            if Path(file).exists():
                self.known_files.append(file)
    
    def _generate_safe_filename(self, file_path, category):
        """生成安全的可视化文件名，包含上级目录信息并避免重名"""
        file_path = Path(file_path)
        
        # 获取文件名（不含扩展名）
        base_name = file_path.stem
        
        # 获取上级目录名，增加唯一性
        parent_dir = file_path.parent.name
        
        # 构建基础文件名：上级目录_原文件名
        safe_name = f"{parent_dir}_{base_name}"
        
        # 清理文件名中的特殊字符
        safe_name = safe_name.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        
        # 构建完整路径
        category_dir = self.output_root / category
        category_dir.mkdir(exist_ok=True)
        
        final_path = category_dir / f"{safe_name}.png"
        
        # 检查重名并处理冲突
        counter = 1
        while final_path.exists():
            conflict_name = f"{safe_name}_{counter}"
            final_path = category_dir / f"{conflict_name}.png"
            counter += 1
            
        return final_path
    
    def _scan_tas_directory(self):
        """扫描data/TAS目录中的所有CSV文件，按目录组织并实现优先处理逻辑"""
        tas_dir = Path("data/TAS")
        csv_files = []
        
        if not tas_dir.exists():
            print(f"⚠️ 警告: {tas_dir} 目录不存在")
            return csv_files
        
        # 按目录组织文件
        dir_files = {}
        for csv_file in tas_dir.rglob("*.csv"):
            dir_path = csv_file.parent
            if dir_path not in dir_files:
                dir_files[dir_path] = []
            dir_files[dir_path].append(csv_file)
        
        # 对每个目录应用优先处理逻辑
        for dir_path, files in dir_files.items():
            file_names = [f.name for f in files]
            
            # 如果存在TA_Average.csv，只处理它，跳过TA_Scan*.csv
            if 'TA_Average.csv' in file_names:
                ta_average_file = dir_path / 'TA_Average.csv'
                csv_files.append(str(ta_average_file))
                print(f"� {dir_path.name}: 发现TA_Average.csv，跳过TA_Scan*.csv")
            else:
                # 否则处理所有文件
                for file in files:
                    csv_files.append(str(file))
                print(f"📁 {dir_path.name}: 处理 {len(files)} 个文件")
        
        print(f"📄 总共选择处理 {len(csv_files)} 个CSV文件")
        
        return csv_files
    
    def load_tas_file(self, file_path):
        """加载TAS文件 - 健壮版本"""
        try:
            print(f"   正在加载: {Path(file_path).name}")
            
            # 首先尝试读取原始CSV
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # 跳过可能的头部信息
            data_start = 0
            for i, line in enumerate(lines):
                if line.strip() and ',' in line:
                    try:
                        # 尝试解析第一行为数字
                        float(line.split(',')[0])
                        data_start = i
                        break
                    except:
                        continue
            
            if data_start >= len(lines) - 5:
                print("   ❌ 找不到有效数据行")
                return None
            
            # 读取数据部分
            data_lines = lines[data_start:]
            
            # 解析第一行为时间延迟
            time_delays = []
            first_line = data_lines[0].strip().split(',')
            for val in first_line:
                try:
                    time_delays.append(float(val))
                except:
                    time_delays.append(0.0)
            
            time_delays = np.array(time_delays)
            
            # 解析其余行
            wavelengths = []
            data_matrix = []
            
            for line in data_lines[1:]:
                if not line.strip():
                    continue
                    
                parts = line.strip().split(',')
                if len(parts) < 2:
                    continue
                
                try:
                    # 第一个值是波长
                    wl = float(parts[0])
                    wavelengths.append(wl)
                    
                    # 其余值是数据
                    row_data = []
                    for val in parts[1:]:
                        try:
                            v = float(val)
                            if np.isfinite(v):
                                row_data.append(v)
                            else:
                                row_data.append(0.0)
                        except:
                            row_data.append(0.0)
                    
                    # 确保长度匹配
                    while len(row_data) < len(time_delays):
                        row_data.append(0.0)
                    row_data = row_data[:len(time_delays)]
                    
                    data_matrix.append(row_data)
                    
                except Exception as e:
                    print(f"   ⚠️ 跳过无效行: {e}")
                    continue
            
            if len(data_matrix) < 10:
                print(f"   ❌ 数据行数太少: {len(data_matrix)}")
                return None
            
            wavelengths = np.array(wavelengths)
            data = np.array(data_matrix)
            
            # 数据验证和清理
            if data.shape[0] < 10 or data.shape[1] < 10:
                print(f"   ❌ 数据形状太小: {data.shape}")
                return None
            
            # 处理异常值和无穷大
            data = np.where(np.isfinite(data), data, 0)
            
            # 移除全零行和列
            non_zero_rows = np.any(np.abs(data) > 1e-10, axis=1)
            non_zero_cols = np.any(np.abs(data) > 1e-10, axis=0)
            
            if np.sum(non_zero_rows) < 5 or np.sum(non_zero_cols) < 5:
                print("   ❌ 有效数据太少")
                return None
            
            data = data[non_zero_rows, :][:, non_zero_cols]
            wavelengths = wavelengths[non_zero_rows]
            time_delays = time_delays[non_zero_cols]
            
            print(f"   ✅ 成功加载: {data.shape} (波长×时间)")
            print(f"   波长范围: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
            print(f"   时间范围: {time_delays.min():.2f} - {time_delays.max():.2f} ps")
            print(f"   数据范围: {data.min():.2e} - {data.max():.2e}")
            
            # 根据先验知识进行光谱裁剪
            file_path_str = str(file_path)
            original_wavelength_range = f"{wavelengths.min():.1f} - {wavelengths.max():.1f} nm"
            
            # 判断光谱类型并确定裁剪范围
            if 'UV' in file_path_str.upper():
                # UV光谱: 380-650 nm
                wl_min, wl_max = 380.0, 650.0
                spectrum_type = "UV"
            elif 'NIR' in file_path_str.upper():
                # NIR光谱: 1100-1620 nm
                wl_min, wl_max = 1100.0, 1620.0
                spectrum_type = "NIR"
            else:
                # 默认VIS光谱: 500-950 nm
                wl_min, wl_max = 500.0, 950.0
                spectrum_type = "VIS"
            
            # 执行光谱裁剪
            mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
            if np.sum(mask) < 5:
                print(f"   ⚠️ 裁剪后有效波长点太少 ({np.sum(mask)}), 跳过此文件")
                return None
            
            wavelengths_cropped = wavelengths[mask]
            data_cropped = data[mask, :]
            
            print(f"   ✂️ 光谱裁剪: {spectrum_type}范围 ({wl_min:.0f}-{wl_max:.0f} nm)")
            print(f"   裁剪前: {original_wavelength_range}")
            print(f"   裁剪后: {wavelengths_cropped.min():.1f} - {wavelengths_cropped.max():.1f} nm")
            print(f"   保留波长点: {len(wavelengths_cropped)}/{len(wavelengths)}")
            
            return {
                'data': data_cropped,
                'wavelengths': wavelengths_cropped,
                'time_delays': time_delays,
                'shape': data_cropped.shape,
                'file_path': str(file_path),
                'spectrum_type': spectrum_type,
                'original_wavelength_range': original_wavelength_range,
                'cropped_wavelength_range': f"{wavelengths_cropped.min():.1f} - {wavelengths_cropped.max():.1f} nm"
            }
            
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
            return None
    
    def analyze_multi_peak_overlap(self, data_info):
        """分析多峰重叠特征"""
        data = data_info['data']
        wavelengths = data_info['wavelengths']
        
        overlap_scores = []
        
        # 寻找有明显信号的时间点
        signal_strength = np.max(np.abs(data), axis=0)
        strong_signal_indices = np.argsort(signal_strength)[-min(10, len(signal_strength)):]
        
        for t_idx in strong_signal_indices:
            spectrum = data[:, t_idx]
            abs_spectrum = np.abs(spectrum)
            
            if np.max(abs_spectrum) < 1e-8:
                continue
            
            # 寻找峰值
            try:
                # 自适应阈值
                prominence = np.std(abs_spectrum) * 0.3
                min_distance = max(3, len(abs_spectrum) // 30)
                
                peaks, properties = signal.find_peaks(
                    abs_spectrum,
                    prominence=prominence,
                    distance=min_distance,
                    height=np.max(abs_spectrum) * 0.1
                )
                
                if len(peaks) >= 3:  # 至少3个峰
                    # 计算峰间重叠度
                    try:
                        peak_widths = signal.peak_widths(abs_spectrum, peaks, rel_height=0.5)[0]
                        peak_distances = np.diff(peaks)
                        
                        if len(peak_distances) > 0:
                            avg_width = np.mean(peak_widths[:-1])
                            avg_distance = np.mean(peak_distances)
                            overlap_ratio = avg_width / avg_distance if avg_distance > 0 else 0
                            
                            if overlap_ratio > 0.3:  # 降低重叠阈值
                                overlap_scores.append({
                                    'time_index': t_idx,
                                    'time_delay': data_info['time_delays'][t_idx],
                                    'num_peaks': len(peaks),
                                    'overlap_ratio': overlap_ratio,
                                    'peaks': peaks.tolist(),
                                    'peak_heights': abs_spectrum[peaks].tolist(),
                                    'peak_widths': peak_widths.tolist()
                                })
                    except:
                        continue
            except:
                continue
        
        if overlap_scores:
            avg_overlap = np.mean([s['overlap_ratio'] for s in overlap_scores])
            max_peaks = max([s['num_peaks'] for s in overlap_scores])
            
            return {
                'is_multi_peak': True,
                'score': avg_overlap,
                'max_peaks': max_peaks,
                'num_overlap_times': len(overlap_scores),
                'details': overlap_scores[:3]  # 保留前3个最好的
            }
        
        return {'is_multi_peak': False, 'score': 0, 'details': []}
    
    def analyze_transient_decay(self, data_info):
        """分析瞬态衰减特征"""
        data = data_info['data']
        time_delays = data_info['time_delays']
        
        # 只分析正时间延迟
        pos_mask = time_delays > 0
        if np.sum(pos_mask) < 5:
            return {'is_transient': False, 'score': 0, 'details': []}
        
        pos_times = time_delays[pos_mask]
        pos_data = data[:, pos_mask]
        
        decay_results = []
        
        # 分析有强信号的波长
        max_signal_per_wavelength = np.max(np.abs(pos_data), axis=1)
        if np.max(max_signal_per_wavelength) < 1e-8:
            return {'is_transient': False, 'score': 0, 'details': []}
        
        strong_wl_indices = np.argsort(max_signal_per_wavelength)[-min(15, len(max_signal_per_wavelength)):]
        
        for wl_idx in strong_wl_indices:
            kinetic = pos_data[wl_idx, :]
            
            if len(kinetic) < 5:
                continue
            
            # 检查衰减趋势
            initial_vals = kinetic[:min(3, len(kinetic))]
            final_vals = kinetic[-min(3, len(kinetic)):]
            
            initial = np.mean(np.abs(initial_vals))
            final = np.mean(np.abs(final_vals))
            
            if initial < 1e-10:
                continue
            
            decay_ratio = initial / final if final > 1e-10 else 100
            
            if decay_ratio > 1.1:  # 进一步降低衰减阈值
                try:
                    # 简单的指数衰减拟合
                    def exp_decay(t, a, tau, c):
                        return a * np.exp(-t / tau) + c
                    
                    # 更好的初始猜测
                    y_data = kinetic
                    max_val = np.max(np.abs(y_data))
                    min_val = np.min(np.abs(y_data))
                    
                    initial_a = max_val - min_val
                    initial_tau = pos_times[len(pos_times)//3]
                    initial_c = min_val
                    
                    popt, pcov = curve_fit(
                        exp_decay,
                        pos_times,
                        y_data,
                        p0=[initial_a, initial_tau, initial_c],
                        bounds=([-np.inf, 0.01, -np.inf], [np.inf, 1000, np.inf]),
                        maxfev=1000
                    )
                    
                    # 计算拟合质量
                    fitted = exp_decay(pos_times, *popt)
                    ss_res = np.sum((y_data - fitted)**2)
                    ss_tot = np.sum((y_data - np.mean(y_data))**2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    if r2 > 0.3 and 0.1 <= abs(popt[1]) <= 200:  # 进一步降低R²阈值，扩大时间常数范围
                        decay_results.append({
                            'wavelength_index': int(wl_idx),
                            'wavelength': float(data_info['wavelengths'][wl_idx]),
                            'time_constant': abs(float(popt[1])),
                            'r2': float(r2),
                            'decay_ratio': float(decay_ratio),
                            'amplitude': float(popt[0]),
                            'offset': float(popt[2])
                        })
                
                except:
                    continue
        
        if decay_results:
            avg_r2 = np.mean([r['r2'] for r in decay_results])
            avg_tau = np.mean([r['time_constant'] for r in decay_results])
            
            return {
                'is_transient': True,
                'score': avg_r2,
                'avg_time_constant': avg_tau,
                'num_decay_wavelengths': len(decay_results),
                'details': decay_results[:3]  # 保留前3个最好的
            }
        
        return {'is_transient': False, 'score': 0, 'details': []}
    
    def analyze_low_snr(self, data_info):
        """分析信噪比"""
        data = data_info['data']
        
        # 更保守的噪声估计
        h, w = data.shape
        
        # 使用边缘区域估计噪声
        edge_size = max(2, min(h//10, w//10, 5))
        
        try:
            # 四个角的数据
            corners = [
                data[:edge_size, :edge_size],
                data[:edge_size, -edge_size:],
                data[-edge_size:, :edge_size],
                data[-edge_size:, -edge_size:]
            ]
            
            noise_stds = []
            for corner in corners:
                if corner.size > 0:
                    corner_flat = corner.flatten()
                    # 移除异常值后计算标准差
                    percentiles = np.percentile(corner_flat, [25, 75])
                    iqr = percentiles[1] - percentiles[0]
                    if iqr > 0:
                        mask = (corner_flat >= percentiles[0] - 1.5*iqr) & \
                               (corner_flat <= percentiles[1] + 1.5*iqr)
                        if np.sum(mask) > 0:
                            noise_stds.append(np.std(corner_flat[mask]))
            
            noise_level = np.mean(noise_stds) if noise_stds else np.std(data) * 0.1
            
            # 估计信号强度（中心区域最大值）
            center_h = slice(h//6, 5*h//6)
            center_w = slice(w//6, 5*w//6)
            center_data = data[center_h, center_w]
            
            if center_data.size > 0:
                signal_strength = np.max(np.abs(center_data))
            else:
                signal_strength = np.max(np.abs(data))
            
            # 计算SNR
            snr = signal_strength / noise_level if noise_level > 0 else float('inf')
            
            return {
                'is_low_snr': snr <= 10.0,  # 调整SNR阈值
                'snr': float(snr),
                'noise_level': float(noise_level),
                'signal_strength': float(signal_strength)
            }
            
        except Exception as e:
            print(f"   ⚠️ SNR分析失败: {e}")
            return {
                'is_low_snr': False,
                'snr': float('inf'),
                'noise_level': 0.0,
                'signal_strength': float(np.max(np.abs(data)))
            }
    
    def create_comprehensive_visualization(self, data_info, analyses, output_file):
        """创建综合可视化"""
        data = data_info['data']
        wavelengths = data_info['wavelengths']
        time_delays = data_info['time_delays']
        file_name = Path(data_info['file_path']).name
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 主要2D热图
        ax1 = plt.subplot(3, 4, 1)
        im = ax1.imshow(data.T, aspect='auto', cmap='rainbow',
                       extent=[wavelengths.min(), wavelengths.max(),
                              time_delays.max(), time_delays.min()])
        ax1.set_xlabel('波长 (nm)')
        ax1.set_ylabel('时间延迟 (ps)')
        ax1.set_title(f'TAS 2D 光谱热图\n{file_name}', fontweight='bold')
        plt.colorbar(im, ax=ax1, label='ΔA')
        
        # 2. 时间切片光谱（多峰重叠分析）
        ax2 = plt.subplot(3, 4, 2)
        if analyses['multi_peak']['is_multi_peak'] and analyses['multi_peak']['details']:
            detail = analyses['multi_peak']['details'][0]
            t_idx = detail['time_index']
            spectrum = data[:, t_idx]
            peaks = detail['peaks']
            
            ax2.plot(wavelengths, np.abs(spectrum), 'b-', linewidth=2, label='光谱')
            if len(peaks) > 0:
                ax2.plot(wavelengths[peaks], np.abs(spectrum[peaks]), 'ro', 
                        markersize=8, label=f'峰 (n={len(peaks)})')
            ax2.set_title(f'多峰识别\nt={time_delays[t_idx]:.2f}ps\n重叠度:{detail["overlap_ratio"]:.3f}')
        else:
            # 显示信号最强的时间切片
            signal_strength = np.max(np.abs(data), axis=0)
            max_t_idx = np.argmax(signal_strength)
            spectrum = data[:, max_t_idx]
            ax2.plot(wavelengths, np.abs(spectrum), 'b-', linewidth=2)
            ax2.set_title(f'光谱切片\nt={time_delays[max_t_idx]:.2f}ps')
        
        ax2.set_xlabel('波长 (nm)')
        ax2.set_ylabel('|ΔA|')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 动力学曲线（瞬态衰减分析）
        ax3 = plt.subplot(3, 4, 3)
        pos_mask = time_delays > 0
        if analyses['transient']['is_transient'] and analyses['transient']['details'] and np.sum(pos_mask) > 0:
            detail = analyses['transient']['details'][0]
            wl_idx = detail['wavelength_index']
            
            pos_times = time_delays[pos_mask]
            pos_data = data[wl_idx, pos_mask]
            
            ax3.plot(pos_times, pos_data, 'bo', markersize=4, label='实验数据')
            
            # 拟合曲线
            try:
                tau = detail['time_constant']
                amp = detail['amplitude']
                offset = detail['offset']
                fitted = amp * np.exp(-pos_times / tau) + offset
                ax3.plot(pos_times, fitted, 'r-', linewidth=2,
                        label=f'拟合 (τ={tau:.2f}ps)')
            except:
                pass
            
            ax3.set_title(f'瞬态衰减\nλ={wavelengths[wl_idx]:.0f}nm\nR²={detail["r2"]:.3f}')
        else:
            # 显示信号最强的波长动力学
            signal_per_wl = np.max(np.abs(data), axis=1)
            max_wl_idx = np.argmax(signal_per_wl)
            kinetic = data[max_wl_idx, :]
            ax3.plot(time_delays, kinetic, 'b-', linewidth=2)
            ax3.set_title(f'动力学曲线\nλ={wavelengths[max_wl_idx]:.0f}nm')
        
        ax3.set_xlabel('时间延迟 (ps)')
        ax3.set_ylabel('ΔA')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 信噪比分析
        ax4 = plt.subplot(3, 4, 4)
        snr = analyses['low_snr']['snr']
        data_flat = data.flatten()
        
        # 移除异常值绘制直方图
        percentiles = np.percentile(data_flat, [1, 99])
        mask = (data_flat >= percentiles[0]) & (data_flat <= percentiles[1])
        clean_data = data_flat[mask]
        
        ax4.hist(clean_data, bins=50, alpha=0.7, density=True, color='skyblue')
        ax4.axvline(0, color='red', linestyle='--', alpha=0.7, label='零基线')
        ax4.set_title(f'数据分布\nSNR = {snr:.2f}')
        ax4.set_xlabel('信号强度')
        ax4.set_ylabel('密度')
        ax4.legend()
        
        # 5-8. 不同时间点的光谱
        n_times = len(time_delays)
        time_indices = [0, n_times//4, n_times//2, 3*n_times//4]
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, (t_idx, color) in enumerate(zip(time_indices, colors)):
            ax = plt.subplot(3, 4, 5+i)
            spectrum = data[:, t_idx]
            ax.plot(wavelengths, spectrum, color=color, linewidth=2)
            ax.set_title(f't = {time_delays[t_idx]:.2f} ps')
            ax.set_xlabel('波长 (nm)')
            ax.set_ylabel('ΔA')
            ax.grid(True, alpha=0.3)
        
        # 9-12. 不同波长的动力学
        n_wls = len(wavelengths)
        wl_indices = [n_wls//6, n_wls//3, 2*n_wls//3, 5*n_wls//6]
        
        for i, wl_idx in enumerate(wl_indices):
            ax = plt.subplot(3, 4, 9+i)
            kinetic = data[wl_idx, :]
            ax.plot(time_delays, kinetic, linewidth=2, color=colors[i])
            ax.set_title(f'λ = {wavelengths[wl_idx]:.0f} nm')
            ax.set_xlabel('时间延迟 (ps)')
            ax.set_ylabel('ΔA')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 尝试保存可视化，处理PIL错误
        try:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            try:
                relative_path = output_file.relative_to(self.output_root)
                print(f"     💾 可视化已保存: {relative_path}")
            except ValueError:
                print(f"     💾 可视化已保存: {output_file}")
        except Exception as e:
            print(f"     ⚠️ 高质量保存失败，尝试低质量保存: {e}")
            try:
                # 降低DPI并移除bbox_inches参数
                plt.savefig(output_file, dpi=100)
                try:
                    relative_path = output_file.relative_to(self.output_root)
                    print(f"     💾 可视化已保存(低质量): {relative_path}")
                except ValueError:
                    print(f"     💾 可视化已保存(低质量): {output_file}")
            except Exception as e2:
                print(f"     ❌ 可视化保存失败: {e2}")
                # 尝试保存为PDF格式
                try:
                    pdf_file = output_file.with_suffix('.pdf')
                    plt.savefig(pdf_file, format='pdf')
                    try:
                        relative_path = pdf_file.relative_to(self.output_root)
                        print(f"     💾 可视化已保存为PDF: {relative_path}")
                    except ValueError:
                        print(f"     💾 可视化已保存为PDF: {pdf_file}")
                except Exception as e3:
                    print(f"     ❌ PDF保存也失败: {e3}")
        
        plt.close()
    
    def run_screening(self):
        """运行筛选"""
        print("🔍 开始筛选TAS挑战性数据")
        print("=" * 50)
        
        # 按目录分组文件，优先处理TA_Average.csv
        files_by_dir = {}
        for file_path in self.known_files:
            if not Path(file_path).exists():
                continue
            dir_path = str(Path(file_path).parent)
            if dir_path not in files_by_dir:
                files_by_dir[dir_path] = []
            files_by_dir[dir_path].append(file_path)
        
        # 对每个目录，优先处理TA_Average.csv，跳过TA_Scan*.csv
        processed_files = []
        for dir_path, files in files_by_dir.items():
            avg_files = [f for f in files if Path(f).name == 'TA_Average.csv']
            scan_files = [f for f in files if Path(f).name.startswith('TA_Scan') and Path(f).name.endswith('.csv')]
            other_files = [f for f in files if f not in avg_files and f not in scan_files]
            
            # 如果存在TA_Average.csv，跳过TA_Scan*.csv文件
            if avg_files:
                processed_files.extend(avg_files)
                print(f"📁 目录 {Path(dir_path).name}: 发现TA_Average.csv，跳过TA_Scan文件")
            else:
                processed_files.extend(scan_files)
            
            # 总是处理其他文件
            processed_files.extend(other_files)
        
        for file_path in processed_files:
            if not Path(file_path).exists():
                print(f"❌ 文件不存在: {file_path}")
                continue
            
            print(f"\n📊 分析文件: {Path(file_path).name}")
            
            # 加载数据
            data_info = self.load_tas_file(file_path)
            if data_info is None:
                continue
            
            # 分析三类特征
            print("   🔬 正在分析多峰重叠...")
            multi_peak = self.analyze_multi_peak_overlap(data_info)
            
            print("   ⚡ 正在分析瞬态衰减...")
            transient = self.analyze_transient_decay(data_info)
            
            print("   📊 正在分析信噪比...")
            low_snr = self.analyze_low_snr(data_info)
            
            analyses = {
                'multi_peak': multi_peak,
                'transient': transient,
                'low_snr': low_snr
            }
            
            # 分类结果
            categories = []
            
            print("   📋 分析结果:")
            if multi_peak['is_multi_peak']:
                categories.append('multi_peak_overlap')
                print(f"     ✅ 多峰重叠: 评分={multi_peak['score']:.3f}, 峰数={multi_peak['max_peaks']}")
            else:
                print(f"     ➖ 多峰重叠: 未检测到")
            
            if transient['is_transient']:
                categories.append('transient_decay')
                print(f"     ✅ 瞬态衰减: R²={transient['score']:.3f}, τ平均={transient['avg_time_constant']:.2f}ps")
            else:
                print(f"     ➖ 瞬态衰减: 未检测到")
            
            if low_snr['is_low_snr']:
                categories.append('low_snr')
                print(f"     ✅ 低信噪比: SNR={low_snr['snr']:.2f}")
            else:
                print(f"     ➖ 信噪比良好: SNR={low_snr['snr']:.2f}")
            
            # 保存结果
            file_result = {
                'file_name': Path(file_path).name,
                'file_path': str(Path(file_path).resolve()),  # 绝对路径
                'relative_path': str(file_path),  # 相对路径
                'shape': data_info['shape'],
                'wavelength_range': [float(data_info['wavelengths'].min()), float(data_info['wavelengths'].max())],
                'time_range': [float(data_info['time_delays'].min()), float(data_info['time_delays'].max())],
                'categories': categories,
                'analyses': self._serialize_analyses(analyses),
                # 添加光谱类型相关信息
                'spectrum_type': data_info.get('spectrum_type', 'VIS'),
                'original_wavelength_range': data_info.get('original_wavelength_range', 'N/A'),
                'cropped_wavelength_range': data_info.get('cropped_wavelength_range', 'N/A')
            }
            
            # 为每个类别保存
            if categories:
                for category in categories:
                    self.results[category].append(file_result)
                    
                    # 生成安全的可视化文件名
                    viz_file = self._generate_safe_filename(file_path, category)
                    
                    try:
                        print(f"   🎨 生成{category}可视化...")
                        self.create_comprehensive_visualization(data_info, analyses, viz_file)
                        try:
                            relative_path = viz_file.relative_to(Path.cwd())
                            print(f"     💾 可视化已保存: {relative_path}")
                        except ValueError:
                            print(f"     💾 可视化已保存: {viz_file}")
                    except Exception as e:
                        print(f"     ❌ 可视化失败: {e}")
            else:
                print("   📝 该文件不属于挑战性数据类别")
        
        # 生成报告
        self.generate_report()
        self.save_results()
    
    def _serialize_analyses(self, analyses):
        """序列化分析结果以便JSON保存"""
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj
        
        return convert(analyses)
    
    def generate_report(self):
        """生成汇总报告"""
        report_file = self.output_root / "TAS数据筛选报告.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# TAS数据筛选报告\n\n")
            f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 统计
            f.write("## 📊 筛选统计\n\n")
            f.write(f"- **多峰重叠数据**: {len(self.results['multi_peak_overlap'])} 个\n")
            f.write(f"- **瞬态衰减数据**: {len(self.results['transient_decay'])} 个\n")
            f.write(f"- **低信噪比数据**: {len(self.results['low_snr'])} 个\n\n")
            
            # 详细结果
            for category, title in [
                ('multi_peak_overlap', '多峰重叠数据'),
                ('transient_decay', '瞬态衰减数据'),
                ('low_snr', '低信噪比数据')
            ]:
                f.write(f"## 🎯 {title}\n\n")
                
                if self.results[category]:
                    for i, item in enumerate(self.results[category], 1):
                        f.write(f"### {i}. {item['file_name']}\n\n")
                        f.write(f"- **文件路径**: `{Path(item['file_path']).resolve()}`\n")
                        f.write(f"- **相对路径**: `{item.get('relative_path', item['file_path'])}`\n")
                        f.write(f"- **光谱类型**: {item.get('spectrum_type', 'VIS')}\n")
                        f.write(f"- **原始波长范围**: {item.get('original_wavelength_range', 'N/A')}\n")
                        f.write(f"- **裁剪后波长范围**: {item.get('cropped_wavelength_range', item['wavelength_range'])}\n")
                        f.write(f"- **数据形状**: {item['shape'][0]}×{item['shape'][1]} (波长×时间)\n")
                        f.write(f"- **时间范围**: {item['time_range'][0]:.2f} - {item['time_range'][1]:.2f} ps\n")
                        
                        # 特征分析结果
                        analyses = item['analyses']
                        if category == 'multi_peak_overlap':
                            mp = analyses['multi_peak']
                            f.write(f"- **重叠评分**: {mp['score']:.3f}\n")
                            f.write(f"- **最大峰数**: {mp['max_peaks']}\n")
                            f.write(f"- **重叠时间点数**: {mp['num_overlap_times']}\n")
                        elif category == 'transient_decay':
                            td = analyses['transient']
                            f.write(f"- **平均拟合R²**: {td['score']:.3f}\n")
                            f.write(f"- **平均时间常数**: {td['avg_time_constant']:.2f} ps\n")
                            f.write(f"- **衰减波长数**: {td['num_decay_wavelengths']}\n")
                        elif category == 'low_snr':
                            ls = analyses['low_snr']
                            f.write(f"- **信噪比**: {ls['snr']:.2f}\n")
                            f.write(f"- **噪声水平**: {ls['noise_level']:.2e}\n")
                            f.write(f"- **信号强度**: {ls['signal_strength']:.2e}\n")
                        
                        # 可视化文件
                        parent_dir = Path(item['file_path']).parent.name
                        safe_name = f"{parent_dir}_{Path(item['file_path']).stem}".replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                        viz_file = f"{category}/{safe_name}.png"
                        f.write(f"- **可视化文件**: {viz_file}\n\n")
                        f.write(f"![{title}可视化]({viz_file})\n\n")
                else:
                    f.write("暂未找到符合条件的数据。\n\n")
            
            # 使用建议
            f.write("## 💡 使用建议\n\n")
            f.write("### 多峰重叠数据\n")
            f.write("- 测试MCR-ALS在复杂光谱重叠下的组分分辨能力\n")
            f.write("- 建议使用更多组分数和严格约束\n")
            f.write("- 可尝试不同的初始化方法\n\n")
            
            f.write("### 瞬态衰减数据\n")
            f.write("- 验证MCR-ALS对时间分辨动力学的分析能力\n")
            f.write("- 关注时间常数的物理合理性\n")
            f.write("- 可结合已知动力学模型验证\n\n")
            
            f.write("### 低信噪比数据\n")
            f.write("- 测试算法在噪声环境下的鲁棒性\n")
            f.write("- 建议增加随机初始化次数\n")
            f.write("- 可尝试预处理降噪技术\n\n")
        
        print(f"\n📋 汇总报告已生成: {report_file}")
    
    def save_results(self):
        """保存结果"""
        results_file = self.output_root / "screening_results.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 详细结果已保存: {results_file}")

def main():
    """主函数"""
    print("🎯 TAS挑战性数据筛选器 - 健壮版")
    print("=" * 50)
    
    screener = RobustTASScreener()
    screener.run_screening()
    
    # 输出最终统计
    print("\n" + "=" * 50)
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