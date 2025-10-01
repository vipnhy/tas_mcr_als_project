#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TAS数据筛选器 - 简化版本，针对已知的工作数据文件
专门筛选三类挑战性数据并生成可视化
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

class SimpleTASScreener:
    """简化的TAS数据筛选器"""
    
    def __init__(self, output_root="experiments/results/data_screening"):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # 已知的工作数据文件
        self.known_files = [
            "data/TAS/TA_Average.csv",
            "data/TAS/TA_Average-crop-TCA-tzs-tzs-tzs-chirp.csv"
        ]
        
        self.results = {
            'multi_peak_overlap': [],
            'transient_decay': [],
            'low_snr': [],
            'analysis_summary': {}
        }
    
    def load_tas_file(self, file_path):
        """加载TAS文件"""
        try:
            # 读取CSV文件，第一行为时间延迟，第一列为波长
            df = pd.read_csv(file_path, index_col=0)
            
            # 获取波长和时间延迟
            wavelengths = df.index.values.astype(float)
            time_delays = df.columns.astype(float)
            data = df.values
            
            # 基本验证
            if data.shape[0] < 10 or data.shape[1] < 10:
                return None
            
            # 处理异常值
            data = np.where(np.isfinite(data), data, 0)
            
            return {
                'data': data,
                'wavelengths': wavelengths,
                'time_delays': time_delays,
                'shape': data.shape,
                'file_path': str(file_path)
            }
            
        except Exception as e:
            print(f"加载失败 {file_path}: {e}")
            return None
    
    def analyze_multi_peak_overlap(self, data_info):
        """分析多峰重叠特征"""
        data = data_info['data']
        wavelengths = data_info['wavelengths']
        
        # 寻找有明显信号的时间点
        signal_strength = np.max(np.abs(data), axis=0)
        strong_signal_idx = np.where(signal_strength > np.percentile(signal_strength, 70))[0]
        
        overlap_scores = []
        
        for idx in strong_signal_idx[:10]:  # 检查前10个强信号时间点
            spectrum = data[:, idx]
            abs_spectrum = np.abs(spectrum)
            
            # 找峰
            prominence = np.std(abs_spectrum) * 0.2
            peaks, properties = signal.find_peaks(
                abs_spectrum,
                prominence=prominence,
                distance=len(abs_spectrum) // 20
            )
            
            if len(peaks) >= 3:  # 至少3个峰
                # 计算峰间重叠
                try:
                    peak_widths = signal.peak_widths(abs_spectrum, peaks, rel_height=0.5)[0]
                    peak_distances = np.diff(peaks)
                    
                    if len(peak_distances) > 0:
                        overlap_ratio = np.mean(peak_widths[:-1]) / np.mean(peak_distances)
                        
                        if overlap_ratio > 0.5:  # 重叠阈值
                            overlap_scores.append({
                                'time_index': idx,
                                'time_delay': data_info['time_delays'][idx],
                                'num_peaks': len(peaks),
                                'overlap_ratio': overlap_ratio,
                                'peaks': peaks,
                                'peak_heights': abs_spectrum[peaks]
                            })
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
                'details': overlap_scores
            }
        
        return {'is_multi_peak': False, 'score': 0}
    
    def analyze_transient_decay(self, data_info):
        """分析瞬态衰减特征"""
        data = data_info['data']
        time_delays = data_info['time_delays']
        
        # 只分析正时间延迟
        pos_mask = time_delays > 0
        if np.sum(pos_mask) < 5:
            return {'is_transient': False, 'score': 0}
        
        pos_times = time_delays[pos_mask]
        pos_data = data[:, pos_mask]
        
        decay_results = []
        
        # 分析有强信号的波长
        max_signal_per_wavelength = np.max(np.abs(pos_data), axis=1)
        strong_wavelengths = np.where(max_signal_per_wavelength > np.percentile(max_signal_per_wavelength, 80))[0]
        
        for wl_idx in strong_wavelengths[:15]:  # 检查前15个强信号波长
            kinetic = pos_data[wl_idx, :]
            
            # 检查衰减趋势
            if len(kinetic) < 5:
                continue
                
            initial = np.mean(kinetic[:3])
            final = np.mean(kinetic[-3:])
            
            if abs(initial) < 1e-10:
                continue
                
            decay_ratio = abs(initial / final) if abs(final) > 1e-10 else 100
            
            if decay_ratio > 1.5:  # 有衰减趋势
                try:
                    # 简单的指数衰减拟合
                    def exp_decay(t, a, tau, c):
                        return a * np.exp(-t / tau) + c
                    
                    # 初始猜测
                    popt, _ = curve_fit(
                        exp_decay,
                        pos_times,
                        kinetic,
                        p0=[initial - final, pos_times[len(pos_times)//2], final],
                        bounds=([-np.inf, 0.01, -np.inf], [np.inf, 1000, np.inf]),
                        maxfev=500
                    )
                    
                    # 计算拟合质量
                    fitted = exp_decay(pos_times, *popt)
                    r2 = 1 - np.sum((kinetic - fitted)**2) / np.sum((kinetic - np.mean(kinetic))**2)
                    
                    if r2 > 0.6 and 0.1 <= abs(popt[1]) <= 100:  # 时间常数合理
                        decay_results.append({
                            'wavelength_index': wl_idx,
                            'wavelength': data_info['wavelengths'][wl_idx],
                            'time_constant': abs(popt[1]),
                            'r2': r2,
                            'decay_ratio': decay_ratio,
                            'amplitude': popt[0]
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
                'details': decay_results
            }
        
        return {'is_transient': False, 'score': 0}
    
    def analyze_low_snr(self, data_info):
        """分析信噪比"""
        data = data_info['data']
        
        # 估计噪声（使用边缘区域）
        h, w = data.shape
        edge_size = max(2, min(h//15, w//15))
        
        noise_regions = [
            data[:edge_size, :edge_size],
            data[:edge_size, -edge_size:],
            data[-edge_size:, :edge_size],
            data[-edge_size:, -edge_size:]
        ]
        
        noise_stds = [np.std(region) for region in noise_regions if region.size > 0]
        noise_level = np.mean(noise_stds) if noise_stds else 0
        
        # 估计信号强度（中心区域最大值）
        center_h = slice(h//4, 3*h//4)
        center_w = slice(w//4, 3*w//4)
        center_data = data[center_h, center_w]
        signal_strength = np.max(np.abs(center_data))
        
        # 计算SNR
        snr = signal_strength / noise_level if noise_level > 0 else float('inf')
        
        return {
            'is_low_snr': snr <= 5.0,
            'snr': snr,
            'noise_level': noise_level,
            'signal_strength': signal_strength
        }
    
    def create_comprehensive_visualization(self, data_info, analyses, output_file):
        """创建综合可视化"""
        data = data_info['data']
        wavelengths = data_info['wavelengths']
        time_delays = data_info['time_delays']
        file_name = Path(data_info['file_path']).name
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 主要2D热图
        ax1 = plt.subplot(3, 4, (1, 2))
        im = ax1.imshow(data, aspect='auto', cmap='RdBu_r',
                       extent=[time_delays.min(), time_delays.max(),
                              wavelengths.max(), wavelengths.min()])
        ax1.set_xlabel('时间延迟 (ps)')
        ax1.set_ylabel('波长 (nm)')
        ax1.set_title(f'TAS 2D 光谱热图\n{file_name}', fontweight='bold')
        plt.colorbar(im, ax=ax1, label='ΔA')
        
        # 2. 时间切片光谱（多峰重叠分析）
        ax2 = plt.subplot(3, 4, 3)
        if analyses['multi_peak']['is_multi_peak']:
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
            ax2.plot(wavelengths, np.abs(data[:, len(time_delays)//4]), 'b-', linewidth=2)
            ax2.set_title('未检测到多峰重叠')
        
        ax2.set_xlabel('波长 (nm)')
        ax2.set_ylabel('|ΔA|')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 动力学曲线（瞬态衰减分析）
        ax3 = plt.subplot(3, 4, 4)
        if analyses['transient']['is_transient']:
            detail = analyses['transient']['details'][0]
            wl_idx = detail['wavelength_index']
            kinetic = data[wl_idx, time_delays > 0]
            pos_times = time_delays[time_delays > 0]
            
            ax3.semilogy(pos_times, np.abs(kinetic), 'bo', markersize=4, label='实验数据')
            
            # 拟合曲线
            tau = detail['time_constant']
            amp = detail['amplitude']
            fitted = amp * np.exp(-pos_times / tau) + kinetic[-1]
            ax3.semilogy(pos_times, np.abs(fitted), 'r-', linewidth=2,
                        label=f'拟合 (τ={tau:.2f}ps)')
            
            ax3.set_title(f'瞬态衰减\nλ={wavelengths[wl_idx]:.0f}nm\nR²={detail["r2"]:.3f}')
        else:
            center_wl = len(wavelengths) // 2
            kinetic = data[center_wl, time_delays > 0]
            pos_times = time_delays[time_delays > 0]
            ax3.plot(pos_times, kinetic, 'b-', linewidth=2)
            ax3.set_title('未检测到明显衰减')
        
        ax3.set_xlabel('时间延迟 (ps)')
        ax3.set_ylabel('ΔA')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 信噪比分析
        ax4 = plt.subplot(3, 4, 5)
        snr = analyses['low_snr']['snr']
        data_flat = data.flatten()
        
        ax4.hist(data_flat, bins=50, alpha=0.7, density=True, color='skyblue')
        ax4.axvline(0, color='red', linestyle='--', alpha=0.7, label='零基线')
        ax4.set_title(f'数据分布\nSNR = {snr:.2f}')
        ax4.set_xlabel('信号强度')
        ax4.set_ylabel('密度')
        ax4.legend()
        
        # 5-8. 不同时间点的光谱
        time_indices = [0, len(time_delays)//4, len(time_delays)//2, 3*len(time_delays)//4]
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, (t_idx, color) in enumerate(zip(time_indices, colors)):
            ax = plt.subplot(3, 4, 6+i)
            spectrum = data[:, t_idx]
            ax.plot(wavelengths, spectrum, color=color, linewidth=2)
            ax.set_title(f't = {time_delays[t_idx]:.2f} ps')
            ax.set_xlabel('波长 (nm)')
            ax.set_ylabel('ΔA')
            ax.grid(True, alpha=0.3)
        
        # 9-12. 不同波长的动力学
        wl_indices = [len(wavelengths)//6, len(wavelengths)//3, 2*len(wavelengths)//3, 5*len(wavelengths)//6]
        
        for i, wl_idx in enumerate(wl_indices):
            ax = plt.subplot(3, 4, 10+i)
            kinetic = data[wl_idx, :]
            ax.plot(time_delays, kinetic, linewidth=2, color=colors[i])
            ax.set_title(f'λ = {wavelengths[wl_idx]:.0f} nm')
            ax.set_xlabel('时间延迟 (ps)')
            ax.set_ylabel('ΔA')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_screening(self):
        """运行筛选"""
        print("🔍 开始筛选TAS挑战性数据")
        print("=" * 50)
        
        for file_path in self.known_files:
            if not Path(file_path).exists():
                print(f"❌ 文件不存在: {file_path}")
                continue
            
            print(f"\n📊 分析文件: {Path(file_path).name}")
            
            # 加载数据
            data_info = self.load_tas_file(file_path)
            if data_info is None:
                print("❌ 加载失败")
                continue
            
            print(f"   数据形状: {data_info['shape']} (波长×时间)")
            print(f"   波长范围: {data_info['wavelengths'].min():.1f} - {data_info['wavelengths'].max():.1f} nm")
            print(f"   时间范围: {data_info['time_delays'].min():.2f} - {data_info['time_delays'].max():.2f} ps")
            
            # 分析三类特征
            multi_peak = self.analyze_multi_peak_overlap(data_info)
            transient = self.analyze_transient_decay(data_info)
            low_snr = self.analyze_low_snr(data_info)
            
            analyses = {
                'multi_peak': multi_peak,
                'transient': transient,
                'low_snr': low_snr
            }
            
            # 分类结果
            categories = []
            
            if multi_peak['is_multi_peak']:
                categories.append('multi_peak_overlap')
                print(f"   ✅ 多峰重叠: 评分={multi_peak['score']:.3f}, 峰数={multi_peak['max_peaks']}")
            
            if transient['is_transient']:
                categories.append('transient_decay')
                print(f"   ✅ 瞬态衰减: R²={transient['score']:.3f}, τ平均={transient['avg_time_constant']:.2f}ps")
            
            if low_snr['is_low_snr']:
                categories.append('low_snr')
                print(f"   ✅ 低信噪比: SNR={low_snr['snr']:.2f}")
            
            if not categories:
                print(f"   ➖ 常规数据 (SNR={low_snr['snr']:.2f})")
            
            # 保存结果
            file_result = {
                'file_name': Path(file_path).name,
                'file_path': file_path,
                'shape': data_info['shape'],
                'wavelength_range': [float(data_info['wavelengths'].min()), float(data_info['wavelengths'].max())],
                'time_range': [float(data_info['time_delays'].min()), float(data_info['time_delays'].max())],
                'categories': categories,
                'analyses': self._serialize_analyses(analyses)
            }
            
            # 为每个类别保存
            for category in categories:
                self.results[category].append(file_result)
                
                # 创建类别目录
                category_dir = self.output_root / category
                category_dir.mkdir(exist_ok=True)
                
                # 生成可视化
                safe_name = Path(file_path).stem.replace(' ', '_').replace('-', '_')
                viz_file = category_dir / f"{safe_name}.png"
                
                try:
                    self.create_comprehensive_visualization(data_info, analyses, viz_file)
                    print(f"     💾 {category} 可视化: {viz_file.relative_to(self.output_root)}")
                except Exception as e:
                    print(f"     ❌ 可视化失败: {e}")
        
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
                        f.write(f"- **文件路径**: `{item['file_path']}`\n")
                        f.write(f"- **数据形状**: {item['shape'][0]}×{item['shape'][1]} (波长×时间)\n")
                        f.write(f"- **波长范围**: {item['wavelength_range'][0]:.1f} - {item['wavelength_range'][1]:.1f} nm\n")
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
                        safe_name = Path(item['file_path']).stem.replace(' ', '_').replace('-', '_')
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
    print("🎯 TAS挑战性数据筛选器 - 简化版")
    print("=" * 50)
    
    screener = SimpleTASScreener()
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