#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成三类挑战性TAS数据的脚本
- 多峰重叠数据（模拟复杂体系）
- 瞬态信号衰减数据（时间分辨验证）
- 低信噪比数据（SNR=5:1）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("experiments/results/data_screening")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def generate_multi_peak_overlap_data():
    """生成多峰重叠复杂体系数据"""
    print("🎯 生成多峰重叠数据...")
    
    # 波长范围 (nm)
    wavelengths = np.linspace(400, 700, 300)
    # 时间延迟 (ps)
    time_delays = np.linspace(-1, 50, 150)
    
    # 创建多个重叠的光谱组分
    data = np.zeros((len(wavelengths), len(time_delays)))
    
    # 组分1: 宽峰在450nm
    center1 = 450
    width1 = 60
    for i, t in enumerate(time_delays):
        if t > 0:
            # 时间演化
            amplitude = 0.8 * np.exp(-t/10) + 0.2
            spectrum = amplitude * np.exp(-((wavelengths - center1)**2) / (2 * width1**2))
            data[:, i] += spectrum
    
    # 组分2: 中等峰在500nm，与组分1重叠
    center2 = 500
    width2 = 40
    for i, t in enumerate(time_delays):
        if t > 1:
            amplitude = 0.6 * np.exp(-t/15) + 0.3
            spectrum = amplitude * np.exp(-((wavelengths - center2)**2) / (2 * width2**2))
            data[:, i] += spectrum
    
    # 组分3: 窄峰在520nm，与组分2严重重叠
    center3 = 520
    width3 = 25
    for i, t in enumerate(time_delays):
        if t > 2:
            amplitude = 0.9 * np.exp(-t/8) + 0.1
            spectrum = amplitude * np.exp(-((wavelengths - center3)**2) / (2 * width3**2))
            data[:, i] += spectrum
    
    # 组分4: 另一个窄峰在550nm
    center4 = 550
    width4 = 30
    for i, t in enumerate(time_delays):
        if t > 3:
            amplitude = 0.7 * np.exp(-t/12) + 0.2
            spectrum = amplitude * np.exp(-((wavelengths - center4)**2) / (2 * width4**2))
            data[:, i] += spectrum
    
    # 添加小量噪声
    noise = np.random.normal(0, 0.02, data.shape)
    data += noise
    
    return wavelengths, time_delays, data

def generate_transient_decay_data():
    """生成瞬态衰减数据"""
    print("⚡ 生成瞬态衰减数据...")
    
    # 波长范围 (nm)
    wavelengths = np.linspace(450, 650, 200)
    # 时间延迟 (ps) - 更长的时间范围以观察衰减
    time_delays = np.linspace(-2, 100, 200)
    
    data = np.zeros((len(wavelengths), len(time_delays)))
    
    # 激发态吸收特征
    for i, wl in enumerate(wavelengths):
        for j, t in enumerate(time_delays):
            if t <= 0:
                # 负时间延迟：基态漂白
                if 480 <= wl <= 520:
                    data[i, j] = -0.3 * np.exp(-((wl-500)**2)/400)
            else:
                # 正时间延迟：多指数衰减
                signal = 0
                
                # 快速组分 (τ1 = 2 ps)
                if 460 <= wl <= 500:
                    signal += 0.8 * np.exp(-t/2) * np.exp(-((wl-480)**2)/300)
                
                # 中等组分 (τ2 = 15 ps)
                if 500 <= wl <= 580:
                    signal += 0.6 * np.exp(-t/15) * np.exp(-((wl-540)**2)/600)
                
                # 慢速组分 (τ3 = 50 ps)
                if 580 <= wl <= 620:
                    signal += 0.4 * np.exp(-t/50) * np.exp(-((wl-600)**2)/400)
                
                # 基态漂白恢复
                if 480 <= wl <= 520:
                    bleach_recovery = -0.3 * np.exp(-t/10) * np.exp(-((wl-500)**2)/400)
                    signal += bleach_recovery
                
                data[i, j] = signal
    
    # 添加少量噪声
    noise = np.random.normal(0, 0.01, data.shape)
    data += noise
    
    return wavelengths, time_delays, data

def generate_low_snr_data():
    """生成低信噪比数据 (SNR ≈ 5:1)"""
    print("📉 生成低信噪比数据...")
    
    # 波长范围 (nm)
    wavelengths = np.linspace(400, 800, 400)
    # 时间延迟 (ps)
    time_delays = np.linspace(-5, 30, 100)
    
    data = np.zeros((len(wavelengths), len(time_delays)))
    
    # 弱信号：小的吸收特征
    for i, t in enumerate(time_delays):
        if t > 0:
            # 主要信号特征
            center1 = 500
            width1 = 50
            amplitude1 = 0.05 * np.exp(-t/8)  # 很弱的信号
            spectrum1 = amplitude1 * np.exp(-((wavelengths - center1)**2) / (2 * width1**2))
            
            center2 = 600
            width2 = 40
            amplitude2 = 0.03 * np.exp(-t/12)  # 更弱的信号
            spectrum2 = amplitude2 * np.exp(-((wavelengths - center2)**2) / (2 * width2**2))
            
            data[:, i] += spectrum1 + spectrum2
    
    # 添加高噪声使SNR约为5:1
    signal_max = np.max(np.abs(data))
    noise_level = signal_max / 5  # SNR = 5:1
    noise = np.random.normal(0, noise_level, data.shape)
    data += noise
    
    return wavelengths, time_delays, data

def save_tas_data(wavelengths, time_delays, data, filename):
    """保存TAS数据为CSV格式"""
    # 创建DataFrame
    df = pd.DataFrame(data, index=wavelengths, columns=time_delays)
    
    # 保存
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / filename
    df.to_csv(output_path)
    print(f"   💾 已保存: {output_path}")
    
    return output_path

def create_preview_plots(wavelengths, time_delays, data, title, output_file):
    """创建预览图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 2D热图
    ax1 = axes[0, 0]
    im = ax1.imshow(data, aspect='auto', cmap='rainbow',
                   extent=[time_delays.min(), time_delays.max(),
                          wavelengths.max(), wavelengths.min()])
    ax1.set_xlabel('时间延迟 (ps)')
    ax1.set_ylabel('波长 (nm)')
    ax1.set_title(f'{title} - 2D热图')
    plt.colorbar(im, ax=ax1, label='ΔA')
    
    # 时间切片
    ax2 = axes[0, 1]
    mid_time_idx = len(time_delays) // 2
    spectrum = data[:, mid_time_idx]
    ax2.plot(wavelengths, spectrum)
    ax2.set_xlabel('波长 (nm)')
    ax2.set_ylabel('ΔA')
    ax2.set_title(f'光谱切片 (t={time_delays[mid_time_idx]:.2f}ps)')
    ax2.grid(True, alpha=0.3)
    
    # 动力学曲线
    ax3 = axes[1, 0]
    mid_wl_idx = len(wavelengths) // 2
    kinetic = data[mid_wl_idx, :]
    ax3.plot(time_delays, kinetic)
    ax3.set_xlabel('时间延迟 (ps)')
    ax3.set_ylabel('ΔA')
    ax3.set_title(f'动力学曲线 (λ={wavelengths[mid_wl_idx]:.0f}nm)')
    ax3.grid(True, alpha=0.3)
    
    # 信号统计
    ax4 = axes[1, 1]
    signal_strength = np.max(np.abs(data), axis=0)
    ax4.plot(time_delays, signal_strength)
    ax4.set_xlabel('时间延迟 (ps)')
    ax4.set_ylabel('最大信号强度')
    ax4.set_title('信号强度随时间变化')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   🎨 预览图已保存: {output_file}")

def main():
    """主函数"""
    print("🏭 生成三类挑战性TAS数据")
    print("=" * 50)
    
    # 确保输出目录存在
    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 生成多峰重叠数据
    wl1, td1, data1 = generate_multi_peak_overlap_data()
    file1 = save_tas_data(wl1, td1, data1, "challenging_multi_peak_overlap.csv")
    create_preview_plots(wl1, td1, data1, "多峰重叠数据", 
                        output_dir / "preview_multi_peak_overlap.png")
    
    # 2. 生成瞬态衰减数据
    wl2, td2, data2 = generate_transient_decay_data()
    file2 = save_tas_data(wl2, td2, data2, "challenging_transient_decay.csv")
    create_preview_plots(wl2, td2, data2, "瞬态衰减数据", 
                        output_dir / "preview_transient_decay.png")
    
    # 3. 生成低信噪比数据
    wl3, td3, data3 = generate_low_snr_data()
    file3 = save_tas_data(wl3, td3, data3, "challenging_low_snr.csv")
    create_preview_plots(wl3, td3, data3, "低信噪比数据", 
                        output_dir / "preview_low_snr.png")
    
    print("\n" + "=" * 50)
    print("🎉 三类挑战性数据生成完成！")
    print(f"📁 输出目录: {output_dir}")
    print("📋 生成的文件:")
    print("  🎯 多峰重叠数据: challenging_multi_peak_overlap.csv")
    print("  ⚡ 瞬态衰减数据: challenging_transient_decay.csv")
    print("  📉 低信噪比数据: challenging_low_snr.csv")
    print("\n💡 接下来可以运行筛选器验证这些数据的分类效果！")

if __name__ == "__main__":
    main()