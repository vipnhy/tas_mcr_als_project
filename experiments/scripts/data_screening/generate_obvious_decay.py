#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
专门生成明显瞬态衰减特征的TAS数据
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("experiments/results/data_screening")

def generate_obvious_transient_decay():
    """生成具有明显瞬态衰减特征的数据"""
    print("⚡ 生成明显瞬态衰减数据...")
    
    # 波长范围 (nm)
    wavelengths = np.linspace(450, 650, 200)
    # 时间延迟 (ps) - 重点在正时间
    time_delays = np.linspace(-2, 80, 180)
    
    data = np.zeros((len(wavelengths), len(time_delays)))
    
    # 为几个特定波长创建明显的指数衰减
    decay_wavelengths = [480, 520, 580]
    decay_constants = [3, 8, 20]  # ps
    
    for wl_center, tau in zip(decay_wavelengths, decay_constants):
        wl_mask = np.abs(wavelengths - wl_center) <= 15
        wl_indices = np.where(wl_mask)[0]
        
        for i in wl_indices:
            for j, t in enumerate(time_delays):
                if t > 0:
                    # 强烈的指数衰减信号
                    amplitude = 1.0 if i == wl_indices[len(wl_indices)//2] else 0.7
                    signal = amplitude * np.exp(-t / tau)
                    data[i, j] += signal
                else:
                    # 负时间基本无信号
                    data[i, j] = 0.0
    
    # 添加极少噪声以保持清晰的衰减特征
    noise = np.random.normal(0, 0.005, data.shape)
    data += noise
    
    return wavelengths, time_delays, data

def save_and_test():
    """生成并测试数据"""
    wl, td, data = generate_obvious_transient_decay()
    
    # 保存数据
    df = pd.DataFrame(data, index=wl, columns=td)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / "obvious_transient_decay.csv"
    df.to_csv(output_file)
    print(f"   💾 已保存: {output_file}")
    
    # 创建验证图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 2D热图
    ax1 = axes[0, 0]
    im = ax1.imshow(data, aspect='auto', cmap='RdBu_r',
                   extent=[td.min(), td.max(), wl.max(), wl.min()])
    ax1.set_xlabel('时间延迟 (ps)')
    ax1.set_ylabel('波长 (nm)')
    ax1.set_title('明显瞬态衰减数据 - 2D热图')
    plt.colorbar(im, ax=ax1, label='ΔA')
    
    # 几个波长的动力学曲线
    ax2 = axes[0, 1]
    pos_mask = td > 0
    pos_td = td[pos_mask]
    
    for wl_val in [480, 520, 580]:
        wl_idx = np.argmin(np.abs(wl - wl_val))
        kinetic = data[wl_idx, pos_mask]
        ax2.semilogy(pos_td, kinetic + 1e-6, label=f'{wl_val} nm')
    
    ax2.set_xlabel('时间延迟 (ps)')
    ax2.set_ylabel('ΔA (log scale)')
    ax2.set_title('动力学衰减曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 时间切片
    ax3 = axes[1, 0]
    t_10ps = np.argmin(np.abs(td - 10))
    spectrum = data[:, t_10ps]
    ax3.plot(wl, spectrum)
    ax3.set_xlabel('波长 (nm)')
    ax3.set_ylabel('ΔA')
    ax3.set_title('t=10ps 光谱')
    ax3.grid(True, alpha=0.3)
    
    # 信号衰减分析
    ax4 = axes[1, 1]
    # 选择520nm波长的衰减
    wl_520_idx = np.argmin(np.abs(wl - 520))
    kinetic_520 = data[wl_520_idx, pos_mask]
    
    # 拟合指数衰减
    from scipy.optimize import curve_fit
    def exp_decay(t, a, tau, c):
        return a * np.exp(-t / tau) + c
    
    try:
        popt, _ = curve_fit(exp_decay, pos_td, kinetic_520, 
                           p0=[1.0, 8.0, 0.0], maxfev=1000)
        fitted = exp_decay(pos_td, *popt)
        
        ax4.plot(pos_td, kinetic_520, 'bo', label='数据')
        ax4.plot(pos_td, fitted, 'r-', label=f'拟合 τ={popt[1]:.2f}ps')
        ax4.set_xlabel('时间延迟 (ps)')
        ax4.set_ylabel('ΔA')
        ax4.set_title('520nm 指数衰减拟合')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        print(f"   ✅ 拟合时间常数: {popt[1]:.2f} ps")
    except:
        ax4.text(0.5, 0.5, '拟合失败', ha='center', va='center', 
                transform=ax4.transAxes)
    
    plt.tight_layout()
    preview_file = RESULTS_DIR / "preview_obvious_transient_decay.png"
    plt.savefig(preview_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   🎨 预览图已保存: {preview_file}")

if __name__ == "__main__":
    save_and_test()