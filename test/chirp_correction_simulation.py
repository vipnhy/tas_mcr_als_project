#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于合成数据的啁啾校正演示脚本

步骤：
1. 生成带有已知啁啾的合成瞬态吸收数据
2. 使用 ChirpCorrector 进行啁啾校正
3. 输出校正前后的可视化图像
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from typing import Dict, Tuple

# 确保可以导入项目模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.chirp_correction import ChirpCorrector

# 项目根目录
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def generate_simulated_tas_data(
    n_wavelengths: int = 180,
    n_times: int = 600,
    wavelength_range: Tuple[float, float] = (360.0, 750.0),
    time_range: Tuple[float, float] = (-0.5, 5.0),
    noise_level: float = 2e-4,
    random_seed: int = 42,
) -> Dict[str, np.ndarray]:
    """生成带啁啾的合成 TAS 数据"""
    rng = np.random.default_rng(random_seed)

    wavelengths = np.linspace(*wavelength_range, n_wavelengths)
    time_delays = np.linspace(*time_range, n_times)

    # 定义两个动力学分量：瞬态激发 + 缓慢衰减
    tau_fast = 0.15  # ps
    tau_slow = 1.2   # ps
    gaussian_width = 0.08

    # 波长依赖的振幅
    amp_fast = np.exp(-0.5 * ((wavelengths - 470.0) / 30.0) ** 2)
    amp_slow = 0.6 * np.exp(-0.5 * ((wavelengths - 620.0) / 50.0) ** 2)

    true_signal = np.zeros((n_wavelengths, n_times), dtype=float)

    for i, wl in enumerate(wavelengths):
        pulse = amp_fast[i] * np.exp(-((time_delays) ** 2) / (2 * gaussian_width ** 2))
        kinetic_fast = np.heaviside(time_delays, 0.0) * np.exp(-np.clip(time_delays, 0, None) / tau_fast)
        kinetic_slow = np.heaviside(time_delays, 0.0) * np.exp(-np.clip(time_delays, 0, None) / tau_slow)
        true_signal[i, :] = pulse + amp_fast[i] * kinetic_fast + amp_slow[i] * kinetic_slow

    # 定义啁啾函数（多项式 + 正弦微扰）
    base = (wavelengths - np.mean(wavelengths)) / (wavelength_range[1] - wavelength_range[0])
    chirp_curve = 0.18 * base + 0.25 * base ** 2 - 0.12 * base ** 3
    chirp_curve += 0.02 * np.sin(2 * np.pi * (wavelengths - wavelength_range[0]) / 150.0)

    # 将啁啾函数转换为时间偏移（ps）
    chirp_offsets = 0.45 * chirp_curve  # 最大偏移约 ±0.2 ps

    # 应用啁啾：对每个波长沿时间轴平移
    chirped_signal = np.zeros_like(true_signal)
    for i in range(n_wavelengths):
        shift = chirp_offsets[i]
        shifted_time = time_delays - shift
        chirped_signal[i, :] = np.interp(
            time_delays, shifted_time, true_signal[i, :], left=0.0, right=0.0
        )

    # 添加噪声
    noise = rng.normal(0.0, noise_level, size=chirped_signal.shape)
    chirped_noisy = chirped_signal + noise

    return {
        "wavelengths": wavelengths,
        "time_delays": time_delays,
        "true_signal": true_signal,
        "chirp_offsets": chirp_offsets,
        "observed": chirped_noisy,
    }


def visualize_before_after(
    wavelengths: np.ndarray,
    time_delays: np.ndarray,
    original: np.ndarray,
    corrected: np.ndarray,
    title_suffix: str = "",
    output_prefix: str = "tas_chirp_simulated",
) -> None:
    """生成校正前后对比图像"""
    plt.rcParams.update({
        "font.size": 11,
        "axes.unicode_minus": False,
        "figure.dpi": 110,
    })

    # 选择代表性波长
    indices = np.linspace(5, len(wavelengths) - 6, 5, dtype=int)
    colors = plt.cm.plasma(np.linspace(0, 1, len(indices)))

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for idx, color in zip(indices, colors):
        axes[0].plot(
            time_delays,
            original[idx, :],
            color=color,
            alpha=0.8,
            label=f"{wavelengths[idx]:.0f} nm",
        )
        axes[1].plot(
            time_delays,
            corrected[idx, :],
            color=color,
            alpha=0.8,
            label=f"{wavelengths[idx]:.0f} nm",
        )

    axes[0].set_title(f"Transient kinetics before chirp correction {title_suffix}")
    axes[1].set_title(f"Transient kinetics after chirp correction {title_suffix}")
    axes[1].set_xlabel("Delay time / ps")
    axes[0].set_ylabel("ΔOD")
    axes[1].set_ylabel("ΔOD")
    axes[0].legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    line_path = RESULTS_DIR / f"{output_prefix}_kinetics.png"
    fig.savefig(line_path, bbox_inches="tight")
    plt.close(fig)

    # 热图对比
    abs_max = float(
        np.nanmax(np.abs(np.concatenate([original.ravel(), corrected.ravel()])))
    )
    if abs_max == 0 or not np.isfinite(abs_max):
        abs_max = 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-abs_max, vmax=abs_max)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    im0 = axes[0].imshow(
        original,
        aspect="auto",
        origin="lower",
        extent=[time_delays.min(), time_delays.max(), wavelengths.min(), wavelengths.max()],
        cmap="RdBu_r",
        norm=norm,
    )
    axes[0].set_title("Before correction")
    axes[0].set_xlabel("Delay time / ps")
    axes[0].set_ylabel("Wavelength / nm")

    im1 = axes[1].imshow(
        corrected,
        aspect="auto",
        origin="lower",
        extent=[time_delays.min(), time_delays.max(), wavelengths.min(), wavelengths.max()],
        cmap="RdBu_r",
        norm=norm,
    )
    axes[1].set_title("After correction")
    axes[1].set_xlabel("Delay time / ps")

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("ΔOD")

    fig.subplots_adjust(wspace=0.06, bottom=0.18, top=0.9)
    heatmap_path = RESULTS_DIR / f"{output_prefix}_heatmap.png"
    fig.savefig(heatmap_path, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ 啁啾校正前后动力学对比图保存至: {line_path}")
    print(f"✅ 啁啾校正前后热图保存至: {heatmap_path}")


def main():
    print("🔬 生成合成啁啾 TAS 数据...")
    simulated = generate_simulated_tas_data()

    wavelengths = simulated["wavelengths"]
    time_delays = simulated["time_delays"]
    observed = simulated["observed"]

    print("⚙️  使用 ChirpCorrector(method='cross_correlation') 进行校正...")
    corrector = ChirpCorrector(method="cross_correlation", verbose=True)
    observed_df = pd.DataFrame(observed.T, index=time_delays, columns=wavelengths)
    corrector.fit_chirp(observed_df)
    corrected_df = corrector.apply_correction(observed_df)
    corrected = corrected_df.T.values

    visualize_before_after(
        wavelengths,
        time_delays,
        observed,
        corrected,
        title_suffix="(synthetic)",
        output_prefix="tas_chirp_simulated",
    )

    # 返回可选统计信息
    stats = corrector.get_correction_stats()
    if stats:
        fit_quality = stats.get("fit_quality", {})
        print("📈 啁啾校正统计:")
        print(f"   方法: {stats.get('method')}")
        print(f"   偏移范围: {fit_quality.get('offset_range', 0):.3f} ps")
        print(f"   标准差: {fit_quality.get('offset_std', 0):.3f} ps")


if __name__ == "__main__":
    main()
