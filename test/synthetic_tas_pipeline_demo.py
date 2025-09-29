#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模拟瞬态吸收光谱数据以测试光谱裁剪与啁啾校正"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import SpectralCropper, ChirpCorrector  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class SyntheticDataset:
    data: pd.DataFrame
    wavelengths: np.ndarray
    time_delays: np.ndarray
    true_chirp: np.ndarray
    noise_band: Tuple[float, float]


def generate_synthetic_tas_dataset(
    wl_start: float = 340.0,
    wl_stop: float = 760.0,
    wl_step: float = 2.0,
    t_start: float = -2.0,
    t_stop: float = 3.0,
    n_time: int = 600,
    noise_band_width: int = 18,
    rng_seed: int = 1234,
) -> SyntheticDataset:
    """生成同时包含边缘噪声和啁啾梯度的瞬态吸收光谱数据"""
    rng = np.random.default_rng(rng_seed)

    wavelengths = np.arange(wl_start, wl_stop + 0.1, wl_step)
    time_delays = np.linspace(t_start, t_stop, n_time)

    n_wavelengths = len(wavelengths)
    noise_band_width = min(noise_band_width, n_wavelengths // 6)

    signal_matrix = np.zeros((n_time, n_wavelengths), dtype=float)

    # 定义啁啾：从-1ps到+1ps线性变化
    valid_slice = slice(noise_band_width, n_wavelengths - noise_band_width)
    valid_wavelengths = wavelengths[valid_slice]
    chirp_centers = np.interp(valid_wavelengths, (valid_wavelengths[0], valid_wavelengths[-1]), (-1.0, 1.0))

    # 设计幅值在低波长为负、高波长为正，用于可视化突变
    amplitudes = np.linspace(-0.8, 0.9, len(valid_wavelengths))

    sigma = 0.18  # ps
    tail_tau = 0.45

    for idx, wl in enumerate(wavelengths):
        column = np.zeros_like(time_delays)
        if idx < noise_band_width or idx >= n_wavelengths - noise_band_width:
            column = rng.normal(0.0, 4e-4, size=time_delays.shape)
        else:
            local_idx = idx - noise_band_width
            center = chirp_centers[local_idx]
            amplitude = amplitudes[local_idx]

            gaussian = amplitude * np.exp(-((time_delays - center) ** 2) / (2 * sigma ** 2))
            tail = 0.6 * amplitude * np.exp(-np.clip(time_delays - center, 0, None) / tail_tau)
            column = gaussian + tail

            # 添加轻微振荡以增加复杂性
            column += 0.08 * amplitude * np.sin(2 * np.pi * (time_delays - center) / 0.6)

        # 添加背景噪声
        column += rng.normal(0.0, 1.2e-4, size=time_delays.shape)
        signal_matrix[:, idx] = column

    df = pd.DataFrame(signal_matrix, index=time_delays, columns=wavelengths)
    return SyntheticDataset(
        data=df,
        wavelengths=wavelengths,
        time_delays=time_delays,
        true_chirp=chirp_centers,
        noise_band=(wavelengths[noise_band_width], wavelengths[-noise_band_width - 1]),
    )


def apply_preprocessing(dataset: SyntheticDataset) -> Dict[str, pd.DataFrame]:
    """先裁剪再啁啾校正"""
    cropper = SpectralCropper(
        auto_detect=True,
        noise_threshold=3.5e-4,
        relative_threshold=0.02,
        margin=2,
        min_valid_span=40,
        verbose=True,
    )
    cropped_df, cropped_wavelengths = cropper.crop(dataset.data)

    corrector = ChirpCorrector(method="cross_correlation", verbose=True)
    corrected_df = corrector.correct_chirp(cropped_df)

    return {
        "original": dataset.data,
        "cropped": cropped_df,
        "corrected": corrected_df,
        "crop_stats": cropper.get_crop_stats(),
        "chirp_stats": corrector.get_correction_stats(),
        "cropped_wavelengths": cropped_wavelengths,
    }


def plot_heatmaps(results: Dict[str, pd.DataFrame], dataset: SyntheticDataset) -> None:
    """绘制校正前后的热图对比"""
    original = results["original"].values
    cropped = results["cropped"].values
    corrected = results["corrected"].values

    arrays = [original, cropped, corrected]
    abs_max = max(np.nanmax(np.abs(arr)) for arr in arrays if arr.size)
    if abs_max == 0 or not np.isfinite(abs_max):
        abs_max = 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-abs_max, vmax=abs_max)
    cmap = plt.get_cmap("RdBu_r")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    titles = ["Original", "After cropping", "After chirp correction"]
    data_list = [original, cropped, corrected]
    wavelength_axes = [dataset.wavelengths, results["cropped"].columns.values, results["corrected"].columns.values]

    for ax, title, data_matrix, wl_axis in zip(axes, titles, data_list, wavelength_axes):
        im = ax.imshow(
            data_matrix,
            aspect="auto",
            origin="lower",
            extent=[wl_axis[0], wl_axis[-1], dataset.time_delays[0], dataset.time_delays[-1]],
            cmap=cmap,
            norm=norm,
        )
        ax.set_title(title)
        ax.set_xlabel("Wavelength / nm")
    axes[0].set_ylabel("Delay time / ps")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("ΔOD")

    fig.subplots_adjust(wspace=0.05, bottom=0.15, top=0.88)
    output = RESULTS_DIR / "synthetic_tas_heatmap_comparison.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 保存热图对比: {output}")


def plot_peak_alignment(results: Dict[str, pd.DataFrame], dataset: SyntheticDataset) -> None:
    """绘制波长对应的峰值时间，展示啁啾校正效果"""
    def extract_peak_times(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        peaks = []
        for col in df.columns:
            spectrum = df[col].values
            idx = np.argmax(np.abs(spectrum))
            peaks.append(df.index.values[idx])
        return df.columns.values.astype(float), np.array(peaks)

    wl_original, peaks_original = extract_peak_times(results["original"])
    wl_cropped, peaks_cropped = extract_peak_times(results["cropped"])
    wl_corrected, peaks_corrected = extract_peak_times(results["corrected"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wl_original, peaks_original, label="Original", color="tab:red", alpha=0.6)
    ax.plot(wl_cropped, peaks_cropped, label="After cropping", color="tab:orange", alpha=0.6)
    ax.plot(wl_corrected, peaks_corrected, label="After chirp correction", color="tab:blue", linewidth=2)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1, label="Target time zero")
    ax.set_xlabel("Wavelength / nm")
    ax.set_ylabel("Peak delay / ps")
    ax.set_title("Chirp alignment across wavelengths")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output = RESULTS_DIR / "synthetic_tas_peak_alignment.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 保存峰值对齐图: {output}")


def main() -> None:
    dataset = generate_synthetic_tas_dataset()
    dataset.data.to_csv(RESULTS_DIR / "synthetic_tas_dataset.csv")
    print("📦 合成数据已保存: synthetic_tas_dataset.csv")

    results = apply_preprocessing(dataset)

    # 打印统计信息
    crop_stats = results["crop_stats"]
    chirp_stats = results["chirp_stats"]

    print("\n🔪 光谱裁剪统计:")
    for key, value in crop_stats.items():
        print(f"  {key}: {value}")

    print("\n⚡ 啁啾校正统计:")
    method = chirp_stats.get("method")
    if method:
        print(f"  method: {method}")
    parameters = chirp_stats.get("parameters", {})
    if parameters:
        offsets = parameters.get("time_offsets")
        if offsets is not None and len(offsets) > 0:
            print(f"  time offset range: {offsets.min():.3f} ~ {offsets.max():.3f} ps")
        print(f"  wavelength count: {len(parameters.get('wavelengths', []))}")
    fit_quality = chirp_stats.get("fit_quality", {})
    if fit_quality:
        print(
            "  fit quality: range={:.3f} ps, std={:.3f} ps, smoothness={:.3f}".format(
                fit_quality.get("offset_range", 0.0),
                fit_quality.get("offset_std", 0.0),
                fit_quality.get("smoothness", 0.0),
            )
        )

    plot_heatmaps(results, dataset)
    plot_peak_alignment(results, dataset)

    print("\n🎯 合成数据测试完成")


if __name__ == "__main__":
    main()
