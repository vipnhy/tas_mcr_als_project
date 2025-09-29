#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""光谱裁剪模块

用于在预处理早期阶段剔除噪声占主导的波长区间（例如低波长白噪声段）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Dict, Any


@dataclass
class CropStatistics:
    """记录裁剪统计信息"""

    original_range: Tuple[float, float]
    cropped_range: Tuple[float, float]
    removed_bands: Dict[str, Tuple[float, float]]
    removed_count: int
    kept_count: int
    auto_detect: bool
    threshold_used: float
    notes: str = ""


class SpectralCropper:
    """针对瞬态吸收光谱的波长裁剪器"""

    def __init__(
        self,
        min_wavelength: Optional[float] = None,
        max_wavelength: Optional[float] = None,
        auto_detect: bool = True,
        noise_threshold: float = 5e-4,
        relative_threshold: float = 0.02,
        margin: int = 1,
        min_valid_span: int = 5,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            min_wavelength: 强制保留的最小波长（用于手动裁剪下限）
            max_wavelength: 强制保留的最大波长（用于手动裁剪上限）
            auto_detect: 是否根据噪声特征自动裁剪两端波长
            noise_threshold: 噪声判定的绝对阈值（ΔOD）
            relative_threshold: 噪声判定的相对阈值，占全谱最大幅值比例
            margin: 自动裁剪后向内保留的额外波长数量，避免裁剪过头
            min_valid_span: 自动检测时要求的最小有效波长数量，防止全部裁剪
            verbose: 是否打印调试信息
        """

        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.auto_detect = auto_detect
        self.noise_threshold = noise_threshold
        self.relative_threshold = relative_threshold
        self.margin = margin
        self.min_valid_span = min_valid_span
        self.verbose = verbose
        self._stats: Optional[CropStatistics] = None

    def crop(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        wavelengths: Optional[np.ndarray] = None,
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], np.ndarray]:
        """执行光谱裁剪"""

        is_dataframe = isinstance(data, pd.DataFrame)
        data_array = data.values if is_dataframe else np.asarray(data)

        if wavelengths is None:
            if is_dataframe:
                wavelengths_array = data.columns.values.astype(float)
            else:
                wavelengths_array = np.arange(data_array.shape[1], dtype=float)
        else:
            wavelengths_array = np.asarray(wavelengths, dtype=float)

        original_range = (float(wavelengths_array.min()), float(wavelengths_array.max()))

        start_idx, end_idx, threshold_used, notes = self._determine_crop_bounds(
            data_array, wavelengths_array
        )

        # 保证索引有效
        start_idx = max(0, min(start_idx, len(wavelengths_array) - 1))
        end_idx = max(start_idx + self.min_valid_span - 1, min(end_idx, len(wavelengths_array) - 1))

        # 应用裁剪
        cropped_wavelengths = wavelengths_array[start_idx : end_idx + 1]
        if is_dataframe:
            cropped_data = data.iloc[:, start_idx : end_idx + 1]
        else:
            cropped_data = data_array[:, start_idx : end_idx + 1]

        self._stats = CropStatistics(
            original_range=original_range,
            cropped_range=(float(cropped_wavelengths.min()), float(cropped_wavelengths.max())),
            removed_bands={
                "low": (original_range[0], float(cropped_wavelengths[0])) if start_idx > 0 else None,
                "high": (float(cropped_wavelengths[-1]), original_range[1])
                if end_idx < len(wavelengths_array) - 1
                else None,
            },
            removed_count=len(wavelengths_array) - len(cropped_wavelengths),
            kept_count=len(cropped_wavelengths),
            auto_detect=self.auto_detect,
            threshold_used=threshold_used,
            notes=notes,
        )

        if self.verbose:
            self._log_stats()

        return (cropped_data, cropped_wavelengths)

    def get_crop_stats(self) -> Dict[str, Any]:
        """返回裁剪统计信息"""
        if self._stats is None:
            return {}
        stats_dict = {
            "original_range": self._stats.original_range,
            "cropped_range": self._stats.cropped_range,
            "removed_bands": self._stats.removed_bands,
            "removed_count": self._stats.removed_count,
            "kept_count": self._stats.kept_count,
            "auto_detect": self._stats.auto_detect,
            "threshold_used": self._stats.threshold_used,
        }
        if self._stats.notes:
            stats_dict["notes"] = self._stats.notes
        return stats_dict

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _determine_crop_bounds(
        self, data_array: np.ndarray, wavelengths: np.ndarray
    ) -> Tuple[int, int, float, str]:
        n_wavelengths = data_array.shape[1]
        max_amplitude = np.nanmax(np.abs(data_array), axis=0)
        smoothed_amplitude = pd.Series(max_amplitude).rolling(window=5, center=True, min_periods=1).median().values

        global_max = float(np.nanmax(smoothed_amplitude)) if np.isfinite(smoothed_amplitude).any() else 0.0
        # 动态阈值：绝对阈值与相对阈值的最大值
        dynamic_threshold = max(self.noise_threshold, self.relative_threshold * global_max)

        start_idx = 0
        end_idx = n_wavelengths - 1
        notes = ""

        if self.auto_detect and global_max > 0:
            while start_idx < n_wavelengths - self.min_valid_span and smoothed_amplitude[start_idx] < dynamic_threshold:
                start_idx += 1
            while end_idx > start_idx + self.min_valid_span - 1 and smoothed_amplitude[end_idx] < dynamic_threshold:
                end_idx -= 1

            if start_idx > 0:
                start_idx = max(0, start_idx - self.margin)
            if end_idx < n_wavelengths - 1:
                end_idx = min(n_wavelengths - 1, end_idx + self.margin)

            if start_idx >= end_idx:
                notes = "Auto detection collapsed full range; falling back to manual limits"
                start_idx, end_idx = 0, n_wavelengths - 1
        else:
            notes = "Auto detection disabled or insufficient signal"

        # 应用手动限制
        if self.min_wavelength is not None:
            manual_start = int(np.searchsorted(wavelengths, self.min_wavelength, side="left"))
            start_idx = max(start_idx, manual_start)
        if self.max_wavelength is not None:
            manual_end = int(np.searchsorted(wavelengths, self.max_wavelength, side="right")) - 1
            end_idx = min(end_idx, manual_end)

        start_idx = min(start_idx, n_wavelengths - self.min_valid_span)
        end_idx = max(end_idx, start_idx + self.min_valid_span - 1)

        return start_idx, end_idx, dynamic_threshold, notes

    def _log_stats(self) -> None:
        if self._stats is None:
            return
        print("🔪 光谱裁剪完成:")
        print(
            f"  原始波长范围: {self._stats.original_range[0]:.1f}-{self._stats.original_range[1]:.1f} nm"
        )
        print(
            f"  裁剪后波长范围: {self._stats.cropped_range[0]:.1f}-{self._stats.cropped_range[1]:.1f} nm"
        )
        if self._stats.removed_bands.get("low"):
            low = self._stats.removed_bands["low"]
            print(f"  移除低波长区间: {low[0]:.1f}-{low[1]:.1f} nm")
        if self._stats.removed_bands.get("high"):
            high = self._stats.removed_bands["high"]
            print(f"  移除高波长区间: {high[0]:.1f}-{high[1]:.1f} nm")
        print(f"  保留波长数量: {self._stats.kept_count} (移除了 {self._stats.removed_count} 个采样点)")
        if self._stats.notes:
            print(f"  备注: {self._stats.notes}")


def crop_wavelengths(
    data: Union[pd.DataFrame, np.ndarray],
    wavelengths: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Tuple[Union[pd.DataFrame, np.ndarray], np.ndarray]:
    """便捷函数"""
    cropper = SpectralCropper(**kwargs)
    return cropper.crop(data, wavelengths=wavelengths)
