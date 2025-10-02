"""分析工具公共辅助函数."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from data.data import read_file
from preprocessing.pipeline import TASPreprocessingPipeline

from .config import AnalysisConfig, DEFAULT_SPECTRAL_RANGES

DEFAULT_PREPROCESS_STEPS = [
    {
        "name": "baseline_correction",
        "processor": "baseline",
        "params": {"method": "als", "lam": 1e6, "p": 0.001},
    },
    {
        "name": "noise_filtering",
        "processor": "noise",
        "params": {"method": "gaussian", "sigma": 1.0},
    },
]


def to_serializable(obj: Any) -> Any:
    """递归转换对象为 JSON 可序列化结构."""

    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, (np.number,)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_list()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="split")
    return obj


def prepare_dataset(config: AnalysisConfig) -> Dict[str, Any]:
    """根据配置加载并可选执行预处理, 返回统一的数据字典."""

    input_cfg = config.input
    spectral_range = list(input_cfg.resolved_wavelength_range())
    default_range = DEFAULT_SPECTRAL_RANGES[input_cfg.spectral_type]
    spectral_range[0] = max(spectral_range[0], default_range[0])
    spectral_range[1] = min(spectral_range[1], default_range[1])

    wavelength_range = [float(spectral_range[0]), float(spectral_range[1])]
    if input_cfg.wavelength_override is not None:
        wavelength_range = [float(x) for x in input_cfg.wavelength_override]

    if input_cfg.delay_range is not None:
        delay_range = [float(x) for x in input_cfg.delay_range]
    else:
        delay_range = [None, None]

    df = read_file(
        input_cfg.file_path,
        file_type=input_cfg.file_type,
        inf_handle=True,
        wavelength_range=wavelength_range,
        delay_range=delay_range,
    )
    if df is None:
        raise RuntimeError(f"无法读取输入数据: {input_cfg.file_path}")

    preprocessing_summary: Dict[str, Any] = {
        "enabled": False,
        "steps": [],
        "history": [],
    }

    if config.mcr.preprocessing.enabled:
        steps = config.mcr.preprocessing.steps or DEFAULT_PREPROCESS_STEPS
        pipeline = TASPreprocessingPipeline(steps=steps, verbose=False)
        processed = pipeline.fit_transform(df)
        if isinstance(processed, pd.DataFrame):
            df = processed
        else:
            df = pd.DataFrame(
                processed,
                index=getattr(pipeline, "time_axis", df.index.values),
                columns=getattr(pipeline, "wavelengths", df.columns.values),
            )
        preprocessing_summary = {
            "enabled": True,
            "steps": steps,
            "history": to_serializable(pipeline.processing_history),
        }

    data_matrix = df.values.astype(float)
    time_axis = df.index.values.astype(float)
    wavelength_axis = df.columns.values.astype(float)

    dataset = {
        "data": data_matrix,
        "time": time_axis,
        "wavelength": wavelength_axis,
        "wavelength_range": (float(wavelength_axis.min()), float(wavelength_axis.max())),
        "delay_range": (float(time_axis.min()), float(time_axis.max())),
        "preprocessing": preprocessing_summary,
    }
    return dataset
