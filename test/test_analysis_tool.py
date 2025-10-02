import json
from pathlib import Path

import numpy as np

from analysis_tool import load_analysis_config
from analysis_tool.utils import prepare_dataset


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.json"
    config_data = {
        "analysis": {
            "name": "test_run",
            "working_dir_mode": "overwrite",
            "output_root": "analysis_runs",
        },
        "input": {
            "file_path": "data/TAS/TA_Average-crop-TCA-tzs-tzs-tzs-chirp.csv",
            "file_type": "handle",
            "spectral_type": "VIS",
            "delay_range": [0.1, 5.0],
        },
        "mcr": {
            "component_range": [2, 4],
            "initialization": {
                "mode": "random",
                "runs_per_component": 2,
                "random_seed": 42,
            },
            "preprocessing": {
                "enabled": False,
            },
        },
        "global_fit": {
            "attempts_per_mcr": 1,
            "init_strategy": "mcr",
            "sort_metric": "lof",
        },
    }
    config_path.write_text(json.dumps(config_data), encoding="utf-8")
    return config_path


def test_load_analysis_config_expands_component_range(tmp_path):
    config_path = _write_config(tmp_path)
    config = load_analysis_config(config_path)

    assert list(config.mcr.iter_components()) == [2, 3, 4]
    assert config.global_fit.sort_metric == "lof"
    assert config.analysis.output_root == "analysis_runs"

    working_dir = config.compute_working_directory(root=tmp_path)
    assert working_dir == tmp_path / "analysis_runs" / "test_run"


def test_prepare_dataset_returns_expected_keys(tmp_path):
    config_path = _write_config(tmp_path)
    config = load_analysis_config(config_path)
    dataset = prepare_dataset(config)

    assert set(dataset.keys()) == {
        "data",
        "time",
        "wavelength",
        "wavelength_range",
        "delay_range",
        "preprocessing",
    }
    assert dataset["data"].ndim == 2
    assert np.isfinite(dataset["data"]).all()
    assert dataset["preprocessing"]["enabled"] is False
