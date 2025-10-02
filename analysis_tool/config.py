"""配置解析与数据结构."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from typing import TypedDict

SpectralType = Literal["UV", "VIS", "NIR"]
SortMetric = Literal["lof", "chi_square", "computation_time"]
InitMode = Literal["random", "svd", "user"]
GlobalInitStrategy = Literal["mcr", "random", "user"]
WorkingDirMode = Literal["timestamp", "overwrite", "custom"]

DEFAULT_SPECTRAL_RANGES: Dict[SpectralType, Tuple[float, float]] = {
    "UV": (200.0, 400.0),
    "VIS": (400.0, 800.0),
    "NIR": (800.0, 1700.0),
}

DEFAULT_GLOBAL_MODELS: List[Dict[str, Any]] = [
    {"type": "gla", "label": "gla"},
    {"type": "gta", "model": "sequential", "label": "gta_sequential"},
    {"type": "gta", "model": "parallel", "label": "gta_parallel"},
    {
        "type": "gta",
        "model": "mixed",
        "label": "gta_mixed_direct",
        "network": [(0, 1, 0), (1, 2, 1), (0, 2, 2)],
    },
    {
        "type": "gta",
        "model": "mixed",
        "label": "gta_mixed_reversible",
        "network": [(0, 1, 0), (1, 2, 1), (1, 0, 2)],
    },
]


@dataclass
class PreprocessingConfig:
    """预处理配置."""

    enabled: bool = False
    steps: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MCRInitializationConfig:
    """MCR 初始化策略配置."""

    mode: InitMode = "random"
    runs_per_component: int = 1
    user_seeds: List[int] = field(default_factory=list)
    random_seed: Optional[int] = None


@dataclass
class MCRConfig:
    """MCR 批量分析配置."""

    component_range: List[int] = field(default_factory=lambda: [3])
    max_iter: int = 200
    tol: float = 1e-7
    enforce_nonneg: bool = True
    spectral_nonneg: bool = False
    constraint_template: Optional[str] = None
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    initialization: MCRInitializationConfig = field(default_factory=MCRInitializationConfig)
    save_intermediate: bool = True

    def iter_components(self) -> Iterable[int]:
        return sorted(set(int(k) for k in self.component_range if k > 0))


class MixedNetworkSpec(TypedDict):
    type: Literal["gta"]
    model: Literal["mixed"]
    network: List[Tuple[int, int, int]]


@dataclass
class GlobalModelSpec:
    """单个全局拟合模型配置."""

    type: Literal["gla", "gta"]
    model: Optional[Literal["sequential", "parallel", "mixed"]] = None
    network: Optional[List[Tuple[int, int, int]]] = None
    label: Optional[str] = None

    def resolved_label(self) -> str:
        if self.label:
            return self.label
        if self.type == "gla":
            return "gla"
        if self.model == "mixed":
            return "gta_mixed"
        return f"gta_{self.model or 'unknown'}"


@dataclass
class GlobalFitConfig:
    """全局拟合批处理配置."""

    models: List[GlobalModelSpec] = field(default_factory=list)
    attempts_per_mcr: int = 3
    init_strategy: GlobalInitStrategy = "mcr"
    random_seed: Optional[int] = None
    sort_metric: SortMetric = "lof"
    sort_order: Literal["asc", "desc"] = "asc"
    top_n: Optional[int] = None
    user_initials: Dict[str, List[List[float]]] = field(default_factory=dict)


@dataclass
class InputConfig:
    """输入数据相关配置."""

    file_path: str
    file_type: str = "handle"
    spectral_type: SpectralType = "VIS"
    wavelength_override: Optional[Tuple[float, float]] = None
    delay_range: Optional[Tuple[float, float]] = None
    lifetime_window_ps: Optional[Tuple[float, float]] = None

    def resolved_wavelength_range(self) -> Tuple[float, float]:
        if self.wavelength_override:
            return tuple(float(x) for x in self.wavelength_override)
        return DEFAULT_SPECTRAL_RANGES[self.spectral_type]


@dataclass
class AnalysisSectionConfig:
    name: str = "analysis"
    output_root: str = "analysis_runs"
    working_dir_mode: WorkingDirMode = "timestamp"
    custom_dir: Optional[str] = None


@dataclass
class AnalysisConfig:
    """综合配置."""

    analysis: AnalysisSectionConfig
    input: InputConfig
    mcr: MCRConfig
    global_fit: GlobalFitConfig

    raw_config: Dict[str, Any] = field(default_factory=dict)

    def compute_working_directory(self, root: Optional[Path] = None) -> Path:
        base = Path(root or os.getcwd())
        preferred_root = base / self.analysis.output_root
        preferred_root.mkdir(parents=True, exist_ok=True)

        mode = self.analysis.working_dir_mode
        if mode == "timestamp":
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return preferred_root / f"{self.analysis.name}_{stamp}"
        if mode == "overwrite":
            return preferred_root / self.analysis.name
        if mode == "custom" and self.analysis.custom_dir:
            return Path(self.analysis.custom_dir)
        raise ValueError(f"Unsupported working_dir_mode: {mode}")


def _ensure_component_range(component_range: Sequence[int] | Sequence[float]) -> List[int]:
    values = list(component_range)
    if len(values) == 2 and all(isinstance(v, (int, float)) for v in values):
        start, end = int(values[0]), int(values[1])
        if start > end:
            start, end = end, start
        return list(range(start, end + 1))
    return [int(v) for v in values]


def _normalize_models(models_raw: Sequence[Dict[str, Any]]) -> List[GlobalModelSpec]:
    specs: List[GlobalModelSpec] = []
    for idx, item in enumerate(models_raw):
        model_type = item.get("type")
        if model_type not in {"gla", "gta"}:
            raise ValueError(f"模型配置 #{idx} 缺少或包含非法 type: {item}")
        model = item.get("model")
        network = item.get("network")
        label = item.get("label")
        if model_type == "gta" and model not in {"sequential", "parallel", "mixed"}:
            raise ValueError(f"GTA 模型配置 #{idx} 缺少 model 字段或取值非法: {item}")
        if model == "mixed":
            if not network or not isinstance(network, list):
                raise ValueError("混合模型必须提供 network 列表")
            network_tuples = []
            for entry in network:
                if len(entry) != 3:
                    raise ValueError(f"network 元素必须为三个值: {entry}")
                network_tuples.append((int(entry[0]), int(entry[1]), int(entry[2])))
            network = network_tuples
        specs.append(GlobalModelSpec(type=model_type, model=model, network=network, label=label))
    return specs


def load_analysis_config(config_path: str | os.PathLike[str]) -> AnalysisConfig:
    """从 JSON 文件加载分析配置."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    analysis_section = data.get("analysis", {})
    analysis_cfg = AnalysisSectionConfig(
        name=analysis_section.get("name", "analysis"),
        output_root=analysis_section.get("output_root", "analysis_runs"),
        working_dir_mode=analysis_section.get("working_dir_mode", "timestamp"),
        custom_dir=analysis_section.get("custom_dir"),
    )
    if analysis_cfg.working_dir_mode not in {"timestamp", "overwrite", "custom"}:
        raise ValueError(f"working_dir_mode 不支持: {analysis_cfg.working_dir_mode}")

    input_section = data.get("input", {})
    spectral_raw = input_section.get("spectral_type", "VIS").upper()
    if spectral_raw not in DEFAULT_SPECTRAL_RANGES:
        raise ValueError(f"不支持的光谱类型: {spectral_raw}")
    wavelength_override = input_section.get("wavelength_override")
    if wavelength_override is not None:
        if len(wavelength_override) != 2:
            raise ValueError("wavelength_override 必须是长度为2的数组")
        wavelength_override = (float(wavelength_override[0]), float(wavelength_override[1]))
    delay_range = input_section.get("delay_range")
    if delay_range is not None:
        if len(delay_range) != 2:
            raise ValueError("delay_range 必须是长度为2的数组")
        delay_range = (float(delay_range[0]), float(delay_range[1]))
    lifetime_window = input_section.get("lifetime_window_ps")
    if lifetime_window is not None:
        if len(lifetime_window) != 2:
            raise ValueError("lifetime_window_ps 必须是长度为2的数组")
        lifetime_window = (float(lifetime_window[0]), float(lifetime_window[1]))

    input_cfg = InputConfig(
        file_path=input_section["file_path"],
        file_type=input_section.get("file_type", "handle"),
        spectral_type=spectral_raw,  # type: ignore[arg-type]
        wavelength_override=wavelength_override,
        delay_range=delay_range,
        lifetime_window_ps=lifetime_window,
    )

    mcr_section = data.get("mcr", {})
    component_range = _ensure_component_range(mcr_section.get("component_range", [3]))
    initialization_section = mcr_section.get("initialization", {})
    init_cfg = MCRInitializationConfig(
        mode=initialization_section.get("mode", "random"),
        runs_per_component=int(initialization_section.get("runs_per_component", 1)),
        user_seeds=[int(x) for x in initialization_section.get("user_seeds", [])],
        random_seed=initialization_section.get("random_seed"),
    )
    if init_cfg.mode not in {"random", "svd", "user"}:
        raise ValueError(f"初始化模式不支持: {init_cfg.mode}")
    preprocessing_section = mcr_section.get("preprocessing", {})
    preprocessing_cfg = PreprocessingConfig(
        enabled=preprocessing_section.get("enabled", False),
        steps=preprocessing_section.get("steps", []),
    )
    mcr_cfg = MCRConfig(
        component_range=component_range,
        max_iter=int(mcr_section.get("max_iter", 200)),
        tol=float(mcr_section.get("tol", 1e-7)),
        enforce_nonneg=bool(mcr_section.get("enforce_nonneg", True)),
        spectral_nonneg=bool(mcr_section.get("spectral_nonneg", False)),
        constraint_template=mcr_section.get("constraint_template"),
        preprocessing=preprocessing_cfg,
        initialization=init_cfg,
        save_intermediate=bool(mcr_section.get("save_intermediate", True)),
    )

    global_section = data.get("global_fit", {})
    models_raw = global_section.get("models")
    if not models_raw:
        models_raw = DEFAULT_GLOBAL_MODELS
    models_cfg = _normalize_models(models_raw)
    init_strategy = global_section.get("init_strategy", "mcr")
    if init_strategy not in {"mcr", "random", "user"}:
        raise ValueError(f"全局拟合初始化策略不支持: {init_strategy}")
    sort_metric = global_section.get("sort_metric", "lof")
    if sort_metric not in {"lof", "chi_square", "computation_time"}:
        raise ValueError(f"排序指标不支持: {sort_metric}")
    sort_order = global_section.get("sort_order", "asc")
    if sort_order not in {"asc", "desc"}:
        raise ValueError(f"排序方向不支持: {sort_order}")

    global_cfg = GlobalFitConfig(
        models=models_cfg,
        attempts_per_mcr=int(global_section.get("attempts_per_mcr", 3)),
        init_strategy=init_strategy,  # type: ignore[arg-type]
        random_seed=global_section.get("random_seed"),
        sort_metric=sort_metric,  # type: ignore[arg-type]
        sort_order=sort_order,  # type: ignore[arg-type]
        top_n=global_section.get("top_n"),
        user_initials=global_section.get("user_initials", {}),
    )

    return AnalysisConfig(
        analysis=analysis_cfg,
        input=input_cfg,
        mcr=mcr_cfg,
        global_fit=global_cfg,
        raw_config=data,
    )
