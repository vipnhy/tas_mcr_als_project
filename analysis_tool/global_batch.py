"""全局拟合批处理执行器."""

from __future__ import annotations

import json
import logging
import math
import os
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from Globalfit import GlobalTargetAnalysis, MCRALSInterface, ParallelModel, SequentialModel
from Globalfit.kinetic_models import MixedModel
from Globalfit.utils import export_results_to_txt

from .config import AnalysisConfig, GlobalModelSpec
from .utils import prepare_dataset

logger = logging.getLogger(__name__)


class GlobalFitBatchRunner:
    """使用多种动力学模型对 MCR 结果进行全局拟合的批处理执行器."""

    def __init__(self, analysis_config: AnalysisConfig, working_dir: Path):
        self.config = analysis_config
        self.working_dir = Path(working_dir)
        self.global_dir = self.working_dir / "global_fit"
        self.global_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = prepare_dataset(self.config)

    # ------------------------------------------------------------------
    def run(self, mcr_summary: Dict[str, Any]) -> Dict[str, Any]:
        """执行全局拟合批处理, 返回综合摘要."""

        all_results: List[Dict[str, Any]] = []
        for run_info in mcr_summary.get("runs", []):
            try:
                mcr_result = self._load_mcr_result(run_info)
            except FileNotFoundError as exc:
                logger.error("跳过 MCR 结果: %s", exc)
                continue

            for model_spec in self.config.global_fit.models:
                model_results = self._run_model_family(mcr_result, run_info, model_spec)
                all_results.extend(model_results)

        ranking = self._sort_results(all_results)
        summary_payload = {
            "summary": {
                "total_results": len(all_results),
                "sorted_by": self.config.global_fit.sort_metric,
                "sort_order": self.config.global_fit.sort_order,
            },
            "results": all_results,
            "ranking": ranking,
        }

        summary_path = self.global_dir / "global_fit_summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary_payload, fh, indent=2, ensure_ascii=False)
        logger.info("全局拟合批处理完成，摘要保存至 %s", summary_path)

        ranking_csv = self.global_dir / "global_fit_ranking.csv"
        if ranking:
            self._write_ranking_csv(ranking_csv, ranking)

        return summary_payload

    # ------------------------------------------------------------------
    def _load_mcr_result(self, run_info: Dict[str, Any]) -> Dict[str, Any]:
        rel_path = run_info["output_dir"]
        mcr_dir = (self.working_dir / rel_path).resolve()
        if not mcr_dir.exists():
            raise FileNotFoundError(f"MCR 结果目录不存在: {mcr_dir}")

        interface = MCRALSInterface(str(mcr_dir))
        if not interface.load_mcr_results():
            raise RuntimeError(f"无法加载 MCR 结果: {mcr_dir}")

        # 使用批处理阶段保存的轴信息
        time_path = mcr_dir / "time_axis.csv"
        wavelength_path = mcr_dir / "wavelength_axis.csv"
        if time_path.exists():
            interface.time_axis = np.loadtxt(time_path, delimiter=",", skiprows=1)
        else:
            interface.time_axis = self.dataset["time"]
        if wavelength_path.exists():
            interface.wavelength_axis = np.loadtxt(wavelength_path, delimiter=",", skiprows=1)
        else:
            interface.wavelength_axis = self.dataset["wavelength"]

        interface.D_original = self.dataset["data"]

        metadata_path = mcr_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        return {
            "interface": interface,
            "metadata": metadata,
            "directory": mcr_dir,
            "run_info": run_info,
        }

    # ------------------------------------------------------------------
    def _run_model_family(
        self,
        mcr_result: Dict[str, Any],
        run_info: Dict[str, Any],
        model_spec: GlobalModelSpec,
    ) -> List[Dict[str, Any]]:
        interface: MCRALSInterface = mcr_result["interface"]
        n_components = interface.C_mcr.shape[1]

        if model_spec.type == "gta" and model_spec.model == "mixed":
            if not model_spec.network:
                logger.warning("混合模型缺少 network 定义，跳过")
                return []
            # 过滤掉超出组分数的边
            network = [
                edge
                for edge in model_spec.network
                if edge[0] < n_components and edge[1] < n_components
            ]
            if not network:
                logger.warning(
                    "混合模型 network 在 %d 个组分下为空，跳过", n_components
                )
                return []
        if model_spec.type == "gla":
            logger.info("跳过 GLA 模型: %s", model_spec.label or "gla")
            return []

        folder_name, display_label = self._resolve_model_names(model_spec, n_components)
        seed_offset = self._seed_offset(run_info, folder_name)

        results: List[Dict[str, Any]] = []
        for attempt in range(self.config.global_fit.attempts_per_mcr):
            try:
                result = self._execute_single_attempt(
                    interface,
                    run_info,
                    model_spec,
                    attempt,
                    seed_offset,
                    folder_name,
                    display_label,
                )
                if result:
                    results.append(result)
            except Exception as exc:
                logger.error(
                    "全局拟合失败: mcr=%s, model=%s, attempt=%d, error=%s",
                    run_info.get("output_dir"),
                    model_spec.resolved_label(),
                    attempt + 1,
                    exc,
                )
        return results

    # ------------------------------------------------------------------
    def _execute_single_attempt(
        self,
        interface: MCRALSInterface,
        run_info: Dict[str, Any],
        model_spec: GlobalModelSpec,
        attempt_index: int,
        seed_offset: int,
        folder_name: str,
        display_label: str,
    ) -> Optional[Dict[str, Any]]:
        n_components = interface.C_mcr.shape[1]
        relative_parts = list(Path(run_info["output_dir"]).parts)
        if relative_parts and relative_parts[0] == "mcr":
            relative_parts = relative_parts[1:]
        model_dir = self.global_dir.joinpath(*relative_parts, folder_name)
        attempt_dir = model_dir / f"attempt_{attempt_index + 1:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        model = self._build_kinetic_model(model_spec, n_components)
        if model is None:
            return None
        k_initial = self._generate_rate_initials(
            interface,
            attempt_index,
            model,
            seed_offset,
        )
        k_bounds = self._build_rate_bounds(k_initial)
        fitter = GlobalTargetAnalysis(
            data_matrix=self.dataset["data"],
            time_axis=self.dataset["time"],
            wavelength_axis=self.dataset["wavelength"],
            kinetic_model=model,
        )
        results = fitter.fit(k_initial=k_initial, k_bounds=k_bounds)

        interface.time_axis = self.dataset["time"]
        interface.wavelength_axis = self.dataset["wavelength"]
        interface.save_global_fit_results(results, output_dir=str(attempt_dir))

        export_results_to_txt(results, str(attempt_dir / "global_fit_report.txt"))

        summary = {
            "mcr_output": run_info["output_dir"],
            "model": display_label,
            "attempt": attempt_index + 1,
            "lof": float(results["lof"]),
            "chi_square": float(results["chi_square"]),
            "computation_time": float(results.get("computation_time", 0.0)),
            "result_dir": os.path.relpath(attempt_dir, self.working_dir).replace("\\", "/"),
            "n_components": n_components,
            "tau_optimal": [float(x) for x in results.get("tau_optimal", [])],
            "k_optimal": [float(x) for x in results.get("k_optimal", [])],
            "model_key": folder_name,
        }
        return summary

    # ------------------------------------------------------------------
    def _generate_rate_initials(
        self,
        interface: MCRALSInterface,
        attempt_index: int,
        model: Any,
        seed_offset: int,
    ) -> List[float]:
        strategy = self.config.global_fit.init_strategy
        user_values = self.config.global_fit.user_initials.get("gta", [])
        n_rates = model.n_rate_constants
        if strategy == "user" and attempt_index < len(user_values):
            values = user_values[attempt_index]
            if len(values) == n_rates:
                return [float(v) for v in values]
            logger.warning("用户提供的速率常数长度与模型不匹配, 使用自动估计")

        lifetimes = interface.estimate_lifetimes_from_mcr()
        rate_constants = interface.estimate_rate_constants_from_lifetimes(lifetimes)
        if model.n_rate_constants <= len(rate_constants):
            return rate_constants[: model.n_rate_constants]
        if strategy == "random":
            return self._random_rates(model.n_rate_constants, attempt_index, seed_offset)
        # 若估计不足, 使用重复填充
        return (rate_constants + rate_constants[: model.n_rate_constants])[: model.n_rate_constants]

    # ------------------------------------------------------------------
    def _random_rates(self, count: int, attempt_index: int, seed_offset: int) -> List[float]:
        low, high = self._rate_window()
        seed_base = self.config.global_fit.random_seed or 0
        seed_value = seed_base + seed_offset * 7 + 1000 + attempt_index + 1
        rng = np.random.default_rng(seed_value)
        return list(rng.uniform(low, high, size=count))

    def _lifetime_window(self) -> Sequence[float]:
        window = self.config.input.lifetime_window_ps
        if window:
            return window
        # 默认使用时间轴范围
        t = self.dataset["time"]
        return (max(min(t), 1e-3), max(t))

    def _rate_window(self) -> Sequence[float]:
        low, high = self._lifetime_window()
        return (1.0 / high, 1.0 / max(low, 1e-6))

    def _build_rate_bounds(self, initial: List[float]) -> List[tuple[float, float]]:
        low, high = self._rate_window()
        result = []
        for k in initial:
            min_k = max(low, k * 0.1)
            max_k = min(high, k * 10.0)
            if min_k > max_k:
                min_k, max_k = max_k * 0.1, min_k * 10.0
            result.append((min_k, max_k))
        return result

    def _seed_offset(self, run_info: Dict[str, Any], model_label: str) -> int:
        base_str = f"{run_info.get('output_dir','')}::{model_label}"
        return zlib.crc32(base_str.encode("utf-8"))

    def _build_kinetic_model(self, model_spec: GlobalModelSpec, n_components: int):
        if model_spec.model == "sequential":
            return SequentialModel(n_components=n_components)
        if model_spec.model == "parallel":
            return ParallelModel(n_components=n_components)
        if model_spec.model == "mixed":
            network = self._prepare_mixed_network(model_spec, n_components)
            if not network:
                return None
            return MixedModel(n_components=n_components, reaction_network=network)
        return None

    # ------------------------------------------------------------------
    def _prepare_mixed_network(self, model_spec: GlobalModelSpec, n_components: int) -> List[Tuple[int, int, int]]:
        if model_spec.network:
            return [
                edge
                for edge in model_spec.network
                if edge[0] < n_components and edge[1] < n_components
            ]

        variant = (model_spec.variant or "direct").lower()
        edges: List[Tuple[int, int, int]] = []
        if n_components < 2:
            return edges

        rate_idx = 0

        if variant == "direct":
            for i in range(n_components - 1):
                edges.append((i, i + 1, rate_idx))
                rate_idx += 1
            if n_components > 2:
                edges.append((0, n_components - 1, rate_idx))
        elif variant == "reversible":
            edges.append((0, 1, rate_idx))
            rate_idx += 1
            edges.append((1, 0, rate_idx))
            rate_idx += 1
            for i in range(1, n_components - 1):
                edges.append((i, i + 1, rate_idx))
                rate_idx += 1
        else:
            logger.warning("未知的混合模型变体: %s", variant)
        return edges

    # ------------------------------------------------------------------
    def _resolve_model_names(
        self, model_spec: GlobalModelSpec, n_components: int
    ) -> Tuple[str, str]:
        letters = [chr(ord("A") + idx) for idx in range(max(n_components, 1))]

        if model_spec.type == "gla":
            display = "GLA"
            folder = "gla"
        elif model_spec.model == "sequential":
            display = "->".join(letters[:n_components])
            folder = "sequential_" + "_to_".join(letters[:n_components])
        elif model_spec.model == "parallel":
            if n_components <= 1:
                display = letters[0]
                folder = "parallel"
            else:
                target = letters[n_components - 1]
                sources = letters[: n_components - 1]
                display = "; ".join(f"{src}->{target}" for src in sources)
                folder = "parallel_" + "__".join(f"{src}_to_{target}" for src in sources)
        elif model_spec.model == "mixed":
            variant = (model_spec.variant or "direct").lower()
            if variant == "direct":
                display = self._format_mixed_direct_label(letters[:n_components])
            elif variant == "reversible":
                display = self._format_mixed_reversible_label(letters[:n_components])
            else:
                display = model_spec.resolved_label()
            folder = f"mixed_{variant}_" + self._slugify(display)
        else:
            display = model_spec.resolved_label()
            folder = self._slugify(display)

        folder = self._slugify(folder)
        return folder, display

    def _format_mixed_direct_label(self, letters: List[str]) -> str:
        if len(letters) <= 1:
            return letters[0]
        if len(letters) == 2:
            seq_part = f"{letters[0]}->{letters[1]}"
        else:
            seq_part = "->".join(letters[:-1])
        extra = f"{letters[0]}->{letters[-1]}" if len(letters) > 2 else None
        if extra:
            return f"{seq_part}; {extra}"
        return seq_part

    def _format_mixed_reversible_label(self, letters: List[str]) -> str:
        if len(letters) <= 1:
            return letters[0]
        parts = [f"{letters[0]}<->{letters[1]}"]
        for idx in range(1, len(letters) - 1):
            parts.append(f"{letters[idx]}->{letters[idx + 1]}")
        return "; ".join(parts)

    def _slugify(self, text: str) -> str:
        replacements = [
            ("<->", "_rev_"),
            ("->", "_to_"),
            (";", "__"),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        text = text.replace(" ", "_")
        sanitized = "".join(ch for ch in text if ch.isalnum() or ch in {"_", "-"})
        sanitized = sanitized.strip("_-")
        return sanitized.lower() or "model"

    # ------------------------------------------------------------------
    def _sort_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return []
        key = self.config.global_fit.sort_metric
        reverse = self.config.global_fit.sort_order == "desc"
        filtered = sorted(results, key=lambda x: x.get(key, math.inf), reverse=reverse)
        top_n = self.config.global_fit.top_n
        if top_n:
            filtered = filtered[:top_n]
        return filtered

    def _write_ranking_csv(self, path: Path, ranking: List[Dict[str, Any]]) -> None:
        import csv

        fieldnames = [
            "rank",
            "mcr_output",
            "model",
            "attempt",
            "lof",
            "chi_square",
            "computation_time",
            "result_dir",
        ]
        with path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for idx, item in enumerate(ranking, start=1):
                row = {**item, "rank": idx}
                writer.writerow({k: row.get(k, "") for k in fieldnames})
            # End of CSV writing function