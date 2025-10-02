"""MCR 批处理执行器."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from mcr.constraint_config import ConstraintConfig
from mcr.mcr_als import MCRALS

from .config import AnalysisConfig
from .utils import prepare_dataset

logger = logging.getLogger(__name__)

class MCRBatchRunner:
    """负责遍历组分与初始化的 MCR-ALS 批处理执行器."""

    def __init__(self, analysis_config: AnalysisConfig, working_dir: Path):
        self.config = analysis_config
        self.working_dir = Path(working_dir)
        self.mcr_dir = self.working_dir / "mcr"
        self.mcr_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        """执行批处理 MCR 分析并返回摘要."""
        dataset = prepare_dataset(self.config)
        summary_runs: List[Dict[str, Any]] = []

        for n_components in self.config.mcr.iter_components():
            seeds = self._resolve_seeds(n_components)
            for run_idx, seed in enumerate(seeds):
                run_summary = self._run_single_mcr(
                    dataset=dataset,
                    n_components=n_components,
                    run_index=run_idx,
                    seed=seed,
                )
                if run_summary:
                    summary_runs.append(run_summary)

        summary_payload = {
            "input": {
                "file_path": str(Path(self.config.input.file_path).as_posix()),
                "file_type": self.config.input.file_type,
                "spectral_type": self.config.input.spectral_type,
                "wavelength_range": dataset["wavelength_range"],
                "delay_range": dataset["delay_range"],
                "data_shape": dataset["data"].shape,
                "preprocessing": dataset["preprocessing"],
            },
            "mcr_config": {
                "component_candidates": list(self.config.mcr.iter_components()),
                "max_iter": self.config.mcr.max_iter,
                "tol": self.config.mcr.tol,
                "enforce_nonneg": self.config.mcr.enforce_nonneg,
                "spectral_nonneg": self.config.mcr.spectral_nonneg,
                "constraint_template": self.config.mcr.constraint_template,
                "initialization": asdict(self.config.mcr.initialization),
            },
            "runs": summary_runs,
        }

        summary_path = self.mcr_dir / "mcr_summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary_payload, fh, indent=2, ensure_ascii=False)
        logger.info("MCR 批处理完成，摘要保存至 %s", summary_path)

        return summary_payload

    # ------------------------------------------------------------------
    # 种子生成
    # ------------------------------------------------------------------
    def _resolve_seeds(self, n_components: int) -> List[Optional[int]]:
        init_cfg = self.config.mcr.initialization
        runs = max(1, int(init_cfg.runs_per_component))

        if init_cfg.mode == "svd":
            return [None for _ in range(runs)]

        if init_cfg.mode == "user":
            if not init_cfg.user_seeds:
                raise ValueError("初始化模式为 user 时必须提供 user_seeds")
            if len(init_cfg.user_seeds) < runs:
                logger.warning(
                    "user_seeds 数量不足，将循环使用提供的种子 (要求 %d, 实际 %d)",
                    runs,
                    len(init_cfg.user_seeds),
                )
            seeds = [init_cfg.user_seeds[i % len(init_cfg.user_seeds)] for i in range(runs)]
            return seeds

        # random 模式
        rng_seed = init_cfg.random_seed
        if rng_seed is not None:
            rng = np.random.default_rng(rng_seed + n_components)
        else:
            rng = np.random.default_rng()
        return [int(rng.integers(0, 2**32 - 1)) for _ in range(runs)]

    # ------------------------------------------------------------------
    # 单次运行
    # ------------------------------------------------------------------
    def _run_single_mcr(
        self,
        dataset: Dict[str, Any],
        n_components: int,
        run_index: int,
        seed: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        data_matrix = dataset["data"]
        time_axis = dataset["time"]
        wavelength_axis = dataset["wavelength"]

        constraint_config = self._build_constraints()

        init_method = "svd" if self.config.mcr.initialization.mode == "svd" else "random"
        rng_seed = seed if init_method == "random" else None

        solver = MCRALS(
            n_components=n_components,
            max_iter=self.config.mcr.max_iter,
            tol=self.config.mcr.tol,
            constraint_config=constraint_config,
            init_method=init_method,
            random_state=rng_seed,
        )

        try:
            solver.fit(data_matrix)
        except Exception as exc:
            logger.error(
                "MCR-ALS 运行失败: components=%s, run=%s, seed=%s, error=%s",
                n_components,
                run_index,
                seed,
                exc,
            )
            return None

        C = solver.C_opt_
        S = solver.S_opt_
        if C is None or S is None:
            logger.error("MCR-ALS 未返回有效的 C/S 矩阵")
            return None

        reconstructed = C @ S.T
        residuals = data_matrix - reconstructed
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((data_matrix - np.mean(data_matrix)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        lof = float(solver.lof_[-1]) if solver.lof_ else float("nan")

        run_dir = (
            self.mcr_dir
            / f"components_{n_components}"
            / f"run_{run_index + 1:02d}{'' if seed is None else f'_seed_{seed}'}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        self._save_mcr_outputs(
            run_dir=run_dir,
            solver=solver,
            C=C,
            S=S,
            reconstructed=reconstructed,
            time_axis=time_axis,
            wavelength_axis=wavelength_axis,
            dataset=dataset,
            n_components=n_components,
            run_index=run_index,
            seed=seed,
            lof=lof,
            r_squared=r_squared,
            ss_res=ss_res,
            ss_tot=ss_tot,
        )

        summary = {
            "components": n_components,
            "run_index": run_index + 1,
            "seed": seed,
            "init_method": init_method,
            "lof": lof,
            "r2": r_squared,
            "iterations": len(solver.lof_),
            "output_dir": os.path.relpath(run_dir, self.working_dir).replace("\\", "/"),
            "residual_norm": float(np.linalg.norm(residuals)),
        }
        return summary

    # ------------------------------------------------------------------
    def _build_constraints(self) -> ConstraintConfig:
        cfg = self.config.mcr
        if cfg.constraint_template:
            template_path = Path(cfg.constraint_template)
            if not template_path.is_absolute():
                template_path = Path(os.getcwd()) / template_path
            constraint_config = ConstraintConfig(str(template_path))
        else:
            constraint_config = ConstraintConfig()

        if not cfg.enforce_nonneg:
            constraint_config.disable_constraint("non_negativity")
        else:
            if not cfg.spectral_nonneg and "non_negativity" in constraint_config.constraints:
                constraint_config.constraints["non_negativity"]["apply_to"] = ["C"]
        return constraint_config

    # ------------------------------------------------------------------
    def _save_mcr_outputs(
        self,
        run_dir: Path,
        solver: MCRALS,
        C: np.ndarray,
        S: np.ndarray,
        reconstructed: np.ndarray,
        time_axis: np.ndarray,
        wavelength_axis: np.ndarray,
        dataset: Dict[str, Any],
        n_components: int,
        run_index: int,
        seed: Optional[int],
        lof: float,
        r_squared: float,
        ss_res: float,
        ss_tot: float,
    ) -> None:
        header = ",".join(f"Component_{i+1}" for i in range(n_components))

        np.savetxt(
            run_dir / "concentration_profiles.csv",
            C,
            delimiter=",",
            header=header,
            comments="",
        )
        np.savetxt(
            run_dir / "pure_spectra.csv",
            S,
            delimiter=",",
            header=header,
            comments="",
        )
        if solver.lof_:
            np.savetxt(
                run_dir / "lof_history.csv",
                np.asarray(solver.lof_),
                delimiter=",",
                header="LOF_%",
                comments="",
            )

        np.savetxt(
            run_dir / "time_axis.csv",
            time_axis,
            delimiter=",",
            header="Time_ps",
            comments="",
        )
        np.savetxt(
            run_dir / "wavelength_axis.csv",
            wavelength_axis,
            delimiter=",",
            header="Wavelength_nm",
            comments="",
        )

        metadata = {
            "components": n_components,
            "run_index": run_index + 1,
            "seed": seed,
            "init_method": self.config.mcr.initialization.mode,
            "max_iter": self.config.mcr.max_iter,
            "tol": self.config.mcr.tol,
            "lof_final": lof,
            "r_squared": r_squared,
            "ss_res": ss_res,
            "ss_tot": ss_tot,
            "data_shape": list(dataset["data"].shape),
            "time_points": len(time_axis),
            "wavelength_points": len(wavelength_axis),
            "preprocessing": dataset["preprocessing"],
            "wavelength_range": dataset["wavelength_range"],
            "delay_range": dataset["delay_range"],
        }

        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        analysis_parameters = {
            "file_path": self.config.input.file_path,
            "file_type": self.config.input.file_type,
            "wavelength_range": dataset["wavelength_range"],
            "delay_range": dataset["delay_range"],
            "n_components": n_components,
            "final_lof": lof,
            "iterations": len(solver.lof_),
            "init_method": self.config.mcr.initialization.mode,
            "random_seed": seed,
            "analysis_name": self.config.analysis.name,
        }
        (run_dir / "analysis_parameters.json").write_text(
            json.dumps(analysis_parameters, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if self.config.mcr.save_intermediate:
            np.savetxt(run_dir / "reconstructed.csv", reconstructed, delimiter=",")
            np.savetxt(run_dir / "residuals.csv", dataset["data"] - reconstructed, delimiter=",")