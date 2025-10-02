#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""高级瞬态吸收光谱分析工作流."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from analysis_tool import (
    AnalysisConfig,
    AnalysisReporter,
    GlobalFitBatchRunner,
    MCRBatchRunner,
    load_analysis_config,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行高级瞬态吸收光谱分析工作流",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="配置文件路径 (JSON)")
    parser.add_argument("--analysis-name", help="覆盖配置中的分析名称")
    parser.add_argument("--spectral-type", choices=["UV", "VIS", "NIR"], help="覆盖光谱类型")
    parser.add_argument("--component-range", nargs="*", type=int, help="覆盖组分数范围/列表")
    parser.add_argument("--no-preprocessing", action="store_true", help="禁用预处理流程")
    parser.add_argument(
        "--sort-metric",
        choices=["lof", "chi_square", "computation_time"],
        help="全局拟合排序指标",
    )
    parser.add_argument("--verbose", action="store_true", help="输出调试日志")
    return parser.parse_args()


def apply_overrides(config: AnalysisConfig, args: argparse.Namespace) -> AnalysisConfig:
    if args.analysis_name:
        config.analysis.name = args.analysis_name
    if args.spectral_type:
        config.input.spectral_type = args.spectral_type
    if args.component_range:
        config.mcr.component_range = list(args.component_range)
    if args.no_preprocessing:
        config.mcr.preprocessing.enabled = False
    if args.sort_metric:
        config.global_fit.sort_metric = args.sort_metric  # type: ignore[assignment]
    return config


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


def copy_config(config_path: Path, working_dir: Path) -> None:
    target = working_dir / "used_config.json"
    shutil.copy2(config_path, target)


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    config = load_analysis_config(args.config)
    config = apply_overrides(config, args)

    working_dir = config.compute_working_directory()
    working_dir.mkdir(parents=True, exist_ok=True)
    logger.info("工作目录: %s", working_dir)

    copy_config(Path(args.config), working_dir)

    # 阶段 1: MCR 批处理
    logger.info("阶段 1/3: 执行 MCR-ALS 批量分析")
    mcr_runner = MCRBatchRunner(config, working_dir)
    mcr_summary = mcr_runner.run()

    # 阶段 2: 全局拟合
    logger.info("阶段 2/3: 执行全局拟合批处理")
    global_runner = GlobalFitBatchRunner(config, working_dir)
    global_summary = global_runner.run(mcr_summary)

    # 阶段 3: 报告
    logger.info("阶段 3/3: 生成总结报告")
    reporter = AnalysisReporter(config, working_dir)
    report_path = reporter.generate(mcr_summary, global_summary)

    logger.info("分析完成! 报告已生成: %s", report_path)

    print("\n===== 工作流完成 =====")
    print(f"工作目录: {working_dir}")
    print(f"MCR 摘要: {working_dir / 'mcr' / 'mcr_summary.json'}")
    print(f"全局拟合摘要: {working_dir / 'global_fit' / 'global_fit_summary.json'}")
    print(f"最终报告: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
