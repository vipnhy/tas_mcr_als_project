#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量运行 TAS MCR-ALS 分析的网格搜索脚本。

- 组分数量: 可配置，默认 1-4
- 惩罚因子: 范围 [0.1, 1.0] 步长 0.1
- 随机初始值: 每个组合运行多次 (默认 5 次)
- 约束配置: 默认、标准、严格、宽松

所有输出将写入 experiments/results/mcr_als_grid/outputs 下的独立目录中，
不会影响主程序的 results 目录。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_main import TASAnalyzer  # noqa: E402

CONSTRAINT_PRESETS: Dict[str, Optional[Path]] = {
    "default": None,
    "standard": PROJECT_ROOT / "mcr" / "constraint_templates" / "standard_constraints.json",
    "strict": PROJECT_ROOT / "mcr" / "constraint_templates" / "strict_constraints.json",
    "relaxed": PROJECT_ROOT / "mcr" / "constraint_templates" / "relaxed_constraints.json",
}


@dataclass
class RunResult:
    constraint: str
    constraint_path: Optional[str]
    n_components: int
    penalty: float
    random_seed: Optional[int]
    final_lof: Optional[float]
    iterations: Optional[int]
    status: str
    output_dir: str
    message: str


def parse_components(option: str) -> List[int]:
    values: List[int] = []
    for part in option.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start_str, end_str = part.split('-', 1)
            start, end = int(start_str), int(end_str)
            if start > end:
                start, end = end, start
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    deduped = sorted(set(values))
    if not deduped:
        raise ValueError("组件数量配置不能为空")
    return deduped


def build_penalty_grid(pmin: float, pmax: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("惩罚因子步长必须大于0")
    penalties: List[float] = []
    current = pmin
    while current <= pmax + 1e-9:
        penalties.append(round(current, 6))
        current += step
    return penalties


def resolve_constraint(tokens: Iterable[str]) -> Dict[str, Optional[Path]]:
    resolved: Dict[str, Optional[Path]] = {}
    for token in tokens:
        name = token.strip().lower()
        if not name:
            continue
        if name in CONSTRAINT_PRESETS:
            resolved[name] = CONSTRAINT_PRESETS[name]
        else:
            path = Path(name)
            if not path.is_file():
                raise FileNotFoundError(f"未找到约束配置文件: {name}")
            resolved[path.stem] = path.resolve()
    if not resolved:
        raise ValueError("至少需要一个约束配置")
    return resolved


def write_summary(summary_path: Path, results: List[RunResult]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(RunResult.__annotations__.keys())
    with summary_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(asdict(item))

    json_path = summary_path.with_suffix('.json')
    with json_path.open('w', encoding='utf-8') as jf:
        json.dump([asdict(item) for item in results], jf, ensure_ascii=False, indent=2)


def write_aggregated_summary(summary_path: Path, results: List[RunResult]) -> None:
    aggregates: Dict[tuple, List[RunResult]] = defaultdict(list)
    for res in results:
        key = (res.constraint, res.n_components, res.penalty)
        aggregates[key].append(res)

    fieldnames = [
        'constraint', 'n_components', 'penalty', 'total_runs', 'successful_runs',
        'avg_lof', 'std_lof', 'min_lof', 'max_lof', 'best_seed', 'avg_iterations'
    ]

    agg_path = summary_path.parent / 'summary_aggregated.csv'
    with agg_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        aggregated_records = []
        for (constraint, n_components, penalty), items in sorted(aggregates.items()):
            total_runs = len(items)
            successful = [item for item in items if item.status == 'success' and item.final_lof is not None]
            lof_values = [item.final_lof for item in successful]
            iter_values = [item.iterations for item in successful if item.iterations is not None]

            if lof_values:
                avg_lof = mean(lof_values)
                std_lof = stdev(lof_values) if len(lof_values) > 1 else 0.0
                min_lof = min(lof_values)
                max_lof = max(lof_values)
                best_seed = min(successful, key=lambda r: r.final_lof if r.final_lof is not None else float('inf')).random_seed
            else:
                avg_lof = std_lof = min_lof = max_lof = None
                best_seed = None

            avg_iterations = mean(iter_values) if iter_values else None

            record = {
                'constraint': constraint,
                'n_components': n_components,
                'penalty': penalty,
                'total_runs': total_runs,
                'successful_runs': len(successful),
                'avg_lof': avg_lof,
                'std_lof': std_lof,
                'min_lof': min_lof,
                'max_lof': max_lof,
                'best_seed': best_seed,
                'avg_iterations': avg_iterations,
            }
            writer.writerow(record)
            aggregated_records.append(record)

    with (summary_path.parent / 'summary_aggregated.json').open('w', encoding='utf-8') as jf:
        json.dump(aggregated_records, jf, ensure_ascii=False, indent=2)


def run_single_experiment(
    analyzer_params: Dict,
    constraint_name: str,
    constraint_path: Optional[Path],
    output_dir: Path,
    max_iter: int,
    tol: float,
    save_plots: bool,
) -> RunResult:
    analyzer_params = analyzer_params.copy()
    analyzer_params.update({
        'constraint_config': str(constraint_path) if constraint_path else None
    })

    analyzer = TASAnalyzer(**{k: analyzer_params[k] for k in [
        'file_path', 'file_type', 'wavelength_range', 'delay_range',
        'n_components', 'language', 'constraint_config', 'penalty',
        'init_method', 'random_seed'
    ]})

    status = 'success'
    message = ''
    final_lof: Optional[float] = None
    iterations: Optional[int] = None

    try:
        if not analyzer.load_data():
            status = 'failed'
            message = '数据加载失败'
        else:
            if not analyzer.run_mcr_als(max_iter=max_iter, tol=tol):
                status = 'failed'
                message = 'MCR-ALS 运行失败'
            else:
                analyzer.save_results(output_dir=str(output_dir))
                if save_plots:
                    analyzer.visualize_results(save_plots=True, output_dir=str(output_dir))
                final_lof = float(analyzer.mcr_solver.lof_[-1]) if analyzer.mcr_solver.lof_ else None
                iterations = len(analyzer.mcr_solver.lof_)
    except Exception as exc:  # noqa: BLE001
        status = 'error'
        message = f"异常: {exc}"
        traceback.print_exc()

    if status != 'success' and not message:
        message = '未知原因失败'

    return RunResult(
        constraint=constraint_name,
        constraint_path=str(constraint_path) if constraint_path else None,
        n_components=analyzer_params['n_components'],
        penalty=float(analyzer_params['penalty']),
        random_seed=analyzer_params['random_seed'],
        final_lof=final_lof,
        iterations=iterations,
        status=status,
        output_dir=str(output_dir),
        message=message
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='批量执行 TAS MCR-ALS 网格搜索实验')
    parser.add_argument('--data-file', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'TAS' / 'TA_Average-crop-TCA-tzs-tzs-tzs-chirp.csv'),
                        help='输入数据文件路径')
    parser.add_argument('--file-type', type=str, default='handle', choices=['handle', 'raw'],
                        help='数据文件类型 (默认: handle)')
    parser.add_argument('--wavelength-range', type=float, nargs=2, default=(420.0, 750.0),
                        help='波长范围 (默认: 420 750)')
    parser.add_argument('--delay-range', type=float, nargs=2, default=(0.1, 50.0),
                        help='时间延迟范围 (默认: 0.1 50)')
    parser.add_argument('--components', type=str, default='1-4',
                        help='组分数量配置，例如 1-4 或 2,3,4')
    parser.add_argument('--penalty-min', type=float, default=0.1,
                        help='惩罚因子最小值 (默认: 0.1)')
    parser.add_argument('--penalty-max', type=float, default=1.0,
                        help='惩罚因子最大值 (默认: 1.0)')
    parser.add_argument('--penalty-step', type=float, default=0.1,
                        help='惩罚因子步长 (默认: 0.1)')
    parser.add_argument('--random-runs', type=int, default=5,
                        help='每个组合的随机初始化次数 (默认: 5)')
    parser.add_argument('--seed-offset', type=int, default=0,
                        help='随机种子偏移量 (默认: 0)')
    parser.add_argument('--constraints', type=str, default='default,standard,strict',
                        help='约束配置，使用逗号分隔，可选 default,standard,strict,relaxed 或配置文件路径')
    parser.add_argument('--language', type=str, default='chinese', choices=['chinese', 'english'],
                        help='输出语言 (默认: chinese)')
    parser.add_argument('--max-iter', type=int, default=200,
                        help='MCR-ALS 最大迭代次数 (默认: 200)')
    parser.add_argument('--tol', type=float, default=1e-7,
                        help='MCR-ALS 收敛容差 (默认: 1e-7)')
    parser.add_argument('--output-root', type=str,
                        default=str(PROJECT_ROOT / 'experiments' / 'results' / 'mcr_als_grid' / 'outputs'),
                        help='实验输出目录 (默认: experiments/results/mcr_als_grid/outputs)')
    parser.add_argument('--save-plots', action='store_true',
                        help='保存图表')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅打印计划，不执行分析')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_file = Path(args.data_file).resolve()
    if not data_file.is_file():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    components = parse_components(args.components)
    penalties = build_penalty_grid(args.penalty_min, args.penalty_max, args.penalty_step)
    constraint_map = resolve_constraint(args.constraints.split(','))

    output_root = Path(args.output_root).resolve()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_root = output_root / f"run_{timestamp}"
    experiment_root.mkdir(parents=True, exist_ok=True)

    base_params = {
        'file_path': str(data_file),
        'file_type': args.file_type,
        'wavelength_range': tuple(args.wavelength_range),
        'delay_range': tuple(args.delay_range),
        'language': args.language,
        'penalty': 0.0,  # will be overwritten per run
        'init_method': 'random',
        'random_seed': None,
        'n_components': components[0],  # placeholder
    }

    summary: List[RunResult] = []
    plan: List[str] = []
    combo_index = 0

    for constraint_name, constraint_path in constraint_map.items():
        for n_components in components:
            for penalty in penalties:
                for idx in range(args.random_runs):
                    seed = args.seed_offset + combo_index * args.random_runs + idx
                    run_dir = experiment_root / constraint_name / f"ncomp_{n_components}" / f"pen_{penalty:.2f}" / f"seed_{seed}"
                    plan.append(str(run_dir))
                    if args.dry_run:
                        continue

                    print(f"运行: constraint={constraint_name}, n_components={n_components}, penalty={penalty:.2f}, seed={seed}")
                    run_dir.mkdir(parents=True, exist_ok=True)
                    params = base_params.copy()
                    params.update({
                        'n_components': n_components,
                        'penalty': penalty,
                        'random_seed': seed,
                    })

                    result = run_single_experiment(
                        analyzer_params=params,
                        constraint_name=constraint_name,
                        constraint_path=constraint_path,
                        output_dir=run_dir,
                        max_iter=args.max_iter,
                        tol=args.tol,
                        save_plots=args.save_plots,
                    )
                    summary.append(result)
                combo_index += 1

    summary_path = experiment_root / 'summary.csv'
    write_summary(summary_path, summary)
    write_aggregated_summary(summary_path, summary)

    manifest = {
        'data_file': str(data_file),
        'components': components,
        'penalties': penalties,
        'random_runs': args.random_runs,
        'constraints': {name: str(path) if path else None for name, path in constraint_map.items()},
        'max_iter': args.max_iter,
        'tol': args.tol,
        'language': args.language,
        'save_plots': args.save_plots,
        'dry_run': args.dry_run,
        'plan_only_dirs': plan if args.dry_run else None,
    }
    with (experiment_root / 'experiment_manifest.json').open('w', encoding='utf-8') as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)

    print(f"实验完成，汇总结果保存在: {summary_path}")
    print(f"聚合统计保存在: {summary_path.parent / 'summary_aggregated.csv'}")


if __name__ == '__main__':
    main()
