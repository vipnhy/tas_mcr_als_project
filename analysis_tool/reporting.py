"""分析结果汇总与报告生成."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .config import AnalysisConfig


class AnalysisReporter:
    """生成 Markdown/JSON 报告的帮助类."""

    def __init__(self, analysis_config: AnalysisConfig, working_dir: Path):
        self.config = analysis_config
        self.working_dir = Path(working_dir)

    def generate(self, mcr_summary: Dict[str, Any], global_summary: Dict[str, Any]) -> Path:
        """生成最终 Markdown 报告, 返回文件路径."""

        report_path = self.working_dir / "final_report.md"
        report_lines: List[str] = []

        report_lines.append(f"# 分析报告 — {self.config.analysis.name}")
        report_lines.append("")
        report_lines.append("## 分析配置")
        report_lines.append("")
        report_lines.append("```json")
        report_lines.append(json.dumps(self.config.raw_config, indent=2, ensure_ascii=False))
        report_lines.append("```")
        report_lines.append("")

        report_lines.extend(self._render_mcr_section(mcr_summary))
        report_lines.extend(self._render_global_section(global_summary))

        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        return report_path

    # ------------------------------------------------------------------
    def _render_mcr_section(self, summary: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        lines.append("## MCR-ALS 批量结果")
        lines.append("")
        runs = summary.get("runs", [])
        if not runs:
            lines.append("*未找到 MCR 运行结果*")
            lines.append("")
            return lines

        lines.append("| 组分数 | 运行编号 | 种子 | LOF (%) | R² | 输出目录 |")
        lines.append("|--------|----------|------|---------|----|----------|")
        for run in runs:
            lines.append(
                "| {components} | {run_index} | {seed} | {lof:.4f} | {r2:.4f} | `{output_dir}` |".format(
                    components=run.get("components"),
                    run_index=run.get("run_index"),
                    seed=run.get("seed", "-"),
                    lof=run.get("lof", 0.0),
                    r2=run.get("r2", 0.0),
                    output_dir=run.get("output_dir"),
                )
            )
        lines.append("")
        return lines

    # ------------------------------------------------------------------
    def _render_global_section(self, summary: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        lines.append("## 全局拟合结果")
        lines.append("")
        ranking = summary.get("ranking", [])
        if not ranking:
            lines.append("*未找到全局拟合结果*")
            lines.append("")
            return lines

        lines.append("| 排名 | 模型 | 尝试 | 组分 | LOF (%) | Chi² | 输出目录 |")
        lines.append("|------|------|------|------|---------|------|----------|")
        for idx, item in enumerate(ranking, start=1):
            lines.append(
                "| {rank} | {model} | {attempt} | {n_components} | {lof:.4f} | {chi:.4e} | `{result_dir}` |".format(
                    rank=idx,
                    model=item.get("model"),
                    attempt=item.get("attempt"),
                    n_components=item.get("n_components"),
                    lof=item.get("lof", 0.0),
                    chi=item.get("chi_square", 0.0),
                    result_dir=item.get("result_dir"),
                )
            )
        lines.append("")

        best = ranking[0]
        lines.append("### 推荐模型")
        lines.append("")
        lines.append(
            "- **模型**: {model} (尝试 {attempt})".format(
                model=best.get("model"), attempt=best.get("attempt")
            )
        )
        lines.append("- **LOF**: {:.4f}%".format(best.get("lof", 0.0)))
        lines.append("- **Chi²**: {:.4e}".format(best.get("chi_square", 0.0)))
        lines.append("- **结果目录**: `{}`".format(best.get("result_dir")))
        lines.append("")
        return lines
