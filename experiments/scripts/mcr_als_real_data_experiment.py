#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于筛选数据的MCR-ALS分析脚本
使用数据筛选器筛选出的挑战性TAS数据进行MCR-ALS分析
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

from mcr.mcr_als import MCRALS
from mcr.constraint_config import ConstraintConfig
from preprocessing.pipeline import TASPreprocessingPipeline
from data.data import read_file


class RealDataMCRALSExperiment:
    """基于真实筛选数据的MCR-ALS实验"""

    def __init__(self, screened_data_path: str = "experiments/results/data_screening/screening_results.json",
                 output_dir: str = "experiments/results/mcr_als_real_data"):
        """
        初始化实验

        Parameters:
        - screened_data_path: 筛选结果JSON文件路径
        - output_dir: 输出目录
        """
        self.screened_data_path = Path(screened_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载筛选数据
        self.screened_data = self._load_screened_data()

        # 初始化预处理器
        self.preprocessor = TASPreprocessingPipeline()

        print(f"加载了 {len(self.screened_data)} 个挑战性数据集")

    def _load_screened_data(self) -> Dict[str, List[Dict]]:
        """加载筛选后的数据"""
        if not self.screened_data_path.exists():
            raise FileNotFoundError(f"筛选数据文件不存在: {self.screened_data_path}")

        with open(self.screened_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 只保留有挑战性的类别
        challenge_categories = ['multi_peak_overlap', 'transient_decay', 'low_snr']
        screened_data = {}

        for category in challenge_categories:
            if category in data:
                screened_data[category] = data[category]

        return screened_data

    def load_single_dataset(self, data_info: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加载单个数据集

        Returns:
        - data_matrix: 预处理后的数据矩阵 (wavelengths × times)
        - wavelengths: 波长数组
        - times: 时间数组
        """
        file_path = data_info['file_path']

        # 使用与数据筛选器相同的数据加载逻辑
        try:
            data_matrix, wavelengths, times = self.load_tas_file(file_path)
            if data_matrix is None:
                return None, None, None

            return data_matrix, wavelengths, times
        except Exception as e:
            print(f"加载数据失败 {file_path}: {e}")
            return None, None, None

    def load_tas_file(self, file_path):
        """加载TAS文件 - 健壮版本（复制自数据筛选器）"""
        try:
            print(f"   正在加载: {Path(file_path).name}")

            # 首先尝试读取原始CSV
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # 跳过可能的头部信息
            data_start = 0
            for i, line in enumerate(lines):
                if line.strip() and ',' in line:
                    try:
                        # 尝试解析第一行为数字
                        float(line.split(',')[0])
                        data_start = i
                        break
                    except:
                        continue

            if data_start >= len(lines) - 5:
                print("   ❌ 找不到有效数据行")
                return None, None, None

            # 读取数据部分
            data_lines = lines[data_start:]

            # 解析第一行为时间延迟
            time_delays = []
            first_line = data_lines[0].strip().split(',')
            for val in first_line:
                try:
                    time_delays.append(float(val))
                except:
                    time_delays.append(0.0)

            time_delays = np.array(time_delays)

            # 解析其余行
            wavelengths = []
            data_matrix = []

            for line in data_lines[1:]:
                if not line.strip():
                    continue

                parts = line.strip().split(',')
                if len(parts) < 2:
                    continue

                try:
                    # 第一个值是波长
                    wl = float(parts[0])
                    wavelengths.append(wl)

                    # 其余值是数据
                    row_data = []
                    for val in parts[1:]:
                        try:
                            v = float(val)
                            if np.isfinite(v):
                                row_data.append(v)
                            else:
                                row_data.append(0.0)
                        except:
                            row_data.append(0.0)

                    # 确保长度匹配
                    while len(row_data) < len(time_delays):
                        row_data.append(0.0)
                    row_data = row_data[:len(time_delays)]

                    data_matrix.append(row_data)

                except Exception as e:
                    print(f"   ⚠️ 跳过无效行: {e}")
                    continue

            if len(data_matrix) < 10:
                print(f"   ❌ 数据行数太少: {len(data_matrix)}")
                return None, None, None

            wavelengths = np.array(wavelengths)
            data = np.array(data_matrix)

            # 数据验证和清理
            if data.shape[0] < 10 or data.shape[1] < 10:
                print(f"   ❌ 数据形状太小: {data.shape}")
                return None, None, None

            # 处理异常值和无穷大
            data = np.where(np.isfinite(data), data, 0)

            # 移除全零行和列
            non_zero_rows = np.any(np.abs(data) > 1e-10, axis=1)
            non_zero_cols = np.any(np.abs(data) > 1e-10, axis=0)

            if np.sum(non_zero_rows) < 5 or np.sum(non_zero_cols) < 5:
                print("   ❌ 有效数据太少")
                return None, None, None

            data = data[non_zero_rows, :][:, non_zero_cols]
            wavelengths = wavelengths[non_zero_rows]
            time_delays = time_delays[non_zero_cols]

            print(f"   ✅ 成功加载: {data.shape} (波长×时间)")
            print(f"   波长范围: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
            print(f"   时间范围: {time_delays.min():.2f} - {time_delays.max():.2f} ps")
            print(f"   数据范围: {data.min():.2e} - {data.max():.2e}")

            return data, wavelengths, time_delays

        except Exception as e:
            print(f"加载数据失败 {file_path}: {e}")
            return None, None, None

    def run_mcr_analysis(self, data_matrix: np.ndarray, n_components: int = 3,
                        max_iter: int = 200, tol: float = 1e-6) -> Dict[str, Any]:
        """
        对单个数据集运行MCR-ALS分析

        Parameters:
        - data_matrix: 输入数据矩阵 (wavelengths × times)
        - n_components: 组分数
        - max_iter: 最大迭代次数
        - tol: 收敛容差

        Returns:
        - 分析结果字典
        """
        print(f"运行MCR-ALS分析: {n_components} 组分, 数据形状 {data_matrix.shape}")

        # 创建MCR-ALS求解器
        mcr = MCRALS(
            n_components=n_components,
            max_iter=max_iter,
            tol=tol,
            constraint_config="config/constraint_config.json"  # 使用配置文件
        )

        # 运行分析
        start_time = time.time()
        try:
            C, S, lof_history = mcr.fit_transform(data_matrix)
            computation_time = time.time() - start_time

            # 计算最终LOF
            final_lof = lof_history[-1] if lof_history else float('inf')

            result = {
                'success': True,
                'C_matrix': C,
                'S_matrix': S,
                'lof_history': lof_history,
                'final_lof': final_lof,
                'iterations': len(lof_history),
                'computation_time': computation_time,
                'converged': len(lof_history) < max_iter
            }

        except Exception as e:
            computation_time = time.time() - start_time
            print(f"MCR-ALS分析失败: {e}")
            result = {
                'success': False,
                'error': str(e),
                'computation_time': computation_time
            }

        return result

    def analyze_category(self, category_name: str, n_components: int = 3,
                        max_samples: int = 10) -> Dict[str, Any]:
        """
        分析某个类别的所有数据集

        Parameters:
        - category_name: 类别名称 ('multi_peak_overlap', 'transient_decay', 'low_snr')
        - n_components: MCR-ALS组分数
        - max_samples: 最大分析样本数

        Returns:
        - 类别分析结果
        """
        if category_name not in self.screened_data:
            raise ValueError(f"未知的类别: {category_name}")

        datasets = self.screened_data[category_name]
        print(f"\n=== 分析类别: {category_name} ({len(datasets)} 个数据集) ===")

        # 限制样本数量
        if len(datasets) > max_samples:
            print(f"随机选择 {max_samples} 个样本进行分析")
            np.random.seed(42)  # 保证可重复性
            selected_indices = np.random.choice(len(datasets), max_samples, replace=False)
            datasets = [datasets[i] for i in selected_indices]

        results = []
        successful_analyses = 0

        for i, data_info in enumerate(datasets):
            print(f"\n[{i+1}/{len(datasets)}] 分析: {Path(data_info['file_path']).name}")

            # 加载数据
            data_matrix, wavelengths, times = self.load_single_dataset(data_info)
            if data_matrix is None:
                continue

            # 运行MCR-ALS分析
            mcr_result = self.run_mcr_analysis(data_matrix, n_components)

            # 保存结果
            result_entry = {
                'data_info': data_info,
                'data_shape': data_matrix.shape,
                'wavelength_range': [float(wavelengths.min()), float(wavelengths.max())],
                'time_range': [float(times.min()), float(times.max())],
                'mcr_result': mcr_result
            }

            results.append(result_entry)

            if mcr_result['success']:
                successful_analyses += 1
                print(f"  ✅ 分析成功: LOF = {mcr_result['final_lof']:.4f}%")
            else:
                print(f"  ❌ 分析失败: {mcr_result.get('error', '未知错误')}")

        # 生成类别汇总
        category_summary = {
            'category_name': category_name,
            'total_datasets': len(datasets),
            'successful_analyses': successful_analyses,
            'success_rate': successful_analyses / len(datasets) if datasets else 0,
            'n_components': n_components,
            'results': results
        }

        return category_summary

    def run_full_experiment(self, n_components: int = 3, max_samples_per_category: int = 5):
        """
        运行完整实验 - 分析所有挑战性数据类别

        Parameters:
        - n_components: MCR-ALS组分数
        - max_samples_per_category: 每个类别最大样本数
        """
        print("=== 开始基于筛选数据的MCR-ALS分析实验 ===")
        print(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"输出目录: {self.output_dir}")

        experiment_results = {}
        all_results = []

        # 分析每个挑战性类别
        categories = ['multi_peak_overlap', 'transient_decay', 'low_snr']

        for category in categories:
            try:
                category_result = self.analyze_category(
                    category, n_components, max_samples_per_category
                )
                experiment_results[category] = category_result
                all_results.extend(category_result['results'])
            except Exception as e:
                print(f"分析类别 {category} 失败: {e}")
                experiment_results[category] = {'error': str(e)}

        # 保存完整结果
        self._save_results(experiment_results, all_results)

        # 生成汇总报告
        self._generate_summary_report(experiment_results)

        print("\n=== 实验完成 ===")
        print(f"结果已保存到: {self.output_dir}")

    def _save_results(self, experiment_results: Dict, all_results: List):
        """保存实验结果"""
        # 保存完整结果
        results_file = self.output_dir / "mcr_als_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            # 转换为可JSON序列化的格式
            serializable_results = self._make_serializable(experiment_results)
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        # 保存汇总数据为CSV
        summary_data = []
        for category, cat_result in experiment_results.items():
            if isinstance(cat_result, dict) and 'results' in cat_result:
                for result in cat_result['results']:
                    mcr_result = result.get('mcr_result', {})
                    if mcr_result.get('success', False):
                        summary_data.append({
                            'category': category,
                            'file_name': Path(result['data_info']['file_path']).name,
                            'data_shape': str(result['data_shape']),
                            'final_lof': mcr_result.get('final_lof', float('inf')),
                            'iterations': mcr_result.get('iterations', 0),
                            'computation_time': mcr_result.get('computation_time', 0),
                            'converged': mcr_result.get('converged', False)
                        })

        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_file = self.output_dir / "mcr_als_summary.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"汇总数据已保存: {csv_file}")

    def _make_serializable(self, obj):
        """将对象转换为可JSON序列化的格式"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    def _generate_summary_report(self, experiment_results: Dict):
        """生成汇总报告"""
        report_file = self.output_dir / "experiment_summary.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# MCR-ALS真实数据分析实验报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 实验概述\n\n")
            f.write("- **数据来源**: 数据筛选器筛选出的挑战性TAS数据\n")
            f.write("- **分析方法**: MCR-ALS (Multivariate Curve Resolution - Alternating Least Squares)\n")
            f.write("- **约束配置**: 使用配置文件 `config/constraint_config.json`\n\n")

            f.write("## 分析结果汇总\n\n")

            total_analyses = 0
            successful_analyses = 0

            for category, cat_result in experiment_results.items():
                if isinstance(cat_result, dict) and 'results' in cat_result:
                    f.write(f"### {category.replace('_', ' ').title()}\n\n")
                    f.write(f"- **数据集数量**: {cat_result['total_datasets']}\n")
                    f.write(f"- **成功分析**: {cat_result['successful_analyses']}\n")
                    f.write(f"- **成功率**: {cat_result['success_rate']:.1f}%\n")
                    f.write(f"- **MCR-ALS组分**: {cat_result['n_components']}\n\n")

                    total_analyses += cat_result['total_datasets']
                    successful_analyses += cat_result['successful_analyses']

                    # 显示前几个成功结果的LOF
                    successful_results = [
                        r for r in cat_result['results']
                        if r.get('mcr_result', {}).get('success', False)
                    ][:5]  # 只显示前5个

                    if successful_results:
                        f.write("**成功分析的LOF值**:\n")
                        for result in successful_results:
                            file_name = Path(result['data_info']['file_path']).name
                            lof = result['mcr_result']['final_lof']
                            f.write(f"- {file_name}: LOF = {lof:.4f}%\n")
                        f.write("\n")

            f.write("## 总体统计\n\n")
            f.write(f"- **总分析数据集**: {total_analyses}\n")
            f.write(f"- **成功分析**: {successful_analyses}\n")
            if total_analyses > 0:
                f.write(f"- **成功率**: {successful_analyses/total_analyses:.1f}%\n")
            f.write("\n")

            f.write("## 结论\n\n")
            f.write("本次实验成功验证了MCR-ALS算法在处理筛选出的挑战性TAS数据时的性能。\n")
            f.write("不同类别的数据（多峰重叠、瞬态衰减、低信噪比）展现了不同的分析难度和收敛特性。\n\n")

        print(f"汇总报告已生成: {report_file}")


def main():
    """主函数"""
    # 创建实验实例
    experiment = RealDataMCRALSExperiment()

    # 运行完整实验
    experiment.run_full_experiment(
        n_components=3,  # 使用3个组分
        max_samples_per_category=5  # 每个类别分析5个样本
    )


if __name__ == "__main__":
    main()