#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCR-ALS实验框架
多轮分析、约束测试和参数扩展性评估的完整实验系统
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict

from mcr.mcr_als import MCRALS
from mcr.constraint_config import ConstraintConfig
from main import generate_synthetic_tas_data


@dataclass
class ExperimentResult:
    """单次实验结果数据结构"""
    experiment_id: str
    n_components: int
    constraint_type: str
    constraint_strength: float
    random_seed: int
    iterations_to_converge: int
    final_lof: float
    converged: bool
    computation_time: float
    C_matrix: np.ndarray = None
    S_matrix: np.ndarray = None
    lof_history: List[float] = None
    
    def to_dict(self):
        """转换为字典格式（排除大矩阵）"""
        result = asdict(self)
        # 排除大矩阵以避免JSON序列化问题
        result.pop('C_matrix', None)
        result.pop('S_matrix', None)
        result.pop('lof_history', None)
        return result


class MCRExperimentRunner:
    """MCR-ALS实验运行器"""
    
    def __init__(self, output_base_dir: str = "mcr_experiments"):
        """
        初始化实验运行器
        
        Parameters:
        - output_base_dir: 实验输出根目录
        """
        self.output_base_dir = Path(output_base_dir)
        self.experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_experiment_dir = self.output_base_dir / f"experiment_{self.experiment_timestamp}"
        
        # 实验结果存储
        self.results: List[ExperimentResult] = []
        self.summary_stats = {}
        
        # 创建输出目录结构
        self._create_directory_structure()
    
    def _create_directory_structure(self):
        """创建分级目录结构"""
        directories = [
            self.current_experiment_dir,
            self.current_experiment_dir / "level1_summary",
            self.current_experiment_dir / "level2_constraint_analysis", 
            self.current_experiment_dir / "level3_component_scaling",
            self.current_experiment_dir / "level4_parameter_tuning",
            self.current_experiment_dir / "level5_individual_runs",
            self.current_experiment_dir / "plots",
            self.current_experiment_dir / "data"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"创建实验目录结构: {self.current_experiment_dir}")
    
    def _create_constraint_configurations(self) -> Dict[str, ConstraintConfig]:
        """创建不同强度的约束配置"""
        configurations = {}
        
        # 1. 基本配置（仅非负性约束）
        basic_config = ConstraintConfig()
        configurations["basic"] = basic_config
        
        # 2. 不同强度的平滑度约束
        smoothness_strengths = [0.1, 0.2, 0.5, 1.0]
        for strength in smoothness_strengths:
            config = ConstraintConfig()
            config.enable_constraint("spectral_smoothness")
            config.set_constraint_parameter("spectral_smoothness", "lambda", strength)
            configurations[f"smoothness_{strength}"] = config
        
        # 3. 组合约束配置
        combined_config = ConstraintConfig()
        combined_config.enable_constraint("spectral_smoothness")
        combined_config.set_constraint_parameter("spectral_smoothness", "lambda", 0.5)
        configurations["combined"] = combined_config
        
        return configurations
    
    def run_single_experiment(self, 
                            data_matrix: np.ndarray,
                            n_components: int,
                            constraint_config: ConstraintConfig,
                            constraint_name: str,
                            constraint_strength: float,
                            random_seed: int,
                            max_iter: int = 200,
                            tolerance: float = 1e-6) -> ExperimentResult:
        """
        运行单次MCR-ALS实验
        
        Returns:
        - ExperimentResult: 实验结果对象
        """
        # 设置随机种子
        np.random.seed(random_seed)
        
        # 创建MCR-ALS求解器
        mcr_solver = MCRALS(
            n_components=n_components,
            max_iter=max_iter,
            tol=tolerance,
            constraint_config=constraint_config
        )
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 运行MCR-ALS分析
            mcr_solver.fit(data_matrix)
            
            # 记录结果
            computation_time = time.time() - start_time
            final_lof = mcr_solver.lof_[-1] if mcr_solver.lof_ else float('inf')
            converged = len(mcr_solver.lof_) < max_iter
            
            experiment_id = f"{constraint_name}_comp{n_components}_seed{random_seed}"
            
            result = ExperimentResult(
                experiment_id=experiment_id,
                n_components=n_components,
                constraint_type=constraint_name,
                constraint_strength=constraint_strength,
                random_seed=random_seed,
                iterations_to_converge=len(mcr_solver.lof_),
                final_lof=final_lof,
                converged=converged,
                computation_time=computation_time,
                C_matrix=mcr_solver.C_opt_.copy() if mcr_solver.C_opt_ is not None else None,
                S_matrix=mcr_solver.S_opt_.copy() if mcr_solver.S_opt_ is not None else None,
                lof_history=mcr_solver.lof_.copy()
            )
            
        except Exception as e:
            print(f"实验失败: {e}")
            result = ExperimentResult(
                experiment_id=f"failed_{constraint_name}_comp{n_components}_seed{random_seed}",
                n_components=n_components,
                constraint_type=constraint_name,
                constraint_strength=constraint_strength,
                random_seed=random_seed,
                iterations_to_converge=-1,
                final_lof=float('inf'),
                converged=False,
                computation_time=time.time() - start_time
            )
        
        return result
    
    def run_multi_round_experiment(self,
                                 data_matrix: np.ndarray,
                                 n_components_range: List[int] = [1, 2, 3, 4],
                                 num_random_runs: int = 5,
                                 max_iter: int = 200,
                                 tolerance: float = 1e-6,
                                 target_lof: float = 0.2) -> None:
        """
        运行完整的多轮实验
        
        Parameters:
        - data_matrix: 输入数据矩阵
        - n_components_range: 组分数量范围测试
        - num_random_runs: 每个配置的随机初始化次数
        - max_iter: 最大迭代次数
        - tolerance: 收敛容差
        - target_lof: 目标LOF值
        """
        print("开始MCR-ALS多轮实验...")
        print(f"目标LOF值: {target_lof}")
        print(f"每个配置运行 {num_random_runs} 次")
        
        # 获取约束配置
        constraint_configs = self._create_constraint_configurations()
        
        total_experiments = len(n_components_range) * len(constraint_configs) * num_random_runs
        experiment_count = 0
        
        # 遍历所有实验配置
        for n_components in n_components_range:
            print(f"\n--- 测试组分数量: {n_components} ---")
            
            for constraint_name, constraint_config in constraint_configs.items():
                print(f"  约束类型: {constraint_name}")
                
                # 获取约束强度（用于记录）
                if "smoothness" in constraint_name and constraint_name != "combined":
                    strength = float(constraint_name.split('_')[1])
                else:
                    strength = 0.0
                
                # 运行多次随机初始化
                for run_idx in range(num_random_runs):
                    experiment_count += 1
                    random_seed = run_idx + 42  # 确定性种子
                    
                    print(f"    运行 {run_idx+1}/{num_random_runs} (种子: {random_seed}) "
                          f"[{experiment_count}/{total_experiments}]")
                    
                    # 运行单次实验
                    result = self.run_single_experiment(
                        data_matrix=data_matrix,
                        n_components=n_components,
                        constraint_config=constraint_config,
                        constraint_name=constraint_name,
                        constraint_strength=strength,
                        random_seed=random_seed,
                        max_iter=max_iter,
                        tolerance=tolerance
                    )
                    
                    # 存储结果
                    self.results.append(result)
                    
                    # 保存单次实验结果
                    self._save_individual_result(result)
                    
                    # 实时反馈
                    status = "✓" if result.final_lof < target_lof else "✗"
                    print(f"      LOF: {result.final_lof:.4f}% {status}")
        
        print(f"\n实验完成! 总计 {len(self.results)} 次实验")
        
        # 生成汇总分析
        self._generate_summary_analysis(target_lof)
        
        # 生成可视化
        self._generate_visualizations()
        
        print(f"结果保存至: {self.current_experiment_dir}")
    
    def _save_individual_result(self, result: ExperimentResult):
        """保存单次实验的详细结果"""
        result_dir = self.current_experiment_dir / "level5_individual_runs" / result.experiment_id
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存基本信息
        with open(result_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 保存矩阵（如果可用）
        if result.C_matrix is not None:
            np.savetxt(result_dir / "concentration_matrix.csv", result.C_matrix, delimiter=',')
        
        if result.S_matrix is not None:
            np.savetxt(result_dir / "spectra_matrix.csv", result.S_matrix, delimiter=',')
        
        # 保存LOF历史
        if result.lof_history:
            lof_df = pd.DataFrame({'iteration': range(len(result.lof_history)),
                                 'lof': result.lof_history})
            lof_df.to_csv(result_dir / "lof_history.csv", index=False)
    
    def _generate_summary_analysis(self, target_lof: float):
        """生成汇总分析报告"""
        print("生成汇总分析报告...")
        
        # 转换为DataFrame以便分析
        df_data = []
        for result in self.results:
            df_data.append(result.to_dict())
        
        df = pd.DataFrame(df_data)
        
        # === 1. 一级汇总报告（保存在顶级目录） ===
        summary_report = self._create_level1_summary(df, target_lof)
        with open(self.current_experiment_dir / "level1_summary" / "experiment_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False)
        
        # === 2. 约束分析报告 ===
        constraint_analysis = self._analyze_constraints(df, target_lof)
        with open(self.current_experiment_dir / "level2_constraint_analysis" / "constraint_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(constraint_analysis, f, indent=2, ensure_ascii=False)
        
        # === 3. 组分扩展性分析 ===
        component_analysis = self._analyze_component_scaling(df, target_lof)
        with open(self.current_experiment_dir / "level3_component_scaling" / "component_scaling.json", 'w', encoding='utf-8') as f:
            json.dump(component_analysis, f, indent=2, ensure_ascii=False)
        
        # === 4. 参数调优分析 ===
        parameter_analysis = self._analyze_parameter_tuning(df, target_lof)
        with open(self.current_experiment_dir / "level4_parameter_tuning" / "parameter_tuning.json", 'w', encoding='utf-8') as f:
            json.dump(parameter_analysis, f, indent=2, ensure_ascii=False)
        
        # === 5. 生成Excel汇总 ===
        self._save_excel_summary(df)
        
        print("汇总分析报告生成完成")
    
    def _create_level1_summary(self, df: pd.DataFrame, target_lof: float) -> Dict[str, Any]:
        """创建一级汇总报告"""
        total_experiments = len(df)
        successful_runs = len(df[df['converged'] == True])
        target_achieved = len(df[df['final_lof'] < target_lof])
        
        summary = {
            "experiment_metadata": {
                "timestamp": self.experiment_timestamp,
                "total_experiments": int(total_experiments),
                "target_lof": target_lof,
                "successful_convergence": int(successful_runs),
                "target_lof_achieved": int(target_achieved)
            },
            "overall_performance": {
                "success_rate": float(successful_runs / total_experiments * 100),
                "target_achievement_rate": float(target_achieved / total_experiments * 100),
                "average_lof": float(df['final_lof'].mean()),
                "median_lof": float(df['final_lof'].median()),
                "best_lof": float(df['final_lof'].min()),
                "average_iterations": float(df[df['converged'] == True]['iterations_to_converge'].mean()),
                "average_computation_time": float(df['computation_time'].mean())
            },
            "best_configurations": self._find_best_configurations(df, target_lof),
            "component_scalability": {
                "optimal_component_count": self._find_optimal_component_count(df),
                "performance_by_components": self._performance_by_components(df)
            }
        }
        
        return summary
    
    def _analyze_constraints(self, df: pd.DataFrame, target_lof: float) -> Dict[str, Any]:
        """约束分析"""
        constraint_performance = {}
        
        for constraint_type in df['constraint_type'].unique():
            constraint_df = df[df['constraint_type'] == constraint_type]
            
            constraint_performance[constraint_type] = {
                "total_runs": len(constraint_df),
                "success_rate": float(len(constraint_df[constraint_df['converged'] == True]) / len(constraint_df) * 100),
                "target_achievement_rate": float(len(constraint_df[constraint_df['final_lof'] < target_lof]) / len(constraint_df) * 100),
                "average_lof": float(constraint_df['final_lof'].mean()),
                "best_lof": float(constraint_df['final_lof'].min()),
                "average_iterations": float(constraint_df[constraint_df['converged'] == True]['iterations_to_converge'].mean()) if len(constraint_df[constraint_df['converged'] == True]) > 0 else None,
                "average_time": float(constraint_df['computation_time'].mean())
            }
        
        return {
            "constraint_comparison": constraint_performance,
            "best_constraint_type": self._find_best_constraint(df, target_lof)
        }
    
    def _analyze_component_scaling(self, df: pd.DataFrame, target_lof: float) -> Dict[str, Any]:
        """组分扩展性分析"""
        component_performance = {}
        
        for n_comp in sorted(df['n_components'].unique()):
            comp_df = df[df['n_components'] == n_comp]
            
            component_performance[str(n_comp)] = {
                "total_runs": len(comp_df),
                "success_rate": float(len(comp_df[comp_df['converged'] == True]) / len(comp_df) * 100),
                "target_achievement_rate": float(len(comp_df[comp_df['final_lof'] < target_lof]) / len(comp_df) * 100),
                "average_lof": float(comp_df['final_lof'].mean()),
                "best_lof": float(comp_df['final_lof'].min()),
                "complexity_impact": self._assess_complexity_impact(comp_df)
            }
        
        return {
            "component_scaling_analysis": component_performance,
            "scalability_assessment": self._assess_scalability(df)
        }
    
    def _analyze_parameter_tuning(self, df: pd.DataFrame, target_lof: float) -> Dict[str, Any]:
        """参数调优分析"""
        # 分析平滑度参数的影响
        smoothness_df = df[df['constraint_type'].str.contains('smoothness') & 
                          (df['constraint_type'] != 'combined')]
        
        parameter_analysis = {}
        
        if not smoothness_df.empty:
            for strength in sorted(smoothness_df['constraint_strength'].unique()):
                strength_df = smoothness_df[smoothness_df['constraint_strength'] == strength]
                
                parameter_analysis[f"smoothness_{strength}"] = {
                    "strength": strength,
                    "success_rate": float(len(strength_df[strength_df['converged'] == True]) / len(strength_df) * 100),
                    "target_achievement_rate": float(len(strength_df[strength_df['final_lof'] < target_lof]) / len(strength_df) * 100),
                    "average_lof": float(strength_df['final_lof'].mean()),
                    "performance_stability": float(strength_df['final_lof'].std())
                }
        
        return {
            "parameter_sensitivity": parameter_analysis,
            "optimal_parameters": self._find_optimal_parameters(df, target_lof)
        }
    
    def _find_best_configurations(self, df: pd.DataFrame, target_lof: float) -> Dict[str, Any]:
        """寻找最佳配置"""
        # 按LOF排序，找出最佳配置
        best_overall = df.loc[df['final_lof'].idxmin()]
        best_achieving_target = df[df['final_lof'] < target_lof]
        
        result = {
            "best_overall_lof": {
                "experiment_id": best_overall['experiment_id'],
                "lof": float(best_overall['final_lof']),
                "constraint_type": best_overall['constraint_type'],
                "n_components": int(best_overall['n_components'])
            }
        }
        
        if not best_achieving_target.empty:
            fastest_target = best_achieving_target.loc[best_achieving_target['computation_time'].idxmin()]
            result["fastest_achieving_target"] = {
                "experiment_id": fastest_target['experiment_id'],
                "lof": float(fastest_target['final_lof']),
                "time": float(fastest_target['computation_time']),
                "constraint_type": fastest_target['constraint_type']
            }
        
        return result
    
    def _find_optimal_component_count(self, df: pd.DataFrame) -> int:
        """寻找最优组分数量"""
        component_avg_lof = df.groupby('n_components')['final_lof'].mean()
        return int(component_avg_lof.idxmin())
    
    def _performance_by_components(self, df: pd.DataFrame) -> Dict[str, float]:
        """按组分数量统计性能"""
        return df.groupby('n_components')['final_lof'].mean().to_dict()
    
    def _find_best_constraint(self, df: pd.DataFrame, target_lof: float) -> str:
        """寻找最佳约束类型"""
        constraint_performance = df.groupby('constraint_type')['final_lof'].mean()
        return constraint_performance.idxmin()
    
    def _assess_complexity_impact(self, comp_df: pd.DataFrame) -> str:
        """评估复杂度影响"""
        avg_time = comp_df['computation_time'].mean()
        if avg_time < 1.0:
            return "低复杂度"
        elif avg_time < 5.0:
            return "中等复杂度"
        else:
            return "高复杂度"
    
    def _assess_scalability(self, df: pd.DataFrame) -> str:
        """评估扩展性"""
        component_success = df.groupby('n_components')['converged'].mean()
        if component_success.min() > 0.8:
            return "优秀扩展性"
        elif component_success.min() > 0.6:
            return "良好扩展性"
        else:
            return "扩展性受限"
    
    def _find_optimal_parameters(self, df: pd.DataFrame, target_lof: float) -> Dict[str, Any]:
        """寻找最优参数"""
        smoothness_df = df[df['constraint_type'].str.contains('smoothness') & 
                          (df['constraint_type'] != 'combined')]
        
        if smoothness_df.empty:
            return {"message": "无平滑度参数数据"}
        
        # 找到达到目标LOF的最佳参数
        target_achieved = smoothness_df[smoothness_df['final_lof'] < target_lof]
        
        if not target_achieved.empty:
            best_param = target_achieved.groupby('constraint_strength')['final_lof'].mean().idxmin()
            return {
                "optimal_smoothness_strength": float(best_param),
                "achieved_lof": float(target_achieved[target_achieved['constraint_strength'] == best_param]['final_lof'].mean())
            }
        else:
            # 如果没有达到目标，返回最佳LOF的参数
            best_param = smoothness_df.groupby('constraint_strength')['final_lof'].mean().idxmin()
            return {
                "best_available_smoothness_strength": float(best_param),
                "best_lof": float(smoothness_df[smoothness_df['constraint_strength'] == best_param]['final_lof'].mean())
            }
    
    def _save_excel_summary(self, df: pd.DataFrame):
        """保存Excel格式汇总"""
        try:
            with pd.ExcelWriter(self.current_experiment_dir / "level1_summary" / "experiment_results.xlsx") as writer:
                # 所有结果
                df.to_excel(writer, sheet_name='All Results', index=False)
                
                # 按约束类型汇总
                constraint_summary = df.groupby('constraint_type').agg({
                    'final_lof': ['mean', 'min', 'std'],
                    'converged': 'mean',
                    'computation_time': 'mean'
                }).round(4)
                constraint_summary.to_excel(writer, sheet_name='Constraint Summary')
                
                # 按组分数量汇总
                component_summary = df.groupby('n_components').agg({
                    'final_lof': ['mean', 'min', 'std'],
                    'converged': 'mean',
                    'computation_time': 'mean'
                }).round(4)
                component_summary.to_excel(writer, sheet_name='Component Summary')
                
        except ImportError:
            print("警告: 无法导入openpyxl，跳过Excel文件生成")
    
    def _generate_visualizations(self):
        """生成可视化图表"""
        print("生成可视化图表...")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 转换为DataFrame
        df_data = [result.to_dict() for result in self.results]
        df = pd.DataFrame(df_data)
        
        # 1. LOF性能对比图
        self._plot_constraint_performance(df)
        
        # 2. 组分扩展性图
        self._plot_component_scaling(df)
        
        # 3. 参数调优图
        self._plot_parameter_tuning(df)
        
        # 4. 收敛性分析图
        self._plot_convergence_analysis(df)
        
        print("可视化图表生成完成")
    
    def _plot_constraint_performance(self, df: pd.DataFrame):
        """约束性能对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Constraint Performance Analysis', fontsize=16)
        
        # LOF均值对比
        constraint_lof = df.groupby('constraint_type')['final_lof'].mean().sort_values()
        axes[0, 0].bar(range(len(constraint_lof)), constraint_lof.values)
        axes[0, 0].set_xticks(range(len(constraint_lof)))
        axes[0, 0].set_xticklabels(constraint_lof.index, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Average LOF (%)')
        axes[0, 0].set_title('Average LOF by Constraint Type')
        axes[0, 0].axhline(y=0.2, color='r', linestyle='--', label='Target LOF')
        axes[0, 0].legend()
        
        # 成功率对比
        success_rate = df.groupby('constraint_type')['converged'].mean() * 100
        axes[0, 1].bar(range(len(success_rate)), success_rate.values)
        axes[0, 1].set_xticks(range(len(success_rate)))
        axes[0, 1].set_xticklabels(success_rate.index, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Success Rate (%)')
        axes[0, 1].set_title('Convergence Success Rate')
        
        # 计算时间对比
        comp_time = df.groupby('constraint_type')['computation_time'].mean()
        axes[1, 0].bar(range(len(comp_time)), comp_time.values)
        axes[1, 0].set_xticks(range(len(comp_time)))
        axes[1, 0].set_xticklabels(comp_time.index, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Average Time (s)')
        axes[1, 0].set_title('Average Computation Time')
        
        # LOF分布箱线图
        constraint_types = df['constraint_type'].unique()
        lof_data = [df[df['constraint_type'] == ct]['final_lof'].values for ct in constraint_types]
        axes[1, 1].boxplot(lof_data, labels=constraint_types)
        axes[1, 1].set_ylabel('LOF (%)')
        axes[1, 1].set_title('LOF Distribution by Constraint')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=0.2, color='r', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.current_experiment_dir / "plots" / "constraint_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_component_scaling(self, df: pd.DataFrame):
        """组分扩展性图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Component Scaling Analysis', fontsize=16)
        
        # LOF vs 组分数量
        component_lof = df.groupby('n_components')['final_lof'].agg(['mean', 'std'])
        axes[0].errorbar(component_lof.index, component_lof['mean'], yerr=component_lof['std'], 
                        marker='o', linewidth=2, markersize=8)
        axes[0].axhline(y=0.2, color='r', linestyle='--', label='Target LOF')
        axes[0].set_xlabel('Number of Components')
        axes[0].set_ylabel('Average LOF (%)')
        axes[0].set_title('LOF vs Component Count')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 成功率 vs 组分数量
        success_rate = df.groupby('n_components')['converged'].mean() * 100
        axes[1].plot(success_rate.index, success_rate.values, 'o-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Success Rate (%)')
        axes[1].set_title('Convergence Rate vs Component Count')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 105])
        
        # 计算时间 vs 组分数量
        comp_time = df.groupby('n_components')['computation_time'].agg(['mean', 'std'])
        axes[2].errorbar(comp_time.index, comp_time['mean'], yerr=comp_time['std'],
                        marker='s', linewidth=2, markersize=8)
        axes[2].set_xlabel('Number of Components')
        axes[2].set_ylabel('Computation Time (s)')
        axes[2].set_title('Computation Time vs Component Count')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.current_experiment_dir / "plots" / "component_scaling.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_tuning(self, df: pd.DataFrame):
        """参数调优图"""
        # 只分析平滑度参数
        smoothness_df = df[df['constraint_type'].str.contains('smoothness') & 
                          (df['constraint_type'] != 'combined')]
        
        if smoothness_df.empty:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Parameter Tuning Analysis (Smoothness)', fontsize=16)
        
        # LOF vs 平滑度强度
        param_lof = smoothness_df.groupby('constraint_strength')['final_lof'].agg(['mean', 'std'])
        axes[0].errorbar(param_lof.index, param_lof['mean'], yerr=param_lof['std'],
                        marker='o', linewidth=2, markersize=8)
        axes[0].axhline(y=0.2, color='r', linestyle='--', label='Target LOF')
        axes[0].set_xlabel('Smoothness Parameter (lambda)')
        axes[0].set_ylabel('Average LOF (%)')
        axes[0].set_title('LOF vs Smoothness Strength')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')
        
        # 目标达成率 vs 平滑度强度
        target_rate = smoothness_df.groupby('constraint_strength').apply(
            lambda x: (x['final_lof'] < 0.2).mean() * 100
        )
        axes[1].plot(target_rate.index, target_rate.values, 'o-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Smoothness Parameter (lambda)')
        axes[1].set_ylabel('Target Achievement Rate (%)')
        axes[1].set_title('Target LOF Achievement vs Smoothness')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')
        axes[1].set_ylim([0, 105])
        
        plt.tight_layout()
        plt.savefig(self.current_experiment_dir / "plots" / "parameter_tuning.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_convergence_analysis(self, df: pd.DataFrame):
        """收敛性分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Convergence Analysis', fontsize=16)
        
        # 迭代次数分布
        converged_df = df[df['converged'] == True]
        axes[0, 0].hist(converged_df['iterations_to_converge'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Iterations to Converge')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Convergence Iterations')
        axes[0, 0].grid(True, alpha=0.3)
        
        # LOF vs 迭代次数
        axes[0, 1].scatter(converged_df['iterations_to_converge'], converged_df['final_lof'], alpha=0.6)
        axes[0, 1].axhline(y=0.2, color='r', linestyle='--', label='Target LOF')
        axes[0, 1].set_xlabel('Iterations to Converge')
        axes[0, 1].set_ylabel('Final LOF (%)')
        axes[0, 1].set_title('Final LOF vs Convergence Speed')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 计算时间 vs LOF
        axes[1, 0].scatter(df['computation_time'], df['final_lof'], alpha=0.6)
        axes[1, 0].axhline(y=0.2, color='r', linestyle='--', label='Target LOF')
        axes[1, 0].set_xlabel('Computation Time (s)')
        axes[1, 0].set_ylabel('Final LOF (%)')
        axes[1, 0].set_title('LOF vs Computation Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 成功率饼图
        converged_count = df['converged'].sum()
        failed_count = len(df) - converged_count
        axes[1, 1].pie([converged_count, failed_count], 
                      labels=[f'Converged ({converged_count})', f'Failed ({failed_count})'],
                      autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Overall Convergence Success Rate')
        
        plt.tight_layout()
        plt.savefig(self.current_experiment_dir / "plots" / "convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()


def run_example_experiment():
    """运行示例实验"""
    print("=== MCR-ALS实验框架示例 ===")
    
    # 生成合成数据
    print("生成合成数据...")
    data_matrix, C_true, S_true = generate_synthetic_tas_data(n_times=80, n_wls=150, n_components=3)
    
    # 创建实验运行器
    runner = MCRExperimentRunner(output_base_dir="mcr_experiments")
    
    # 运行完整实验
    runner.run_multi_round_experiment(
        data_matrix=data_matrix,
        n_components_range=[1, 2, 3, 4],  # 测试1-4个组分
        num_random_runs=5,  # 每个配置运行5次
        max_iter=200,
        tolerance=1e-6,
        target_lof=0.2  # 目标LOF < 0.2%
    )
    
    print(f"实验完成! 结果保存在: {runner.current_experiment_dir}")
    return runner


if __name__ == "__main__":
    # 运行示例实验
    runner = run_example_experiment()
    
    # 显示汇总信息
    print("\n=== 实验汇总 ===")
    print(f"总实验次数: {len(runner.results)}")
    print(f"成功收敛: {sum(1 for r in runner.results if r.converged)}")
    print(f"达到目标LOF(<0.2%): {sum(1 for r in runner.results if r.final_lof < 0.2)}")
    
    best_result = min(runner.results, key=lambda x: x.final_lof)
    print(f"最佳LOF: {best_result.final_lof:.4f}% ({best_result.constraint_type}, {best_result.n_components}组分)")