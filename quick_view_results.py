#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速查看MCR-ALS分析结果的脚本
Usage: python quick_view_results.py [run_directory]
"""

import sys
import os
from pathlib import Path
import pandas as pd
import json

def view_results(run_dir=None):
    """查看分析结果"""
    
    # 如果没有指定目录，自动查找最新的
    if run_dir is None:
        output_base = Path("experiments/results/mcr_als_grid/outputs")
        if not output_base.exists():
            print("❌ 输出目录不存在，请先运行分析")
            return
        
        run_dirs = [d for d in output_base.iterdir() if d.is_dir() and d.name.startswith("run_")]
        if not run_dirs:
            print("❌ 没有找到运行结果")
            return
        
        run_dir = sorted(run_dirs)[-1]
    else:
        run_dir = Path(run_dir)
    
    print(f"📊 查看分析结果: {run_dir.name}")
    print("=" * 60)
    
    # 读取配置信息
    manifest_file = run_dir / "experiment_manifest.json"
    if manifest_file.exists():
        with open(manifest_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("🔧 实验配置:")
        print(f"   数据文件: {Path(config['data_file']).name}")
        print(f"   组分数: {config['components']}")
        print(f"   惩罚因子: {config['penalties']}")
        print(f"   随机运行次数: {config['random_runs']}")
        print(f"   约束条件: {list(config['constraints'].keys())}")
        print(f"   最大迭代: {config['max_iter']}")
        print()
    
    # 读取汇总结果
    summary_file = run_dir / "summary.csv"
    agg_file = run_dir / "summary_aggregated.csv"
    
    if not summary_file.exists():
        print("❌ 找不到汇总文件")
        return
    
    df_detail = pd.read_csv(summary_file)
    df_agg = pd.read_csv(agg_file) if agg_file.exists() else None
    
    print("📈 关键指标:")
    print(f"   总运行次数: {len(df_detail)}")
    print(f"   成功率: {sum(df_detail['status'] == 'success')}/{len(df_detail)} ({sum(df_detail['status'] == 'success')/len(df_detail)*100:.1f}%)")
    print(f"   LOF值范围: {df_detail['final_lof'].min():.4f}% - {df_detail['final_lof'].max():.4f}%")
    print()
    
    # 最佳结果
    print("🏆 最佳结果 TOP 3:")
    best_results = df_detail.nsmallest(3, 'final_lof')
    for i, (_, row) in enumerate(best_results.iterrows(), 1):
        print(f"   {i}. 组分{int(row['n_components'])}, 惩罚{row['penalty']:.1f}, 种子{int(row['random_seed'])}")
        print(f"      LOF: {row['final_lof']:.4f}%, 迭代: {int(row['iterations'])}次")
    print()
    
    # 参数组合排名
    if df_agg is not None:
        print("🎯 参数组合性能排名:")
        print("   排名 | 组分 | 惩罚 | 最佳LOF | 平均LOF | 标准差 | 推荐")
        print("   " + "-" * 55)
        
        df_agg_sorted = df_agg.sort_values('min_lof')
        for rank, (_, row) in enumerate(df_agg_sorted.iterrows(), 1):
            stability = "★★★" if row['std_lof'] < 1e-6 else "★★☆" if row['std_lof'] < 1e-3 else "★☆☆"
            recommend = "🔥" if row['min_lof'] < 90.4 else "✅" if row['min_lof'] < 90.7 else "⚡"
            print(f"    {rank}   |  {int(row['n_components'])}   | {row['penalty']:.1f}  | {row['min_lof']:.4f} | {row['avg_lof']:.4f} |  {stability}  | {recommend}")
        print()
    
    # 建议
    best_row = df_detail.loc[df_detail['final_lof'].idxmin()]
    print("💡 建议:")
    print(f"   推荐使用: {int(best_row['n_components'])}组分, 惩罚因子{best_row['penalty']:.1f}")
    print(f"   最佳种子: {int(best_row['random_seed'])}")
    print(f"   预期LOF: {best_row['final_lof']:.4f}%")
    print()
    
    # 文件位置
    print("📁 详细结果文件:")
    print(f"   📊 Excel汇总: {run_dir / '结果汇总_Excel版.csv'}")
    print(f"   📝 详细报告: {run_dir / '结果解释报告.md'}")
    print(f"   📈 原始数据: {summary_file}")
    print(f"   🔍 最佳结果: {best_row['output_dir']}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        view_results(sys.argv[1])
    else:
        view_results()