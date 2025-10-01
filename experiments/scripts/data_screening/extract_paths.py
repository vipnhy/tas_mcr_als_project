#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TAS筛选数据路径提取器
专门提取和显示筛选出的挑战性数据的路径信息
"""

import json
from pathlib import Path

RESULTS_DIR = Path("experiments/results/data_screening")

def extract_screened_data_paths():
    """提取筛选出的数据路径"""
    results_file = RESULTS_DIR / "screening_results.json"

    if not results_file.exists():
        print("❌ 找不到筛选结果文件")
        return

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print("🎯 TAS挑战性数据筛选结果 - 路径信息")
    print("=" * 60)

    total_files = 0

    for category, title in [
        ('multi_peak_overlap', '多峰重叠数据'),
        ('transient_decay', '瞬态衰减数据'),
        ('low_snr', '低信噪比数据')
    ]:
        files = results.get(category, [])
        if files:
            print(f"\n🎯 {title} ({len(files)} 个文件):")
            print("-" * 40)

            for i, file_info in enumerate(files, 1):
                print(f"{i}. {file_info['file_name']}")
                print(f"   📂 绝对路径: {file_info['file_path']}")
                print(f"   📁 相对路径: {file_info.get('relative_path', file_info['file_path'])}")
                print(f"   📊 数据形状: {file_info['shape'][0]}×{file_info['shape'][1]}")
                print(f"   📈 波长范围: {file_info['wavelength_range'][0]:.1f} - {file_info['wavelength_range'][1]:.1f} nm")
                print(f"   ⏱️  时间范围: {file_info['time_range'][0]:.2f} - {file_info['time_range'][1]:.2f} ps")
                print()

            total_files += len(files)

    print("=" * 60)
    print(f"📊 总计筛选出 {total_files} 个挑战性数据文件")
    print("\n💡 使用提示:")
    print("- 绝对路径可用于脚本直接访问文件")
    print("- 相对路径显示文件在项目中的位置")
    print("- 可视化文件保存在对应类别目录中")

    # 生成路径列表文件
    paths_file = RESULTS_DIR / "screened_data_paths.txt"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(paths_file, 'w', encoding='utf-8') as f:
        f.write("# TAS挑战性数据筛选结果 - 文件路径列表\n")
        f.write(f"# 生成时间: {Path('.').resolve()}\n\n")

        for category, title in [('multi_peak_overlap', '多峰重叠数据'),
                               ('transient_decay', '瞬态衰减数据'),
                               ('low_snr', '低信噪比数据')]:
            files = results.get(category, [])
            if files:
                f.write(f"## {title}\n")
                for file_info in files:
                    f.write(f"# {file_info['file_name']}\n")
                    f.write(f"{file_info['file_path']}\n")
                    f.write(f"{file_info.get('relative_path', file_info['file_path'])}\n\n")

    print(f"\n📄 路径列表已保存: {paths_file}")

if __name__ == "__main__":
    extract_screened_data_paths()