#!/usr/bin/env python3
"""
测试可视化工具的多语言功能
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from visualize_results import TASResultsVisualizer

def test_visualization():
    """测试可视化功能"""
    # 测试输出目录
    test_dir = "analysis_runs/demo_vis_batch_20251002_102611/global_fit/components_4/run_02_seed_4009801342/gta_sequential/attempt_02"

    if not Path(test_dir).exists():
        print(f"测试目录不存在: {test_dir}")
        return

    print("测试中文可视化...")
    visualizer_zh = TASResultsVisualizer(test_dir, language='zh')
    zh_report = visualizer_zh.generate_visualization_report(f"test_report_zh.html")
    print(f"中文报告生成: {zh_report}")

    print("测试英文可视化...")
    visualizer_en = TASResultsVisualizer(test_dir, language='en')
    en_report = visualizer_en.generate_visualization_report(f"test_report_en.html")
    print(f"英文报告生成: {en_report}")

    print("测试完成！")

if __name__ == "__main__":
    test_visualization()