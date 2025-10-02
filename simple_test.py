#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
simple_test.py - 简化的全局拟合测试
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("Testing Globalfit imports...")

try:
    from Globalfit import MCRALSInterface
    print("✓ MCRALSInterface imported")

    # 测试接口
    interface = MCRALSInterface("results")
    print("✓ MCRALSInterface initialized")

    data_dict = interface.prepare_for_global_fitting(
        data_file="data/TAS/TA_Average-crop-TCA-tzs-tzs-tzs-chirp.csv"
    )
    print("✓ Data prepared for global fitting")
    print(f"  Data shape: {data_dict['data_matrix'].shape}")
    print(f"  Components: {data_dict['n_components']}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()