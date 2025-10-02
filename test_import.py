#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_import.py - 测试Globalfit模块导入
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("Project root:", project_root)
print("Current dir:", os.getcwd())
print("Python path:", sys.path[:3])

try:
    print("Testing lmfit import...")
    import lmfit
    print("✓ lmfit imported successfully")

    print("Testing numpy import...")
    import numpy as np
    print("✓ numpy imported successfully")

    print("Testing Globalfit.kinetic_models import...")
    import Globalfit.kinetic_models
    print("✓ kinetic_models imported successfully")

except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()