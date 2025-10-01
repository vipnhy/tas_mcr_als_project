"""
Globalfit - 全局拟合模块

该模块提供瞬态吸收光谱(TAS)数据的全局拟合分析功能，
包括全局寿命分析(GLA)和全局目标分析(GTA)。

主要组件:
- kinetic_models: 动力学模型定义
- model: 全局拟合核心算法
- interface: MCR-ALS输出接口
- utils: 工具函数
"""

__version__ = "1.0.0"
__author__ = "TAS Analysis Team"

from .kinetic_models import (
    SequentialModel,
    ParallelModel,
    KineticModelBase
)

from .model import (
    GlobalFitter,
    GlobalLifetimeAnalysis,
    GlobalTargetAnalysis
)

from .interface import MCRALSInterface

__all__ = [
    'SequentialModel',
    'ParallelModel',
    'KineticModelBase',
    'GlobalFitter',
    'GlobalLifetimeAnalysis',
    'GlobalTargetAnalysis',
    'MCRALSInterface'
]
