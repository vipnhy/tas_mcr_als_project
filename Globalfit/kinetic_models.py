"""
kinetic_models.py - 动力学模型定义

该模块定义了常见的光物理/光化学动力学模型，用于描述
瞬态吸收光谱中的浓度演化过程。
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Tuple, Optional, Callable
from abc import ABC, abstractmethod


class KineticModelBase(ABC):
    """动力学模型基类"""
    
    def __init__(self, n_components: int):
        """
        初始化动力学模型
        
        参数:
        - n_components: 组分数量
        """
        self.n_components = n_components
        
    @abstractmethod
    def differential_equations(self, t: float, y: np.ndarray, 
                               rate_constants: List[float]) -> np.ndarray:
        """
        定义微分方程组
        
        参数:
        - t: 时间
        - y: 当前浓度向量
        - rate_constants: 速率常数列表
        
        返回:
        - dydt: 浓度变化率
        """
        pass
    
    @abstractmethod
    def get_initial_concentrations(self) -> np.ndarray:
        """
        获取初始浓度
        
        返回:
        - y0: 初始浓度向量
        """
        pass
    
    def solve(self, time_points: np.ndarray, 
              rate_constants: List[float]) -> np.ndarray:
        """
        求解微分方程组
        
        参数:
        - time_points: 时间点数组
        - rate_constants: 速率常数列表
        
        返回:
        - C: 浓度矩阵 (时间点 × 组分)
        """
        y0 = self.get_initial_concentrations()
        
        solution = solve_ivp(
            lambda t, y: self.differential_equations(t, y, rate_constants),
            t_span=(time_points[0], time_points[-1]),
            y0=y0,
            t_eval=time_points,
            method='RK45',
            dense_output=True
        )
        
        if not solution.success:
            raise RuntimeError(f"ODE求解失败: {solution.message}")
        
        # 转置以得到 (时间点 × 组分) 的形状
        return solution.y.T


class SequentialModel(KineticModelBase):
    """
    顺序反应模型: A → B → C → ...
    
    例如: A --k1--> B --k2--> C --k3--> D
    
    该模型描述了一系列的一级反应，每个中间体依次转化为下一个产物。
    """
    
    def __init__(self, n_components: int):
        """
        初始化顺序反应模型
        
        参数:
        - n_components: 组分数量
        """
        super().__init__(n_components)
        self.n_rate_constants = n_components - 1  # 需要 n-1 个速率常数
    
    def differential_equations(self, t: float, y: np.ndarray, 
                               rate_constants: List[float]) -> np.ndarray:
        """
        顺序反应的微分方程组
        
        对于 A → B → C:
        d[A]/dt = -k1 * [A]
        d[B]/dt = k1 * [A] - k2 * [B]
        d[C]/dt = k2 * [B]
        """
        if len(rate_constants) != self.n_rate_constants:
            raise ValueError(f"需要 {self.n_rate_constants} 个速率常数，"
                           f"但提供了 {len(rate_constants)} 个")
        
        dydt = np.zeros(self.n_components)
        
        # 第一个组分 A: d[A]/dt = -k1 * [A]
        dydt[0] = -rate_constants[0] * y[0]
        
        # 中间组分: d[B]/dt = k1 * [A] - k2 * [B]
        for i in range(1, self.n_components - 1):
            dydt[i] = (rate_constants[i-1] * y[i-1] - 
                      rate_constants[i] * y[i])
        
        # 最后一个组分: d[C]/dt = k2 * [B]
        if self.n_components > 1:
            dydt[-1] = rate_constants[-1] * y[-2]
        
        return dydt
    
    def get_initial_concentrations(self) -> np.ndarray:
        """
        初始浓度: 第一个组分浓度为1，其余为0
        """
        y0 = np.zeros(self.n_components)
        y0[0] = 1.0
        return y0


class ParallelModel(KineticModelBase):
    """
    平行反应模型: A → B, A → C, A → D, ...
    
    例如: A --k1--> B
              --k2--> C
              --k3--> D
    
    该模型描述了一个激发态通过多个平行通道衰减到不同的产物。
    """
    
    def __init__(self, n_components: int):
        """
        初始化平行反应模型
        
        参数:
        - n_components: 组分数量 (包括初始激发态)
        """
        super().__init__(n_components)
        self.n_rate_constants = n_components - 1  # 需要 n-1 个速率常数
    
    def differential_equations(self, t: float, y: np.ndarray, 
                               rate_constants: List[float]) -> np.ndarray:
        """
        平行反应的微分方程组
        
        对于 A → B, A → C:
        d[A]/dt = -(k1 + k2) * [A]
        d[B]/dt = k1 * [A]
        d[C]/dt = k2 * [A]
        """
        if len(rate_constants) != self.n_rate_constants:
            raise ValueError(f"需要 {self.n_rate_constants} 个速率常数，"
                           f"但提供了 {len(rate_constants)} 个")
        
        dydt = np.zeros(self.n_components)
        
        # 初始组分 A: d[A]/dt = -(k1 + k2 + ...) * [A]
        total_rate = sum(rate_constants)
        dydt[0] = -total_rate * y[0]
        
        # 产物组分: d[B]/dt = k1 * [A], d[C]/dt = k2 * [A]
        for i in range(1, self.n_components):
            dydt[i] = rate_constants[i-1] * y[0]
        
        return dydt
    
    def get_initial_concentrations(self) -> np.ndarray:
        """
        初始浓度: 第一个组分(激发态)浓度为1，其余为0
        """
        y0 = np.zeros(self.n_components)
        y0[0] = 1.0
        return y0


class MixedModel(KineticModelBase):
    """
    混合模型: 可自定义的复杂动力学模型
    
    例如: A → B → C 和 A → D (顺序 + 平行)
    
    该模型允许用户自定义任意复杂的反应网络。
    """
    
    def __init__(self, n_components: int, 
                 reaction_network: List[Tuple[int, int, float]]):
        """
        初始化混合模型
        
        参数:
        - n_components: 组分数量
        - reaction_network: 反应网络列表，每个元素为 (from_idx, to_idx, k_idx)
          其中 from_idx 和 to_idx 是组分索引，k_idx 是对应的速率常数索引
        """
        super().__init__(n_components)
        self.reaction_network = reaction_network
        self.n_rate_constants = max([k_idx for _, _, k_idx in reaction_network]) + 1
    
    def differential_equations(self, t: float, y: np.ndarray, 
                               rate_constants: List[float]) -> np.ndarray:
        """
        根据反应网络构建微分方程组
        """
        if len(rate_constants) < self.n_rate_constants:
            raise ValueError(f"需要至少 {self.n_rate_constants} 个速率常数，"
                           f"但提供了 {len(rate_constants)} 个")
        
        dydt = np.zeros(self.n_components)
        
        # 根据反应网络计算每个组分的变化率
        for from_idx, to_idx, k_idx in self.reaction_network:
            rate = rate_constants[k_idx] * y[from_idx]
            dydt[from_idx] -= rate
            dydt[to_idx] += rate
        
        return dydt
    
    def get_initial_concentrations(self) -> np.ndarray:
        """
        初始浓度: 第一个组分浓度为1，其余为0
        """
        y0 = np.zeros(self.n_components)
        y0[0] = 1.0
        return y0


def create_exponential_decay_model(n_components: int) -> Callable:
    """
    创建指数衰减模型 (用于GLA)
    
    该模型将浓度轮廓表示为多个指数衰减的线性组合:
    C(t) = Σ A_i * exp(-t / τ_i)
    
    参数:
    - n_components: 组分数量
    
    返回:
    - model_func: 模型函数，接受参数 (t, amplitudes, lifetimes)
    """
    def exponential_decay(t: np.ndarray, 
                         amplitudes: np.ndarray, 
                         lifetimes: np.ndarray) -> np.ndarray:
        """
        计算多指数衰减
        
        参数:
        - t: 时间数组 (n_times,)
        - amplitudes: 振幅数组 (n_components,)
        - lifetimes: 寿命数组 (n_components,)
        
        返回:
        - C: 浓度矩阵 (n_times, n_components)
        """
        if len(amplitudes) != n_components or len(lifetimes) != n_components:
            raise ValueError(f"振幅和寿命数组的长度必须等于 {n_components}")
        
        C = np.zeros((len(t), n_components))
        for i in range(n_components):
            C[:, i] = amplitudes[i] * np.exp(-t / lifetimes[i])
        
        return C
    
    return exponential_decay
