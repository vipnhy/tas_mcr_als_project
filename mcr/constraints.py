# mcr/constraints.py
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from typing import Dict, Any


def non_negativity(matrix: np.ndarray) -> np.ndarray:
    """
    Applies the non-negativity constraint.
    Sets all negative values in the input matrix to zero.

    Parameters:
    - matrix (np.ndarray): The input matrix (C or S).

    Returns:
    - np.ndarray: The matrix with non-negativity applied.
    """
    # Create a copy to avoid modifying the original matrix
    result = matrix.copy()
    result[result < 0] = 0
    return result


def spectral_smoothness(matrix: np.ndarray, lambda_smooth: float = 1e-3, order: int = 2) -> np.ndarray:
    """
    Applies spectral smoothness constraint using second derivative penalty.
    
    Parameters:
    - matrix (np.ndarray): The spectral matrix S (n_wavelengths x n_components)
    - lambda_smooth (float): Smoothing penalty parameter
    - order (int): Derivative order (default: 2 for second derivative)
    
    Returns:
    - np.ndarray: The smoothed spectral matrix
    """
    result = matrix.copy()
    n_points, n_components = result.shape
    
    if n_points < order + 1:
        # 如果数据点太少，无法应用平滑约束
        return result
    
    # 构建差分矩阵（二阶导数）
    D = _build_difference_matrix(n_points, order)
    
    # 对每个组分独立应用平滑约束
    for i in range(n_components):
        spectrum = result[:, i]
        
        # 构建惩罚项矩阵： (I + λD'D)
        I = np.eye(n_points)
        penalty_matrix = I + lambda_smooth * (D.T @ D)
        
        # 解线性系统: (I + λD'D) * s_smooth = s_original
        try:
            result[:, i] = np.linalg.solve(penalty_matrix, spectrum)
        except np.linalg.LinAlgError:
            # 如果求解失败，使用原始光谱
            result[:, i] = spectrum
    
    return result


def _build_difference_matrix(n: int, order: int) -> np.ndarray:
    """
    构建差分矩阵用于计算导数
    
    Parameters:
    - n (int): 数据点数量
    - order (int): 导数阶数
    
    Returns:
    - np.ndarray: 差分矩阵
    """
    if order == 1:
        # 一阶差分
        D = np.zeros((n-1, n))
        for i in range(n-1):
            D[i, i] = -1
            D[i, i+1] = 1
    elif order == 2:
        # 二阶差分
        D = np.zeros((n-2, n))
        for i in range(n-2):
            D[i, i] = 1
            D[i, i+1] = -2
            D[i, i+2] = 1
    else:
        # 高阶差分（递归构建）
        D1 = _build_difference_matrix(n, 1)
        D = D1[:-1, :] - D1[1:, :]
        for _ in range(order - 2):
            D_prev = D
            m = D.shape[0]
            D = np.zeros((m-1, n))
            for i in range(m-1):
                D[i] = D_prev[i] - D_prev[i+1]
    
    return D


def apply_constraint_from_config(matrix: np.ndarray, constraint_name: str, 
                                constraint_config: Dict[str, Any]) -> np.ndarray:
    """
    根据配置应用约束
    
    Parameters:
    - matrix (np.ndarray): 输入矩阵
    - constraint_name (str): 约束名称
    - constraint_config (dict): 约束配置
    
    Returns:
    - np.ndarray: 应用约束后的矩阵
    """
    constraint_type = constraint_config.get('type', constraint_name)
    parameters = constraint_config.get('parameters', {})
    
    if constraint_type == 'non_negativity':
        return non_negativity(matrix)
    
    elif constraint_type == 'spectral_smoothness':
        lambda_smooth = parameters.get('lambda', 1e-3)
        order = parameters.get('order', 2)
        return spectral_smoothness(matrix, lambda_smooth, order)
    
    else:
        print(f"警告: 未知的约束类型 '{constraint_type}'，跳过约束应用")
        return matrix


def validate_component_count(n_components: int, constraint_config: Dict[str, Any]) -> bool:
    """
    验证组分数量约束
    
    Parameters:
    - n_components (int): 组分数量
    - constraint_config (dict): 约束配置
    
    Returns:
    - bool: 是否满足约束
    """
    parameters = constraint_config.get('parameters', {})
    min_comp = parameters.get('min_components', 1)
    max_comp = parameters.get('max_components', 4)
    
    if not (min_comp <= n_components <= max_comp):
        print(f"组分数量 {n_components} 不在允许范围 [{min_comp}, {max_comp}] 内")
        return False
    
    return True


# 未来可以添加更多约束函数
def closure_constraint(matrix: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    归一化约束：确保每行或每列的和为1
    
    Parameters:
    - matrix (np.ndarray): 输入矩阵
    - axis (int): 归一化轴 (0: 列, 1: 行)
    
    Returns:
    - np.ndarray: 归一化后的矩阵
    """
    result = matrix.copy()
    sums = np.sum(result, axis=axis, keepdims=True)
    # 避免除零
    sums[sums == 0] = 1
    return result / sums


def unimodality_constraint(matrix: np.ndarray) -> np.ndarray:
    """
    单峰性约束：确保每个光谱只有一个峰
    （简化实现：仅应用平滑约束）
    
    Parameters:
    - matrix (np.ndarray): 光谱矩阵
    
    Returns:
    - np.ndarray: 应用约束后的矩阵
    """
    # 简化实现：使用更强的平滑约束
    return spectral_smoothness(matrix, lambda_smooth=1e-2, order=2)