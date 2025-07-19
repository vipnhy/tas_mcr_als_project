# mcr/constraints.py
import numpy as np

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

# 未来可以添加更多约束函数
# def norm_constraint(matrix: np.ndarray, axis=1) -> np.ndarray:
#     ...