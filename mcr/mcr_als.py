# mcr/mcr_als.py
import numpy as np
from . import constraints
from .constraint_config import ConstraintConfig
from typing import Optional, Union, Dict, Any

import numpy.typing as npt


class MCRALS:
    """
    Multivariate Curve Resolution - Alternating Least Squares (MCR-ALS)
    
    This class implements the MCR-ALS algorithm to resolve mixed signals
    into pure components' contributions (C) and spectra (S).
    """

    def __init__(self, n_components: int, max_iter: int = 100, tol: float = 1e-6,
                 constraint_config: Optional[Union[str, ConstraintConfig]] = None,
                 penalty: float = 0.0, init_method: str = "svd",
                 random_state: Optional[int] = None):
        """
        Initializes the MCRALS solver.

        Parameters:
        - n_components (int): The number of pure components to resolve.
        - max_iter (int): The maximum number of iterations for the ALS optimization.
        - tol (float): The tolerance for the change in lack of fit (LOF) to determine convergence.
        - constraint_config (str or ConstraintConfig, optional): 约束配置文件路径或约束配置对象
        """
        if n_components <= 0:
            raise ValueError("Number of components must be a positive integer.")
        
        # 加载约束配置
        if isinstance(constraint_config, str):
            self.constraint_config = ConstraintConfig(constraint_config)
        elif isinstance(constraint_config, ConstraintConfig):
            self.constraint_config = constraint_config
        else:
            self.constraint_config = ConstraintConfig()  # 使用默认配置
        
        # 验证组分数量约束
        if not self.constraint_config.validate_component_count(n_components):
            component_constraint = self.constraint_config.get_constraint("component_count_range")
            if component_constraint:
                params = component_constraint.get("parameters", {})
                default_comp = params.get("default_components", 3)
                max_comp = params.get("max_components", 4)
                min_comp = params.get("min_components", 1)
                suggested_comp = min(max(n_components, min_comp), max_comp)
                raise ValueError(f"组分数量 {n_components} 不在允许范围 [{min_comp}, {max_comp}] 内。"
                               f"建议使用 {suggested_comp} 个组分。")
        
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.penalty = max(0.0, float(penalty))
        self.penalty_scale = 0.1
        self.init_method = init_method.lower()
        if self.init_method not in {"svd", "random"}:
            raise ValueError("init_method must be 'svd' or 'random'")
        self.random_state = random_state
        
        # Results will be stored here after fitting
        self.C_opt_ = None  # Optimal concentration matrix
        self.S_opt_ = None  # Optimal spectra matrix
        self.residuals_ = None
        self.lof_ = [] # Lack of fit history

    def _initial_guess_svd(self, D: np.ndarray) -> np.ndarray:
        """
        Generates an initial guess for the S matrix using Singular Value Decomposition (SVD).
        The most common method is to use the right singular vectors (V.T).
        """
        # We only need the first n_components
        _, _, Vh = np.linalg.svd(D, full_matrices=False)
        # Vh is (n, n), we take the first k rows and transpose it to get (n, k)
        S_initial = Vh[:self.n_components, :].T
        return S_initial

    def _initial_guess_random(self, D: np.ndarray) -> npt.NDArray[np.float64]:
        """Generates a randomised initial guess for the S matrix based on SVD with perturbations."""
        base = self._initial_guess_svd(D)
        rng = np.random.default_rng(self.random_state)
        noise = rng.standard_normal(size=base.shape)
        scale = np.linalg.norm(base) / np.sqrt(base.size)
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        perturbed = base + 0.05 * scale * noise
        return perturbed

    def _calculate_lof(self, D: np.ndarray, D_reconstructed: np.ndarray) -> float:
        """Calculates the Lack of Fit in percent."""
        residual_sum_of_squares = np.sum( (D - D_reconstructed)**2 )
        total_sum_of_squares = np.sum(D**2)
        return 100 * np.sqrt(residual_sum_of_squares / total_sum_of_squares)
    
    def _apply_constraints(self, matrix: np.ndarray, matrix_type: str) -> np.ndarray:
        """
        应用约束到指定矩阵
        
        Parameters:
        - matrix (np.ndarray): 输入矩阵 (C 或 S)
        - matrix_type (str): 矩阵类型 ("C" 或 "S")
        
        Returns:
        - np.ndarray: 应用约束后的矩阵
        """
        result = matrix.copy()
        
        # 获取适用于此矩阵类型的约束
        applicable_constraints = self.constraint_config.get_constraints_for_matrix(matrix_type)
        
        for constraint_name in applicable_constraints:
            constraint_config = self.constraint_config.get_constraint(constraint_name)
            if constraint_config and constraint_config.get('enabled', False):
                try:
                    result = constraints.apply_constraint_from_config(
                        result, constraint_name, constraint_config
                    )
                except Exception as e:
                    print(f"警告: 应用约束 '{constraint_name}' 时出错: {e}")
        
        return result

    def fit(self, D: np.ndarray, S_initial: np.ndarray = None):
        """
        Executes the MCR-ALS algorithm on the data matrix D.

        Parameters:
        - D (np.ndarray): The experimental data matrix (m_times x n_wavelengths).
        - S_initial (np.ndarray, optional): An initial guess for the S matrix.
          If None, SVD or random initialisation will be used based on init_method.
        """
        _, n = D.shape

        if S_initial is None:
            if self.init_method == "svd":
                S = self._initial_guess_svd(D)
            else:
                S = self._initial_guess_random(D)
        else:
            if S_initial.shape != (n, self.n_components):
                raise ValueError(f"Initial S must have shape ({n}, {self.n_components})")
            S = S_initial.copy()

        # Apply constraints to initial guess
        S = self._apply_constraints(S, "S")

        self.lof_ = []
        identity = np.eye(self.n_components, dtype=D.dtype)
        converged = False

        for iteration in range(self.max_iter):
            # Step 1: solve for C given S
            if self.penalty > 0:
                sts_base = S.T @ S
                scale = np.trace(sts_base) / self.n_components
                if not np.isfinite(scale) or scale == 0:
                    scale = 1.0
                reg_strength = self.penalty * self.penalty_scale * scale
                sts = sts_base + reg_strength * identity
                try:
                    C = np.linalg.solve(sts, S.T @ D.T).T
                except np.linalg.LinAlgError:
                    C = D @ np.linalg.pinv(S).T
            else:
                C = D @ np.linalg.pinv(S).T

            C = self._apply_constraints(C, "C")

            # Step 2: solve for S given C
            if self.penalty > 0:
                ctc_base = C.T @ C
                scale_c = np.trace(ctc_base) / self.n_components
                if not np.isfinite(scale_c) or scale_c == 0:
                    scale_c = 1.0
                reg_strength_c = self.penalty * self.penalty_scale * scale_c
                ctc = ctc_base + reg_strength_c * identity
                try:
                    S = np.linalg.solve(ctc, C.T @ D).T
                except np.linalg.LinAlgError:
                    S = D.T @ np.linalg.pinv(C).T
            else:
                S = D.T @ np.linalg.pinv(C).T

            S = self._apply_constraints(S, "S")

            # Normalize component scales to avoid numerical blow-up
            component_norms = np.linalg.norm(S, axis=0)
            component_norms[component_norms == 0] = 1.0
            S = S / component_norms
            C = C * component_norms

            # Convergence check
            D_reconstructed = C @ S.T
            lof = self._calculate_lof(D, D_reconstructed)
            self.lof_.append(lof)

            if len(self.lof_) > 1 and abs(self.lof_[-2] - self.lof_[-1]) < self.tol:
                print(f"Converged at iteration {iteration + 1} with LOF = {lof:.4f}%")
                converged = True
                break

        if not self.lof_:
            # Ensure at least one LOF value is recorded
            D_reconstructed = C @ S.T
            self.lof_.append(self._calculate_lof(D, D_reconstructed))

        if not converged:
            print(f"Maximum iterations ({self.max_iter}) reached. LOF = {self.lof_[-1]:.4f}%")

        # Store results
        self.C_opt_ = C
        self.S_opt_ = S
        self.residuals_ = D - (C @ S.T)

        return self
    
    def get_constraint_info(self) -> Dict[str, Any]:
        """获取当前约束配置信息"""
        active_constraints = self.constraint_config.get_active_constraints()
        constraint_info = {}
        
        for name in active_constraints:
            constraint_info[name] = self.constraint_config.get_constraint(name)
        
        return constraint_info
    
    def set_constraint_parameter(self, constraint_name: str, parameter_name: str, value: Any):
        """设置约束参数"""
        if constraint_name in self.constraint_config.constraints:
            if 'parameters' not in self.constraint_config.constraints[constraint_name]:
                self.constraint_config.constraints[constraint_name]['parameters'] = {}
            self.constraint_config.constraints[constraint_name]['parameters'][parameter_name] = value
        else:
            raise ValueError(f"约束 '{constraint_name}' 不存在")