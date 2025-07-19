# mcr/mcr_als.py
import numpy as np
from . import constraints

class MCRALS:
    """
    Multivariate Curve Resolution - Alternating Least Squares (MCR-ALS)
    
    This class implements the MCR-ALS algorithm to resolve mixed signals
    into pure components' contributions (C) and spectra (S).
    """

    def __init__(self, n_components: int, max_iter: int = 100, tol: float = 1e-6):
        """
        Initializes the MCRALS solver.

        Parameters:
        - n_components (int): The number of pure components to resolve.
        - max_iter (int): The maximum number of iterations for the ALS optimization.
        - tol (float): The tolerance for the change in lack of fit (LOF) to determine convergence.
        """
        if n_components <= 0:
            raise ValueError("Number of components must be a positive integer.")
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        
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

    def _calculate_lof(self, D: np.ndarray, D_reconstructed: np.ndarray) -> float:
        """Calculates the Lack of Fit in percent."""
        residual_sum_of_squares = np.sum( (D - D_reconstructed)**2 )
        total_sum_of_squares = np.sum(D**2)
        return 100 * np.sqrt(residual_sum_of_squares / total_sum_of_squares)

    def fit(self, D: np.ndarray, S_initial: np.ndarray = None):
        """
        Executes the MCR-ALS algorithm on the data matrix D.

        Parameters:
        - D (np.ndarray): The experimental data matrix (m_times x n_wavelengths).
        - S_initial (np.ndarray, optional): An initial guess for the S matrix.
          If None, SVD will be used to generate one.
        """
        m, n = D.shape
        
        # 1. Initial Guess for S
        if S_initial is None:
            S = self._initial_guess_svd(D)
        else:
            if S_initial.shape != (n, self.n_components):
                raise ValueError(f"Initial S must have shape ({n}, {self.n_components})")
            S = S_initial.copy()

        # Apply non-negativity to initial guess (often a good practice)
        S = constraints.non_negativity(S)

        # 2. Iterative Optimization Loop
        for i in range(self.max_iter):
            # --- Step A: Solve for C, given S ---
            # C = D * pinv(S) = D * S * (S.T * S)^-1
            # Using pseudoinverse (pinv) for numerical stability
            C = D @ np.linalg.pinv(S)
            
            # Apply constraints on C
            C = constraints.non_negativity(C)

            # --- Step B: Solve for S, given C ---
            # S = pinv(C) * D = (C.T * C)^-1 * C.T * D
            # Transposing the equation: S.T = pinv(C) * D -> S = D.T * pinv(C.T)
            S = D.T @ np.linalg.pinv(C)

            # Apply constraints on S
            S = constraints.non_negativity(S)

            # --- Convergence Check ---
            D_reconstructed = C @ S.T
            lof = self._calculate_lof(D, D_reconstructed)
            
            if i > 0 and abs(self.lof_[-1] - lof) < self.tol:
                print(f"Converged at iteration {i+1} with LOF = {lof:.4f}%")
                break
            
            self.lof_.append(lof)
            
            if i == self.max_iter - 1:
                print(f"Maximum iterations ({self.max_iter}) reached. LOF = {lof:.4f}%")

        # 3. Store Results
        self.C_opt_ = C
        self.S_opt_ = S
        self.residuals_ = D - (C @ S.T)
        
        return self