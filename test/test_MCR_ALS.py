import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mcr.mcr_als import MCRALS
from data.data import read_file
from main import generate_synthetic_tas_data  # 导入合成数据生成函数



if __name__ == '__main__':
    # Generate data
    n_comps = 2
    D, C_true, S_true = generate_synthetic_tas_data(n_components=n_comps)
    
    # Initialize and run MCR-ALS
    mcr_solver = MCRALS(n_components=n_comps, max_iter=200, tol=1e-7)
    mcr_solver.fit(D)
    
    # Retrieve results
    C_resolved = mcr_solver.C_opt_
    S_resolved = mcr_solver.S_opt_
    
    # --- 3. Visualize Results ---
    # In a real scenario, this would be in `utils/plotting.py`
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # True vs Resolved Concentrations
    axes[0, 0].set_title("Concentration Profiles (C)")
    axes[0, 0].plot(C_true, linestyle='--', label=[f'True Comp {i+1}' for i in range(n_comps)])
    axes[0, 0].plot(C_resolved, linestyle='-', label=[f'Resolved Comp {i+1}' for i in range(n_comps)])
    axes[0, 0].set_xlabel("Time (a.u.)")
    axes[0, 0].set_ylabel("Concentration (a.u.)")
    axes[0, 0].legend()
    
    # True vs Resolved Spectra
    axes[0, 1].set_title("Pure Spectra (S.T)")
    axes[0, 1].plot(S_true, linestyle='--')
    axes[0, 1].plot(S_resolved, linestyle='-')
    axes[0, 1].set_xlabel("Wavelength (a.u.)")
    axes[0, 1].set_ylabel("Absorbance (a.u.)")

    # LOF convergence
    axes[1, 0].set_title("Lack of Fit (LOF) Convergence")
    axes[1, 0].plot(mcr_solver.lof_)
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("LOF (%)")
    
    # Residuals
    im = axes[1, 1].imshow(mcr_solver.residuals_, aspect='auto', cmap='coolwarm', 
                           vmin=-np.max(np.abs(mcr_solver.residuals_)), 
                           vmax=np.max(np.abs(mcr_solver.residuals_)))
    axes[1, 1].set_title("Residuals (D - C*S.T)")
    axes[1, 1].set_xlabel("Wavelength index")
    axes[1, 1].set_ylabel("Time index")
    fig.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()