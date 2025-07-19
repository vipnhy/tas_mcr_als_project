# main.py
import numpy as np
import matplotlib.pyplot as plt
from mcr.mcr_als import MCRALS

# --- 1. Generate Synthetic Data (for demonstration) ---
# In a real scenario, this would be in `data/synthetic_data.py`
def generate_synthetic_tas_data(n_times=100, n_wls=200, n_components=2):
    """Generates simple synthetic data for testing."""
    # Time axis
    t = np.linspace(0, 10, n_times)
    
    # Concentration profiles (C) - e.g., A -> B -> C kinetics
    C = np.zeros((n_times, n_components))
    C[:, 0] = np.exp(-0.5 * t)  # Species 1 decays
    C[:, 1] = (np.exp(-0.1 * t) - np.exp(-0.5 * t)) # Species 2 grows and decays
    
    # Normalize concentrations for clarity
    for i in range(n_components):
        C[:, i] /= np.max(C[:, i])

    # Spectra (S.T) - e.g., Gaussian peaks
    wls = np.linspace(400, 800, n_wls)
    S = np.zeros((n_wls, n_components))
    S[:, 0] = np.exp(-((wls - 500)**2) / (2 * 30**2))  # Peak at 500 nm
    S[:, 1] = np.exp(-((wls - 650)**2) / (2 * 40**2))  # Peak at 650 nm
    
    # Normalize spectra
    for i in range(n_components):
        S[:, i] /= np.max(S[:, i])

    # Create the data matrix D
    D_clean = C @ S.T
    
    # Add some noise
    noise_level = 0.02
    noise = np.random.randn(*D_clean.shape) * np.max(D_clean) * noise_level
    D_noisy = D_clean + noise
    
    return D_noisy, C, S

# --- 2. Run MCR-ALS Analysis ---
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