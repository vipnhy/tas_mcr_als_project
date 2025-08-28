# main.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
from mcr.mcr_als import MCRALS
from data.data import read_file

# 设置中文字体支持
def setup_chinese_fonts():
    """设置matplotlib中文字体"""
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
    for font in chinese_fonts:
        try:
            rcParams['font.sans-serif'] = [font]
            rcParams['axes.unicode_minus'] = False
            break
        except:
            continue
    else:
        print("警告: 未找到合适的中文字体")

# 初始化字体
setup_chinese_fonts()

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

# --- 2. Load Real TAS Data ---
def load_tas_data(file_path, file_type="handle", wavelength_range=(400, 800), delay_range=(0, 10)):
    """Load TAS data using the read_file function from data.py"""
    df = read_file(file_path, file_type=file_type, inf_handle=True, 
                   wavelength_range=wavelength_range, delay_range=delay_range)
    
    if df is None:
        raise ValueError("Failed to load data. Please check the file path and format.")
    
    # Convert DataFrame to matrix (D) where rows are time points and columns are wavelengths
    D = df.values
    time_axis = df.index.values  # Time delays
    wavelength_axis = df.columns.values  # Wavelengths
    
    print(f"Data shape: {D.shape}")
    print(f"Time range: {time_axis.min():.2f} to {time_axis.max():.2f}")
    print(f"Wavelength range: {wavelength_axis.min():.1f} to {wavelength_axis.max():.1f} nm")
    
    return D, time_axis, wavelength_axis

# --- 3. Run MCR-ALS Analysis ---
if __name__ == '__main__':
    # Load real TAS data
    file_path = "data/TAS/TA_Average-crop-TCA-tzs-tzs-tzs-chirp.csv"
    
    try:
        # Load data with specified wavelength and time ranges
        D, time_axis, wavelength_axis = load_tas_data(
            file_path, 
            file_type="handle",
            wavelength_range=(420, 750),  # Adjust based on your data
            delay_range=(0.1, 50)         # Adjust based on your data
        )
        
        # Determine number of components (you may want to adjust this)
        n_comps = 3  # Start with 3 components for TAS data
        
        # Initialize and run MCR-ALS
        print(f"\nRunning MCR-ALS with {n_comps} components...")
        mcr_solver = MCRALS(n_components=n_comps, max_iter=200, tol=1e-7)
        mcr_solver.fit(D)
        
        # Retrieve results
        C_resolved = mcr_solver.C_opt_
        S_resolved = mcr_solver.S_opt_
        
        print(f"MCR-ALS completed after {len(mcr_solver.lof_)} iterations")
        print(f"Final LOF: {mcr_solver.lof_[-1]:.4f}%")
        
        # --- 4. Visualize Results ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Concentration Profiles
        axes[0, 0].set_title("Concentration Profiles (C)")
        for i in range(n_comps):
            axes[0, 0].plot(time_axis, C_resolved[:, i], 
                           label=f'Component {i+1}', linewidth=2)
        axes[0, 0].set_xlabel("Time Delay (ps)")
        axes[0, 0].set_ylabel("Concentration (a.u.)")
        axes[0, 0].set_xscale('log')  # Log scale often better for TAS
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Pure Spectra
        axes[0, 1].set_title("Pure Spectra (S.T)")
        for i in range(n_comps):
            axes[0, 1].plot(wavelength_axis, S_resolved[:, i], 
                           label=f'Component {i+1}', linewidth=2)
        axes[0, 1].set_xlabel("Wavelength (nm)")
        axes[0, 1].set_ylabel("ΔA (a.u.)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # LOF convergence
        axes[1, 0].set_title("Lack of Fit (LOF) Convergence")
        axes[1, 0].plot(mcr_solver.lof_, 'b-', linewidth=2)
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("LOF (%)")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals
        im = axes[1, 1].imshow(mcr_solver.residuals_, aspect='auto', cmap='coolwarm', 
                               vmin=-np.max(np.abs(mcr_solver.residuals_)), 
                               vmax=np.max(np.abs(mcr_solver.residuals_)),
                               origin='lower')
        axes[1, 1].set_title("Residuals (D - C*S.T)")
        axes[1, 1].set_xlabel("Wavelength Index")
        axes[1, 1].set_ylabel("Time Index")
        fig.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()
        plt.show()
        
        # Additional analysis: Show original data
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original TAS data heatmap
        im1 = ax1.imshow(D, aspect='auto', cmap='coolwarm', 
                        extent=[wavelength_axis.min(), wavelength_axis.max(),
                               time_axis.min(), time_axis.max()],
                        origin='lower')
        ax1.set_title("Original TAS Data")
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Time Delay (ps)")
        fig2.colorbar(im1, ax=ax1, label='ΔA')
        
        # Reconstructed data
        D_reconstructed = C_resolved @ S_resolved.T
        im2 = ax2.imshow(D_reconstructed, aspect='auto', cmap='coolwarm',
                        extent=[wavelength_axis.min(), wavelength_axis.max(),
                               time_axis.min(), time_axis.max()],
                        vmin=im1.get_clim()[0], vmax=im1.get_clim()[1],
                        origin='lower')
        ax2.set_title("Reconstructed Data (C*S.T)")
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Time Delay (ps)")
        fig2.colorbar(im2, ax=ax2, label='ΔA')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        print("Falling back to synthetic data for demonstration...")
        
        # Fallback to synthetic data if real data fails
        D, C_true, S_true = generate_synthetic_tas_data(n_components=2)
        time_axis = np.linspace(0, 10, D.shape[0])
        wavelength_axis = np.linspace(400, 800, D.shape[1])
        
        n_comps = 2
        mcr_solver = MCRALS(n_components=n_comps, max_iter=200, tol=1e-7)
        mcr_solver.fit(D)
        
        C_resolved = mcr_solver.C_opt_
        S_resolved = mcr_solver.S_opt_