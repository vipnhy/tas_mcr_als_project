import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mcr.mcr_als import MCRALS
from data.data import read_file

def test_real_tas_data():
    """测试使用实际TAS数据进行MCR-ALS分析"""
    
    print("Testing MCR-ALS with real TAS data...")
    
    # 数据文件路径
    file_path = "data/TAS/TA_Average-crop-TCA-tzs-tzs-tzs-chirp.csv"
    
    try:
        # 读取数据
        print(f"Loading data from: {file_path}")
        df = read_file(file_path, file_type="handle", inf_handle=True, 
                      wavelength_range=(420, 750), delay_range=(0.1, 50))
        
        if df is None:
            raise ValueError("Failed to load data")
        
        # 转换为矩阵格式
        D = df.values
        time_axis = df.index.values
        wavelength_axis = df.columns.values
        
        print(f"Data loaded successfully!")
        print(f"Data shape: {D.shape}")
        print(f"Time range: {time_axis.min():.3f} to {time_axis.max():.1f} ps")
        print(f"Wavelength range: {wavelength_axis.min():.1f} to {wavelength_axis.max():.1f} nm")
        print(f"Data range: {np.min(D):.6f} to {np.max(D):.6f}")
        
        # 检查数据质量
        nan_count = np.isnan(D).sum()
        inf_count = np.isinf(D).sum()
        print(f"NaN values: {nan_count}, Inf values: {inf_count}")
        
        # MCR-ALS分析
        n_comps = 3
        print(f"\nRunning MCR-ALS with {n_comps} components...")
        
        mcr_solver = MCRALS(n_components=n_comps, max_iter=100, tol=1e-6)
        mcr_solver.fit(D)
        
        # 获取结果
        C_resolved = mcr_solver.C_opt_
        S_resolved = mcr_solver.S_opt_
        
        print(f"MCR-ALS completed after {len(mcr_solver.lof_)} iterations")
        print(f"Final LOF: {mcr_solver.lof_[-1]:.4f}%")
        
        # 可视化结果
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始数据
        im1 = axes[0, 0].imshow(D, aspect='auto', cmap='coolwarm',
                               extent=[wavelength_axis.min(), wavelength_axis.max(),
                                     time_axis.min(), time_axis.max()],
                               origin='lower')
        axes[0, 0].set_title("Original TAS Data")
        axes[0, 0].set_xlabel("Wavelength (nm)")
        axes[0, 0].set_ylabel("Time Delay (ps)")
        plt.colorbar(im1, ax=axes[0, 0], label='ΔA')
        
        # 重构数据
        D_reconstructed = C_resolved @ S_resolved.T
        im2 = axes[0, 1].imshow(D_reconstructed, aspect='auto', cmap='coolwarm',
                               extent=[wavelength_axis.min(), wavelength_axis.max(),
                                     time_axis.min(), time_axis.max()],
                               vmin=im1.get_clim()[0], vmax=im1.get_clim()[1],
                               origin='lower')
        axes[0, 1].set_title("Reconstructed Data")
        axes[0, 1].set_xlabel("Wavelength (nm)")
        axes[0, 1].set_ylabel("Time Delay (ps)")
        plt.colorbar(im2, ax=axes[0, 1], label='ΔA')
        
        # 残差
        im3 = axes[0, 2].imshow(mcr_solver.residuals_, aspect='auto', cmap='coolwarm',
                               vmin=-np.max(np.abs(mcr_solver.residuals_)), 
                               vmax=np.max(np.abs(mcr_solver.residuals_)),
                               origin='lower')
        axes[0, 2].set_title("Residuals")
        axes[0, 2].set_xlabel("Wavelength Index")
        axes[0, 2].set_ylabel("Time Index")
        plt.colorbar(im3, ax=axes[0, 2])
        
        # 浓度轮廓
        for i in range(n_comps):
            axes[1, 0].plot(time_axis, C_resolved[:, i], 
                           label=f'Component {i+1}', linewidth=2)
        axes[1, 0].set_title("Concentration Profiles")
        axes[1, 0].set_xlabel("Time Delay (ps)")
        axes[1, 0].set_ylabel("Concentration (a.u.)")
        axes[1, 0].set_xscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 纯光谱
        for i in range(n_comps):
            axes[1, 1].plot(wavelength_axis, S_resolved[:, i], 
                           label=f'Component {i+1}', linewidth=2)
        axes[1, 1].set_title("Pure Spectra")
        axes[1, 1].set_xlabel("Wavelength (nm)")
        axes[1, 1].set_ylabel("ΔA (a.u.)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # LOF收敛
        axes[1, 2].plot(mcr_solver.lof_, 'b-', linewidth=2)
        axes[1, 2].set_title("LOF Convergence")
        axes[1, 2].set_xlabel("Iteration")
        axes[1, 2].set_ylabel("LOF (%)")
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return False

def test_data_loading():
    """测试数据加载功能"""
    
    print("Testing data loading functionality...")
    
    file_path = "data/TAS/TA_Average-crop-TCA-tzs-tzs-tzs-chirp.csv"
    
    try:
        # 测试不同的参数设置
        print("Test 1: Basic loading...")
        df1 = read_file(file_path, file_type="handle")
        if df1 is not None:
            print(f"Full data shape: {df1.shape}")
        else:
            raise ValueError("Failed to load basic data")
        
        print("Test 2: With wavelength range...")
        df2 = read_file(file_path, file_type="handle", 
                       wavelength_range=(450, 700))
        print(f"Wavelength filtered shape: {df2.shape}")
        
        print("Test 3: With time range...")
        df3 = read_file(file_path, file_type="handle", 
                       delay_range=(1, 100))
        print(f"Time filtered shape: {df3.shape}")
        
        print("Test 4: With both ranges...")
        df4 = read_file(file_path, file_type="handle", 
                       wavelength_range=(450, 700), 
                       delay_range=(1, 100))
        print(f"Both filtered shape: {df4.shape}")
        
        # 显示数据信息
        print(f"\nData info:")
        print(f"Time axis: {df4.index.min():.3f} to {df4.index.max():.1f} ps")
        print(f"Wavelength axis: {df4.columns.min():.1f} to {df4.columns.max():.1f} nm")
        print(f"Data range: {df4.values.min():.6f} to {df4.values.max():.6f}")
        
        return True
        
    except Exception as e:
        print(f"Error during data loading test: {e}")
        return False

if __name__ == '__main__':
    print("=== Testing Real TAS Data Analysis ===\n")
    
    # 测试数据加载
    if test_data_loading():
        print("✓ Data loading test passed\n")
    else:
        print("✗ Data loading test failed\n")
    
    # 测试完整分析
    if test_real_tas_data():
        print("✓ Real data analysis test passed")
    else:
        print("✗ Real data analysis test failed")
