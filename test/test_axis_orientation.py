import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from data.data import read_file

def demonstrate_axis_orientation():
    """演示y轴方向的修正"""
    
    # 读取数据
    file_path = "data/TAS/TA_Average-crop-TCA-tzs-tzs-tzs-chirp.csv"
    df = read_file(file_path, file_type="handle", inf_handle=True, 
                  wavelength_range=(420, 750), delay_range=(0.1, 50))
    
    D = df.values
    time_axis = df.index.values
    wavelength_axis = df.columns.values
    
    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 错误的方向（原来的方式）
    im1 = ax1.imshow(D, aspect='auto', cmap='coolwarm', 
                    extent=[wavelength_axis.min(), wavelength_axis.max(),
                           time_axis.max(), time_axis.min()])
    ax1.set_title("错误的y轴方向\n(最大时间在下，0时间在上)")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Time Delay (ps)")
    fig.colorbar(im1, ax=ax1, label='ΔA')
    
    # 正确的方向（修正后的方式）
    im2 = ax2.imshow(D, aspect='auto', cmap='coolwarm',
                    extent=[wavelength_axis.min(), wavelength_axis.max(),
                           time_axis.min(), time_axis.max()],
                    origin='lower')
    ax2.set_title("正确的y轴方向\n(0时间在下，最大时间在上)")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Time Delay (ps)")
    fig.colorbar(im2, ax=ax2, label='ΔA')
    
    plt.tight_layout()
    plt.show()
    
    print("y轴方向对比:")
    print(f"时间范围: {time_axis.min():.2f} 到 {time_axis.max():.2f} ps")
    print("左图: 错误方向 - 时间轴倒置")
    print("右图: 正确方向 - 0在下方，最大值在上方")

if __name__ == '__main__':
    demonstrate_axis_orientation()
