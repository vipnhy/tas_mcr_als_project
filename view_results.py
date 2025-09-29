#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的图片查看脚本 - 显示TAS预处理结果
"""

import matplotlib.pyplot as plt
from PIL import Image
import os

def show_images():
    """显示生成的预处理对比图片"""
    image_files = [
        'tas_preprocessing_heatmap_comparison.png',
        'tas_preprocessing_spectra_comparison.png', 
        'tas_preprocessing_kinetics_comparison.png',
        'tas_preprocessing_statistics_comparison.png'
    ]
    
    print("📊 TAS预处理结果可视化")
    print("=" * 50)
    
    for img_file in image_files:
        if os.path.exists(img_file):
            print(f"✅ 找到图片: {img_file}")
            
            # 使用PIL打开图片
            img = Image.open(img_file)
            
            # 显示图片信息
            print(f"   尺寸: {img.size}")
            print(f"   模式: {img.mode}")
            
            # 如果需要显示图片（在有GUI环境中）
            try:
                img.show()
            except:
                print(f"   💡 请手动查看文件: {img_file}")
        else:
            print(f"❌ 未找到图片: {img_file}")
    
    print("\n🎯 预处理效果总结:")
    print("1. 热图对比 - 显示原始数据vs预处理后的2D谱图")
    print("2. 光谱对比 - 显示不同延迟时间的光谱变化")  
    print("3. 动力学对比 - 显示不同波长的动力学曲线")
    print("4. 统计对比 - 显示数据质量改进统计")

if __name__ == '__main__':
    show_images()
