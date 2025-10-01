#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真实TAS数据预处理测试脚本

使用项目中的真实瞬态吸收光谱数据测试预处理模块功能
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import TwoSlopeNorm

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 导入预处理模块
from preprocessing import (
    create_standard_pipeline, 
    create_gentle_pipeline,
    create_aggressive_pipeline,
    create_chirp_corrected_pipeline,
    create_comprehensive_pipeline,
    preprocess_tas_data,
    TASPreprocessingPipeline,
    ChirpCorrector
)

# 设置matplotlib中文显示
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 10

def read_file_auto(file, inf_handle=False, wavelength_range=None, delay_range=None):
    """
    参考api.py的自动识别文件类型并读取文件
    file: 文件路径
    inf_handle: 是否处理无穷值
    wavelength_range: 波长范围
    delay_range: 延迟时间范围
    """
    import pandas as pd
    import numpy as np
    
    detected_type = "raw"  # 默认类型
    
    try:
        print("开始文件类型自动检测...")
        # 尝试多种编码方式读取文件
        encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        file_content = None
        used_encoding = None
        
        for encoding in encodings_to_try:
            try:
                print(f"尝试使用编码 {encoding} 读取文件: {file}")
                with open(file, 'r', encoding=encoding) as f:
                    first_line = f.readline().strip()
                    second_line = f.readline().strip()
                    third_line = f.readline().strip()
                    file_content = (first_line, second_line, third_line)
                    used_encoding = encoding
                    print(f"✅ 成功使用编码 {encoding} 读取文件")
                    break
            except UnicodeDecodeError as e:
                print(f"编码 {encoding} 读取失败: {str(e)}")
                continue
            except Exception as e:
                print(f"读取文件时发生异常: {str(e)}")
                continue
        
        if file_content is None:
            print("⚠️ 文件编码检测失败，使用默认 raw 类型")
            detected_type = "raw"
        else:
            first_line, second_line, third_line = file_content
            print(f"✅ 使用编码: {used_encoding}")
            tab_char = '\t'
            comma_char = ','
            print(f"文件内容分析 - 第1行长度: {len(first_line)}, 制表符数: {first_line.count(tab_char)}, 逗号数: {first_line.count(comma_char)}")
            print(f"文件内容分析 - 第2行长度: {len(second_line)}, 制表符数: {second_line.count(tab_char)}, 逗号数: {second_line.count(comma_char)}")
            
            # 更精确的分隔符检测逻辑
            tab_count = first_line.count(tab_char) + second_line.count(tab_char)
            comma_count = first_line.count(comma_char) + second_line.count(comma_char)
            
            # 如果制表符明显多于逗号，判定为handle类型
            if tab_count > 0 and tab_count >= comma_count:
                detected_type = "handle"
                print(f"🎯 自动检测：文件类型为 handle（制表符分隔，制表符:{tab_count} vs 逗号:{comma_count}）")
            else:
                detected_type = "raw"
                print(f"🎯 自动检测：文件类型为 raw（逗号分隔，逗号:{comma_count} vs 制表符:{tab_count}）")
        
    except Exception as e:
        print(f"⚠️ 文件类型自动检测失败，使用默认 raw 类型: {e}")
        detected_type = "raw"
    
    print(f"� 最终使用的文件类型: {detected_type}")
    
    # 根据检测到的类型处理文件
    df = None
    if detected_type == "raw":
        try:
            print("📖 按 raw 类型处理文件")
            # 尝试多种编码读取raw类型文件
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            for encoding in encodings:
                try:
                    print(f"尝试使用编码 {encoding} 读取 raw 类型文件")
                    df = pd.read_csv(file, index_col=0, header=0, sep=",", encoding=encoding)
                    print(f"✅ raw类型成功使用编码: {encoding}")
                    break
                except UnicodeDecodeError as e:
                    print(f"编码 {encoding} 读取失败: {str(e)}")
                    continue
                except Exception as e:
                    print(f"编码 {encoding} 处理异常: {str(e)}")
                    if encoding == encodings[-1]:  # 最后一个编码也失败了
                        raise e
                    continue
            
            if df is None:
                raise Exception("无法使用任何编码读取文件")
            
            # 检查是否需要去掉最后11行（raw类型的特征）
            original_shape = df.shape
            if df.shape[0] > 11:
                df = df.iloc[:-11, :]
                print(f"删除最后11行，数据形状从 {original_shape} 变为 {df.shape}")
            df = df.T
            print(f"转置后数据形状: {df.shape}")
            df.index = df.index.str.replace("0.000000.1", "0")
            df.index = pd.to_numeric(df.index)
            df.columns = pd.to_numeric(df.columns)
            print("✅ raw 类型文件处理成功")
        except Exception as e:
            # 如果raw类型处理失败，尝试handle类型
            print(f"⚠️ 按raw类型处理失败，尝试handle类型: {str(e)}")
            try:
                # 尝试多种编码读取handle类型文件
                encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
                df = None
                for encoding in encodings:
                    try:
                        print(f"尝试使用编码 {encoding} 读取 handle 类型文件（自动切换）")
                        df = pd.read_csv(file, index_col=0, header=0, sep="\t", encoding=encoding)
                        print(f"✅ handle类型使用编码: {encoding}")
                        break
                    except UnicodeDecodeError as e:
                        print(f"编码 {encoding} 读取失败: {str(e)}")
                        continue
                    except Exception as e_inner:
                        print(f"编码 {encoding} 处理异常: {str(e_inner)}")
                        if encoding == encodings[-1]:
                            raise e_inner
                        continue
                
                if df is None:
                    raise Exception("无法使用任何编码读取文件")
                
                # handle类型文件不需要转置，因为数据排列已经是正确的（行=时间，列=波长）
                # 处理索引：如果索引是字符串类型才进行替换操作
                if df.index.dtype == 'object':
                    df.index = df.index.str.replace("0.000000000E+0.1", "0")
                    df.index = df.index.str.replace("0.00000E+0.1", "0")
                # 确保索引和列都是数值类型
                try:
                    df.index = pd.to_numeric(df.index)
                except (ValueError, TypeError):
                    print("⚠️ 索引转为数值类型失败，保持原样")
                df.columns = pd.to_numeric(df.columns)
                detected_type = "handle"
                print("✅ 自动切换到handle类型处理成功")
            except Exception as e2:
                print(f"❌ 文件读取失败，尝试了raw和handle两种格式都无法正确解析：\nraw格式错误: {e}\nhandle格式错误: {e2}")
                raise Exception(f"文件读取失败，尝试了raw和handle两种格式都无法正确解析：\nraw格式错误: {e}\nhandle格式错误: {e2}")
                
    elif detected_type == "handle":
        try:
            print("📖 按 handle 类型处理文件")
            # 尝试多种编码读取handle类型文件
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            for encoding in encodings:
                try:
                    print(f"尝试使用编码 {encoding} 读取 handle 类型文件")
                    df = pd.read_csv(file, index_col=0, header=0, sep="\t", encoding=encoding)
                    print(f"✅ handle类型成功使用编码: {encoding}")
                    break
                except UnicodeDecodeError as e:
                    print(f"编码 {encoding} 读取失败: {str(e)}")
                    continue
                except Exception as e_inner:
                    print(f"编码 {encoding} 处理异常: {str(e_inner)}")
                    if encoding == encodings[-1]:
                        raise e_inner
                    continue
            
            if df is None:
                raise Exception("无法使用任何编码读取文件")
            
            # handle类型文件不需要转置，因为数据排列已经是正确的（行=时间，列=波长）
            # 处理索引：如果索引是字符串类型才进行替换操作
            if df.index.dtype == 'object':
                df.index = df.index.str.replace("0.000000000E+0.1", "0")  # 将数据0点改为0
                df.index = df.index.str.replace("0.00000E+0.1", "0")  # bug暂修，后续修复
            # 确保索引和列都是数值类型
            try:
                df.index = pd.to_numeric(df.index)
            except (ValueError, TypeError):
                print("⚠️ 索引转为数值类型失败，保持原样")
            df.columns = pd.to_numeric(df.columns)
            print("✅ handle 类型文件处理成功")
        except Exception as e:
            print(f"❌ handle类型文件处理失败: {str(e)}")
            raise Exception(f"handle类型文件读取失败: {e}")
    
    # 处理无穷值
    if inf_handle:
        print("🔧 处理无穷值和NaN值...")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill(axis=0)  # 使用现代的填充方法
        print("✅ 无穷值处理完成")
    
    # 设置默认的波长和时间范围
    if wavelength_range is None:
        wavelength_range = [None, None]
    if delay_range is None:
        delay_range = [None, None]
        
    # 选择波长范围和时间范围
    if wavelength_range[0] is not None or wavelength_range[1] is not None:
        print(f"🔍 筛选波长范围: {wavelength_range}")
        df = select_wavelength(df, wavelength_range)
        
    if delay_range[0] is not None or delay_range[1] is not None:
        print(f"🔍 筛选延迟时间范围: {delay_range}")
        df = select_delay(df, delay_range)
    
    # 标签小数点后保留两位
    df.index = df.index.map(lambda x: round(x, 2))
    df.columns = df.columns.map(lambda x: round(x, 2))
    
    print(f"✅ 数据处理完成，最终数据形状: {df.shape}")
    return df

def select_wavelength(df, wavelength_range):
    """选择波长范围"""
    if wavelength_range[0] is None:
        wavelength_range[0] = df.columns.min()
    if wavelength_range[1] is None:
        wavelength_range[1] = df.columns.max()
    df = df.loc[:, (df.columns >= wavelength_range[0]) &
                (df.columns <= wavelength_range[1])]
    return df

def select_delay(df, delay_range):
    """选择延迟时间范围"""
    if delay_range[0] is None:
        delay_range[0] = df.index.min()
    if delay_range[1] is None:
        delay_range[1] = df.index.max()
    df = df.loc[(df.index >= delay_range[0]) &
                (df.index <= delay_range[1]), :]
    return df

def load_real_tas_data():
    """加载真实的TAS数据（使用自动检测模式）"""
    # 数据文件路径
    data_file = Path("D:/TAS/tas_mcr_als_project/data/TAS/TA_Average.csv")
    
    if not data_file.exists():
        # 如果绝对路径不存在，尝试相对路径
        data_file = Path("data/TAS/TA_Average.csv")
        
    if not data_file.exists():
        print(f"❌ 找不到数据文件: {data_file}")
        return None
    
    print(f"📁 加载数据文件: {data_file}")
    
    try:
        # 使用自动检测的数据读取函数
        df = read_file_auto(
            str(data_file), 
            inf_handle=True,
            wavelength_range=(400, 800),  # 400-800nm
            delay_range=(0, 100)  # 0-100ps
        )
        
        return df
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

def analyze_data_quality(data):
    """分析数据质量"""
    print("\n" + "="*50)
    print("📊 数据质量分析")
    print("="*50)
    
    print(f"数据维度: {data.shape}")
    print(f"延迟时间范围: {data.index.min():.2f} - {data.index.max():.2f}")
    print(f"波长范围: {data.columns.min():.1f} - {data.columns.max():.1f} nm")
    print(f"数据范围: {data.values.min():.6f} - {data.values.max():.6f}")
    print(f"数据均值: {data.values.mean():.6f}")
    print(f"数据标准差: {data.values.std():.6f}")
    
    # 检查NaN和无穷值
    nan_count = np.isnan(data.values).sum()
    inf_count = np.isinf(data.values).sum()
    print(f"NaN值数量: {nan_count}")
    print(f"无穷值数量: {inf_count}")
    
    # 估算信噪比
    # 使用高频成分估算噪声
    if data.shape[1] > 4:
        noise_estimate = np.std(np.diff(data.values, n=2, axis=1)) / np.sqrt(6)
        signal_estimate = np.std(data.values)
        snr_estimate = signal_estimate / noise_estimate if noise_estimate > 0 else float('inf')
        print(f"估算信噪比: {snr_estimate:.2f}")
    
def test_preprocessing_pipelines(data):
    """测试不同的预处理管道"""
    print("\n" + "="*50)
    print("🔧 预处理管道测试")
    print("="*50)
    
    results = {}
    
    # 1. 标准管道
    print("\n1. 标准预处理管道")
    try:
        pipeline_std = create_standard_pipeline(verbose=True)
        processed_std = pipeline_std.fit_transform(data.copy())
        results['standard'] = {
            'data': processed_std,
            'pipeline': pipeline_std,
            'name': '标准管道'
        }
        print("✅ 标准管道处理完成")
    except Exception as e:
        print(f"❌ 标准管道失败: {e}")
        results['standard'] = None
    
    # 2. 温和管道
    print("\n2. 温和预处理管道")
    try:
        pipeline_gentle = create_gentle_pipeline(verbose=False)
        processed_gentle = pipeline_gentle.fit_transform(data.copy())
        results['gentle'] = {
            'data': processed_gentle,
            'pipeline': pipeline_gentle,
            'name': '温和管道'
        }
        print("✅ 温和管道处理完成")
    except Exception as e:
        print(f"❌ 温和管道失败: {e}")
        results['gentle'] = None
    
    # 3. 激进管道
    print("\n3. 激进预处理管道")
    try:
        pipeline_aggressive = create_aggressive_pipeline(verbose=False)
        processed_aggressive = pipeline_aggressive.fit_transform(data.copy())
        results['aggressive'] = {
            'data': processed_aggressive,
            'pipeline': pipeline_aggressive,
            'name': '激进管道'
        }
        print("✅ 激进管道处理完成")
    except Exception as e:
        print(f"❌ 激进管道失败: {e}")
        results['aggressive'] = None
    
    # 4. 自定义管道（针对TAS数据优化）
    print("\n4. TAS优化管道")
    try:
        tas_optimized_steps = [
            {'name': 'outlier_detection', 'processor': 'outlier', 
             'params': {'method': 'z_score', 'threshold': 2.5}, 'strategy': 'interpolate'},
            {'name': 'baseline_correction', 'processor': 'baseline', 
             'params': {'method': 'als', 'lam': 1e7, 'p': 0.001}},
            {'name': 'noise_filtering', 'processor': 'noise', 
             'params': {'method': 'gaussian', 'sigma': 0.8}},
            {'name': 'data_smoothing', 'processor': 'smooth', 
             'params': {'method': 'savgol', 'window_length': 7, 'polyorder': 3}}
        ]
        
        pipeline_tas = TASPreprocessingPipeline(steps=tas_optimized_steps, verbose=False)
        processed_tas = pipeline_tas.fit_transform(data.copy())
        results['tas_optimized'] = {
            'data': processed_tas,
            'pipeline': pipeline_tas,
            'name': 'TAS优化管道'
        }
        print("✅ TAS优化管道处理完成")
    except Exception as e:
        print(f"❌ TAS优化管道失败: {e}")
        results['tas_optimized'] = None
    
    return results

def create_comprehensive_visualization(original_data, processed_results):
    """创建全面的可视化对比"""
    print("\n" + "="*50)
    print("📈 创建可视化图表")
    print("="*50)
    
    # 准备数据
    original_wavelengths = np.asarray(original_data.columns.values, dtype=float)
    original_delays = np.asarray(original_data.index.values, dtype=float)

    def ensure_axis_length(candidates, expected_length):
        """从候选列表中选择与数据长度匹配的轴，如果没有匹配则回退第一个可用选项"""
        for axis in candidates:
            if axis is None:
                continue
            axis_array = np.asarray(axis, dtype=float)
            if axis_array.size == expected_length:
                return axis_array
        for axis in candidates:
            if axis is None:
                continue
            axis_array = np.asarray(axis, dtype=float)
            if axis_array.size:
                return axis_array
        return np.arange(expected_length, dtype=float)

    def extract_result_entry(key, result):
        """统一提取结果中的数据、名称和坐标轴信息"""
        if result is None:
            return None

        name = key
        pipeline = None
        data_obj = result

        if isinstance(result, dict):
            data_obj = result.get('data')
            name = result.get('name', key)
            pipeline = result.get('pipeline')

        if data_obj is None:
            return None

        data_values = data_obj.values if hasattr(data_obj, 'values') else np.asarray(data_obj)

        if data_values.ndim == 1:
            data_values = data_values.reshape(1, -1)

        wl_candidates = []
        delay_candidates = []

        if hasattr(data_obj, 'columns'):
            wl_candidates.append(data_obj.columns.values)
        if pipeline is not None and getattr(pipeline, 'wavelengths', None) is not None:
            wl_candidates.append(pipeline.wavelengths)
        wl_candidates.append(original_wavelengths)

        if hasattr(data_obj, 'index'):
            delay_candidates.append(data_obj.index.values)
        if pipeline is not None and getattr(pipeline, 'time_axis', None) is not None:
            delay_candidates.append(pipeline.time_axis)
        delay_candidates.append(original_delays)

        wavelengths = ensure_axis_length(wl_candidates, data_values.shape[1])
        delays = ensure_axis_length(delay_candidates, data_values.shape[0])

        return {
            'key': key,
            'name': name,
            'data': data_obj,
            'values': np.asarray(data_values, dtype=float),
            'wavelengths': np.asarray(wavelengths, dtype=float),
            'delays': np.asarray(delays, dtype=float),
            'pipeline': pipeline
        }

    def find_nearest_index(axis_values, target_value):
        if axis_values.size == 0:
            return 0
        idx = int(np.argmin(np.abs(axis_values - target_value)))
        return max(0, min(idx, axis_values.size - 1))

    processed_entries = []
    for key, res in processed_results.items():
        entry = extract_result_entry(key, res)
        if entry is not None:
            processed_entries.append(entry)
    
    # 选择几个代表性的延迟时间
    delay_indices = [
        len(original_delays) // 8,      # 早期时间
        len(original_delays) // 4,      # 早期-中期
        len(original_delays) // 2,      # 中期
        3 * len(original_delays) // 4,  # 中期-后期
        -2                              # 后期
    ]
    
    # 1. 2D热图对比
    print("生成2D热图对比...")
    n_plots = 1 + len(processed_entries)

    fig1, axes = plt.subplots(1, min(n_plots, 4), figsize=(20, 5))
    if isinstance(axes, np.ndarray):
        axes_list = list(np.atleast_1d(axes))
    else:
        axes_list = [axes]

    all_arrays = [original_data.values]
    for entry in processed_entries:
        all_arrays.append(entry['values'])

    abs_max = max(np.nanmax(np.abs(arr)) for arr in all_arrays if arr.size) if all_arrays else 0.0
    if not np.isfinite(abs_max) or abs_max == 0:
        abs_max = 1.0

    norm = TwoSlopeNorm(vcenter=0.0, vmin=-abs_max, vmax=abs_max)
    cmap = plt.get_cmap('rainbow')
    
    # 原始数据
    im0 = axes_list[0].imshow(original_data.values, aspect='auto', origin='lower',
                        extent=[original_wavelengths[0], original_wavelengths[-1], original_delays[0], original_delays[-1]],
                        cmap=cmap, norm=norm)
    axes_list[0].set_title('原始数据', fontsize=12, fontweight='bold')
    axes_list[0].set_xlabel('波长 (nm)')
    axes_list[0].set_ylabel('延迟时间 (ps)')
    plt.colorbar(im0, ax=axes_list[0], shrink=0.8)
    
    # 处理后数据
    plot_idx = 1
    for entry in processed_entries:
        if plot_idx >= len(axes_list):
            break

        im = axes_list[plot_idx].imshow(
            entry['values'],
            aspect='auto',
            origin='lower',
            extent=[
                entry['wavelengths'][0],
                entry['wavelengths'][-1],
                entry['delays'][0],
                entry['delays'][-1]
            ],
            cmap=cmap,
            norm=norm
        )
        axes_list[plot_idx].set_title(entry['name'], fontsize=12, fontweight='bold')
        axes_list[plot_idx].set_xlabel('波长 (nm)')
        if plot_idx == 0:
            axes_list[plot_idx].set_ylabel('延迟时间 (ps)')
        plt.colorbar(im, ax=axes_list[plot_idx], shrink=0.8)
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('tas_preprocessing_heatmap_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 保存热图对比: tas_preprocessing_heatmap_comparison.png")
    
    # 2. 不同时间点的光谱对比
    print("生成光谱对比图...")
    fig2, axes = plt.subplots(len(delay_indices), 1, figsize=(12, 3*len(delay_indices)))
    if len(delay_indices) == 1:
        axes = [axes]
    
    colors = ['black', 'red', 'blue', 'green', 'orange']
    
    for i, delay_idx in enumerate(delay_indices):
        delay_time = original_delays[delay_idx]
        
        # 原始数据
        axes[i].plot(original_wavelengths, original_data.iloc[delay_idx], 
                    color='black', linewidth=2, alpha=0.7, label='原始数据')
        
        # 处理后数据
        color_idx = 1
        for entry in processed_entries:
            if color_idx >= len(colors):
                break

            dataset_delays = entry['delays']
            target_idx = find_nearest_index(dataset_delays, delay_time)
            data_slice = entry['values'][target_idx]

            axes[i].plot(
                entry['wavelengths'],
                data_slice,
                color=colors[color_idx],
                linewidth=1.5,
                alpha=0.8,
                label=entry['name']
            )
            color_idx += 1
        
        axes[i].set_title(f'延迟时间: {delay_time:.2f} ps', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('波长 (nm)')
        axes[i].set_ylabel('ΔOD')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tas_preprocessing_spectra_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 保存光谱对比: tas_preprocessing_spectra_comparison.png")
    
    # 3. 动力学曲线对比
    print("生成动力学曲线对比...")
    # 选择几个代表性波长
    wavelength_indices = [
        len(original_wavelengths) // 4,
        len(original_wavelengths) // 2,
        3 * len(original_wavelengths) // 4
    ]
    wavelength_indices = [min(max(idx, 0), len(original_wavelengths) - 1) for idx in wavelength_indices]

    target_wavelengths = original_wavelengths[wavelength_indices]

    fig3, axes = plt.subplots(len(wavelength_indices), 1, figsize=(12, 3*len(wavelength_indices)))
    if len(wavelength_indices) == 1:
        axes = [axes]
    
    for i, wl_idx in enumerate(wavelength_indices):
        wavelength = original_wavelengths[wl_idx]
        
        # 原始数据
        axes[i].semilogx(original_delays, original_data.iloc[:, wl_idx], 
                        color='black', linewidth=2, alpha=0.7, label='原始数据')
        
        # 处理后数据
        color_idx = 1
        for entry in processed_entries:
            if color_idx >= len(colors):
                break

            dataset_wavelengths = entry['wavelengths']
            nearest_idx = find_nearest_index(dataset_wavelengths, target_wavelengths[i])
            data_slice = entry['values'][:, nearest_idx]

            axes[i].semilogx(
                entry['delays'],
                data_slice,
                color=colors[color_idx],
                linewidth=1.5,
                alpha=0.8,
                label=f"{entry['name']} ({dataset_wavelengths[nearest_idx]:.1f} nm)"
            )
            color_idx += 1
        
        axes[i].set_title(f'波长: {wavelength:.1f} nm', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('延迟时间 (ps)')
        axes[i].set_ylabel('ΔOD')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tas_preprocessing_kinetics_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 保存动力学对比: tas_preprocessing_kinetics_comparison.png")
    
    # 4. 统计信息对比
    print("生成统计信息对比...")
    fig4, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 数据范围对比
    names = ['原始数据']
    data_ranges = [(original_data.values.min(), original_data.values.max())]
    stds = [original_data.values.std()]
    snrs = []
    
    # 估算原始数据SNR
    try:
        noise_orig = np.std(np.diff(original_data.values, n=2, axis=1)) / np.sqrt(6)
        signal_orig = np.std(original_data.values)
        snr_orig = signal_orig / noise_orig if noise_orig > 0 else 0
        snrs.append(snr_orig)
    except:
        snrs.append(0)
    
    for key, result in processed_results.items():
        if result is not None:
            # 检查result是否为字典格式，如果不是则直接使用
            if isinstance(result, dict) and 'data' in result:
                data = result['data']
                name = result.get('name', key)
            else:
                data = result  # 直接使用数据
                name = key
                
            names.append(name)
            
            # 处理数据类型差异
            if hasattr(data, 'values'):
                data_values = data.values
            else:
                data_values = data  # 假设已经是numpy数组
            
            data_ranges.append((data_values.min(), data_values.max()))
            stds.append(data_values.std())
            
            # 估算SNR
            try:
                noise = np.std(np.diff(data_values, n=2, axis=1)) / np.sqrt(6)
                signal = np.std(data_values)
                snr = signal / noise if noise > 0 else 0
                snrs.append(snr)
            except:
                snrs.append(0)
    
    # 数据范围
    mins, maxs = zip(*data_ranges)
    x_pos = np.arange(len(names))
    
    axes[0, 0].bar(x_pos, mins, alpha=0.7, label='最小值')
    axes[0, 0].bar(x_pos, maxs, alpha=0.7, label='最大值')
    axes[0, 0].set_title('数据范围对比')
    axes[0, 0].set_ylabel('ΔOD')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 0].legend()
    
    # 标准差对比
    axes[0, 1].bar(x_pos, stds, alpha=0.7, color='orange')
    axes[0, 1].set_title('数据标准差对比')
    axes[0, 1].set_ylabel('标准差')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
    
    # SNR对比
    axes[1, 0].bar(x_pos, snrs, alpha=0.7, color='green')
    axes[1, 0].set_title('信噪比对比')
    axes[1, 0].set_ylabel('SNR')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
    
    # 处理时间对比
    processing_times = [0]  # 原始数据无处理时间
    for key, result in processed_results.items():
        if result is not None:
            # 检查result是否为字典格式且包含pipeline信息
            if isinstance(result, dict) and 'pipeline' in result:
                summary = result['pipeline'].get_processing_summary()
                processing_times.append(summary.get('total_processing_time', 0))
            else:
                processing_times.append(0)  # 无法获取处理时间信息
    
    axes[1, 1].bar(x_pos, processing_times, alpha=0.7, color='red')
    axes[1, 1].set_title('处理时间对比')
    axes[1, 1].set_ylabel('时间 (秒)')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('tas_preprocessing_statistics_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 保存统计对比: tas_preprocessing_statistics_comparison.png")

def print_processing_summaries(processed_results):
    """打印处理摘要"""
    print("\n" + "="*50)
    print("📋 预处理摘要报告")
    print("="*50)
    
    for key, result in processed_results.items():
        if result is not None:
            # 检查result是否为字典格式，如果不是则跳过摘要打印
            if isinstance(result, dict) and 'name' in result and 'pipeline' in result:
                print(f"\n🔧 {result['name']}:")
                
                summary = result['pipeline'].get_processing_summary()
                
                print(f"   处理步骤数: {summary.get('total_steps', 0)}")
                print(f"   总处理时间: {summary.get('total_processing_time', 0):.3f}s")
                
                if 'steps_summary' in summary:
                    for step in summary['steps_summary']:
                        print(f"   - {step['name']}: {step['time']:.3f}s")
                
                if 'data_statistics' in summary:
                    stats = summary['data_statistics']
                    print(f"   数据改进:")
                    print(f"     - 信噪比提升: {stats.get('snr_improvement', 1):.2f}x")
                    print(f"     - 噪声降低: {stats.get('noise_reduction', 0)*100:.1f}%")
                    print(f"     - 数据保真度: {stats.get('data_preservation', 0):.3f}")
            else:
                print(f"\n🔧 {key}: 处理完成 (数据形状: {result.shape if hasattr(result, 'shape') else 'N/A'})")

def test_chirp_correction(data, time_delays):
    """测试啁啾校正功能"""
    print("\n" + "="*50)
    print("⚡ 啁啾校正测试")
    print("="*50)
    
    # 确保数据格式正确
    if isinstance(data, pd.DataFrame):
        data_array = data.values
        wavelengths = data.index.values if hasattr(data.index, 'values') else np.arange(data.shape[0])
    else:
        data_array = data
        wavelengths = np.arange(data.shape[0])
    
    print(f"数据形状: {data_array.shape}")
    print(f"波长范围: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
    print(f"时间延迟范围: {time_delays.min():.2f} - {time_delays.max():.2f} ps")
    
    # 初始化啁啾校正器
    chirp_corrector = ChirpCorrector(method='cross_correlation')
    
    results = {}
    
    # 测试不同的啁啾校正方法
    methods_to_test = ['cross_correlation', 'solvent_response', 'polynomial']
    
    for method in methods_to_test:
        print(f"\n🔍 测试啁啾校正方法: {method}")
        try:
            # 重新初始化校正器
            if method == 'solvent_response':
                corrector = ChirpCorrector(method=method, solvent_wavelengths=[400, 450])
            elif method == 'polynomial':
                corrector = ChirpCorrector(method=method, polynomial_order=3)
            else:
                corrector = ChirpCorrector(method=method)
            
            # 执行啁啾校正
            corrected_data = corrector.correct_chirp(data_array, time_delays)
            
            if corrected_data is not None:
                results[method] = {
                    'original': data_array,
                    'corrected': corrected_data,
                    'correction_stats': corrector.get_correction_stats() if hasattr(corrector, 'get_correction_stats') else {}
                }
                print(f"  ✅ {method} 校正成功")
                print(f"     校正前数据范围: {data_array.min():.2e} - {data_array.max():.2e}")
                print(f"     校正后数据范围: {corrected_data.min():.2e} - {corrected_data.max():.2e}")
            else:
                print(f"  ❌ {method} 校正失败")
                
        except Exception as e:
            print(f"  ❌ {method} 校正出错: {str(e)}")
            continue
    
    # 可视化啁啾校正结果
    if results:
        print(f"\n📊 生成啁啾校正对比可视化...")
        visualize_chirp_correction_results(results, wavelengths, time_delays)
    
    return results

def visualize_chirp_correction_results(results, wavelengths, time_delays):
    """可视化啁啾校正结果"""
    n_methods = len(results)
    if n_methods == 0:
        return
    
    # 创建比较图
    fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))
    if n_methods == 1:
        axes = axes.reshape(2, 1)
    
    # 选择部分波长进行展示
    wavelength_indices = np.linspace(0, len(wavelengths)-1, 5, dtype=int)
    selected_wavelengths = wavelengths[wavelength_indices]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_wavelengths)))
    
    for i, (method, data_dict) in enumerate(results.items()):
        original = data_dict['original']
        corrected = data_dict['corrected']
        
        # 上方子图：校正前
        ax1 = axes[0, i]
        for j, wl_idx in enumerate(wavelength_indices):
            ax1.plot(time_delays, original[wl_idx, :], 
                    color=colors[j], alpha=0.7, linewidth=1,
                    label=f'{selected_wavelengths[j]:.0f} nm')
        
        ax1.set_title(f'{method} - 校正前', fontsize=12, fontweight='bold')
        ax1.set_xlabel('时间延迟 (ps)')
        ax1.set_ylabel('ΔOD')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        
        # 下方子图：校正后
        ax2 = axes[1, i]
        for j, wl_idx in enumerate(wavelength_indices):
            ax2.plot(time_delays, corrected[wl_idx, :], 
                    color=colors[j], alpha=0.7, linewidth=1,
                    label=f'{selected_wavelengths[j]:.0f} nm')
        
        ax2.set_title(f'{method} - 校正后', fontsize=12, fontweight='bold')
        ax2.set_xlabel('时间延迟 (ps)')
        ax2.set_ylabel('ΔOD')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        
        # 设置相同的y轴范围便于比较
        y_min = min(original.min(), corrected.min()) * 1.1
        y_max = max(original.max(), corrected.max()) * 1.1
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # 保存图像
    output_path = project_root / "tas_chirp_correction_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 啁啾校正对比图保存至: {output_path}")
    
    # 创建更详细的2D热图比较
    fig_2d, axes_2d = plt.subplots(1, n_methods*2, figsize=(4*n_methods*2, 6))
    if n_methods == 1:
        axes_2d = [axes_2d] if not isinstance(axes_2d, np.ndarray) else axes_2d
    
    heatmap_arrays = []
    for data_dict in results.values():
        if data_dict is None:
            continue
        heatmap_arrays.append(np.asarray(data_dict['original']))
        heatmap_arrays.append(np.asarray(data_dict['corrected']))

    abs_max = max(np.nanmax(np.abs(arr)) for arr in heatmap_arrays if arr.size) if heatmap_arrays else 1.0
    if abs_max == 0 or not np.isfinite(abs_max):
        abs_max = 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-abs_max, vmax=abs_max)
    cmap = plt.get_cmap('rainbow')

    plot_idx = 0
    for method, data_dict in results.items():
        original = data_dict['original']
        corrected = data_dict['corrected']
        
        # 选择适当的波长和时间范围进行显示
        wl_slice = slice(0, min(50, original.shape[0]))
        time_slice = slice(0, min(100, original.shape[1]))
        
        # 原始数据热图
        im1 = axes_2d[plot_idx].imshow(
            original[wl_slice, time_slice],
            aspect='auto',
            cmap=cmap,
            norm=norm,
            extent=[
                time_delays[time_slice].min(),
                time_delays[time_slice].max(),
                wavelengths[wl_slice].max(),
                wavelengths[wl_slice].min(),
            ],
        )
        axes_2d[plot_idx].set_title(f'{method} - 原始', fontweight='bold')
        axes_2d[plot_idx].set_xlabel('时间延迟 (ps)')
        axes_2d[plot_idx].set_ylabel('波长 (nm)')
        plt.colorbar(im1, ax=axes_2d[plot_idx], shrink=0.6)
        
        # 校正后数据热图
        im2 = axes_2d[plot_idx + 1].imshow(
            corrected[wl_slice, time_slice],
            aspect='auto',
            cmap=cmap,
            norm=norm,
            extent=[
                time_delays[time_slice].min(),
                time_delays[time_slice].max(),
                wavelengths[wl_slice].max(),
                wavelengths[wl_slice].min(),
            ],
        )
        axes_2d[plot_idx+1].set_title(f'{method} - 校正后', fontweight='bold')
        axes_2d[plot_idx+1].set_xlabel('时间延迟 (ps)')
        axes_2d[plot_idx+1].set_ylabel('波长 (nm)')
        plt.colorbar(im2, ax=axes_2d[plot_idx+1], shrink=0.6)
        
        plot_idx += 2
    
    plt.tight_layout()
    
    # 保存2D热图
    output_path_2d = project_root / "tas_chirp_correction_heatmap.png"
    plt.savefig(output_path_2d, dpi=300, bbox_inches='tight')
    print(f"✅ 啁啾校正2D热图保存至: {output_path_2d}")
    
    plt.close('all')

def main():
    """主函数"""
    print("🔬 TAS真实数据预处理测试")
    print("="*60)
    
    # 1. 加载真实数据
    original_data = load_real_tas_data()
    if original_data is None:
        print("❌ 无法加载数据，测试终止")
        return
    
    print(f"✅ 数据加载成功: {original_data.shape}")
    
    # 1.5. 生成时间延迟数组（假设数据的列是时间延迟）
    # 这里假设时间延迟从-1 ps 到 1000 ps
    n_time_points = original_data.shape[1] if isinstance(original_data, pd.DataFrame) else original_data.shape[1]
    time_delays = np.linspace(-1, 1000, n_time_points)  # ps
    print(f"⏰ 时间延迟范围: {time_delays.min():.2f} - {time_delays.max():.2f} ps")
    
    # 2. 数据质量分析
    analyze_data_quality(original_data)
    
    # 3. 测试啁啾校正功能
    try:
        chirp_results = test_chirp_correction(original_data, time_delays)
        print(f"✅ 啁啾校正测试完成，测试了 {len(chirp_results)} 种方法")
    except Exception as e:
        print(f"❌ 啁啾校正测试失败: {str(e)}")
        chirp_results = {}
    
    # 4. 测试预处理管道
    processed_results = test_preprocessing_pipelines(original_data)
    
    # 5. 测试包含啁啾校正的完整管道
    print("\n" + "="*50)
    print("🌟 完整预处理管道测试（包含啁啾校正）")
    print("="*50)
    
    try:
        # 测试啁啾校正管道
        chirp_pipeline = create_chirp_corrected_pipeline(chirp_method='cross_correlation', verbose=True)
        processed_with_chirp = chirp_pipeline.fit_transform(original_data.copy())
        print("✅ 啁啾校正管道测试成功")
        
        # 测试综合管道
        comprehensive_pipeline = create_comprehensive_pipeline(chirp_method='cross_correlation', verbose=True)
        processed_comprehensive = comprehensive_pipeline.fit_transform(original_data.copy())
        print("✅ 综合预处理管道测试成功")
        
        # 将结果添加到处理结果中
        processed_results['chirp_corrected'] = processed_with_chirp
        processed_results['comprehensive'] = processed_comprehensive
        
    except Exception as e:
        print(f"❌ 完整管道测试失败: {str(e)}")
    
    # 6. 创建可视化
    create_comprehensive_visualization(original_data, processed_results)
    
    # 7. 打印摘要
    print_processing_summaries(processed_results)
    
    # 8. 显示图片
    print(f"\n📊 已生成可视化文件:")
    print("   - tas_preprocessing_heatmap_comparison.png")
    print("   - tas_preprocessing_spectra_comparison.png") 
    print("   - tas_preprocessing_kinetics_comparison.png")
    print("   - tas_preprocessing_statistics_comparison.png")
    if chirp_results:
        print("   - tas_chirp_correction_comparison.png")
        print("   - tas_chirp_correction_heatmap.png")
    
    # 显示图片
    try:
        plt.show()
    except:
        print("\n💡 请手动查看生成的PNG文件")
    
    print("\n🎉 TAS真实数据预处理测试完成！")

if __name__ == '__main__':
    main()
