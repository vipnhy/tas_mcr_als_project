#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_main.py - 参数化运行TAS MCR-ALS分析

用法:
1. 命令行参数:
   python run_main.py --file_path "data/TAS/TA_Average.csv" --n_components 3 --wavelength_range 420 750

2. 配置文件:
   python run_main.py --config config.json

3. 交互式输入:
   python run_main.py
"""

import argparse
import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcr.mcr_als import MCRALS
from data.data import read_file


def setup_matplotlib_fonts(language='chinese'):
    """
    设置matplotlib字体以支持中文或英文显示
    
    参数:
    - language: 'chinese' 或 'english'
    """
    if language == 'chinese':
        # 尝试设置中文字体
        chinese_fonts = [
            'SimHei',      # 黑体
            'Microsoft YaHei',  # 微软雅黑
            'SimSun',      # 宋体
            'KaiTi',       # 楷体
            'FangSong',    # 仿宋
            'DejaVu Sans'  # 备用字体
        ]
        
        font_found = False
        for font in chinese_fonts:
            try:
                rcParams['font.sans-serif'] = [font]
                rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                
                # 测试字体是否可用
                test_font = fm.FontProperties(family=font)
                if test_font.get_name() in [f.name for f in fm.fontManager.ttflist]:
                    print(f"使用中文字体: {font}")
                    font_found = True
                    break
            except:
                continue
        
        if not font_found:
            print("警告: 未找到合适的中文字体，将使用默认字体")
            rcParams['font.sans-serif'] = ['DejaVu Sans']
    else:
        # 使用英文字体
        rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
        rcParams['axes.unicode_minus'] = False
        print("使用英文字体显示")


def get_labels(language='chinese'):
    """
    根据语言返回相应的标签字典
    
    参数:
    - language: 'chinese' 或 'english'
    
    返回:
    - labels: 包含所有标签的字典
    """
    if language == 'chinese':
        labels = {
            'concentration_profiles': '浓度轮廓 (Concentration Profiles)',
            'pure_spectra': '纯光谱 (Pure Spectra)',
            'lof_convergence': 'LOF收敛曲线',
            'residuals': '残差图 (D - C*S.T)',
            'original_data': '原始TAS数据',
            'reconstructed_data': '重构数据 (C*S.T)',
            'residuals_comparison': '残差 (原始-重构)',
            'time_delay': '时间延迟 (ps)',
            'wavelength': '波长 (nm)',
            'concentration': '浓度 (a.u.)',
            'absorption': 'ΔA (a.u.)',
            'iteration': '迭代次数',
            'lof_percent': 'LOF (%)',
            'wavelength_index': '波长索引',
            'time_index': '时间索引',
            'component': '组分',
            'delta_a': 'ΔA'
        }
    else:
        labels = {
            'concentration_profiles': 'Concentration Profiles',
            'pure_spectra': 'Pure Spectra',
            'lof_convergence': 'LOF Convergence',
            'residuals': 'Residuals (D - C*S.T)',
            'original_data': 'Original TAS Data',
            'reconstructed_data': 'Reconstructed Data (C*S.T)',
            'residuals_comparison': 'Residuals (Original - Reconstructed)',
            'time_delay': 'Time Delay (ps)',
            'wavelength': 'Wavelength (nm)',
            'concentration': 'Concentration (a.u.)',
            'absorption': 'ΔA (a.u.)',
            'iteration': 'Iteration',
            'lof_percent': 'LOF (%)',
            'wavelength_index': 'Wavelength Index',
            'time_index': 'Time Index',
            'component': 'Component',
            'delta_a': 'ΔA'
        }
    
    return labels


class TASAnalyzer:
    """TAS MCR-ALS分析器类"""
    
    def __init__(self, file_path, file_type="handle", wavelength_range=(400, 800), 
                 delay_range=(0, 10), n_components=3, language='chinese',
                 constraint_config=None, penalty=0.0, init_method='svd',
                 random_seed=None):
        """
        初始化分析器
        
        参数:
        - file_path: TAS数据文件路径
        - file_type: 文件类型 ("handle" 或 "raw")
        - wavelength_range: 波长范围 (min, max)
        - delay_range: 时间延迟范围 (min, max)
        - n_components: MCR-ALS组分数量
        - language: 界面语言 ('chinese' 或 'english')
        """
        self.file_path = file_path
        self.file_type = file_type
        self.wavelength_range = wavelength_range
        self.delay_range = delay_range
        self.n_components = n_components
        self.language = language
        self.constraint_config = constraint_config
        self.penalty = penalty
        self.init_method = init_method
        self.random_seed = random_seed
        
        # 设置字体和标签
        setup_matplotlib_fonts(language)
        self.labels = get_labels(language)
        
        # 分析结果存储
        self.D = None
        self.time_axis = None
        self.wavelength_axis = None
        self.mcr_solver = None
        self.C_resolved = None
        self.S_resolved = None
    
    def load_data(self):
        """加载TAS数据"""
        print(f"正在加载数据: {self.file_path}")
        print(f"文件类型: {self.file_type}")
        print(f"波长范围: {self.wavelength_range} nm")
        print(f"时间延迟范围: {self.delay_range} ps")
        
        try:
            df = read_file(
                self.file_path, 
                file_type=self.file_type, 
                inf_handle=True,
                wavelength_range=self.wavelength_range, 
                delay_range=self.delay_range
            )
            
            if df is None:
                raise ValueError("数据加载失败，请检查文件路径和格式")
            
            # 转换为矩阵格式
            self.D = df.values
            self.time_axis = df.index.values  # 时间延迟
            self.wavelength_axis = df.columns.values  # 波长
            
            print(f"数据形状: {self.D.shape}")
            print(f"时间范围: {self.time_axis.min():.2f} 到 {self.time_axis.max():.2f} ps")
            print(f"波长范围: {self.wavelength_axis.min():.1f} 到 {self.wavelength_axis.max():.1f} nm")
            
            # 检查数据质量
            if np.any(np.isnan(self.D)):
                print("警告: 数据中包含NaN值")
            if np.any(np.isinf(self.D)):
                print("警告: 数据中包含无穷值")
                
            return True
            
        except Exception as e:
            print(f"数据加载错误: {e}")
            return False
    
    def run_mcr_als(self, max_iter=200, tol=1e-7):
        """运行MCR-ALS分析"""
        if self.D is None:
            print("错误: 请先加载数据")
            return False
        
        print(f"\n开始运行MCR-ALS分析，组分数量: {self.n_components}")
        print(f"最大迭代次数: {max_iter}, 收敛容差: {tol}")
        
        try:
            # 初始化和运行MCR-ALS
            self.mcr_solver = MCRALS(
                    n_components=self.n_components,
                    max_iter=max_iter,
                    tol=tol,
                    constraint_config=self.constraint_config,
                    penalty=self.penalty,
                    init_method=self.init_method,
                    random_state=self.random_seed
                )
            self.mcr_solver.fit(self.D)
            
            # 获取结果
            self.C_resolved = self.mcr_solver.C_opt_
            self.S_resolved = self.mcr_solver.S_opt_
            
            print(f"MCR-ALS完成，迭代次数: {len(self.mcr_solver.lof_)}")
            print(f"最终LOF: {self.mcr_solver.lof_[-1]:.4f}%")
            
            return True
            
        except Exception as e:
            print(f"MCR-ALS分析错误: {e}")
            return False
    
    def visualize_results(self, save_plots=False, output_dir="results"):
        """可视化分析结果"""
        if self.mcr_solver is None:
            print("错误: 请先运行MCR-ALS分析")
            return
        
        print("\n正在生成可视化图表...")
        
        # 创建输出目录
        if save_plots and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 主要结果图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 浓度轮廓
        axes[0, 0].set_title(self.labels['concentration_profiles'])
        for i in range(self.n_components):
            component_label = f"{self.labels['component']} {i+1}"
            axes[0, 0].plot(self.time_axis, self.C_resolved[:, i], 
                           label=component_label, linewidth=2)
        axes[0, 0].set_xlabel(self.labels['time_delay'])
        axes[0, 0].set_ylabel(self.labels['concentration'])
        axes[0, 0].set_xscale('log')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 纯光谱
        axes[0, 1].set_title(self.labels['pure_spectra'])
        for i in range(self.n_components):
            component_label = f"{self.labels['component']} {i+1}"
            axes[0, 1].plot(self.wavelength_axis, self.S_resolved[:, i], 
                           label=component_label, linewidth=2)
        axes[0, 1].set_xlabel(self.labels['wavelength'])
        axes[0, 1].set_ylabel(self.labels['absorption'])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # LOF收敛曲线
        axes[1, 0].set_title(self.labels['lof_convergence'])
        axes[1, 0].plot(self.mcr_solver.lof_, 'b-', linewidth=2)
        axes[1, 0].set_xlabel(self.labels['iteration'])
        axes[1, 0].set_ylabel(self.labels['lof_percent'])
        axes[1, 0].grid(True, alpha=0.3)
        
        # 残差图
        im = axes[1, 1].imshow(self.mcr_solver.residuals_, aspect='auto', cmap='coolwarm', 
                               vmin=-np.max(np.abs(self.mcr_solver.residuals_)), 
                               vmax=np.max(np.abs(self.mcr_solver.residuals_)),
                               origin='lower')
        axes[1, 1].set_title(self.labels['residuals'])
        axes[1, 1].set_xlabel(self.labels['wavelength_index'])
        axes[1, 1].set_ylabel(self.labels['time_index'])
        fig.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(output_dir, "mcr_als_results.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 原始数据与重构数据对比
        fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # 原始TAS数据热图
        im1 = ax1.imshow(self.D, aspect='auto', cmap='coolwarm', 
                        extent=[self.wavelength_axis.min(), self.wavelength_axis.max(),
                               self.time_axis.min(), self.time_axis.max()],
                        origin='lower')
        ax1.set_title(self.labels['original_data'])
        ax1.set_xlabel(self.labels['wavelength'])
        ax1.set_ylabel(self.labels['time_delay'])
        fig2.colorbar(im1, ax=ax1, label=self.labels['delta_a'])
        
        # 重构数据
        D_reconstructed = self.C_resolved @ self.S_resolved.T
        im2 = ax2.imshow(D_reconstructed, aspect='auto', cmap='coolwarm',
                        extent=[self.wavelength_axis.min(), self.wavelength_axis.max(),
                               self.time_axis.min(), self.time_axis.max()],
                        vmin=im1.get_clim()[0], vmax=im1.get_clim()[1],
                        origin='lower')
        ax2.set_title(self.labels['reconstructed_data'])
        ax2.set_xlabel(self.labels['wavelength'])
        ax2.set_ylabel(self.labels['time_delay'])
        fig2.colorbar(im2, ax=ax2, label=self.labels['delta_a'])
        
        # 残差热图
        residuals = self.D - D_reconstructed
        im3 = ax3.imshow(residuals, aspect='auto', cmap='coolwarm',
                        extent=[self.wavelength_axis.min(), self.wavelength_axis.max(),
                               self.time_axis.min(), self.time_axis.max()],
                        vmin=-np.max(np.abs(residuals)), vmax=np.max(np.abs(residuals)),
                        origin='lower')
        ax3.set_title(self.labels['residuals_comparison'])
        ax3.set_xlabel(self.labels['wavelength'])
        ax3.set_ylabel(self.labels['time_delay'])
        fig2.colorbar(im3, ax=ax3, label=self.labels['delta_a'])
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(output_dir, "data_comparison.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, output_dir="results"):
        """保存分析结果"""
        if self.mcr_solver is None:
            print("错误: 请先运行MCR-ALS分析")
            return
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\n正在保存结果到: {output_dir}")
        
        # 保存浓度矩阵
        np.savetxt(os.path.join(output_dir, "concentration_profiles.csv"), 
                   self.C_resolved, delimiter=',', 
                   header=','.join([f'Component_{i+1}' for i in range(self.n_components)]))
        
        # 保存光谱矩阵
        np.savetxt(os.path.join(output_dir, "pure_spectra.csv"), 
                   self.S_resolved, delimiter=',',
                   header=','.join([f'Component_{i+1}' for i in range(self.n_components)]))
        
        # 保存LOF历史
        np.savetxt(os.path.join(output_dir, "lof_history.csv"), 
                   self.mcr_solver.lof_, delimiter=',', header='LOF_%')
        
        # 保存参数信息
        params = {
            'file_path': self.file_path,
            'file_type': self.file_type,
            'wavelength_range': self.wavelength_range,
            'delay_range': self.delay_range,
            'n_components': self.n_components,
            'final_lof': float(self.mcr_solver.lof_[-1]),
            'iterations': len(self.mcr_solver.lof_),
            'data_shape': self.D.shape,
            'penalty': self.penalty,
            'init_method': self.init_method,
            'random_seed': self.random_seed,
            'constraint_config': self.constraint_config if isinstance(self.constraint_config, str) else None
        }
        
        with open(os.path.join(output_dir, "analysis_parameters.json"), 'w') as f:
            json.dump(params, f, indent=2)
        
        print("结果保存完成!")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TAS MCR-ALS 参数化分析工具')
    
    # 文件路径和类型
    parser.add_argument('--file_path', type=str, 
                       help='TAS数据文件路径')
    parser.add_argument('--file_type', type=str, default='handle', 
                       choices=['handle', 'raw'],
                       help='文件类型 (默认: handle)')
    
    # 数据范围参数
    parser.add_argument('--wavelength_range', nargs=2, type=float, 
                       default=[400, 800],
                       help='波长范围 (min max), 默认: 400 800')
    parser.add_argument('--delay_range', nargs=2, type=float, 
                       default=[0, 10],
                       help='时间延迟范围 (min max), 默认: 0 10')
    
    # MCR-ALS参数
    parser.add_argument('--n_components', type=int, default=3,
                       help='MCR-ALS组分数量 (默认: 3)')
    parser.add_argument('--max_iter', type=int, default=200,
                       help='最大迭代次数 (默认: 200)')
    parser.add_argument('--tol', type=float, default=1e-7,
                       help='收敛容差 (默认: 1e-7)')
    parser.add_argument('--penalty', type=float, default=None,
                       help='正则化惩罚因子 (默认: 0.0)')
    parser.add_argument('--init_method', type=str, default=None,
                       choices=['svd', 'random'],
                       help="初始值策略: 'svd' 或 'random' (默认: svd)")
    parser.add_argument('--random_seed', type=int, default=None,
                       help='随机初始化的随机种子 (默认: None)')
    parser.add_argument('--constraint_config', type=str, default=None,
                       help='约束配置文件路径 (默认: 使用内置配置)')
    
    # 输出选项
    parser.add_argument('--save_plots', action='store_true',
                       help='保存图表到文件')
    parser.add_argument('--save_results', action='store_true',
                       help='保存数值结果到文件')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='输出目录 (默认: results)')
    parser.add_argument('--language', type=str, default='chinese',
                       choices=['chinese', 'english'],
                       help='界面语言 (默认: chinese)')
    
    # 配置文件选项
    parser.add_argument('--config', type=str,
                       help='从JSON配置文件读取参数')
    
    return parser.parse_args()


def load_config(config_file):
    """从JSON文件加载配置"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"已加载配置文件: {config_file}")
        return config
    except Exception as e:
        print(f"配置文件加载错误: {e}")
        return None


def interactive_input():
    """交互式参数输入"""
    print("=== TAS MCR-ALS 交互式参数设置 ===\n")
    
    # 文件路径
    file_path = input("请输入TAS数据文件路径: ").strip()
    if not file_path:
        file_path = "data/TAS/TA_Average-crop-TCA-tzs-tzs-tzs-chirp.csv"
        print(f"使用默认路径: {file_path}")
    
    # 文件类型
    file_type = input("文件类型 (handle/raw) [默认: handle]: ").strip()
    if not file_type:
        file_type = "handle"
    
    # 波长范围
    wl_input = input("波长范围 (min max) [默认: 420 750]: ").strip()
    if wl_input:
        wavelength_range = list(map(float, wl_input.split()))
    else:
        wavelength_range = [420, 750]
    
    # 时间延迟范围
    delay_input = input("时间延迟范围 (min max) [默认: 0.1 50]: ").strip()
    if delay_input:
        delay_range = list(map(float, delay_input.split()))
    else:
        delay_range = [0.1, 50]
    
    # 组分数量
    n_comp_input = input("MCR-ALS组分数量 [默认: 3]: ").strip()
    if n_comp_input:
        n_components = int(n_comp_input)
    else:
        n_components = 3
    
    penalty_input = input("正则化惩罚因子 [默认: 0.0]: ").strip()
    penalty = float(penalty_input) if penalty_input else 0.0

    init_input = input("初始值策略 (svd/random) [默认: svd]: ").strip().lower()
    init_method = init_input if init_input in ['svd', 'random'] else 'svd'

    seed_input = input("随机种子 (留空表示随机): ").strip()
    random_seed = int(seed_input) if seed_input else None

    constraint_input = input("约束配置文件路径 (可选): ").strip()
    constraint_config = constraint_input if constraint_input else None

    # 语言选择
    lang_input = input("界面语言 (chinese/english) [默认: chinese]: ").strip().lower()
    if lang_input in ['english', 'en', 'e']:
        language = 'english'
    else:
        language = 'chinese'
    
    return {
        'file_path': file_path,
        'file_type': file_type,
        'wavelength_range': wavelength_range,
        'delay_range': delay_range,
        'n_components': n_components,
        'language': language,
        'penalty': penalty,
        'init_method': init_method,
        'random_seed': random_seed,
        'constraint_config': constraint_config
    }


def main():
    """主函数"""
    args = parse_arguments()
    
    # 确定参数来源
    if args.config:
        # 从配置文件读取
        config = load_config(args.config)
        if config is None:
            return
        params = config
    elif args.file_path:
        # 从命令行参数读取
        params = {
            'file_path': args.file_path,
            'file_type': args.file_type,
            'wavelength_range': tuple(args.wavelength_range),
            'delay_range': tuple(args.delay_range),
            'n_components': args.n_components,
            'language': args.language
        }
    else:
        # 交互式输入
        params = interactive_input()
        # 为交互式输入添加默认语言
        if 'language' not in params:
            params['language'] = 'chinese'
    
    # 合并可选参数和默认值
    if args.constraint_config is not None:
        params['constraint_config'] = args.constraint_config
    else:
        params.setdefault('constraint_config', None)

    if args.penalty is not None:
        params['penalty'] = args.penalty
    else:
        params.setdefault('penalty', 0.0)

    if args.init_method is not None:
        params['init_method'] = args.init_method
    else:
        params.setdefault('init_method', 'svd')

    if args.random_seed is not None:
        params['random_seed'] = args.random_seed
    else:
        params.setdefault('random_seed', None)

    params['penalty'] = float(params.get('penalty', 0.0))
    params['init_method'] = str(params.get('init_method', 'svd')).lower()
    if params['init_method'] not in ['svd', 'random']:
        params['init_method'] = 'svd'

    print("\n=== 分析参数 ===")
    for key, value in params.items():
        print(f"{key}: {value}")
    print("=" * 30)
    
    # 创建分析器并运行分析
    analyzer = TASAnalyzer(**params)
    
    # 加载数据
    if not analyzer.load_data():
        print("数据加载失败，程序退出")
        return
    
    # 运行MCR-ALS分析
    if not analyzer.run_mcr_als(max_iter=args.max_iter, tol=args.tol):
        print("MCR-ALS分析失败，程序退出")
        return
    
    # 可视化结果
    analyzer.visualize_results(save_plots=args.save_plots, output_dir=args.output_dir)
    
    # 保存结果
    if args.save_results:
        analyzer.save_results(output_dir=args.output_dir)
    
    print("\n分析完成!")


if __name__ == '__main__':
    main()
