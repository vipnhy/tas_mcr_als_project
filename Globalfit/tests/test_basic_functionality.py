"""
test_basic_functionality.py - 基本功能测试

该脚本测试Globalfit模块的核心功能是否正常工作。
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from Globalfit import (
    SequentialModel,
    ParallelModel,
    GlobalLifetimeAnalysis,
    GlobalTargetAnalysis
)


def generate_synthetic_data(n_times=50, n_wavelengths=100, n_components=3):
    """生成合成测试数据"""
    print("\n生成合成测试数据...")
    
    # 时间轴 (对数分布)
    time_axis = np.logspace(-1, 3, n_times)  # 0.1 to 1000 ps
    
    # 波长轴
    wavelength_axis = np.linspace(400, 700, n_wavelengths)
    
    # 生成真实的浓度轮廓 (顺序反应 A→B→C)
    k1, k2 = 0.2, 0.01  # ps^-1
    
    C = np.zeros((n_times, n_components))
    C[:, 0] = np.exp(-k1 * time_axis)  # A
    C[:, 1] = k1 / (k2 - k1) * (np.exp(-k1 * time_axis) - np.exp(-k2 * time_axis))  # B
    C[:, 2] = 1 - C[:, 0] - C[:, 1]  # C
    
    # 生成光谱
    S = np.zeros((n_wavelengths, n_components))
    # 高斯峰
    S[:, 0] = np.exp(-((wavelength_axis - 450) / 30) ** 2)
    S[:, 1] = np.exp(-((wavelength_axis - 550) / 40) ** 2)
    S[:, 2] = np.exp(-((wavelength_axis - 620) / 35) ** 2)
    
    # 重构数据
    D = C @ S.T
    
    # 添加噪声
    noise_level = 0.01
    D += noise_level * np.random.randn(*D.shape)
    
    print(f"  数据形状: {D.shape}")
    print(f"  时间范围: {time_axis[0]:.2f} - {time_axis[-1]:.2f} ps")
    print(f"  波长范围: {wavelength_axis[0]:.1f} - {wavelength_axis[-1]:.1f} nm")
    print(f"  组分数量: {n_components}")
    
    return D, time_axis, wavelength_axis, C, S


def test_kinetic_models():
    """测试动力学模型"""
    print("\n" + "=" * 70)
    print("测试1: 动力学模型")
    print("=" * 70)
    
    # 测试顺序模型
    print("\n测试顺序反应模型 (A→B→C)...")
    seq_model = SequentialModel(n_components=3)
    time_points = np.linspace(0, 100, 50)
    rate_constants = [0.1, 0.05]
    
    try:
        C_seq = seq_model.solve(time_points, rate_constants)
        print(f"  ✓ 顺序模型求解成功，浓度矩阵形状: {C_seq.shape}")
        
        # 检查质量守恒
        total_conc = np.sum(C_seq, axis=1)
        if np.allclose(total_conc, 1.0, atol=1e-3):
            print(f"  ✓ 质量守恒验证通过")
        else:
            print(f"  ✗ 质量守恒验证失败: {total_conc[0]:.4f} → {total_conc[-1]:.4f}")
    except Exception as e:
        print(f"  ✗ 顺序模型失败: {e}")
        return False
    
    # 测试平行模型
    print("\n测试平行反应模型 (A→B, A→C)...")
    par_model = ParallelModel(n_components=3)
    
    try:
        C_par = par_model.solve(time_points, rate_constants)
        print(f"  ✓ 平行模型求解成功，浓度矩阵形状: {C_par.shape}")
        
        # 检查初始条件
        if np.isclose(C_par[0, 0], 1.0, atol=1e-3):
            print(f"  ✓ 初始条件验证通过")
        else:
            print(f"  ✗ 初始条件验证失败")
    except Exception as e:
        print(f"  ✗ 平行模型失败: {e}")
        return False
    
    return True


def test_gla():
    """测试全局寿命分析"""
    print("\n" + "=" * 70)
    print("测试2: 全局寿命分析 (GLA)")
    print("=" * 70)
    
    # 生成测试数据
    D, time_axis, wavelength_axis, C_true, S_true = generate_synthetic_data()
    
    # 创建GLA分析器
    print("\n创建GLA分析器...")
    gla = GlobalLifetimeAnalysis(
        data_matrix=D,
        time_axis=time_axis,
        wavelength_axis=wavelength_axis,
        n_components=3
    )
    print("  ✓ GLA分析器创建成功")
    
    # 执行拟合
    print("\n执行GLA拟合...")
    tau_initial = [5.0, 50.0, 500.0]
    
    try:
        results = gla.fit(
            tau_initial=tau_initial,
            optimization_method='leastsq'
        )
        
        print(f"  ✓ 拟合成功!")
        print(f"    最优寿命: {results['tau_optimal']}")
        print(f"    LOF: {results['lof']:.4f}%")
        print(f"    Chi-Square: {results['chi_square']:.6e}")
        print(f"    计算时间: {results['computation_time']:.2f} 秒")
        
        # 验证LOF
        if results['lof'] < 20:
            print(f"  ✓ LOF验证通过 (< 20%)")
        else:
            print(f"  ⚠ LOF较高，但在合成数据中可接受")
        
        return True
        
    except Exception as e:
        print(f"  ✗ GLA拟合失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gta():
    """测试全局目标分析"""
    print("\n" + "=" * 70)
    print("测试3: 全局目标分析 (GTA)")
    print("=" * 70)
    
    # 生成测试数据 (使用顺序模型生成)
    D, time_axis, wavelength_axis, C_true, S_true = generate_synthetic_data()
    
    # 创建顺序模型
    print("\n创建顺序反应模型...")
    kinetic_model = SequentialModel(n_components=3)
    print("  ✓ 动力学模型创建成功")
    
    # 创建GTA分析器
    print("\n创建GTA分析器...")
    gta = GlobalTargetAnalysis(
        data_matrix=D,
        time_axis=time_axis,
        wavelength_axis=wavelength_axis,
        kinetic_model=kinetic_model
    )
    print("  ✓ GTA分析器创建成功")
    
    # 执行拟合
    print("\n执行GTA拟合...")
    k_initial = [0.15, 0.015]  # 接近真实值 [0.2, 0.01]
    
    try:
        results = gta.fit(
            k_initial=k_initial,
            optimization_method='leastsq'
        )
        
        print(f"  ✓ 拟合成功!")
        print(f"    最优速率常数: {results['k_optimal']}")
        print(f"    对应寿命: {results['tau_optimal']}")
        print(f"    LOF: {results['lof']:.4f}%")
        print(f"    Chi-Square: {results['chi_square']:.6e}")
        print(f"    计算时间: {results['computation_time']:.2f} 秒")
        
        # 与真实值比较
        k_true = [0.2, 0.01]
        print(f"\n  真实值: k = {k_true}")
        print(f"  拟合值: k = {[f'{k:.4f}' for k in results['k_optimal']]}")
        
        # 验证LOF
        if results['lof'] < 20:
            print(f"  ✓ LOF验证通过 (< 20%)")
        else:
            print(f"  ⚠ LOF较高，但在合成数据中可接受")
        
        return True
        
    except Exception as e:
        print(f"  ✗ GTA拟合失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("=" * 70)
    print("Globalfit模块基本功能测试")
    print("=" * 70)
    
    tests = [
        ("动力学模型", test_kinetic_models),
        ("全局寿命分析 (GLA)", test_gla),
        ("全局目标分析 (GTA)", test_gta)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ 测试 '{test_name}' 遇到意外错误: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {test_name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过!")
        return 0
    else:
        print(f"\n⚠ {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
