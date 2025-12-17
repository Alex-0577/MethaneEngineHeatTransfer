#!/usr/bin/env python3
"""
发动机传热分析调试脚本
使用pytest进行针对性调试
"""

import os
import sys
import pytest
import logging

# 设置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_specific_test(test_name):
    """运行特定的测试"""
    logger.info(f"运行测试: {test_name}")
    
    # 使用pytest运行特定测试
    result = pytest.main([
        "test_engine_analysis.py", 
        f"-k {test_name}",
        "-v", "-s", "--tb=short"
    ])
    
    return result == 0

def debug_geometry_loading():
    """调试几何加载问题"""
    logger.info("=== 调试几何加载 ===")
    
    try:
        from MethaneEngineHeatTransfer import EngineGeometry
        
        # 测试几何加载器
        geometry = EngineGeometry()
        logger.info("几何加载器创建成功")
        
        # 测试文件验证
        validation = geometry.validate_geometry_file("AE-1305.txt")
        logger.info(f"文件验证结果: {validation}")
        
        return True
        
    except Exception as e:
        logger.error(f"几何加载调试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def debug_axial_distribution():
    """调试轴向分布计算"""
    logger.info("=== 调试轴向分布计算 ===")
    
    try:
        from MethaneEngineHeatTransfer import LOX_MethaneEngineHeatTransfer
        
        # 创建简化版本的发动机实例用于调试
        engine = LOX_MethaneEngineHeatTransfer(
            refprop_path=None,
            use_cea=False,  # 禁用CEA以简化调试
            engine_shape_file=None
        )
        
        # 设置基本几何参数
        engine.set_geometric_parameters(
            d_throat=0.0625,
            L_chamber=0.257,
            delta_wall=0.002,
            number_of_fins=36,
            t_fin=0.0008,
            h_fin=0.008,
            delta_fin=0.012
        )
        
        logger.info("发动机实例创建成功")
        
        # 测试局部几何计算
        diameter, area = engine.calculate_local_geometry(0.1, 0.257)
        logger.info(f"局部几何计算: 直径={diameter:.4f}m, 面积={area:.6f}m²")
        
        # 测试流通面积计算
        flow_area, b_channel, radius = engine.calculate_flow_area(0.1, 0.257)
        logger.info(f"流通面积计算: 面积={flow_area:.6f}m², 通道宽={b_channel:.4f}m")
        
        return True
        
    except Exception as e:
        logger.error(f"轴向分布调试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def debug_heat_balance():
    """调试热平衡计算"""
    logger.info("=== 调试热平衡计算 ===")
    
    try:
        from MethaneEngineHeatTransfer import LOX_MethaneEngineHeatTransfer
        from unittest.mock import Mock, patch
        
        with patch('MethaneEngineHeatTransfer.REFPROPFluid') as mock_refprop:
            # 模拟物性计算
            mock_refprop_instance = Mock()
            mock_refprop_instance.get_methane_properties.return_value = {
                'density': 100, 'viscosity': 1e-5, 'conductivity': 0.1,
                'specific_heat': 2000, 'prandtl': 0.7
            }
            mock_refprop_instance.get_combustion_gas_properties.return_value = {
                'density': 10, 'viscosity': 1e-5, 'conductivity': 0.2,
                'specific_heat': 1200, 'prandtl': 0.6, 'gamma': 1.2
            }
            mock_refprop.return_value = mock_refprop_instance
            
            engine = LOX_MethaneEngineHeatTransfer()
            engine.set_geometric_parameters(
                d_throat=0.0625, L_chamber=0.257, delta_wall=0.002,
                number_of_fins=36, t_fin=0.0008, h_fin=0.008, delta_fin=0.012
            )
            
            # 测试热平衡求解
            with patch('scipy.optimize.fsolve') as mock_fsolve:
                mock_fsolve.return_value = [800.0, 400.0]
                
                T_wg, T_wc = engine.solve_heat_balance(
                    T_gas=3400, P_chamber=10e6, c_star=1750,
                    m_dot_coolant=4.5, T_coolant_in=110, P_coolant_in=18e6,
                    geometry_params={
                        'local_area': 0.01, 'throat_area': 0.003,
                        'local_diameter': 0.1, 'local_channel_width': 0.003
                    },
                    material_properties={
                        'thermal_conductivity_points': [[300, 400], [800, 350]]
                    },
                    operating_conditions={
                        'coolant_velocity': 12, 'hydraulic_diameter': 0.008, 'mach_number': 0.5
                    }
                )
                
                logger.info(f"热平衡求解结果: T_wg={T_wg:.1f}K, T_wc={T_wc:.1f}K")
                return True
                
    except Exception as e:
        logger.error(f"热平衡调试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """主调试函数"""
    logger.info("开始发动机传热分析调试")
    
    # 运行各个调试模块
    debug_modules = [
        ("几何加载", debug_geometry_loading),
        ("轴向分布", debug_axial_distribution),
        ("热平衡", debug_heat_balance)
    ]
    
    results = []
    for name, debug_func in debug_modules:
        logger.info(f"\n{'='*50}")
        logger.info(f"调试模块: {name}")
        logger.info('='*50)
        
        success = debug_func()
        results.append((name, success))
        
        if success:
            logger.info(f"✓ {name} 调试成功")
        else:
            logger.error(f"✗ {name} 调试失败")
    
    # 汇总结果
    logger.info("\n" + "="*60)
    logger.info("调试结果汇总:")
    logger.info("="*60)
    
    for name, success in results:
        status = "✓ 成功" if success else "✗ 失败"
        logger.info(f"{name:15} {status}")
    
    # 运行pytest测试
    logger.info("\n运行pytest测试套件...")
    pytest_result = pytest.main([
        "test_engine_analysis.py",
        "-v", "--tb=short", "-x"  # 遇到第一个错误就停止
    ])
    
    if pytest_result == 0:
        logger.info("✓ 所有测试通过")
    else:
        logger.error("✗ 测试失败")
    
    return all(success for _, success in results) and (pytest_result == 0)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)