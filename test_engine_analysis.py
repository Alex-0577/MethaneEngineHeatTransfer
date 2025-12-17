import pytest
import numpy as np
import os
import sys
import json
from unittest.mock import Mock, patch, MagicMock
import unittest

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TestEngineAnalysis:
    """发动机传热分析测试类"""
    
    @pytest.fixture
    def mock_params(self):
        """模拟参数配置"""
        return {
            "engine_geometry": {
                "d_throat": 0.0625,
                "L_chamber": 0.257,
                "delta_wall": 0.002,
                "number_of_fins": 36,
                "t_fin": 0.0008,
                "h_fin": 0.008,
                "delta_fin": 0.012
            },
            "calculation_settings": {
                "num_segments": 10,  # 测试时使用较少的分段
                "max_allowable_temp": 1100.0
            }
        }
    
    @pytest.fixture
    def mock_geometry_data(self):
        """模拟几何数据"""
        return {
            'points': [
                (0, 100), (50, 80), (100, 50), (150, 30), 
                (200, 30), (250, 40), (300, 60)
            ],
            'header': {
                'Lc': 169.83604,
                'Rc': 50.0,
                'Rt': 15.0,
                'Re': 80.0
            }
        }
    
    def test_geometry_loader_initialization(self):
        """测试几何加载器初始化"""
        from MethaneEngineHeatTransfer import EngineGeometry
        
        geometry = EngineGeometry()
        assert geometry.shape_data is None
        
    def test_geometry_loading(self, mock_geometry_data):
        """测试几何数据加载"""
        from MethaneEngineHeatTransfer import EngineGeometry
        
        geometry = EngineGeometry()
        
        # 模拟文件读取
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            with patch('MethaneEngineHeatTransfer.logger') as mock_logger:
                # 模拟文件内容
                mock_file.return_value.__iter__.return_value = [
                    "# 测试几何文件",
                    "Lc=169.83604",
                    "Rc=50.0",
                    "#------数据开始------",
                    "0.0 100.0",
                    "50.0 80.0",
                    "100.0 50.0"
                ]
                
                result = geometry.load_geometry_from_file("test_file.txt")
                assert result is True
                assert geometry.shape_data is not None
                assert len(geometry.shape_data['points']) > 0
    
    def test_diameter_calculation(self, mock_geometry_data):
        """测试直径计算"""
        from MethaneEngineHeatTransfer import EngineGeometry
        
        geometry = EngineGeometry()
        geometry.shape_data = mock_geometry_data
        
        # 测试不同位置的直径计算
        diameter1 = geometry.get_diameter_at_position(0.05)  # 50mm位置
        diameter2 = geometry.get_diameter_at_position(0.15)  # 150mm位置
        
        assert diameter1 is not None
        assert diameter2 is not None
        assert diameter1 > diameter2  # 收敛段直径应减小
    
    def test_engine_initialization(self, mock_params):
        """测试发动机初始化"""
        with patch('MethaneEngineHeatTransfer.CEACalculator') as mock_cea:
            with patch('MethaneEngineHeatTransfer.REFPROPFluid') as mock_refprop:
                from MethaneEngineHeatTransfer import LOX_MethaneEngineHeatTransfer
                
                mock_cea_instance = Mock()
                mock_cea.return_value = mock_cea_instance
                mock_refprop_instance = Mock()
                mock_refprop.return_value = mock_refprop_instance
                
                engine = LOX_MethaneEngineHeatTransfer(
                    refprop_path=None,
                    use_cea=True,
                    engine_shape_file=None
                )
                
                assert engine is not None
                assert engine.geometry_loader is not None
    
    def test_geometry_parameter_setting(self, mock_params):
        """测试几何参数设置"""
        with patch('MethaneEngineHeatTransfer.CEACalculator') as mock_cea:
            with patch('MethaneEngineHeatTransfer.REFPROPFluid') as mock_refprop:
                from MethaneEngineHeatTransfer import LOX_MethaneEngineHeatTransfer
                
                engine = LOX_MethaneEngineHeatTransfer()
                
                geo_params = mock_params['engine_geometry']
                engine.set_geometric_parameters(
                    d_throat=geo_params['d_throat'],
                    L_chamber=geo_params['L_chamber'],
                    delta_wall=geo_params['delta_wall'],
                    number_of_fins=geo_params['number_of_fins'],
                    t_fin=geo_params['t_fin'],
                    h_fin=geo_params['h_fin'],
                    delta_fin=geo_params['delta_fin']
                )
                
                assert engine._geometry_configured is True
                assert engine.geometry['diameter']['throat'] == geo_params['d_throat']
    
    def test_local_geometry_calculation(self, mock_params):
        """测试局部几何计算"""
        with patch('MethaneEngineHeatTransfer.CEACalculator') as mock_cea:
            with patch('MethaneEngineHeatTransfer.REFPROPFluid') as mock_refprop:
                from MethaneEngineHeatTransfer import LOX_MethaneEngineHeatTransfer
                
                engine = LOX_MethaneEngineHeatTransfer()
                geo_params = mock_params['engine_geometry']
                engine.set_geometric_parameters(**geo_params)
                
                # 测试不同位置的几何计算
                diameter1, area1 = engine.calculate_local_geometry(0.0, geo_params['L_chamber'])
                diameter2, area2 = engine.calculate_local_geometry(geo_params['L_chamber']/2, geo_params['L_chamber'])
                
                assert diameter1 > 0
                assert area1 > 0
                assert diameter2 > 0
                assert area2 > 0
    
    def test_flow_area_calculation(self, mock_params):
        """测试流通面积计算"""
        with patch('MethaneEngineHeatTransfer.CEACalculator') as mock_cea:
            with patch('MethaneEngineHeatTransfer.REFPROPFluid') as mock_refprop:
                from MethaneEngineHeatTransfer import LOX_MethaneEngineHeatTransfer
                
                engine = LOX_MethaneEngineHeatTransfer()
                geo_params = mock_params['engine_geometry']
                engine.set_geometric_parameters(**geo_params)
                
                flow_area, b_channel, radius = engine.calculate_flow_area(0.1, geo_params['L_chamber'])
                
                assert flow_area > 0
                assert b_channel > 0
                assert radius > 0
    
    @pytest.mark.parametrize("temperature,pressure", [
        (300, 10e6),    # 正常条件
        (100, 5e6),     # 低温条件
        (500, 20e6)     # 高温高压条件
    ])
    def test_methane_properties(self, temperature, pressure):
        """测试甲烷物性计算"""
        with patch('MethaneEngineHeatTransfer.REFPROPFluid') as mock_refprop_class:
            from MethaneEngineHeatTransfer import REFPROPFluid
            
            # 模拟REFPROP返回数据
            mock_refprop = Mock()
            mock_refprop.get_methane_properties.return_value = {
                'density': 100 + 0.1 * pressure/1e6 - 0.5 * (temperature-300),
                'viscosity': 1e-5 * (temperature/300)**0.7,
                'conductivity': 0.03 + 0.0001 * (temperature-300),
                'specific_heat': 2200 + 2 * (temperature-300),
                'prandtl': 0.7 + 0.001 * (temperature-300)
            }
            mock_refprop_class.return_value = mock_refprop
            
            fluid = REFPROPFluid()
            props = fluid.get_methane_properties(temperature, pressure)
            
            assert 'density' in props
            assert 'viscosity' in props
            assert props['density'] > 0
    
    def test_heat_balance_solution(self, mock_params):
        """测试热平衡方程求解"""
        with patch('MethaneEngineHeatTransfer.CEACalculator') as mock_cea:
            with patch('MethaneEngineHeatTransfer.REFPROPFluid') as mock_refprop:
                from MethaneEngineHeatTransfer import LOX_MethaneEngineHeatTransfer
                from scipy.optimize import fsolve
                
                engine = LOX_MethaneEngineHeatTransfer()
                geo_params = mock_params['engine_geometry']
                engine.set_geometric_parameters(**geo_params)
                
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
                
                # 模拟fsolve
                with patch('scipy.optimize.fsolve') as mock_fsolve:
                    mock_fsolve.return_value = [800.0, 400.0]  # 模拟解
                    
                    try:
                        T_wg, T_wc = engine.solve_heat_balance(
                            T_gas=3400, P_chamber=10e6, c_star=1750,
                            m_dot_coolant=4.5, T_coolant_in=110, P_coolant_in=18e6,
                            geometry_params={
                                'local_area': 0.01,
                                'throat_area': 0.003,
                                'local_diameter': 0.1,
                                'local_channel_width': 0.003
                            },
                            material_properties={
                                'thermal_conductivity_points': [[300, 400], [800, 350]]
                            },
                            operating_conditions={
                                'coolant_velocity': 12,
                                'hydraulic_diameter': 0.008,
                                'mach_number': 0.5
                            }
                        )
                        
                        assert T_wg > 0
                        assert T_wc > 0
                        assert T_wg > T_wc
                        
                    except Exception as e:
                        pytest.fail(f"热平衡求解失败: {e}")
    
    def test_axial_distribution_calculation(self, mock_params):
        """测试轴向分布计算"""
        with patch('MethaneEngineHeatTransfer.CEACalculator') as mock_cea:
            with patch('MethaneEngineHeatTransfer.REFPROPFluid') as mock_refprop:
                from MethaneEngineHeatTransfer import LOX_MethaneEngineHeatTransfer
                
                engine = LOX_MethaneEngineHeatTransfer()
                geo_params = mock_params['engine_geometry']
                engine.set_geometric_parameters(**geo_params)
                
                # 模拟物性计算
                mock_refprop_instance = Mock()
                mock_refprop_instance.get_methane_properties.return_value = {
                    'density': 100, 'viscosity': 1e-5, 'conductivity': 0.1,
                    'specific_heat': 2000, 'prandtl': 0.7, 'speed_of_sound': 400
                }
                mock_refprop_instance.get_combustion_gas_properties.return_value = {
                    'density': 10, 'viscosity': 1e-5, 'conductivity': 0.2,
                    'specific_heat': 1200, 'prandtl': 0.6, 'gamma': 1.2,
                    'specific_heat_ratio': 1.2
                }
                mock_refprop.return_value = mock_refprop_instance
                
                # 模拟fsolve
                with patch('scipy.optimize.fsolve') as mock_fsolve:
                    mock_fsolve.return_value = [800.0, 400.0]
                    
                    try:
                        axial_results = engine.calculate_axial_distribution(
                            T_gas=3400, P_chamber=10e6, c_star=1750,
                            m_dot_coolant=4.5, T_coolant_in=110, P_coolant_in=18e6,
                            material_properties={
                                'thermal_conductivity_points': [[300, 400], [800, 350]]
                            },
                            operating_conditions={
                                'coolant_velocity': 12,
                                'hydraulic_diameter': 0.008
                            },
                            num_segments=5,  # 测试使用较少分段
                            mixture_ratio=3.5
                        )
                        
                        assert len(axial_results) == 5
                        for result in axial_results:
                            assert 'axial_position' in result
                            assert 'gas_side_wall_temp' in result
                            assert 'coolant_side_wall_temp' in result
                            
                    except Exception as e:
                        pytest.fail(f"轴向分布计算失败: {e}")

class TestIntegration:
    """集成测试类"""
    
    def test_end_to_end_analysis(self):
        """端到端分析测试"""
        # 这个测试会运行完整的分析流程
        with patch('MethaneEngineHeatTransfer.setup_logging') as mock_logging:
            with patch('MethaneEngineHeatTransfer.load_parameters') as mock_load_params:
                with patch('MethaneEngineHeatTransfer.CEACalculator') as mock_cea:
                    with patch('MethaneEngineHeatTransfer.REFPROPFluid') as mock_refprop:
                        
                        # 模拟参数加载
                        mock_load_params.return_value = {
                            "engine_geometry": {
                                "d_throat": 0.0625, "L_chamber": 0.257,
                                "delta_wall": 0.002, "number_of_fins": 36,
                                "t_fin": 0.0008, "h_fin": 0.008, "delta_fin": 0.012
                            },
                            "calculation_settings": {"num_segments": 5}
                        }
                        
                        # 模拟主函数
                        from MethaneEngineHeatTransfer import main
                        result = main()
                        
                        # 检查是否返回了结果（即使是None也表示执行完成）
                        assert result is not None

def test_parameter_validation():
    """测试参数验证"""
    from MethaneEngineHeatTransfer import load_parameters
    
    # 测试默认参数加载
    with patch('builtins.open', side_effect=FileNotFoundError()):
        params = load_parameters('nonexistent.json')
        assert params is not None
        assert 'engine_geometry' in params

if __name__ == "__main__":
    # 直接运行测试，并将输出保存到test_log.txt
    import sys
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    with open('test_log.txt', 'w', encoding='utf-8') as f:
        sys.stdout = f
        sys.stderr = f
        try:
            result = pytest.main([__file__, "-v", "-s"])
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    
    print(f"测试完成，结果已保存到 test_log.txt，退出代码: {result}")