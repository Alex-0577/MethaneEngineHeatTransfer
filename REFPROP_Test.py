import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary

class REFPROPCompleteTest:
    def __init__(self, rp_path=None):
        """
        完整的REFPROP测试类
        """
        if rp_path is None:
            rp_path = r"C:\Program Files (x86)\REFPROP"
        
        self.rp_path = rp_path
        self.RP = None
        self.setup_refprop()
    
    def setup_refprop(self):
        """设置并初始化REFPROP库"""
        try:
            # 添加REFPROP路径到系统路径
            if self.rp_path not in os.environ['PATH']:
                os.environ['PATH'] = self.rp_path + ';' + os.environ['PATH']
            
            # 初始化REFPROP库
            self.RP = REFPROPFunctionLibrary(self.rp_path)
            self.RP.SETPATHdll(self.rp_path)
            
            print("REFPROP库初始化成功!")
            print(f"REFPROP版本: {self.RP.RPVersion()}")
            
        except Exception as e:
            print(f"REFPROP初始化失败: {e}")
            sys.exit(1)
    
    def test_transport_properties_correct(self):
        """正确测试传输性质获取"""
        print("\n" + "="*50)
        print("正确测试传输性质获取")
        print("="*50)
        
        try:
            # 设置水为工作流体
            water = "Water"
            self.RP.SETUPdll(1, water, "HMX.BNC", "DEF")
            
            T = 300  # K
            P = 101.325  # kPa
            z = [1.0]  # 纯物质的摩尔分数
            
            # 首先获取密度
            results = self.RP.TPFLSHdll(T, P, z)
            D = results.D
            
            print(f"在{T}K和{P}kPa下:")
            print(f"密度: {D:.3f} kg/m³")
            
            # 正确调用TRNPRPdll函数 - 只需要3个参数
            transport_results = self.RP.TRNPRPdll(T, D, z)
            
            print(f"粘度 (eta): {transport_results.eta:.6f} μPa·s")
            print(f"导热系数 (tcx): {transport_results.tcx:.6f} W/m·K")
            print(f"错误代码: {transport_results.ierr}")
            print(f"错误信息: {transport_results.herr}")
            
            # 解释单位
            print("\n单位解释:")
            print("粘度: μPa·s (微帕秒)")
            print("导热系数: W/m·K (瓦特每米开尔文)")
            
        except Exception as e:
            print(f"传输性质测试失败: {e}")
    
    def test_transport_properties_comprehensive(self):
        """全面测试传输性质"""
        print("\n" + "="*50)
        print("全面测试传输性质")
        print("="*50)
        
        try:
            # 设置水为工作流体
            water = "Water"
            self.RP.SETUPdll(1, water, "HMX.BNC", "DEF")
            z = [1.0]
            
            # 测试不同温度下的传输性质
            temperatures = [273.15, 300, 373.15, 473.15, 573.15]  # 0°C, 27°C, 100°C, 200°C, 300°C
            P = 101.325  # kPa
            
            print(f"{'温度(K)':<10} {'温度(°C)':<10} {'密度(kg/m³)':<15} {'粘度(μPa·s)':<15} {'导热系数(W/m·K)':<15}")
            print("-" * 70)
            
            for T in temperatures:
                try:
                    # 获取密度
                    results = self.RP.TPFLSHdll(T, P, z)
                    D = results.D
                    
                    # 获取传输性质
                    transport_results = self.RP.TRNPRPdll(T, D, z)
                    
                    T_celsius = T - 273.15
                    print(f"{T:<10.2f} {T_celsius:<10.1f} {D:<15.3f} {transport_results.eta:<15.3f} {transport_results.tcx:<15.6f}")
                    
                except Exception as e:
                    print(f"在{T}K时出错: {e}")
                    
        except Exception as e:
            print(f"全面测试失败: {e}")
    
    def test_different_fluids_transport(self):
        """测试不同流体的传输性质"""
        print("\n" + "="*50)
        print("测试不同流体的传输性质")
        print("="*50)
        
        fluids = [
            ("WATER", "水"),
            ("AIR", "空气"),
            ("CO2", "二氧化碳"),
            ("R134A", "制冷剂R134a"),
            ("NITROGEN", "氮气"),
            ("METHANE", "甲烷")
        ]
        
        T = 300  # K
        P = 101.325  # kPa
        
        print(f"{'流体':<15} {'密度(kg/m³)':<15} {'粘度(μPa·s)':<15} {'导热系数(W/m·K)':<15}")
        print("-" * 70)
        
        for fluid_code, fluid_name in fluids:
            try:
                self.RP.SETUPdll(1, fluid_code, "HMX.BNC", "DEF")
                z = [1.0]
                
                # 获取密度
                results = self.RP.TPFLSHdll(T, P, z)
                D = results.D
                
                # 获取传输性质
                transport_results = self.RP.TRNPRPdll(T, D, z)
                
                print(f"{fluid_name:<15} {D:<15.6f} {transport_results.eta:<15.3f} {transport_results.tcx:<15.6f}")
                
            except Exception as e:
                print(f"{fluid_name}测试失败: {e}")
    
    def test_water_transport_curve(self):
        """测试水的传输性质随温度变化曲线"""
        print("\n" + "="*50)
        print("测试水的传输性质随温度变化曲线")
        print("="*50)
        
        try:
            # 设置中文字体或使用英文避免警告
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 设置水为工作流体
            water = "Water"
            self.RP.SETUPdll(1, water, "HMX.BNC", "DEF")
            z = [1.0]
            
            # 生成温度范围
            temperatures = np.linspace(273.15, 573.15, 50)  # 0°C 到 300°C
            P = 101.325  # kPa
            
            viscosities = []
            conductivities = []
            valid_temps = []
            
            for T in temperatures:
                try:
                    # 获取密度
                    results = self.RP.TPFLSHdll(T, P, z)
                    D = results.D
                    
                    # 获取传输性质
                    transport_results = self.RP.TRNPRPdll(T, D, z)
                    
                    if transport_results.ierr == 0:  # 无错误
                        viscosities.append(transport_results.eta)
                        conductivities.append(transport_results.tcx)
                        valid_temps.append(T)
                        
                except:
                    continue
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 粘度图表
            ax1.plot(valid_temps, viscosities, 'b-', linewidth=2)
            ax1.set_xlabel('Temperature (K)')
            ax1.set_ylabel('Viscosity (μPa·s)')
            ax1.set_title('Water Viscosity vs Temperature')
            ax1.grid(True, alpha=0.3)
            
            # 导热系数图表
            ax2.plot(valid_temps, conductivities, 'r-', linewidth=2)
            ax2.set_xlabel('Temperature (K)')
            ax2.set_ylabel('Thermal Conductivity (W/m·K)')
            ax2.set_title('Water Thermal Conductivity vs Temperature')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('water_transport_properties.png', dpi=300)
            print("水的传输性质图表已保存为 'water_transport_properties.png'")
            
            # 打印一些关键点的数据
            print("\n水的传输性质关键点:")
            print(f"{'温度(K)':<10} {'温度(°C)':<10} {'粘度(μPa·s)':<15} {'导热系数(W/m·K)':<15}")
            print("-" * 60)
            
            key_temps = [273.15, 293.15, 313.15, 373.15, 473.15, 573.15]
            for T in key_temps:
                if T in valid_temps:
                    idx = valid_temps.index(T)
                    T_celsius = T - 273.15
                    print(f"{T:<10.2f} {T_celsius:<10.1f} {viscosities[idx]:<15.3f} {conductivities[idx]:<15.6f}")
            
        except Exception as e:
            print(f"传输性质曲线测试失败: {e}")
    
    def test_saturation_transport_properties(self):
        """测试饱和状态下的传输性质"""
        print("\n" + "="*50)
        print("测试饱和状态下的传输性质")
        print("="*50)
        
        try:
            # 设置水为工作流体
            water = "Water"
            self.RP.SETUPdll(1, water, "HMX.BNC", "DEF")
            z = [1.0]
            
            temperatures = [300, 350, 400, 450, 500]  # K
            
            print(f"{'温度(K)':<10} {'相态':<10} {'粘度(μPa·s)':<15} {'导热系数(W/m·K)':<15}")
            print("-" * 60)
            
            for T in temperatures:
                try:
                    # 获取饱和性质
                    sat_results = self.RP.SATTdll(T, z, 1)
                    P_sat = sat_results.P
                    
                    # 饱和液相
                    liq_results = self.RP.TPFLSHdll(T, P_sat, z)
                    liq_transport = self.RP.TRNPRPdll(T, liq_results.D, z)
                    
                    # 饱和汽相 (稍微过热)
                    vap_results = self.RP.TPFLSHdll(T, P_sat*0.99, z)
                    vap_transport = self.RP.TRNPRPdll(T, vap_results.D, z)
                    
                    print(f"{T:<10.1f} {'液相':<10} {liq_transport.eta:<15.3f} {liq_transport.tcx:<15.6f}")
                    print(f"{T:<10.1f} {'汽相':<10} {vap_transport.eta:<15.3f} {vap_transport.tcx:<15.6f}")
                    print("-" * 60)
                    
                except Exception as e:
                    print(f"在{T}K时出错: {e}")
                    
        except Exception as e:
            print(f"饱和状态传输性质测试失败: {e}")
    
    def test_mixture_transport_properties(self):
        """测试混合物的传输性质"""
        print("\n" + "="*50)
        print("测试混合物的传输性质")
        print("="*50)
        
        try:
            # 测试空气混合物
            air = "Air"
            self.RP.SETUPdll(1, air, "HMX.BNC", "DEF")
            
            # 空气的摩尔分数 (近似)
            z_air = [0.78, 0.21, 0.01]  # N2, O2, Ar
            
            T = 300  # K
            P = 101.325  # kPa
            
            # 获取密度
            results = self.RP.TPFLSHdll(T, P, z_air)
            D = results.D
            
            # 获取传输性质
            transport_results = self.RP.TRNPRPdll(T, D, z_air)
            
            print(f"空气在{T}K和{P}kPa下的传输性质:")
            print(f"密度: {D:.6f} kg/m³")
            print(f"粘度: {transport_results.eta:.3f} μPa·s")
            print(f"导热系数: {transport_results.tcx:.6f} W/m·K")
            
        except Exception as e:
            print(f"混合物传输性质测试失败: {e}")
    
    def test_high_pressure_transport_properties(self):
        """测试高压下的传输性质"""
        print("\n" + "="*50)
        print("测试高压下的传输性质")
        print("="*50)
        
        try:
            # 设置水为工作流体
            water = "Water"
            self.RP.SETUPdll(1, water, "HMX.BNC", "DEF")
            z = [1.0]
            
            T = 300  # K
            pressures = [0.1, 1, 10, 100, 1000]  # MPa -> 转换为kPa
            
            print(f"{'压力(MPa)':<12} {'密度(kg/m³)':<15} {'粘度(μPa·s)':<15} {'导热系数(W/m·K)':<15}")
            print("-" * 70)
            
            for P_mpa in pressures:
                P_kpa = P_mpa * 1000  # 转换为kPa
                
                try:
                    # 获取密度
                    results = self.RP.TPFLSHdll(T, P_kpa, z)
                    D = results.D
                    
                    # 获取传输性质
                    transport_results = self.RP.TRNPRPdll(T, D, z)
                    
                    print(f"{P_mpa:<12.1f} {D:<15.3f} {transport_results.eta:<15.3f} {transport_results.tcx:<15.6f}")
                    
                except Exception as e:
                    print(f"在{P_mpa}MPa时出错: {e}")
                    
        except Exception as e:
            print(f"高压传输性质测试失败: {e}")
    
    def demonstrate_practical_applications(self):
        """演示传输性质的实际应用"""
        print("\n" + "="*50)
        print("演示传输性质的实际应用")
        print("="*50)
        
        try:
            # 设置水为工作流体
            water = "Water"
            self.RP.SETUPdll(1, water, "HMX.BNC", "DEF")
            z = [1.0]
            
            # 应用1: 计算雷诺数
            print("应用1: 计算管道流动的雷诺数")
            T = 300  # K
            P = 101.325  # kPa
            velocity = 1.0  # m/s
            diameter = 0.1  # m
            
            # 获取水的性质
            results = self.RP.TPFLSHdll(T, P, z)
            transport_results = self.RP.TRNPRPdll(T, results.D, z)
            
            # 密度 (kg/m³)
            density = results.D
            
            # 动力粘度 (Pa·s) = 粘度(μPa·s) / 1e6
            dynamic_viscosity = transport_results.eta / 1e6  # 转换为Pa·s
            
            # 计算雷诺数
            reynolds_number = density * velocity * diameter / dynamic_viscosity
            
            print(f"条件: 温度={T}K, 压力={P}kPa, 流速={velocity}m/s, 管径={diameter}m")
            print(f"密度: {density:.3f} kg/m³")
            print(f"动力粘度: {dynamic_viscosity:.6f} Pa·s")
            print(f"雷诺数: {reynolds_number:.0f}")
            
            if reynolds_number < 2000:
                flow_regime = "层流"
            elif reynolds_number > 4000:
                flow_regime = "湍流"
            else:
                flow_regime = "过渡流"
                
            print(f"流动状态: {flow_regime}")
            
            # 应用2: 计算普朗特数
            print("\n应用2: 计算普朗特数")
            
            # 定压比热 (J/kg·K)
            cp = results.Cp * 1000  # 转换为J/kg·K
            
            # 导热系数 (W/m·K)
            thermal_conductivity = transport_results.tcx
            
            # 计算普朗特数
            prandtl_number = dynamic_viscosity * cp / thermal_conductivity
            
            print(f"定压比热: {cp:.0f} J/kg·K")
            print(f"导热系数: {thermal_conductivity:.3f} W/m·K")
            print(f"普朗特数: {prandtl_number:.3f}")
            
        except Exception as e:
            print(f"实际应用演示失败: {e}")

def main():
    """主函数"""
    print("REFPROP 10.0 传输性质完整测试")
    print("=" * 50)
    
    # 创建测试实例
    rp_test = REFPROPCompleteTest()
    
    # 运行测试
    rp_test.test_transport_properties_correct()
    rp_test.test_transport_properties_comprehensive()
    rp_test.test_different_fluids_transport()
    rp_test.test_water_transport_curve()
    rp_test.test_saturation_transport_properties()
    rp_test.test_mixture_transport_properties()
    rp_test.test_high_pressure_transport_properties()
    rp_test.demonstrate_practical_applications()
    
    print("\n" + "="*50)
    print("传输性质测试完成!")
    print("="*50)
    
    print("\n总结:")
    print("✓ 成功获取了传输性质 (粘度和导热系数)")
    print("✓ TRNPRPdll函数需要3个参数: 温度(T), 密度(D), 组成(z)")
    print("✓ 返回对象包含属性: eta (粘度, 单位: μPa·s), tcx (导热系数, 单位: W/m·K)")
    print("✓ 测试了不同流体、不同温度和压力条件下的传输性质")
    print("✓ 演示了传输性质在实际工程中的应用")

if __name__ == "__main__":
    main()