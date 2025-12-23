import os
import numpy as np
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary

class MethaneTwoPhaseCalculator:
    def __init__(self, refprop_path=None):
        """
        初始化 REFPROP 计算器
        
        参数:
            refprop_path: REFPROP 安装路径
        """
        if refprop_path is None:
            # 常见的 REFPROP 安装路径
            default_paths = [
                r'C:\Program Files (x86)\REFPROP',
                r'C:\Program Files\REFPROP',
                r'/usr/local/REFPROP',
                r'C:\REFPROP'
            ]
            for path in default_paths:
                if os.path.exists(path):
                    refprop_path = path
                    break
        
        if refprop_path is None or not os.path.exists(refprop_path):
            raise FileNotFoundError(f"未找到 REFPROP 安装路径，请手动指定。尝试过的路径: {default_paths}")
        
        # 初始化 REFPROP
        self.rp = REFPROPFunctionLibrary(refprop_path)
        
        # 新版本的 SETUPdll 只需要 5 个参数
        # 参数说明: 物质数量, 物质名称, 热力学模型, 参考状态, 错误信息单位
        self.rp.SETUPdll(1, 'METHANE', "HMX.BNC", "DEF")
        
        # 或者使用以下格式（如果上面的不工作）:
        # self.rp.SETUPdll(1, 'METHANE', 'HMX.BNC', 'DEF', 0)
        
        print(f"REFPROP 版本: {self.rp.RPVersion()}")
        print("甲烷物性计算器初始化完成")
    
    def get_phase_properties_TQ(self, T, Q, properties='P;D;H;S;VIS;TCX;CP;CV;W'):
        """
        通过温度和干度计算物性
        
        参数:
            T: 温度 (K)
            Q: 干度 (0=饱和液体, 1=饱和蒸气, 0-1之间为两相混合物)
            properties: 要计算的物性字符串，用分号分隔
        
        返回:
            物性值列表
        """
        result = self.rp.REFPROP2dll('METHANE', 'TQ', properties, 0, 0, T, Q, [1.0])
        
        if result.ierr != 0:
            raise ValueError(f"计算失败 (错误码: {result.ierr}): {result.herr}")
        
        return result.Output
    
    def get_phase_properties_TP(self, T, P, properties='D;H;S;QMASS;VIS;TCX;CP;CV;W'):
        """
        通过温度和压力计算物性（自动判断相态）
        
        参数:
            T: 温度 (K)
            P: 压力 (kPa)
            properties: 要计算的物性字符串，用分号分隔
        
        返回:
            物性值列表
        """
        result = self.rp.REFPROP2dll('METHANE', 'TP', properties, 0, 0, T, P, [1.0])
        
        if result.ierr != 0:
            raise ValueError(f"计算失败 (错误码: {result.ierr}): {result.herr}")
        
        return result.Output
    
    def calculate_methane_at_110K(self):
        """
        计算甲烷在110K时的各种物性
        """
        print("\n" + "="*60)
        print("甲烷在110K时的物性计算")
        print("="*60)
        
        T = 110.0  # 温度 110K
        
        try:
            # 1. 计算饱和压力
            props = self.get_phase_properties_TQ(T, 0, 'P')
            P_sat = props[0]
            print(f"1. 110K时的饱和压力: {P_sat:.2f} kPa")
            
            # 2. 计算饱和液体的性质
            print("\n2. 饱和液体的性质 (Q=0):")
            liq_props = self.get_phase_properties_TQ(T, 0, 'P;D;H;S;VIS;TCX;CP;CV;W')
            property_names = ['压力', '密度', '焓', '熵', '粘度', '热导率', '定压比热', '定容比热', '音速']
            units = ['kPa', 'kg/m³', 'kJ/kg', 'kJ/kg·K', 'μPa·s', 'W/m·K', 'kJ/kg·K', 'kJ/kg·K', 'm/s']
            
            # 处理比热单位：REFPROP 输出的 CP 可能为 J/mol·K 或 kJ/kg·K 等
            # 对甲烷使用摩尔质量进行转换（16.04 g/mol）以得到 J/kg·K
            MOLWT = 16.04  # g/mol
            processed = []
            for i, (name, unit, value) in enumerate(zip(property_names, units, liq_props)):
                if name == '定压比热' or name == '定容比热':
                    try:
                        raw_cp = float(value)
                        # 假设 REFPROP 返回的 raw_cp 单位为 J/mol·K（常见），转换为 J/kg·K:
                        cp_j_per_kgk = raw_cp * 1000.0 / MOLWT
                        # 同时以 kJ/kg·K 显示
                        out_val = cp_j_per_kgk / 1000.0
                        out_unit = 'kJ/kg·K'
                        processed.append((name, out_val, out_unit, cp_j_per_kgk))
                    except Exception:
                        processed.append((name, value, unit, None))
                else:
                    processed.append((name, value, unit, None))

            for name, out_val, out_unit, _ in processed:
                if out_unit is None:
                    # find original index to print raw
                    idx = property_names.index(name)
                    print(f"   {name}: {liq_props[idx]:.4f} {units[idx]}")
                else:
                    print(f"   {name}: {out_val:.4f} {out_unit}")
            
            # 3. 计算饱和蒸气的性质
            print("\n3. 饱和蒸气的性质 (Q=1):")
            vap_props = self.get_phase_properties_TQ(T, 1, 'P;D;H;S;VIS;TCX;CP;CV;W')
            
            # vapor properties: 同样转换比热
            processed_v = []
            for i, (name, unit, value) in enumerate(zip(property_names, units, vap_props)):
                if name == '定压比热' or name == '定容比热':
                    try:
                        raw_cp = float(value)
                        cp_j_per_kgk = raw_cp * 1000.0 / MOLWT
                        out_val = cp_j_per_kgk / 1000.0
                        out_unit = 'kJ/kg·K'
                        processed_v.append((name, out_val, out_unit, cp_j_per_kgk))
                    except Exception:
                        processed_v.append((name, value, unit, None))
                else:
                    processed_v.append((name, value, unit, None))

            for name, out_val, out_unit, _ in processed_v:
                if out_unit is None:
                    idx = property_names.index(name)
                    print(f"   {name}: {vap_props[idx]:.4f} {units[idx]}")
                else:
                    print(f"   {name}: {out_val:.4f} {out_unit}")
            
            # 4. 计算不同干度下的两相混合物性质
            print("\n4. 不同干度下的两相混合物性质 (110K):")
            print("-"*80)
            print(f"{'干度(Q)':<10} {'压力(kPa)':<12} {'密度(kg/m³)':<15} {'焓(kJ/kg)':<12} {'熵(kJ/kg·K)':<12}")
            print("-"*80)
            
            qualities = [0.0, 0.25, 0.5, 0.75, 1.0]
            for Q in qualities:
                try:
                    props = self.get_phase_properties_TQ(T, Q, 'P;D;H;S')
                    print(f"{Q:<10.2f} {props[0]:<12.2f} {props[1]:<15.2f} "
                          f"{props[2]:<12.2f} {props[3]:<12.2f}")
                except Exception as e:
                    print(f"干度 Q={Q} 计算失败: {e}")
            
            return liq_props, vap_props
            
        except Exception as e:
            print(f"计算失败: {e}")
            return None, None
    
    def flash_calculation_examples(self):
        """
        闪蒸计算示例
        """
        print("\n" + "="*60)
        print("闪蒸计算示例")
        print("="*60)
        
        # 示例1: 已知T和P，求物性
        print("\n示例1: T=110K, P=100kPa 时的状态")
        try:
            props = self.get_phase_properties_TP(110, 100, 'D;H;S;QMASS;VIS;TCX;CP;CV;W')
            print(f"  干度: {props[3]:.4f}")
            print(f"  密度: {props[0]:.2f} kg/m³")
            print(f"  焓: {props[1]:.2f} kJ/kg")
            print(f"  音速: {props[8]:.2f} m/s")
        except Exception as e:
            print(f"  计算失败: {e}")
        
        # 示例2: 已知T和H，求P和Q
        print("\n示例2: T=110K, H=500 kJ/kg 时的状态")
        try:
            result = self.rp.REFPROP2dll('METHANE', 'TH', 'P;QMASS;D;S', 0, 0, 110, 500, [1.0])
            if result.ierr == 0:
                print(f"  压力: {result.Output[0]:.2f} kPa")
                print(f"  干度: {result.Output[1]:.4f}")
                print(f"  密度: {result.Output[2]:.2f} kg/m³")
            else:
                print(f"  计算失败: {result.herr}")
        except Exception as e:
            print(f"  计算失败: {e}")
    
    def get_saturation_table(self, T_min=100, T_max=190, n_points=10):
        """
        生成饱和表
        """
        temperatures = np.linspace(T_min, T_max, n_points)
        
        print("\n" + "="*60)
        print(f"甲烷饱和表 ({T_min}K 到 {T_max}K)")
        print("="*60)
        print(f"{'温度(K)':<10} {'饱和压力(kPa)':<15} {'密度_液(kg/m³)':<15} {'密度_气(kg/m³)':<15} {'焓_液(kJ/kg)':<15} {'焓_气(kJ/kg)':<15}")
        print("-"*85)
        
        for T in temperatures:
            try:
                # 计算饱和液体
                liq = self.get_phase_properties_TQ(T, 0, 'P;D;H')
                # 计算饱和蒸气
                vap = self.get_phase_properties_TQ(T, 1, 'P;D;H')
                
                print(f"{T:<10.2f} {liq[0]:<15.2f} {liq[1]:<15.2f} {vap[1]:<15.2f} {liq[2]:<15.2f} {vap[2]:<15.2f}")
            except Exception as e:
                print(f"{T:<10.2f} 计算失败")


def simple_example():
    """
    简化示例，直接调用
    """
    try:
        # 请修改为您的 REFPROP 安装路径
        REFPROP_PATH = r'C:\Program Files (x86)\REFPROP'
        
        # 创建计算器实例
        calc = MethaneTwoPhaseCalculator(REFPROP_PATH)
        
        # 计算110K时的物性
        calc.calculate_methane_at_110K()
        
        # 闪蒸计算示例
        calc.flash_calculation_examples()
        
        # 生成饱和表
        calc.get_saturation_table(100, 150, 6)
        
    except Exception as e:
        print(f"错误: {e}")
        print("\n可能的原因:")
        print("1. REFPROP 未安装或路径不正确")
        print("2. ctREFPROP 库未正确安装")
        print("3. 版本不兼容")
        print("\n解决方法:")
        print("1. 确认 REFPROP 已安装，并修改代码中的 REFPROP_PATH")
        print("2. 安装 ctREFPROP: pip install ctREFPROP")
        print("3. 查看 REFPROP 的安装说明")


def quick_calculation():
    """
    快速计算示例
    """
    try:
        # 设置 REFPROP 路径
        refprop_path = r'C:\Program Files (x86)\REFPROP'
        
        # 初始化
        rp = REFPROPFunctionLibrary(refprop_path)
        rp.SETUPdll(1, 'METHANE', "HMX.BNC", "DEF")
        
        print("快速计算甲烷在110K时的饱和压力:")
        
        # 计算饱和压力
        result = rp.REFPROP2dll('METHANE', 'TQ', 'P', 0, 0, 110, 0, [1.0])
        if result.ierr == 0:
            P_sat = result.Output[0]
            print(f"饱和压力: {P_sat:.2f} kPa")
            
            # 计算饱和液体性质
            result = rp.REFPROP2dll('METHANE', 'TQ', 'D;H;S;VIS;TCX;CP;W', 0, 0, 110, 0, [1.0])
            if result.ierr == 0:
                print(f"\n饱和液体性质:")
                print(f"密度: {result.Output[0]:.2f} kg/m³")
                print(f"焓: {result.Output[1]:.2f} kJ/kg")
                print(f"熵: {result.Output[2]:.2f} kJ/kg·K")
                print(f"粘度: {result.Output[3]:.2f} μPa·s")
                print(f"热导率: {result.Output[4]:.2f} W/m·K")
                print(f"定压比热: {result.Output[5]:.2f} kJ/kg·K")
                print(f"音速: {result.Output[6]:.2f} m/s")
            
            # 计算两相混合物 (干度=0.5)
            result = rp.REFPROP2dll('METHANE', 'TQ', 'D;H;S;Q', 0, 0, 110, 0.5, [1.0])
            if result.ierr == 0:
                print(f"\n两相混合物 (干度=0.5):")
                print(f"密度: {result.Output[0]:.2f} kg/m³")
                print(f"焓: {result.Output[1]:.2f} kJ/kg")
                print(f"熵: {result.Output[2]:.2f} kJ/kg·K")
                print(f"干度: {result.Output[3]:.2f}")
                
        else:
            print(f"计算失败: {result.herr}")
            
    except Exception as e:
        print(f"错误: {e}")
        print("\n请确保:")
        print("1. REFPROP 已正确安装")
        print("2. ctREFPROP 库已安装: pip install ctREFPROP")
        print("3. 代码中的路径正确")


if __name__ == "__main__":
    # 运行简化示例
    simple_example()
    
    # 或者运行快速计算
    # quick_calculation()