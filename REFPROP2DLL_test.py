import os
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary

def calculate_supercritical_methane(T=700, P=8000):
    """
    计算超临界甲烷的物性
    T: 温度 (K) - 超临界约700K
    P: 压力 (kPa) - 建议 > 4600 kPa (超临界压力)
    """
    try:
        # 初始化REFPROP - 传入DLL路径
        rp = REFPROPFunctionLibrary(r"C:\Program Files (x86)\REFPROP")
        rp.SETUPdll(1, "METHANE.FLD", "HMX.BNC", "DEF")  # 注意使用.FLD后缀
        
        print(f"计算超临界甲烷物性: T={T}K, P={P}kPa")
        
        # 方法1: 使用TP模式，iFlag=1强制单相计算
        result = rp.REFPROP2dll(
            'METHANE',          # 物质
            'TP',               # 输入模式: 温度+压力
            'D;H;S;CP;CV;W;VIS;TCX;PRANDTL',  # 输出物性
            1,                  # iMass=1: 质量单位
            1,                  # iFlag=1: 强制单相计算
            T,                  # 温度
            P,                  # 压力
            [1.0]               # 组成
        )
        
        if result.ierr == 0:
            # 解析结果
            props = {
                '密度_kg_m3': result.Output[0],
                '焓_kJ_kg': result.Output[1],
                '熵_kJ_kgK': result.Output[2],
                'Cp_kJ_kgK': result.Output[3],
                'Cv_kJ_kgK': result.Output[4],
                '音速_m_s': result.Output[5],
                '粘度_μPa_s': result.Output[6],
                '热导率_W_mK': result.Output[7],
                '普朗特数': result.Output[8]
            }
            
            print("超临界甲烷物性:")
            for key, value in props.items():
                print(f"  {key}: {value:.4f}")
            
            return props
            
        else:
            print(f"计算失败: {result.herr}")
            return None
            
    except Exception as e:
        print(f"错误: {e}")
        return None


def calculate_by_density(T=700, D=100):
    """
    通过温度和密度计算（避免压力输入问题）
    D: 密度 (kg/m³)
    """
    try:
        rp = REFPROPFunctionLibrary(r"C:\Program Files (x86)\REFPROP")
        rp.SETUPdll(1, "METHANE.FLD", "HMX.BNC", "DEF")
        
        # 使用TD模式（温度+密度）
        result = rp.REFPROP2dll(
            'METHANE',
            'TD',               # 输入模式: 温度+密度
            'P;H;S;CP;CV;W;VIS;TCX;PRANDTL',
            1, 0,               # iMass=1, iFlag=0
            T, D,               # 温度, 密度
            [1.0]
        )
        
        if result.ierr == 0:
            return {
                '压力_kPa': result.Output[0],
                '焓_kJ_kg': result.Output[1],
                '熵_kJ_kgK': result.Output[2],
                'Cp_kJ_kgK': result.Output[3],
                'Cv_kJ_kgK': result.Output[4],
                '音速_m_s': result.Output[5],
                '粘度_μPa_s': result.Output[6],
                '热导率_W_mK': result.Output[7],
                '普朗特数': result.Output[8]
            }
        else:
            print(f"TD计算失败: {result.herr}")
            return None
            
    except Exception as e:
        print(f"错误: {e}")
        return None


# 改进版本：更健壮的处理方式
def calculate_supercritical_methane_improved(T=700, P=8000):
    """
    改进版本：包含更多错误处理和兼容性考虑
    """
    try:
        # 检查DLL文件是否存在
        dll_path = r"C:\Program Files (x86)\REFPROP\REFPROP.DLL"
        if not os.path.exists(dll_path):
            print(f"REFPROP DLL未找到: {dll_path}")
            return None
            
        # 初始化REFPROP
        rp = REFPROPFunctionLibrary(dll_path)
        
        # 尝试不同的物质名称格式
        fluid_names = ['METHANE.FLD', 'METHANE', 'CH4.FLD']
        setup_success = False
        
        for fluid_name in fluid_names:
            try:
                rp.SETUPdll((1, fluid_names, "HMX.BNC", "DEF"))
                setup_success = True
                print(f"成功加载物质文件: {fluid_name}")
                break
            except:
                continue
                
        if not setup_success:
            print("无法加载甲烷物质文件")
            return None
        
        print(f"计算超临界甲烷物性: T={T}K, P={P}kPa")
        
        # 执行计算
        result = rp.REFPROP2dll(
            'METHANE', 'TP', 'D;H;S;CP;CV;W;VIS;TCX;PRANDTL',
            1, 1, T, P, [1.0]
        )
        
        if result.ierr == 0:
            props = {
                '密度_kg_m3': result.Output[0],
                '焓_kJ_kg': result.Output[1],
                '熵_kJ_kgK': result.Output[2],
                'Cp_kJ_kgK': result.Output[3],
                'Cv_kJ_kgK': result.Output[4],
                '音速_m_s': result.Output[5],
                '粘度_μPa_s': result.Output[6],
                '热导率_W_mK': result.Output[7],
                '普朗特数': result.Output[8]
            }
            
            print("超临界甲烷物性:")
            for key, value in props.items():
                print(f"  {key}: {value:.4f}")
            
            return props
        else:
            print(f"REFPROP计算错误 {result.ierr}: {result.herr}")
            return None
            
    except Exception as e:
        print(f"程序错误: {e}")
        return None


# 使用示例
if __name__ == "__main__":
    # 使用改进版本
    props = calculate_supercritical_methane_improved(T=700, P=8000)
    
    if props is None:
        print("\n尝试通过密度计算...")
        # 这里需要先创建实例
        dll_path = r"C:\Program Files (x86)\REFPROP"
        if os.path.exists(dll_path):
            rp = REFPROPFunctionLibrary(dll_path)
            rp.SETUPdll(1, "METHANE.FLD", "HMX.BNC", "DEF")
            # 通过密度计算
            props2 = calculate_by_density(T=700, D=100)