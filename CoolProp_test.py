import CoolProp.CoolProp as CP

def calculate_supercritical_methane_coolprop(T=700, P=8000):
    """
    使用CoolProp计算超临界甲烷物性
    CoolProp无架构问题，且支持更高温度范围
    """
    try:
        # 将压力转换为Pa（CoolProp使用SI单位）
        P_Pa = P * 1000
        
        print(f"使用CoolProp计算超临界甲烷物性: T={T}K, P={P}kPa")
        
        # 计算各项物性
        props = {
            '密度_kg_m3': CP.PropsSI('D', 'T', T, 'P', P_Pa, 'Methane'),
            '焓_kJ_kg': CP.PropsSI('H', 'T', T, 'P', P_Pa, 'Methane') / 1000,  # J/kg -> kJ/kg
            '熵_kJ_kgK': CP.PropsSI('S', 'T', T, 'P', P_Pa, 'Methane') / 1000, # J/kg/K -> kJ/kg/K
            'Cp_kJ_kgK': CP.PropsSI('C', 'T', T, 'P', P_Pa, 'Methane') / 1000,  # 定压比热
            'Cv_kJ_kgK': CP.PropsSI('O', 'T', T, 'P', P_Pa, 'Methane') / 1000,  # 定容比热
            '音速_m_s': CP.PropsSI('A', 'T', T, 'P', P_Pa, 'Methane'),           # 音速
            '粘度_μPa_s': CP.PropsSI('V', 'T', T, 'P', P_Pa, 'Methane') * 1e6,  # Pa·s -> μPa·s
            '热导率_W_mK': CP.PropsSI('L', 'T', T, 'P', P_Pa, 'Methane'),        # 热导率
            '普朗特数': CP.PropsSI('Prandtl', 'T', T, 'P', P_Pa, 'Methane')     # 普朗特数
        }
        
        print("CoolProp计算的超临界甲烷物性:")
        for key, value in props.items():
            print(f"  {key}: {value:.4f}")
        
        return props
        
    except Exception as e:
        print(f"CoolProp计算错误: {e}")
        return None

# 安装CoolProp: pip install CoolProp