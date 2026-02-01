# 导入必要的类
from MethaneEngineHeatTransfer import REFPROPFluid

# 创建REFPROPFluid实例（不需要CEA计算器）
fluid_calculator = REFPROPFluid(refprop_path=None, cea_calculator=None)

# 设置条件：150K，6MPa（注意单位转换：6MPa = 6e6 Pa）
T = 150  # 温度 [K]
P = 6e6  # 压力 [Pa] (6MPa)

# 获取甲烷物性
methane_props = fluid_calculator.get_methane_properties(T, P)

# 提取比热
specific_heat = methane_props['specific_heat']

print(f"甲烷在{T}K、{P/1e6}MPa下的比热为: {specific_heat:.2f} J/(kg·K)")