import json
import os
import sys
import numpy as np
from scipy.optimize import fsolve

# 添加当前目录到路径以便导入模块
sys.path.append('.')

try:
    from MethaneEngineHeatTransfer import LOX_MethaneEngineHeatTransfer, CEACalculator, REFPROPFluid, EngineGeometry
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保MethaneEngineHeatTransfer.py在当前目录中")
    sys.exit(1)

# 尝试导入pandas用于Excel输出
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    print("警告: 未找到pandas库，将无法生成Excel文件")
    HAS_PANDAS = False

def load_geometry_points(geometry_file):
    """从几何文件加载数据点"""
    points = []
    try:
        with open(geometry_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        data_section = False
        for line in lines:
            line = line.strip()
            if line.startswith('#------') or line.startswith('# ='):
                data_section = True
                continue
            if data_section and line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x_mm = float(parts[0])  # 轴向位置 (mm)
                        r_mm = float(parts[1])  # 半径 (mm)
                        points.append((x_mm, r_mm))
                    except ValueError:
                        continue
        return points
    except Exception as e:
        print(f"加载几何文件错误: {e}")
        return []

def calculate_mach_number_from_area_ratio(eps, gamma):
    """从面积比计算马赫数（等熵流动）"""
    def equation(M):
        return (1/M) * ((2/(gamma+1)) * (1 + (gamma-1)/2 * M**2))**((gamma+1)/(2*(gamma-1))) - eps
    
    # 初始猜测，根据eps选择亚音速或超音速解
    if eps >= 1:
        M_guess = 0.1  # 亚音速
    else:
        M_guess = 2.0  # 超音速
    
    try:
        M = fsolve(equation, M_guess)[0]
        return max(M, 0.01)  # 避免除零
    except:
        return  1.0  # 默认值

def create_excel_output(data_rows, metadata, output_file):
    """创建Excel格式的输出文件"""
    if not HAS_PANDAS:
        print(f"无法创建Excel文件 {output_file}，请安装pandas库")
        return False
    
    try:
        # 创建DataFrame
        df = pd.DataFrame(data_rows)
        
        # 创建Excel写入器
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 写入数据工作表
            df.to_excel(writer, sheet_name='燃气属性数据', index=False)
            
            # 创建元数据工作表
            metadata_df = pd.DataFrame({
                '参数': list(metadata.keys()),
                '值': list(metadata.values())
            })
            metadata_df.to_excel(writer, sheet_name='计算参数', index=False)
            
            # 获取工作表对象进行格式设置
            workbook = writer.book
            worksheet_data = writer.sheets['燃气属性数据']
            worksheet_meta = writer.sheets['计算参数']
            
            # 设置列宽
            for column in worksheet_data.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 20)
                worksheet_data.column_dimensions[column_letter].width = adjusted_width
            
            for column in worksheet_meta.columns:
                column_letter = column[0].column_letter
                worksheet_meta.column_dimensions[column_letter].width = 20
            
            # 添加冻结窗格
            worksheet_data.freeze_panes = 'A2'
            
        print(f"Excel文件已创建: {output_file}")
        return True
        
    except Exception as e:
        print(f"创建Excel文件错误: {e}")
        return False

def main():
    # 参数配置
    Pc_MPa = 5.0  # 燃烧室压力 5 MPa
    mixture_ratio = 3.0  # 混合比 O/F=3
    geometry_file = "engine_shape.txt"  # 几何文件
    output_txt_file = "CEA_test_result.out"  # 文本输出文件
    output_excel_file = "CEA_test_result.xlsx"  # Excel输出文件
    
    # 加载参数文件
    try:
        with open('parameters.json', 'r', encoding='utf-8') as f:
            params = json.load(f)
        geometry_file = params.get('file_paths', {}).get('engine_shape_file', geometry_file)
    except:
        print("使用默认参数")
    
    # 初始化计算器
    cea_calculator = CEACalculator()
    fluid_props = REFPROPFluid(cea_calculator=cea_calculator)
    geometry_loader = EngineGeometry()
    
    # 加载几何数据
    points = load_geometry_points(geometry_file)
    if not points:
        print(f"无法加载几何文件: {geometry_file}")
        return
    
    print(f"加载几何点数量: {len(points)}")
    
    # 找到喉部（最小半径）
    throat_point = min(points, key=lambda p: p[1])
    r_throat_mm = throat_point[1]
    A_throat = np.pi * (r_throat_mm / 1000)**2  # 转换为平方米
    print(f"喉部半径: {r_throat_mm} mm, 喉部面积: {A_throat:.6f} m²")
    
    # 获取燃烧室条件（滞止条件）
    try:
        chamber_props = cea_calculator.get_combustion_properties(Pc_MPa, mixture_ratio)
        T0 = chamber_props['temperature']  # 滞止温度
        P0 = Pc_MPa * 1e6  # 滞止压力 (Pa)
        gamma = chamber_props['gamma']  # 比热比
        MolWt = chamber_props['molecular_weight']  # 分子量
        
        print(f"燃烧室条件:")
        print(f"滞止温度 T0: {T0:.2f} K")
        print(f"滞止压力 P0: {P0/1e6:.2f} MPa") 
        print(f"比热比 γ: {gamma:.3f}")
        print(f"分子量: {MolWt:.3f} g/mol")
    except Exception as e:
        print(f"CEA计算错误: {e}")
        return
    
    # 准备数据存储
    data_rows = []  # 用于Excel输出的数据行
    metadata = {    # 元数据
        '燃烧室压力 (MPa)': Pc_MPa,
        '混合比 (O/F)': mixture_ratio,
        '几何文件': geometry_file,
        '数据点数量': len(points),
        '喉部半径 (mm)': r_throat_mm,
        '喉部面积 (m²)': A_throat,
        '滞止温度 (K)': T0,
        '滞止压力 (MPa)': P0/1e6,
        '比热比': gamma,
        '分子量 (g/mol)': MolWt
    }
    
    # 打开文本输出文件
    try:
        with open(output_txt_file, 'w', encoding='utf-8') as f:
            # 写入文件头信息
            f.write("# CEA Test Results - LOX/CH4 Engine Gas Properties\n")
            f.write("# =================================================\n")
            f.write(f"# Chamber Pressure: {Pc_MPa} MPa\n")
            f.write(f"# Mixture Ratio (O/F): {mixture_ratio}\n")
            f.write(f"# Geometry File: {geometry_file}\n")
            f.write(f"# Number of Points: {len(points)}\n")
            f.write(f"# Throat Radius: {r_throat_mm} mm\n")
            f.write(f"# Throat Area: {A_throat:.6f} m²\n")
            f.write(f"# Stagnation Temperature: {T0:.2f} K\n")
            f.write(f"# Stagnation Pressure: {P0/1e6:.2f} MPa\n")
            f.write(f"# Specific Heat Ratio: {gamma:.3f}\n")
            f.write(f"# Molecular Weight: {MolWt:.3f} g/mol\n")
            f.write("#\n")
            f.write("# Columns:\n")
            f.write("# 1. Position (mm) - Axial position from injector\n")
            f.write("# 2. Radius (mm) - Local radius\n")
            f.write("# 3. Area_ratio - Local area / throat area\n")
            f.write("# 4. Mach - Local Mach number\n")
            f.write("# 5. Pressure (MPa) - Local static pressure\n")
            f.write("# 6. Temperature (K) - Local static temperature\n")
            f.write("# 7. Density (kg/m³) - Gas density\n")
            f.write("# 8. Speed_of_sound (m/s) - Local speed of sound\n")
            f.write("# 9. Velocity (m/s) - Gas velocity\n")
            f.write("# 10. Gamma - Specific heat ratio\n")
            f.write("# 11. Gas_constant (J/kg-K) - Specific gas constant\n")
            f.write("# 12. Molecular_weight (g/mol) - Molecular weight\n")
            f.write("# 13. Specific_heat_cp (J/kg-K) - Constant pressure specific heat\n")
            f.write("# 14. Specific_heat_cv (J/kg-K) - Constant volume specific heat\n")
            f.write("# 15. Viscosity (Pa-s) - Dynamic viscosity\n")
            f.write("# 16. Conductivity (W/m-K) - Thermal conductivity\n")
            f.write("# 17. Prandtl - Prandtl number\n")
            f.write("#\n")
            
            # 写入表头
            headers = [
                "Position(mm)", "Radius(mm)", "Area_ratio", "Mach", "Pressure(MPa)", "Temperature(K)",
                "Density(kg/m³)", "Speed_of_sound(m/s)", "Velocity(m/s)", "Gamma", 
                "Gas_constant(J/kg-K)", "Molecular_weight(g/mol)", "Specific_heat_cp(J/kg-K)",
                "Specific_heat_cv(J/kg-K)", "Viscosity(Pa-s)", "Conductivity(W/m-K)", "Prandtl"
            ]
            f.write("\t".join(headers) + "\n")
            
            # 计算每个点的燃气属性
            for i, (x_mm, r_mm) in enumerate(points):
                # 计算当地几何
                r_local_m = r_mm / 1000  # 半径 (m)
                A_local = np.pi * r_local_m**2  # 当地面积 (m²)
                eps = A_local / A_throat  # 面积比
                
                # 计算马赫数
                M = calculate_mach_number_from_area_ratio(eps, gamma)
                
                # 等熵关系计算当地条件
                T_ratio = 1 / (1 + (gamma-1)/2 * M**2)
                P_ratio = T_ratio**(gamma/(gamma-1))
                
                T_local = T0 * T_ratio  # 当地温度
                P_local = P0 * P_ratio  # 当地压力
                
                # 计算燃气物性
                try:
                    gas_props = fluid_props.get_combustion_gas_properties(T_local, P_local, mixture_ratio, Pc_MPa, eps)
                    
                    density = gas_props['density']
                    viscosity = gas_props['viscosity']
                    conductivity = gas_props['conductivity']
                    specific_heat = gas_props['specific_heat']
                    prandtl = gas_props['prandtl']
                    speed_of_sound = gas_props['speed_of_sound']
                    gas_constant = gas_props.get('gas_constant', 8314.46/MolWt)
                    
                    # 计算其他参数
                    velocity = M * speed_of_sound
                    specific_heat_cv = specific_heat / gamma  # cp/gamma = cv
                    
                except Exception as e:
                    print(f"点 {i} 物性计算错误: {e}")
                    continue
                
                # 格式化结果
                results = [
                    f"{x_mm:.3f}", f"{r_mm:.3f}", f"{eps:.6f}", f"{M:.6f}",
                    f"{P_local/1e6:.6f}", f"{T_local:.2f}", f"{density:.6f}",
                    f"{speed_of_sound:.2f}", f"{velocity:.2f}", f"{gamma:.6f}",
                    f"{gas_constant:.6f}", f"{MolWt:.6f}", f"{specific_heat:.6f}",
                    f"{specific_heat_cv:.6f}", f"{viscosity:.6e}", f"{conductivity:.6e}",
                    f"{prandtl:.6f}"
                ]
                
                # 写入文本文件
                f.write("\t".join(results) + "\n")
                
                # 准备Excel数据行
                data_row = {
                    'Position_mm': x_mm,
                    'Radius_mm': r_mm,
                    'Area_ratio': eps,
                    'Mach': M,
                    'Pressure_MPa': P_local/1e6,
                    'Temperature_K': T_local,
                    'Density_kg_m3': density,
                    'Speed_of_sound_m_s': speed_of_sound,
                    'Velocity_m_s': velocity,
                    'Gamma': gamma,
                    'Gas_constant_J_kg_K': gas_constant,
                    'Molecular_weight_g_mol': MolWt,
                    'Specific_heat_cp_J_kg_K': specific_heat,
                    'Specific_heat_cv_J_kg_K': specific_heat_cv,
                    'Viscosity_Pa_s': viscosity,
                    'Conductivity_W_m_K': conductivity,
                    'Prandtl': prandtl
                }
                data_rows.append(data_row)
                
                # 同时在控制台显示进度
                if i % 10 == 0:  # 每10个点显示一次进度
                    print(f"已完成 {i}/{len(points)} 个点的计算")
            
            f.write(f"#\n# Calculation completed successfully\n")
            f.write(f"# Total points processed: {len(points)}\n")
        
        print(f"\n文本文件已保存: {output_txt_file}")
        print(f"共处理 {len(points)} 个数据点")
        
        # 创建Excel文件
        if data_rows:
            create_excel_output(data_rows, metadata, output_excel_file)
        else:
            print("无有效数据，跳过Excel文件创建")
        
    except Exception as e:
        print(f"写入输出文件错误: {e}")
        return

if __name__ == "__main__":
    main()