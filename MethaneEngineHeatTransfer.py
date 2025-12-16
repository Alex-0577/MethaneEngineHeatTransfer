"""
液氧/甲烷发动机再生冷却和膜冷却传热数值研究
集成CEA燃烧计算、REFPROP物性计算和真实发动机几何的传热分析程序

主要功能：
- 基于NASA CEA计算燃烧产物热力学性质
- 使用REFPROP精确计算流体物性参数
- 支持真实发动机几何数据加载和分析
- 实现再生冷却和膜冷却传热计算
- 提供轴向分布分析和性能评估
"""

import re
import math
import numpy as np
from scipy.optimize import fsolve
import os
import logging
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
from scipy.optimize import fsolve
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import functools
from functools import lru_cache


def setup_logging():
    """
    设置日志配置
    
    功能：
    - 创建文件和控制台日志处理器
    - 设置不同级别的日志输出
    - 配置日志格式和时间格式
    
    返回：
    Logger: 配置完成的日志记录器
    """
    # 创建logger
    logger = logging.getLogger("MethaneEngineHeatTransfer")
    logger.setLevel(logging.DEBUG)  # 设置最低级别为DEBUG，确保所有日志都被捕获
    
    # 清除已有的handler，避免重复
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建文件handler，用于写入所有日志到文件
    file_handler = logging.FileHandler('log.txt', mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别的日志
    
    # 创建控制台handler，只输出重要信息
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 控制台只显示INFO及以上级别的日志
    
    # 创建日志格式
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加handler到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_parameters(parameter_file='parameters.json'):
    """
    从JSON文件加载所有可配置参数
    
    参数:
    parameter_file: str - 参数文件路径
    
    返回:
    dict: 包含所有配置参数的字典
    """
    import json
    default_params = {
        "engine_geometry": {
            "d_throat": 0.0625,
            "L_chamber": 0.257,
            "delta_wall": 0.002,
            "number_of_fins": 36,
            "t_fin": 0.0008,
            "h_fin": 0.008,
            "delta_fin": 0.012
        },
        "combustion_analysis": {
            "Pc_MPa": 10.0,
            "mixture_ratios": [2.5, 3.0, 3.5, 4.0, 4.5],
            "eps_values": [10, 20, 30, 40, 50]
        },
        "material_properties": {
            "thermal_conductivity_points": [[300, 400.0], [800, 350.0]]
        },
        "operating_conditions": {
            "coolant_velocity": 12,
            "hydraulic_diameter": 0.008
        },
        "engine_conditions": {
            "T_gas": 3400.0,
            "P_chamber_MPa": 10.0,
            "c_star": 1750,
            "m_dot_coolant": 4.5,
            "T_coolant_in": 110.0,
            "P_coolant_in_MPa": 18.0,
            "mixture_ratio": 3.4
        },
        "calculation_settings": {
            "num_segments": 60,
            "max_allowable_temp": 1100.0
        },
        "file_paths": {
            "refprop_path": r"C:\Program Files (x86)\REFPROP",
            "engine_shape_file": "AE-1305.txt",
            "parameter_file": "parameters.json"
        },
        "safety_factors": {
            "temperature_safety_margin": 0.1,
            "pressure_safety_margin": 0.15
        }
    }
    
    try:
        with open(parameter_file, 'r', encoding='utf-8') as f:
            loaded_params = json.load(f)
            # 深度合并默认参数和加载的参数
            def deep_update(default, update):
                for key, value in update.items():
                    if isinstance(value, dict) and key in default and isinstance(default[key], dict):
                        deep_update(default[key], value)
                    else:
                        default[key] = value
                return default
            
            params = deep_update(default_params.copy(), loaded_params)
            logger.info(f"参数文件加载成功: {parameter_file}")
            return params
    except FileNotFoundError:
        logger.warning(f"参数文件 {parameter_file} 未找到，使用默认参数")
        return default_params
    except Exception as e:
        logger.error(f"参数文件加载失败: {e}，使用默认参数")
        return default_params


# 初始化日志
logger = setup_logging()
# 加载全局参数
params = load_parameters()


def setup_chinese_font():
    """
    配置matplotlib中文字体支持
    
    功能：
    - 自动检测系统中可用的中文字体
    - 确保数学符号和减号的正确显示
    - 设置全局字体参数
    - 解决中文显示和保存问题
    """
    try:
        # 重新组织字体优先级，确保数学符号支持
        font_families = [
            # 优先使用支持数学符号的字体
            'DejaVu Sans',           # 支持完整数学符号
            'Arial Unicode MS',      # 支持Unicode数学符号
            'STIXGeneral',           # 专业数学字体
            # 中文字体（确保包含基本数学符号）
            'SimHei', 
            'Microsoft YaHei', 
            'SimSun', 
            'KaiTi',
            # 跨平台备选
            'WenQuanYi Micro Hei',
            'Noto Sans CJK SC'
        ]
        
        # 设置字体家族
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = font_families
        
        # 关键设置：禁用Unicode减号，使用ASCII减号
        plt.rcParams['axes.unicode_minus'] = False
        
        # 额外设置：确保文本渲染使用支持数学符号的字体
        plt.rcParams['mathtext.fontset'] = 'stix'  # 使用STIX数学字体
        plt.rcParams['mathtext.default'] = 'regular'  # 默认数学字体样式
        
        logger.info("字体配置完成，已禁用Unicode减号，使用ASCII减号")
        
    except Exception as e:
        logger.warning(f"字体设置异常: {e}，使用默认设置")
        # 回退到最安全的设置
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['mathtext.fontset'] = 'cm'  # 使用Computer Modern数学字体


class CEACalculator:
    """
    NASA CEA计算器集成类
    
    功能：
    - 计算液氧/甲烷燃烧产物的热力学性质
    - 提供燃烧温度、比热比、特征速度等关键参数
    - 在CEA不可用时使用简化燃烧模型作为备选
    """
    
    def __init__(self, oxName='LOX', fuelName='CH4', cr = 6.25):
        """
        初始化CEA计算器
        
        参数:
        oxName: str - 氧化剂名称 (默认'LOX')
        fuelName: str - 燃料名称 (默认'CH4')
        cr: float - 燃烧室收缩比 (默认6.25)
        """
        self.oxName = oxName
        self.fuelName = fuelName
        self.cea_obj = None
        self.MPa_to_psi = 145.0377  # 1 MPa = 145.0377 psi
        self.psi_to_MPa = 1 / 145.0377
        self.cr = cr
        self.setup_cea()
    
    def setup_cea(self):
        """
        设置CEA计算环境
        
        功能：
        - 尝试加载rocketcea库
        - 配置推进剂组合
        - 处理初始化失败情况
        
        返回：
        bool: CEA环境是否成功初始化
        """
        try:
            from rocketcea.cea_obj import CEA_Obj
            self.cea_obj = CEA_Obj(oxName=self.oxName, fuelName=self.fuelName, fac_CR=self.cr)
            logger.info(f"CEA环境初始化成功 - 推进剂: {self.oxName}/{self.fuelName}，收缩比: {self.cr}")
            return True
        except Exception as e:
            logger.warning(f"CEA环境初始化失败: {e}，将使用简化燃烧模型")
            self.cea_obj = None
            return False
        
    def MPa_to_psi(self, pressure_MPa):
        """将MPa转换为psi"""
        return pressure_MPa * self.MPa_to_psi

    def psi_to_MPa(self, pressure_psi):
        """将psi转换为MPa"""
        return pressure_psi * self.psi_to_MPa
    
    def get_combustion_properties(self, Pc, MR, eps=20.0):
        """
        获取燃烧产物的综合热力学性质
        
        参数:
        Pc: float - 燃烧室压力 [MPa]
        MR: float - 混合比 (O/F)
        eps: float - 面积比 (默认20.0)
        
        返回:
        dict: 包含燃烧产物性质的字典，包括：
            - temperature: 燃烧温度 [K]
            - pressure: 燃烧室压力 [Pa]
            - specific_heat: 比热 [J/(kg·K)]
            - gamma: 比热比
            - molecular_weight: 分子量 [g/mol]
            - cstar: 特征速度 [m/s]
            - isp: 比冲 [s]
            - mixture_ratio: 混合比
        """
        if self.cea_obj is None:
            return self._get_simplified_combustion_properties(Pc, MR, eps)
        
        try:
            Pc_psi = Pc * self.MPa_to_psi  # 转换为psi供CEA使用

            PinjOverPcomb = 1.0 + 0.54 / self.cr**2.2

            PinjOverPcomb = self.cea_obj.get_Pinj_over_Pcomb( Pc=Pc_psi * PinjOverPcomb, MR=MR )

            Pc_cea_psi = Pc_psi * PinjOverPcomb

            # 基本性能参数
            Isp = float(self.cea_obj.get_Isp(Pc=Pc_cea_psi, MR=MR, eps=eps))
            Cstar = float(self.cea_obj.get_Cstar(Pc=Pc_cea_psi, MR=MR)) * 3.2808  # ft/s转换为m/s
            Tcomb = float(self.cea_obj.get_Tcomb(Pc=Pc_cea_psi, MR=MR)) / 1.8  # Rankine(?)转换为开尔文
            
            # 热力学性质
            molwt_gamma = self.cea_obj.get_Chamber_MolWt_gamma(Pc=Pc_cea_psi, MR=MR)
            MolWt = float(molwt_gamma[0])
            Gamma = float(molwt_gamma[1])
            
            # 计算其他热力学参数
            R_specific = 8314.46 / MolWt
            Cp_avg = Gamma * R_specific / (Gamma - 1)
            
            return {
                'temperature': Tcomb,      # 燃烧温度 [K]
                'pressure': Pc * 1e6,     # 燃烧室压力 [Pa]
                'specific_heat': Cp_avg,  # 比热 [J/(kg·K)]
                'gamma': Gamma,           # 比热比
                'molecular_weight': MolWt,# 分子量 [g/mol]
                'cstar': Cstar,           # 特征速度 [m/s]
                'isp': Isp,               # 比冲 [s]
                'mixture_ratio': MR       # 混合比
            }
            
        except Exception as e:
            logger.error(f"CEA燃烧计算失败: {e}，使用简化模型")
            return self._get_simplified_combustion_properties(Pc, MR, eps)
    
    def _get_simplified_combustion_properties(self, Pc, MR, eps):
        """
        当CEA不可用时使用的简化燃烧模型
        
        参数:
        Pc: float - 燃烧室压力 [MPa]
        MR: float - 混合比 (O/F)
        eps: float - 面积比
        
        返回:
        dict: 基于经验公式的简化燃烧产物性质
        """
        # 基于经验公式的简化计算
        T_comb = 3000 + 200 * (MR - 3.0)  # 燃烧温度经验公式
        gamma = 1.2 + 0.02 * (MR - 3.0)   # 比热比经验公式
        
        return {
            'temperature': T_comb,
            'pressure': Pc * 1e6,
            'specific_heat': 1500 + 100 * (MR - 3.0),
            'gamma': gamma,
            'molecular_weight': 20.0,
            'cstar': 1600 + 100 * (MR - 3.0),
            'isp': 300 + 20 * (MR - 3.0),
            'mixture_ratio': MR
        }


class EngineGeometry:
    """
    发动机几何形状处理类
    
    功能：
    - 从文件加载发动机几何数据
    - 提供轴向位置直径插值计算
    - 管理几何参数和验证
    """
    
    def __init__(self, shape_file_path=None):
        """
        初始化几何处理器
        
        参数:
        shape_file_path: str - 几何文件路径 (可选)
        """
        self.shape_data = None
        if shape_file_path:
            self.load_geometry_from_file(shape_file_path)
    
    def load_geometry_from_file(self, file_path):
        """
        从文件加载发动机几何数据
        
        参数:
        file_path: str - 几何数据文件路径
        
        功能：
        - 解析文件头部信息
        - 读取几何坐标点数据
        - 验证数据完整性
        """
        # 重置几何数据
        logger.info(f"开始加载几何文件: {file_path}")
        self.shape_data = {'points': [], 'header': {}}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            logger.info(f"文件总行数: {len(lines)}")
            
            # 预览前几行内容
            for i in range(min(5, len(lines))):
                logger.debug(f"文件预览行: {lines[i].strip()}")
            
            header_params = {}
            data_section_start = False
            points = []
            
            # 正则表达式匹配键值对（例如：Lc=169.83604）
            # 修改点：使用正则表达式匹配所有键值对，而不是仅第一个
            pattern = re.compile(r'(\w+)=\s*([\d.-]+)')
            
            for i, line in enumerate(lines):
                line = line.strip()
                logger.debug(f"第{i+1}行内容: {line}")
                
                # 检查数据段开始标记
                if line.startswith('#------') or line.startswith('# ='):
                    data_section_start = True
                    logger.debug(f"第{i+1}行检测到数据段分隔符")
                    continue
                
                # 解析头部参数（在数据段开始前）
                if not data_section_start and '=' in line:
                    # 修改点：使用findall匹配所有键值对，而不是搜索第一个
                    matches = pattern.findall(line)
                    if matches:
                        for key, value in matches:
                            try:
                                numeric_value = float(value)
                                header_params[key] = numeric_value
                                logger.debug(f"解析头部参数: {key} = {numeric_value}")
                            except ValueError as e:
                                logger.warning(f"忽略无效数值: {key}={value}, 错误: {e}")
                    else:
                        logger.debug(f"第{i+1}行未找到有效参数")
                
                # 解析数据点（在数据段开始后）
                if data_section_start and line and not line.startswith('#'):
                    # 原有数据解析逻辑保持不变
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            x = float(parts[0])
                            r = float(parts[1])
                            points.append((x, r))
                        except ValueError as e:
                            logger.warning(f"忽略无效数据行 {i+1}: {line}, 错误: {e}")
            
            # 存储结果
            self.shape_data['points'] = points
            self.shape_data['header'] = header_params
            
            # 验证关键参数是否存在
            expected_params = ['Lc', 'Rc', 'R1', 'R2', 'bc', 'Rt', 'Le', 'Re', 'c1', 'c2']
            for param in expected_params:
                if param not in header_params:
                    logger.warning(f"关键参数 {param} 未在头部信息中找到")
            
            logger.info(f"几何数据加载成功 - 有效点数量: {len(points)}")
            logger.info(f"- 头部信息: {header_params}")
            
            return True
            
        except Exception as e:
            logger.error(f"加载几何文件失败: {e}")
            return False
    
    def _validate_geometry_continuity(self):
        """验证几何数据的连续性"""
        if not self.shape_data or not self.shape_data['points']:
            return
            
        points = self.shape_data['points']
        x_values = [p[0] for p in points]
        
        # 检查x坐标是否单调递增
        if not all(x_values[i] <= x_values[i+1] for i in range(len(x_values)-1)):
            logger.warning("几何点x坐标非单调递增，可能影响插值精度")
        
        # 检查数据范围
        min_x, max_x = min(x_values), max(x_values)
        logger.info(f"几何数据范围: x=[{min_x:.1f}, {max_x:.1f}]mm")

    def validate_geometry_file(self, file_path):
        """
        验证几何文件格式和内容
        
        参数:
        file_path: str - 几何文件路径
        
        返回:
        dict: 验证结果，包含格式正确性、点数等信息
        """
        validation_result = {
            'file_exists': False,
            'format_correct': False,
            'point_count': 0,
            'header_info': {},
            'sample_points': []
        }
        
        if not os.path.exists(file_path):
            logger.error(f"几何文件不存在: {file_path}")
            return validation_result
        
        validation_result['file_exists'] = True
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            validation_result['total_lines'] = len(lines)
            
            # 分析文件结构
            header_lines = []
            data_lines = []
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith('#') or '=' in stripped:
                    header_lines.append(stripped)
                elif any(c.isdigit() for c in stripped):
                    data_lines.append(stripped)
            
            validation_result['header_lines'] = len(header_lines)
            validation_result['data_lines'] = len(data_lines)
            
            # 尝试解析数据点
            test_points = []
            for line in data_lines[:5]:  # 只测试前5行数据
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        r = float(parts[1])
                        test_points.append((x, r))
                    except ValueError:
                        pass
            
            validation_result['sample_points'] = test_points
            validation_result['point_count'] = len(test_points)
            validation_result['format_correct'] = len(test_points) > 0
            
            logger.info(f"几何文件验证: 存在={validation_result['file_exists']}, "
                    f"格式正确={validation_result['format_correct']}, "
                    f"样例点数={validation_result['point_count']}")
            
        except Exception as e:
            logger.error(f"几何文件验证失败: {e}")
        
        return validation_result

    def get_diameter_at_position(self, x_position_m):
        """
        获取指定轴向位置的直径
        
        参数:
        x_position_m: float - 轴向位置 [m]
        
        返回:
        float: 当地直径 [m]，如无法计算返回None
        """
        if not self.shape_data:
            return None
        
        x_position_mm = x_position_m * 1000  # 转换为毫米
        
        points = self.shape_data['points']
        if not points:
            return None
        
        # 找到最近的上下点进行线性插值
        for i in range(len(points) - 1):
            x1, r1 = points[i]
            x2, r2 = points[i + 1]
            
            if x1 <= x_position_mm <= x2:
                # 线性插值
                fraction = (x_position_mm - x1) / (x2 - x1) if x2 != x1 else 0
                radius_mm = r1 + fraction * (r2 - r1)
                diameter_m = 2 * radius_mm / 1000  # 转换为米
                return diameter_m
        
        # 如果超出范围，返回端点值
        if x_position_mm <= points[0][0]:
            return 2 * points[0][1] / 1000
        else:
            return 2 * points[-1][1] / 1000
    
    def get_total_length(self):
        """
        获取发动机总长度
        
        返回:
        float: 发动机总长度 [m]，如无法计算返回None
        """
        if not self.shape_data or not self.shape_data['points']:
            return None
        max_x = max(point[0] for point in self.shape_data['points'])
        return max_x / 1000  # 转换为米


class REFPROPFluid:
    """
    REFPROP流体物性计算类 - 增强版
    
    功能：
    - 集成CEA燃烧计算功能
    - 提供精确的流体物性参数计算
    - 支持高温高压条件下的物性计算
    - 包含温度范围检查和错误处理机制
    """
    
    def __init__(self, refprop_path=None, cea_calculator=None):
        """
        初始化REFPROP接口和CEA计算器
        
        参数:
        refprop_path: str - REFPROP安装路径
        cea_calculator: CEACalculator - CEA计算器实例
        """
        self.RP = None
        self.cea_calculator = cea_calculator
        self.load_refprop(refprop_path)

        self._cache_hits = 0
        self._cache_misses = 0

    @lru_cache(maxsize=100)
    def get_methane_properties_cached(self, T, P):
        """带缓存的甲烷物性计算"""
        return self.get_methane_properties(T, P)
    
    @lru_cache(maxsize=50)
    def get_combustion_gas_properties_cached(self, T, P, mixture_ratio):
        """带缓存的燃烧产物物性计算"""
        return self.get_combustion_gas_properties(T, P, mixture_ratio)
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
            'total_calls': total
        }
        
    def load_refprop(self, refprop_path):
        """
        加载REFPROP库
        
        参数:
        refprop_path: str - REFPROP安装路径
        
        功能：
        - 自动搜索常见安装位置
        - 验证库文件完整性
        - 设置工作流体和参数
        """
        try:
            # 如果未指定路径，尝试常见安装位置
            if refprop_path is None:
                possible_paths = [
                    r"C:\Program Files\REFPROP",
                    r"C:\Program Files (x86)\REFPROP",
                    r"/usr/local/REFPROP"  # Linux路径
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        refprop_path = path
                        break
            
            if refprop_path and os.path.exists(refprop_path):
                # 使用pyrefprop库
                self.RP = REFPROPFunctionLibrary(refprop_path)
                self.RP.SETPATHdll(refprop_path)
                logger.info(f"REFPROP加载成功: {refprop_path}")
                logger.info(f"REFPROP版本: {self.RP.RPVersion()}")
            else:
                logger.warning("未找到REFPROP库，将使用默认物性值")
                self.RP = None
                
        except Exception as e:
            logger.error(f"REFPROP加载失败: {e}，将使用默认物性值")
            self.RP = None
    
    def get_methane_properties(self, T, P):
        """
        获取甲烷在指定温度和压力下的物性参数 - 修正版
        
        参数:
        T: float - 温度 [K]
        P: float - 压力 [Pa]
        
        返回:
        dict: 包含密度、粘度、热导率、比热等物性的字典
        """
        if self.RP is None:
            logger.debug(f"REFPROP不可用，使用默认甲烷物性计算: T={T}K, P={P/1e6:.2f}MPa")
            return self._get_default_methane_properties(T, P)
        
        try:
            # 设置甲烷为当前工作流体
            methane = "METHANE"
            logger.debug(f"设置REFPROP工作流体: {methane}")
            self.RP.SETUPdll(1, methane, "HMX.BNC", "DEF")
            
            # 转换压力单位: Pa -> kPa (REFPROP标准单位)
            P_kpa = P / 1000.0
            
            # 摩尔分数 (纯物质)
            z = [1.0]
            
            # 执行温度-压力闪蒸计算
            results = self.RP.TPFLSHdll(T, P_kpa, z)
            
            if results.ierr != 0:
                logger.warning(f"TPFLSH计算警告，代码: {results.ierr}, 信息: {results.herr}")
            else:
                logger.debug(f"TPFLSH计算成功: 密度={results.D:.2f}kg/m³, 温度={T}K")
            
            # 计算输运性质
            transport_results = self.RP.TRNPRPdll(T, results.D, z)
            
            if transport_results.ierr != 0:
                logger.warning(f"TRNPRP计算警告，代码: {transport_results.ierr}, 信息: {transport_results.herr}")
            else:
                logger.debug(f"TRNPRP计算成功: 粘度={transport_results.eta*1e-6:.6f}Pa·s, "
                             f"热导率={transport_results.tcx:.4f}W/m·K")
            
            # 组装返回结果
            result_dict = {
                'density': results.D,  # kg/m³
                'viscosity': transport_results.eta * 1e-6,  # μPa·s -> Pa·s
                'conductivity': transport_results.tcx,  # W/m·K
                'specific_heat': results.Cp * 1000,  # kJ/kg·K -> J/(kg·K)
                'prandtl': (transport_results.eta * 1e-6 * results.Cp * 1000) / transport_results.tcx,
                'temperature': T,
                'pressure': P,
                'speed_of_sound': results.w,  # m/s
                'enthalpy': results.h,  # kJ/kg
                'entropy': results.s  # kJ/kg·K
            }

            logger.debug(f"甲烷物性计算结果汇总:")
            logger.debug(f"- 密度: {result_dict['density']:.2f} kg/m³")
            logger.debug(f"- 粘度: {result_dict['viscosity']:.2e} Pa·s")
            logger.debug(f"- 热导率: {result_dict['conductivity']:.3f} W/(m·K)")
            logger.debug(f"- 比热: {result_dict['specific_heat']:.0f} J/(kg·K)")
            logger.debug(f"- 普朗特数: {result_dict['prandtl']:.3f}")

            return result_dict

            
        except Exception as e:
            logger.error(f"REFPROP甲烷物性计算失败: {e}，使用默认值")
            logger.debug(f"异常详情: {str(e)}")
            return self._get_default_methane_properties(T, P)
    
    def _get_default_methane_properties(self, T, P):
        """
        当REFPROP不可用时使用的甲烷默认物性计算
        
        参数:
        T: float - 温度 [K]
        P: float - 压力 [Pa]
        
        返回:
        dict: 基于甲烷临界参数和工程经验公式的物性估算
        """
        logger.debug(f"使用默认甲烷物性计算: T={T}K, P={P/1e6:.2f}MPa")

        # 甲烷临界参数
        T_critical = 190.56    # 临界温度 [K]
        P_critical = 4.599e6   # 临界压力 [Pa]
        
        # 根据状态选择不同的计算模型
        is_supercritical = (P > P_critical and T > T_critical)
        logger.debug(f"甲烷状态: 超临界={is_supercritical}, T_critical={T_critical}K, P_critical={P_critical/1e6:.2f}MPa")
        
        if is_supercritical:
            # 超临界状态物性计算
            density = 160.0 * (P/P_critical)**0.3 * (T_critical/T)**0.7
            logger.debug(f"超临界状态密度计算: ρ={density:.2f}kg/m³")
        else:
            # 亚临界状态物性计算
            density = 200.0 * (P/P_critical)**0.2 * (T_critical/T)**0.8
            logger.debug(f"亚临界状态密度计算: ρ={density:.2f}kg/m³")
        
        # 输运性质计算
        viscosity = 1.2e-5 * (T/300)**0.7
        conductivity = 0.033 * (T/300)**0.8
        specific_heat = 2200 + 5*(T - 300)

        logger.debug(f"默认甲烷物性计算结果:")
        logger.debug(f"- 密度: {density:.2f} kg/m³")
        logger.debug(f"- 粘度: {viscosity:.2e} Pa·s")
        logger.debug(f"- 热导率: {conductivity:.3f} W/(m·K)")
        logger.debug(f"- 比热: {specific_heat:.0f} J/(kg·K)")
        
        return {
            'density': density,
            'viscosity': viscosity,
            'conductivity': conductivity,
            'specific_heat': specific_heat,
            'prandtl': viscosity * specific_heat / conductivity,
            'temperature': T,
            'pressure': P,
            'speed_of_sound': np.sqrt(1.3 * 518.3 * T)  # 估算音速
        }
    
    def get_combustion_gas_properties(self, T, P, mixture_ratio=3.5, Pc_MPa=10.0, eps=20.0):
        """增强温度范围检查，T>2000K时强制使用CEA增强模型"""
        # 定义REFPROP有效温度范围
        REFPROP_T_MAX = 2000.0  # [K]
        logger.debug(f"开始计算燃烧产物物性: T={T}K, P={P/1e6:.2f}MPa, O/F={mixture_ratio}")
        
        if T < 0:
            logger.error(f"输入温度{T}K无效，使用默认值")
            return self._get_default_combustion_gas_properties(3000, P, mixture_ratio)
        
        # 温度超限检查 - 强制使用CEA增强模型
        if T > REFPROP_T_MAX:
            if not hasattr(self, '_high_temp_warned'):
                logger.warning(f"温度{T}K超出REFPROP范围(>{REFPROP_T_MAX}K)，强制使用CEA增强模型")
                self._high_temp_warned = True
            logger.debug(f"高温处理: 使用CEA增强模型替代REFPROP")
            
            # 强制使用CEA增强计算
            if self.cea_calculator is not None:
                cea_props = self.cea_calculator.get_combustion_properties(Pc_MPa, mixture_ratio, eps)
                T_ref = cea_props['temperature']
                gamma = cea_props['gamma']
                MolWt = cea_props['molecular_weight']
                
                return self._calculate_gas_properties_from_cea(T, P, T_ref, gamma, MolWt, mixture_ratio)
            else:
                logger.warning("CEA计算器不可用，回退到简化模型")
                return self._get_default_combustion_gas_properties(T, P, mixture_ratio)
        
        # 正常温度范围内的原有逻辑
        if self.cea_calculator is not None:
            cea_props = self.cea_calculator.get_combustion_properties(Pc_MPa, mixture_ratio, eps)
            T_ref = cea_props['temperature']
            gamma = cea_props['gamma']
            MolWt = cea_props['molecular_weight']
            
            return self._calculate_gas_properties_from_cea(T, P, T_ref, gamma, MolWt, mixture_ratio)
        else:
            return self._get_default_combustion_gas_properties(T, P, mixture_ratio)
        
    def reset_warning_flags(self):
        """
        重置警告标志，用于新一轮计算
        
        功能：
        - 清除高温警告标志
        - 重置错误报告集合
        - 准备新的计算周期
        """
        if hasattr(self, '_high_temp_warned'):
            delattr(self, '_high_temp_warned')
        if hasattr(self, '_reported_errors'):
            delattr(self, '_reported_errors')
        if hasattr(self, '_reported_exceptions'):
            delattr(self, '_reported_exceptions')
    
    def _get_mixture_composition(self, mixture_ratio):
        """
        根据氧燃比确定燃烧产物组成
        
        参数:
        mixture_ratio: float - 混合比 (O/F)
        
        返回:
        str: REFPROP格式的混合物组成字符串
        """
        if mixture_ratio < 2.5:
            # 富燃料燃烧产物组成
            return "H2O[0.30]&CO2[0.15]&CO[0.20]&H2[0.15]&O2[0.15]&N2[0.05]"
        elif mixture_ratio > 4.0:
            # 贫燃料燃烧产物组成
            return "H2O[0.25]&CO2[0.25]&CO[0.05]&H2[0.02]&O2[0.38]&N2[0.05]"
        else:
            # 化学计量比附近产物组成 (O/F ≈ 3.5)
            return "H2O[0.35]&CO2[0.20]&CO[0.10]&H2[0.05]&O2[0.25]&N2[0.05]"
        
    def _calculate_gas_properties_from_cea(self, T, P, T_ref, gamma, MolWt, mixture_ratio):
        """
        基于CEA计算结果计算燃气物性
        
        参数:
        T: float - 实际温度 [K]
        P: float - 压力 [Pa]
        T_ref: float - CEA参考温度 [K]
        gamma: float - 比热比
        MolWt: float - 分子量 [g/mol]
        mixture_ratio: float - 混合比
        
        返回:
        dict: 燃气物性字典，包含密度、粘度、热导率等参数
        """
        # 气体常数
        R_specific = 8314.46 / MolWt
        
        # 密度计算 (理想气体状态方程)
        density = P / (R_specific * T)
        
        # 比热计算
        cp = gamma * R_specific / (gamma - 1)
        
        # 粘度计算 (基于Sutherland公式修正)
        viscosity = 5.0e-5 * (T/T_ref)**0.7
        
        # 热导率计算
        conductivity = 0.10 * (T/T_ref)**0.8
        
        # 音速计算
        speed_of_sound = np.sqrt(gamma * R_specific * T)
        
        return {
            'density': density,
            'viscosity': viscosity,
            'conductivity': conductivity,
            'specific_heat': cp,
            'prandtl': viscosity * cp / conductivity,
            'temperature': T,
            'pressure': P,
            'speed_of_sound': speed_of_sound,
            'gamma': gamma,
            'molecular_weight': MolWt,
            'mixture_ratio': mixture_ratio,
            'source': 'CEA_enhanced'
        }
        
    def get_combustion_gas_properties(self, T, P, mixture_ratio=3.5, Pc_MPa=10.0, eps=20.0):
        """
        增强的燃烧产物物性计算方法 - 结合CEA和REFPROP
        
        参数:
        T: float - 温度 [K]
        P: float - 压力 [Pa]
        mixture_ratio: float - 混合比 (O/F) (默认3.5)
        Pc_MPa: float - 燃烧室压力 [MPa] (默认10.0)
        eps: float - 面积比 (默认20.0)
        
        返回:
        dict: 包含燃烧产物物性的字典
        """
        # 如果提供了CEA计算器，优先使用CEA计算结果
        logger.debug(f"开始计算燃烧产物物性: T={T}K, P={P/1e6:.2f}MPa, O/F={mixture_ratio}")

        if self.cea_calculator is not None:
            cea_props = self.cea_calculator.get_combustion_properties(Pc_MPa, mixture_ratio, eps)
            
            # 使用CEA计算的温度作为参考，但根据实际温度调整物性
            T_ref = cea_props['temperature']
            gamma = cea_props['gamma']
            MolWt = cea_props['molecular_weight']
            
            # 基于CEA结果计算更精确的物性
            results = self._calculate_gas_properties_from_cea(T, P, T_ref, gamma, MolWt, mixture_ratio)
        else:
            # 回退到原来的方法
            results = self._get_default_combustion_gas_properties(T, P, mixture_ratio)
        logger.debug(f"燃烧产物物性计算结果汇总:")
        logger.debug(f"- 密度: {results['density']:.2f} kg/m³")
        logger.debug(f"- 粘度: {results['viscosity']:.2e} Pa·s")
        logger.debug(f"- 热导率: {results['conductivity']:.3f} W/(m·K)")
        logger.debug(f"- 比热: {results['specific_heat']:.0f} J/(kg·K)")
        return results

    def _get_default_combustion_gas_properties(self, T, P, mixture_ratio=3.5):
        """
        基于论文的热力学计算完善燃烧产物物性模型
        
        参数:
        T: float - 温度 [K]
        P: float - 压力 [Pa]
        mixture_ratio: float - 混合比 (O/F) (默认3.5)
        
        返回:
        dict: 基于论文经验公式的燃烧产物物性估算
        """
        logger.debug(f"使用默认燃烧产物物性计算: T={T}K, P={P/1e6:.2f}MPa, O/F={mixture_ratio}")
        
        # 根据混合比调整燃气组成和参数（论文第3节）
        if mixture_ratio < 2.5:  # 富燃料
            gas_constant = 320  # [J/(kg·K)]
            k_g = 1.25         # 比热比
            Pr_g = 0.75        # 普朗特数
            logger.debug("燃气状态: 富燃料")
        elif mixture_ratio > 4.0:  # 贫燃料  
            gas_constant = 280
            k_g = 1.18
            Pr_g = 0.72
            logger.debug("燃气状态: 贫燃料")
        else:  # 化学计量比附近 (O/F ≈ 3.5)
            gas_constant = 300
            k_g = 1.22
            Pr_g = 0.74
            logger.debug("燃气状态: 化学计量比附近")
        # 密度计算（理想气体状态方程）
        density = P / (gas_constant * T)
        
        # 粘度计算 - 基于萨瑟兰公式修正（论文公式相关）
        viscosity = 5.0e-5 * (T/3000)**0.7  # [Pa·s]
        
        # 热导率计算 - 基于高温气体经验关系
        conductivity = 0.10 * (T/3000)**0.8  # [W/(m·K)]
        
        # 比热计算 - 考虑温度变化的影响
        specific_heat = gas_constant * k_g / (k_g - 1)  # [J/(kg·K)]
        
        # 音速计算
        speed_of_sound = np.sqrt(k_g * gas_constant * T)

        logger.debug(f"默认燃烧产物物性计算结果:")
        logger.debug(f"- 气体常数: {gas_constant} J/(kg·K)")
        logger.debug(f"- 比热比: {k_g:.3f}")
        logger.debug(f"- 密度: {density:.3f} kg/m³")
        logger.debug(f"- 粘度: {viscosity:.2e} Pa·s")
        logger.debug(f"- 热导率: {conductivity:.4f} W/(m·K)")
        logger.debug(f"- 比热: {specific_heat:.0f} J/(kg·K)")
        logger.debug(f"- 音速: {speed_of_sound:.0f} m/s")
        
        return {
            'density': density,
            'viscosity': viscosity,
            'conductivity': conductivity,
            'specific_heat': specific_heat,
            'prandtl': viscosity * specific_heat / conductivity,
            'temperature': T,
            'pressure': P,
            'mixture_ratio': mixture_ratio,
            'speed_of_sound': speed_of_sound,
            'gas_constant': gas_constant,
            'specific_heat_ratio': k_g
        }


class LOX_MethaneEngineHeatTransfer:
    """
    液氧/甲烷发动机再生冷却和膜冷却传热计算主类 - 修正版
    
    基于论文《液氧/甲烷发动机再生冷却和膜冷却传热数值研究》的计算模型
    集成CEA燃烧计算、REFPROP物性计算和真实发动机几何数据
    功能：计算发动机轴向温度分布、热流密度、冷却性能等关键参数
    
    主要功能模块：
    - 燃气侧对流换热系数计算
    - 冷却剂侧换热系数计算  
    - 壁面热传导计算
    - 膜冷却效率计算
    - 热平衡方程求解
    - 轴向分布分析
    - 冷却性能评估
    """

    def __init__(self, refprop_path=None, use_cea=True, engine_shape_file=None):
        """
        初始化发动机传热计算器
        
        参数:
        refprop_path: REFPROP数据库安装路径 [str]，如未指定则自动搜索常见安装位置
        use_cea: 是否启用CEA燃烧计算 [bool]，True表示启用NASA CEA计算燃烧产物性质
        engine_shape_file: 发动机几何数据文件路径 [str]，包含发动机轮廓坐标数据
        """
        # 物理常数
        self.R = 8.314  # 通用气体常数 [J/(mol·K)]
        
        # 甲烷临界参数
        self.T_CRITICAL_METHANE = 190.56
        self.P_CRITICAL_METHANE = 4.599e6
        
        # 几何参数存储
        self.geometry = {
            'diameter': {},
            'length': {}, 
            'wall_thickness': {},
            'cooling_channels': {}
        }
        
        self._geometry_configured = False
        
        self.geometry_loader = EngineGeometry(engine_shape_file)
        self._update_geometry_from_shape_file()
        
        # 创建CEA计算器（如果启用）
        self.cea_calculator = None
        cr = 0
        if use_cea:
            Rc = self.geometry_loader.shape_data['header'].get('Rc', None)
            Rt = self.geometry_loader.shape_data['header'].get('Rt', None)
            if Rc is not None and Rt is not None:
                try:
                    cr = (float(Rc) / float(Rt)) ** 2
                except (ValueError, TypeError):
                    logger.warning("几何文件中Rc或Rt参数无效，使用默认压比6.25")
                    cr = 6.25
            self.cea_calculator = CEACalculator(cr=cr)
            logger.debug(f"chamber_ratio = {cr:.3f}，已初始化CEA计算器")
        
        # 创建物性计算实例（传递CEA计算器）
        self.fluid_props = REFPROPFluid(refprop_path, self.cea_calculator)

    def _update_geometry_from_shape_file(self):
        """
        根据几何文件更新几何参数
        
        从发动机几何文件中提取关键几何参数并更新类属性
        包括总长度、喉部直径等关键尺寸
        """
        if self.geometry_loader.shape_data:
            # 更新总长度
            total_length = self.geometry_loader.get_total_length()
            if total_length:
                self.geometry['length']['chamber'] = total_length
                logger.info(f"根据几何文件更新总长度为: {total_length:.3f} m")
            
            # 增强喉部直径提取逻辑
            header = self.geometry_loader.shape_data['header']
            throat_diameter = None
            
            # 方案1: 从头部信息提取Rt
            if 'Rt' in header:
                try:
                    rt_value = header['Rt']
                    if isinstance(rt_value, str):
                        import re
                        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", rt_value)
                        if numbers:
                            throat_radius_mm = float(numbers[0])
                            throat_diameter = 2 * throat_radius_mm / 1000
                            logger.info(f"从头部信息Rt参数获取喉部直径: {throat_diameter:.3f} m")
                    else:
                        throat_radius_mm = float(rt_value)
                        throat_diameter = 2 * throat_radius_mm / 1000
                        logger.info(f"从头部信息Rt参数获取喉部直径: {throat_diameter:.3f} m")
                        
                except (ValueError, TypeError) as e:
                    logger.warning(f"头部信息Rt参数解析失败: {e}")
            
            # 方案2: 从几何点数据中搜索最小半径
            if throat_diameter is None and self.geometry_loader.shape_data['points']:
                points = self.geometry_loader.shape_data['points']
                min_radius_point = min(points, key=lambda p: p[1])  # 找到最小半径的点
                throat_radius_mm = min_radius_point[1]
                throat_diameter = 2 * throat_radius_mm / 1000
                throat_position = min_radius_point[0] / 1000
                logger.info(f"从几何点数据搜索到喉部: 位置={throat_position:.3f}m, 直径={throat_diameter:.3f}m")
            
            # 方案3: 使用默认值
            if throat_diameter is None:
                throat_diameter = 0.0625
                logger.warning(f"喉部直径提取失败，使用默认值: {throat_diameter:.3f} m")
            
            self.geometry['diameter']['throat'] = throat_diameter
            logger.info(f"最终设置的喉部直径为: {throat_diameter:.3f} m")

    def set_geometric_parameters(self, d_throat, L_chamber, delta_wall, 
                                number_of_fins, t_fin, h_fin, delta_fin):
        """
        配置推力室几何参数
        
        参数:
        d_throat: 喉部直径 [m]，发动机最窄截面直径
        L_chamber: 燃烧室长度 [m]，从喷注器到喉部的轴向距离
        delta_wall: 内壁厚度 [m]，推力室内壁材料厚度
        b_channel: 冷却通道宽度 [m]，再生冷却通道的宽度
        t_fin: 肋片厚度 [m]，冷却通道间肋片的厚度
        h_fin: 肋片高度 [m]，冷却通道的高度方向尺寸
        delta_fin: 通道高度 [m]，冷却通道的径向高度
        """
        # 存储基本几何参数
        self.geometry['diameter']['throat'] = d_throat
        self.geometry['length']['chamber'] = L_chamber
        self.geometry['wall_thickness']['inner'] = delta_wall
        
        # 冷却通道参数
        self.geometry['cooling_channels'] = {
            'number_of_fins': number_of_fins,
            'fin_thickness': t_fin,
            'fin_height': h_fin,
            'height': delta_fin
        }
        
        self._geometry_configured = True
        logger.info("几何参数配置完成")

    def calculate_local_geometry(self, axial_position, total_length):
        """
        计算指定轴向位置的局部几何参数
        
        参数:
        axial_position: 轴向位置 [m]，从喷注器开始的轴向距离
        total_length: 推力室总长度 [m]，发动机总轴向长度
        
        返回:
        tuple: (当地直径 [m], 当地截面积 [m²])
        """
        if self.geometry_loader.shape_data and self.geometry_loader.shape_data['points']:
            local_diameter = self.geometry_loader.get_diameter_at_position(axial_position)
            if local_diameter is not None and local_diameter > 0:
                local_area = np.pi * (local_diameter / 2)**2
                logger.debug(f"使用真实几何数据: 直径={local_diameter:.4f}m, 面积={local_area:.6f}m²")
                return local_diameter, local_area
            else:
                logger.debug(f"位置{axial_position:.3f}m真实几何计算失败，使用简化模型")
        else:
            logger.warning("几何数据未加载或无效，始终使用简化模型")
        
        # 备用简化模型
        return self._calculate_simplified_geometry(axial_position, total_length)

    def _calculate_simplified_geometry(self, axial_position, total_length):
        """
        计算推力室轴向位置的局部几何参数（简化模型）
        
        参数:
        axial_position: 轴向位置 [m]
        total_length: 推力室总长度 [m]
        
        返回:
        tuple: (当地直径 [m], 当地面积 [m²])
        """
        logger.debug(f"使用简化模型计算几何参数: 位置={axial_position:.3f}m, 总长={total_length:.3f}m")

        if not self._geometry_configured:
            raise ValueError("请先配置几何参数")
        
        d_throat = self.geometry['diameter']['throat']
        relative_position = axial_position / total_length
        logger.debug(f"相对位置: {relative_position:.3f}")
        
        if relative_position < 0.3:
            # 燃烧室段: 直径基本恒定
            local_diameter = d_throat * 2.0
            logger.debug("燃烧室段: 直径基本恒定")
        elif relative_position < 0.6:
            # 收敛段: 直径线性减小
            progress = (relative_position - 0.3) / 0.3
            local_diameter = d_throat * (2.0 - progress)
            logger.debug("收敛段: 直径线性减小")
        else:
            # 扩张段: 直径线性增大
            progress = (relative_position - 0.6) / 0.4
            local_diameter = d_throat * (1.0 + progress * 0.5)
            logger.debug("扩张段: 直径线性增大")
        
        local_area = np.pi * (local_diameter / 2)**2
        logger.debug(f"简化几何计算结果: 直径={local_diameter:.4f}m, 面积={local_area:.6f}m²")
        
        return local_diameter, local_area

    def calculate_coolant_velocity(self, m_dot, T, P, flow_area):
        """
        根据质量流量守恒计算冷却剂流速
        
        参数:
        m_dot: float - 质量流量 [kg/s]
        T: float - 冷却剂温度 [K]
        P: float - 冷却剂压力 [Pa]
        flow_area: float - 流通面积 [m²]
        
        返回:
        float: 冷却剂流速 [m/s]
        """
        logger.debug(f"计算冷却剂流速: ṁ={m_dot:.3f}kg/s, T={T}K, P={P/1e6:.2f}MPa, A={flow_area:.6f}m²")
        
        # 获取冷却剂物性
        coolant_props = self.fluid_props.get_methane_properties(T, P)
        density = coolant_props['density']

        logger.debug(f"冷却剂密度: ρ={density:.3f}kg/m³")
        
        # 质量流量守恒: ṁ = ρ·v·A ⇒ v = ṁ/(ρ·A)
        if density <= 0 or flow_area <= 0:
            logger.warning(f"无效的密度或流通面积: ρ={density}, A={flow_area}")
            return 10.0
        
        # 基本流速计算
        velocity = m_dot / (density * flow_area)
        logger.debug(f"基本流速计算结果: v={velocity:.2f}m/s")
        
        # 马赫数检查
        speed_of_sound = coolant_props.get('speed_of_sound', 400)  # 默认音速
        mach_number = velocity / speed_of_sound if speed_of_sound > 0 else 0
        logger.debug(f"冷却剂马赫数: Ma={mach_number:.3f} (音速={speed_of_sound:.2f}m/s)")
        
        # 压缩性修正（Ma > 0.3时触发）
        if mach_number > 0.3:
            # 简单的一维可压缩流修正
            gamma = 1.3  # 甲烷的比热比
            correction_factor = 1 / np.sqrt(1 + (gamma - 1)/2 * mach_number**2)
            corrected_velocity = velocity * correction_factor
            
            logger.debug(f"流速压缩性修正: Ma={mach_number:.3f} > 0.3, 修正因子={correction_factor:.3f}")
            logger.debug(f"修正后流速: {corrected_velocity:.2f}m/s")
            
            return corrected_velocity
        
        logger.debug(f"流速计算完成: {velocity:.2f}m/s (亚音速，无需压缩性修正)")
        return velocity

    def calculate_flow_area(self, axial_position, total_length):
        """
        计算指定轴向位置的冷却剂流通面积
        
        参数:
        axial_position: float - 轴向位置 [m]
        total_length: float - 总长度 [m]
        
        返回:
        float: 流通面积 [m²]
        """
        # 获取当地几何参数
        logger.debug(f"计算流通面积: 位置={axial_position:.3f}m, 总长={total_length:.3f}m")

        d_local, A_local = self.calculate_local_geometry(axial_position, total_length)
        local_radius = d_local / 2
        
        # 计算周长
        circumference = 2 * math.pi * local_radius
        logger.debug(f"当地直径={d_local:.4f}m, 周长={circumference:.4f}m")
        
        # 获取冷却通道参数
        channels = self.geometry['cooling_channels']
        n_fins = channels['number_of_fins']
        t_fin = channels['fin_thickness']
        h_fin = channels['fin_height']

        logger.debug(f"冷却通道参数: 数量={n_fins}, 翼片厚度={t_fin:.4f}m, 翼片高度={h_fin:.4f}m")
        
        # 计算单个通道宽度
        arc_per_fin = circumference / n_fins
        b_channel = arc_per_fin - t_fin

        if b_channel <= 0:
            logger.warning(f"计算出的通道宽度无效: {b_channel:.6f}m, 使用最小值")
            b_channel = max(b_channel, 1e-6)
        
        # 总流通面积 = 通道数量 × 单个通道截面积
        # 单个通道截面积 = 通道宽度 × 通道高度
        single_channel_area = b_channel * h_fin
        total_flow_area = n_fins * single_channel_area

        logger.debug(f"流通面积计算:")
        logger.debug(f"- 单个弧长: {arc_per_fin:.4f}m")
        logger.debug(f"- 通道宽度: {b_channel:.4f}m")
        logger.debug(f"- 单个通道面积: {single_channel_area:.6f}m²")
        logger.debug(f"- 总流通面积: {total_flow_area:.6f}m²")
        
        return total_flow_area, b_channel, local_radius

    def gas_side_heat_transfer_coefficient(self, P_c, c_star, T_gas, T_wg, Ma, 
                                         d_local, A_local, A_throat):
        """
        计算燃气侧对流换热系数 - 基于巴兹公式
        
        参数:
        P_c: 燃烧室压力 [Pa]，燃烧室工作压力
        c_star: 特征速度 [m/s]，推进剂特征速度
        T_gas: 燃气温度 [K]，燃烧产物温度
        T_wg: 燃气侧壁温 [K]，燃气接触面壁温
        Ma: 当地马赫数，当地燃气流动马赫数
        d_local: 当地直径 [m]，计算位置的直径
        A_local: 当地截面积 [m²]，计算位置的流通面积
        A_throat: 喉部面积 [m²]，喉部参考面积
        
        返回:
        float: 燃气侧换热系数 [W/(m²·K)]
        """
        logger.debug(f"计算燃气侧换热系数: Pc={P_c/1e6:.2f}MPa, T_gas={T_gas}K, Ma={Ma:.3f}")

        # 绝热壁温计算
        k_g = 1.2  # 燃气比热比
        T_aw = T_gas * (1 + (k_g - 1) / 2 * Ma**2)
        logger.debug(f"绝热壁温 T_aw={T_aw:.2f}K")

        d_throat = self.geometry['diameter']['throat']
        eps = (d_local / d_throat) ** 2

        # 获取燃气物性（以T_aw为温度）
        gas_props = self.fluid_props.get_combustion_gas_properties(T_aw, P_c, eps=eps)
        mu_g = gas_props['viscosity']
        cp_g = gas_props['specific_heat']
        Pr_g = gas_props['prandtl']

        logger.debug(f"燃气物性: μ={mu_g:.2e}Pa·s, cp={cp_g:.2f}J/(kg·K), Pr={Pr_g:.3f}")

        # 温度恢复系数修正 - 论文公式(3)
        term1 = 0.5 * (T_wg / T_aw) * (1 + (k_g - 1)/2 * Ma**2) + 0.5
        term2 = 1 + (k_g - 1)/2 * Ma**2
        recovery_factor = term1**(-0.68) * term2**(-0.12)
        logger.debug(f"恢复系数修正: term1={term1:.3f}, term2={term2:.3f}, factor={recovery_factor:.3f}")
        
        # 巴兹公式计算基础换热系数 - 论文公式(2)
        h_g = (0.026 * mu_g**0.2 * cp_g / d_local**0.2 / Pr_g**0.6 * 
                    (P_c / c_star)**0.8 * (A_throat / A_local)**0.9) * recovery_factor
        
        logger.debug(f"燃气侧换热系数计算结果: h_g={h_g:.0f} W/(m²·K)")
        logger.debug(f"几何参数: d_local={d_local:.3f}m, A_local={A_local:.6f}m², A_throat/A_local={A_throat/A_local:.3f}")

        return h_g

    def gas_side_heat_transfer_coefficient_enhanced(self, P_c, c_star, T_gas, T_wg, Ma, 
                                                  d_local, A_local, A_throat, mixture_ratio=3.5):
        """
        增强的燃气侧对流换热系数计算 - 使用CEA计算的燃气性质
        
        参数:
        P_c: 燃烧室压力 [Pa]
        c_star: 特征速度 [m/s]
        T_gas: 燃气温度 [K]
        T_wg: 燃气侧壁温 [K]
        Ma: 当地马赫数
        d_local: 当地直径 [m]
        A_local: 当地截面积 [m²]
        A_throat: 喉部面积 [m²]
        mixture_ratio: 混合比 (O/F)，氧化剂与燃料质量比
        
        返回:
        float: 燃气侧换热系数 [W/(m²·K)]
        """
        logger.debug(f"增强计算燃气侧换热系数: Pc={P_c/1e6:.2f}MPa, T_gas={T_gas}K, O/F={mixture_ratio}, Ma={Ma:.3f}")

        # 使用CEA增强的物性计算方法
        Pc_MPa = P_c / 1e6  # 转换为MPa供CEA使用
        
        # 基于半径比计算面积比：eps = (d_local / d_throat)^2
        d_throat = self.geometry['diameter']['throat']
        eps = (d_local / d_throat) ** 2

        # 传入正确的面积比参数
        gas_props_cea = self.fluid_props.get_combustion_gas_properties(
            T_gas, P_c, mixture_ratio, Pc_MPa, eps  # 添加eps参数
        )
        k_g = gas_props_cea['gamma']

        logger.debug(f"燃气比热比 k_g={k_g:.3f} (基于CEA)")
        
        T_aw = T_gas * (1 + (k_g - 1) / 2 * Ma**2)

        # 获取燃气物性（使用增强方法）
        gas_props = self.fluid_props.get_combustion_gas_properties(
            T_aw, P_c, mixture_ratio, Pc_MPa, eps
        )
        mu_g = gas_props['viscosity']
        cp_g = gas_props['specific_heat']
        Pr_g = gas_props['prandtl']

        # 温度恢复系数修正
        term1 = 0.5 * (T_wg / T_aw) * (1 + (k_g - 1)/2 * Ma**2) + 0.5
        term2 = 1 + (k_g - 1)/2 * Ma**2
        recovery_factor = term1**(-0.68) * term2**(-0.12)
        
        # 巴兹公式计算基础换热系数
        h_g = (0.026 * mu_g**0.2 * cp_g / d_local**0.2 / Pr_g**0.6 * 
                    (P_c / c_star)**0.8 * (A_throat / A_local)**0.9) * recovery_factor
        
        logger.debug(f"增强燃气侧换热系数计算结果: h_g={h_g:.0f} W/(m²·K)")

        return h_g

    def calculate_thermal_conductivity(self, temperature, material_properties):
        """
        使用两个温度电导率点之间的线性插值计算给定温度下的热导率
        
        参数:
        temperature: 温度 [K]
        material_properties: dict - 反应材料导热率的两个导热率-温度数对
        
        返回:
        float: 导热率 [W/(m·K)]
        """
        logger.debug(f"计算热导率: T={temperature}K")

        if 'thermal_conductivity_points' not in material_properties:
            # 如果未提供积分，则回退到常数值
            default_k = material_properties.get('thermal_conductivity', 350.0)
            logger.debug(f"未提供导热率数据点，使用默认值: k={default_k} W/(m·K)")
            return default_k
        
        points = material_properties['thermal_conductivity_points']
        if len(points) != 2:
            logger.error(f"导热率数据点数量不正确，必须为两个点，实际数量: {len(points)}")
            raise ValueError("导热率数据点必须恰好包含两个点")
        
        T1, k1 = points[0]
        T2, k2 = points[1]

        logger.debug(f"导热率数据点: T1={T1}K, k1={k1} W/(m·K); T2={T2}K, k2={k2} W/(m·K)")
        
        # 线性插值: k = k1 + (k2 - k1) * (T - T1) / (T2 - T1)
        if T2 == T1:
            logger.warning("导热率数据点温度相同，无法插值，返回k1")
            return k1  # 避免除以零
        
        k_value = k1 + (k2 - k1) * (temperature - T1) / (T2 - T1)
        logger.debug(f"插值计算热导率: k={k_value} W/(m·K)")
        return k_value

    
    def analyze_combustion_performance(self, Pc_MPa, mixture_ratios, eps_values):
        """
        分析燃烧性能参数随混合比和面积比的变化
        
        参数:
        Pc_MPa: 燃烧室压力 [MPa]，燃烧室工作压力
        mixture_ratios: 混合比数组 [list]，氧燃比取值范围
        eps_values: 面积比数组 [list]，喷管面积比取值范围
        
        返回:
        dict: 包含燃烧性能分析结果的字典，包括温度、比冲等参数
        """
        if self.cea_calculator is None:
            logger.warning("CEA计算器不可用，无法进行燃烧性能分析")
            return {}
        
        results = {
            'mixture_ratio_analysis': [],
            'area_ratio_analysis': [],
            'optimal_mixture_ratio': None,
            'max_spscific_impulse': 0.0
        }
        
        # 混合比影响分析
        logger.info("进行混合比影响分析...")
        for MR in mixture_ratios:
            try:
                data = self.cea_calculator.get_combustion_properties(Pc_MPa, MR)
                results['mixture_ratio_analysis'].append({
                    'mixture_ratio': MR,
                    'combustion_temperature': data['temperature'],
                    'specific_impulse': data['isp'],
                    'characteristic_velocity': data['cstar'],
                    'specific_heat_ratio': data['gamma']
                })

                # 寻找最优混合比（比冲最大化）
                if data['isp'] > results['max_specific_impulse']:
                    results['max_specific_impulse'] = data['isp']
                    results['optimal_mixture_ratio'] = MR
            except Exception as e:
                logger.error(f"混合比{MR}分析失败: {e}")
        
        # 面积比影响分析
        logger.info("进行面积比影响分析...")
        for eps in eps_values:
            try:
                data = self.cea_calculator.get_combustion_properties(Pc_MPa, 3.5, eps)
                results['area_ratio_analysis'].append({
                    'area_ratio': eps,
                    'specific_impulse': data['isp'],
                    'combustion_temperature': data['temperature']
                })
            except Exception as e:
                logger.error(f"面积比{eps}分析失败: {e}")

        # 验证CEA输出合理性
        if results['mixture_ratio_analysis']:
            isp_values = [x['specific_impulse'] for x in results['mixture_ratio_analysis']]
            if max(isp_values) - min(isp_values) < 10:  # 比冲变化过小
                logger.warning("CEA计算的比冲变化较小，可能存在问题")
            else:
                logger.info(f"最优混合比: O/F={results['optimal_mixture_ratio']:.1f}, "
                        f"最大比冲: {results['max_specific_impulse']:.1f} s")
        
        return results

    def coolant_side_heat_transfer_coefficient(self, T_c, P_c, T_wc, v_c, d_h):
        """
        计算冷却剂侧对流换热系数
        
        参数:
        T_c: 冷却剂温度 [K]，冷却剂主流温度
        P_c: 冷却剂压力 [Pa]，冷却剂工作压力
        T_wc: 冷却剂侧壁温 [K]，冷却剂接触面壁温
        v_c: 冷却剂流速 [m/s]，冷却剂流动速度
        d_h: 水力直径 [m]，冷却通道水力直径
        
        返回:
        tuple: (换热系数 [W/(m²·K)], 物性参数字典)
        """
        logger.debug(f"计算冷却剂侧换热系数: T_c={T_c}K, P_c={P_c/1e6:.2f}MPa, T_wc={T_wc}K, v_c={v_c}m/s, d_h={d_h}m")

        # 获取甲烷物性
        methane_props = self.fluid_props.get_methane_properties(T_c, P_c)
        rho = methane_props['density']
        lambda_c = methane_props['conductivity']
        cp_c = methane_props['specific_heat']
        mu_c = methane_props['viscosity']

        logger.debug(f"冷却剂物性: ρ={rho:.1f}kg/m³, λ={lambda_c:.3f}W/(m·K), cp={cp_c:.0f}J/(kg·K), μ={mu_c:.2e}Pa·s")

        # 验证流速的物理合理性
        if v_c <= 0:
            logger.warning(f"计算得到的冷却剂流速无效: {v_c} m/s, 使用默认值")
            v_c = 10.0  # 安全默认值
        
        # 防止T_wc为零或负数，避免无效运算
        T_wc_safe = max(T_wc, 100.0)
        if T_wc <= 0:
            logger.warning(f"冷却剂侧壁温T_wc={T_wc}过低，已自动修正为1.0K，T_c={T_c}")
        
        # 冷却剂侧换热系数计算 - 论文公式(8)
        try:
            h_c = (0.023 * v_c**0.8 * d_h**-0.2 * 
                   (rho**0.8 * lambda_c**0.6 * cp_c**0.4 / mu_c**0.4) * 
                   (T_c / T_wc_safe)**0.45)
            logger.debug(f"冷却剂侧换热系数计算结果: h_c={h_c:.0f} W/(m²·K)")
        except Exception as e:
            logger.error(f"冷却剂侧换热系数计算异常: {e}, T_c={T_c}, T_wc={T_wc}, v_c={v_c}, d_h={d_h}")
            h_c = 1.0
            logger.warning(f"使用默认换热系数: h_c={h_c} W/(m²·K)")
        return h_c, methane_props

    def fin_correction_factor(self, h_c, lambda_wall, b_channel, temperature):
        """
        计算肋片传热修正系数
        
        参数:
        h_c: 冷却剂侧换热系数 [W/(m²·K)]
        lambda_wall: 壁面材料热导率 [W/(m·K)]
        temperature: 热导率计算的壁温 [K]
        
        返回:
        float: 肋片修正系数，考虑肋片效率的修正因子
        """
        logger.debug(f"计算肋片修正系数: h_c={h_c:.0f} W/(m²·K), λ_wall={lambda_wall:.3f}W/(m·K), b_channel={b_channel:.4f}m, T={temperature}K")
        
        channels = self.geometry['cooling_channels']
        b_R = b_channel                  # 通道宽度
        t_R = channels['fin_thickness']  # 肋片厚度
        h_R = channels['fin_height']     # 肋片高度

        logger.debug(f"肋片参数: b_R={b_R:.4f}m, t_R={t_R:.4f}m, h_R={h_R:.4f}m")
        
        # 肋片效率计算 - 论文公式(6)-(7)
        s_R = h_R * np.sqrt(2 * h_c / (lambda_wall * t_R))
        k_r = b_R / (b_R + t_R) + (2 * h_c / (b_R + t_R)) * (np.tanh(s_R) / s_R)

        logger.debug(f"肋片修正系数计算: s_R={s_R:.3f}, k_r={k_r:.3f}")
        
        return k_r

    def wall_heat_conduction(self, T_wg, T_wc, lambda_wall, material_properties):
        """
        计算壁面热传导热流密度
        
        参数:
        T_wg: 燃气侧壁温 [K]，高温侧壁温
        T_wc: 冷却剂侧壁温 [K]，低温侧壁温
        lambda_wall: 壁面材料热导率 [W/(m·K)]
        material_properties: 壁面材料热导率性质，两个热导率-温度数对
        
        返回:
        float: 壁面热流密度 [W/m²]
        """
        logger.debug(f"计算壁面热传导: T_wg={T_wg}K, T_wc={T_wc}K, λ_wall={lambda_wall:.3f}W/(m·K)")
        
        # 使用平均温度计算导热系数
        T_avg = (T_wg + T_wc) / 2
        lambda_wall = self.calculate_thermal_conductivity(T_avg, material_properties)
        
        delta_wall = self.geometry['wall_thickness']['inner']
        
        q_wall = lambda_wall / delta_wall * (T_wg - T_wc)

        logger.debug(f"壁面热传导计算:")
        logger.debug(f"- 平均壁温: {T_avg:.1f}K")
        logger.debug(f"- 壁面导热系数: {lambda_wall:.1f} W/(m·K)")
        logger.debug(f"- 壁厚: {delta_wall:.3f}m")
        logger.debug(f"- 热流密度: {q_wall/1e6:.3f} MW/m²")

        return q_wall

    def liquid_film_cooling_efficiency(self, Re_lf):
        """
        计算液膜冷却效率 - 基于图3的关系曲线
        
        参数:
        Re_lf: 液膜雷诺数，表征液膜流动状态
        
        返回:
        float: 液膜冷却效率 [0-1]，1表示完全有效
        """
        logger.debug(f"计算液膜冷却效率: Re_lf={Re_lf:.0f}")

        if Re_lf < 100:
            efficiency = 0.95
            regime = "低雷诺数区 (Re < 100)"
        elif Re_lf < 1000:
            efficiency = 0.95 - 0.0003 * (Re_lf - 100)
            regime = "过渡区 (100 ≤ Re < 1000)"
        else:
            efficiency = 0.8 - 0.0001 * (Re_lf - 1000)
            regime = "高雷诺数区 (Re ≥ 1000)"
        
        # 确保效率在合理范围内
        efficiency = max(0.1, min(1.0, efficiency))
        
        logger.debug(f"液膜冷却效率计算结果:")
        logger.debug(f"- 雷诺数范围: {regime}")
        logger.debug(f"- 计算效率: {efficiency:.3f}")
        logger.debug(f"- 雷诺数: Re_lf={Re_lf:.1f}")
        
        if efficiency < 0.5:
            logger.warning(f"液膜冷却效率较低: {efficiency:.3f}，可能影响冷却效果")
        
        return efficiency

    def gas_film_cooling_efficiency(self, X, m_g, m_gf, d, x, mu_g):
        """
        计算气膜冷却效率 - 论文公式(13)
        
        参数:
        X: 无量纲距离，冷却剂注入位置到计算点的无量纲距离
        m_g: 燃气质量流量 [kg/s]，主流燃气流量
        m_gf: 气膜质量流量 [kg/s]，冷却剂薄膜流量
        d: 当地直径 [m]，计算位置直径
        x: 轴向距离 [m]，从喷注器开始的轴向距离
        mu_g: 燃气粘度 [Pa·s]，燃气动力粘度
        
        返回:
        float: 气膜冷却效率 [0-1]
        """
        logger.debug(f"计算气膜冷却效率: X={X:.3f}, m_g={m_g:.3f}kg/s, m_gf={m_gf:.3f}kg/s, d={d:.3f}m, x={x:.3f}m, μ_g={mu_g:.2e}Pa·s")

        # 获取燃气物性
        gas_props = self.fluid_props.get_combustion_gas_properties(3000, 10e6)
        c_pg = gas_props['specific_heat']
        c_pgf = c_pg  # 简化假设
        
        X0 = (3.08 + (m_g/(np.pi*(d/2)**2) * mu_g**0.25 * 
                     (m_gf/(np.pi*d))**(-1.25) * x)**0.8)**1.25 - X
        
        eta_gf = 1 / (1 + c_pg/c_pgf * (0.325 * (X + X0)**0.8 - 1))
        return eta_gf

    def pressure_drop_calculation(self, L_channel, d_h, rho_c, v_c, 
                                delta_rho, delta_v, T_avg, P_avg):
        """
        计算冷却通道压降
        
        参数:
        L_channel: 通道长度 [m]，冷却通道轴向长度
        d_h: 水力直径 [m]，冷却通道水力直径
        rho_c: 冷却剂密度 [kg/m³]，冷却剂平均密度
        v_c: 冷却剂流速 [m/s]，冷却剂平均流速
        delta_rho: 密度变化量 [kg/m³]，沿程密度变化
        delta_v: 速度变化量 [m/s]，沿程速度变化
        T_avg: 平均温度 [K]，冷却剂平均温度
        P_avg: 平均压力 [Pa]，冷却剂平均压力
        
        返回:
        float: 总压降 [Pa]，沿程摩擦损失和动量损失之和
        """
        logger.debug(f"计算冷却通道压降: L={L_channel}m, d_h={d_h}m, ρ={rho_c}kg/m³, v={v_c}m/s")

        # 获取物性用于雷诺数计算
        methane_props = self.fluid_props.get_methane_properties(T_avg, P_avg)
        mu_c = methane_props['viscosity']
        
        # 雷诺数计算
        Re = rho_c * v_c * d_h / mu_c
        logger.debug(f"流动状态: Re={Re:.0f}, ρ={rho_c:.1f}kg/m³, μ={mu_c:.2e}Pa·s")
        
        # 摩擦系数计算
        if Re < 2300:
            f_friction = 64 / Re  # 层流
            flow_regime = "层流"
        else:
            f_friction = 0.316 * Re**(-0.25)  # 湍流
            flow_regime = "湍流"

        logger.debug(f"流动状态: {flow_regime}, 摩擦系数 f={f_friction:.4f}")
        
        # 沿程摩擦损失 - 论文公式(9)
        delta_p_friction = f_friction * (L_channel / d_h) * (rho_c * v_c**2 / 2)
        
        # 动量损失 - 论文公式(10)
        delta_p_momentum = 0.5 * (delta_rho * v_c) * delta_v
        
        total_pressure_drop = delta_p_friction + delta_p_momentum

        logger.debug(f"压降计算结果:")
        logger.debug(f"- 摩擦损失: {delta_p_friction/1e3:.1f} kPa")
        logger.debug(f"- 动量损失: {delta_p_momentum/1e3:.1f} kPa")
        logger.debug(f"- 总压降: {total_pressure_drop/1e3:.1f} kPa")

        return total_pressure_drop

    # 这个函数最后可能要进行类似NozzleDesigner.py中的读图线性插值
    def liquid_film_cooling_efficiency(self, Re_lf):
        """
        计算液膜冷却效率 - 基于论文图3的关系曲线
        
        参数:
        Re_lf: float - 液膜雷诺数
        
        返回:
        float: 液膜冷却效率 [0-1]
        """
        logger.debug(f"计算液膜冷却效率: Re_lf={Re_lf:.0f}")

        # 基于论文图3的液膜冷却效率与雷诺数关系
        if Re_lf < 100:
            result = 0.95
        elif Re_lf < 1000:
            result = 0.95 - 0.0003 * (Re_lf - 100)
        else:
            result = 0.8 - 0.0001 * (Re_lf - 1000)
        logger.debug(f"液膜冷却效率计算结果: efficiency={result:.3f}")
        return result

    def gas_film_cooling_efficiency(self, X, m_g, m_gf, d, x, mu_g):
        """
        计算气膜冷却效率 - 论文公式(13)
        
        参数:
        X: float - 无量纲距离
        m_g: float - 燃气质量流量 [kg/s]
        m_gf: float - 气膜质量流量 [kg/s]
        d: float - 当地直径 [m]
        x: float - 轴向距离 [m]
        mu_g: float - 燃气粘度 [Pa·s]
        
        返回:
        float: 气膜冷却效率 [0-1]
        """
        logger.debug(f"计算气膜冷却效率: X={X:.3f}, m_g={m_g:.3f}kg/s, m_gf={m_gf:.3f}kg/s, d={d:.3f}m, x={x:.3f}m, μ_g={mu_g:.2e}Pa·s")

        gas_props = self.fluid_props.get_combustion_gas_properties(3000, 10e6)
        c_pg = gas_props['specific_heat']
        c_pgf = c_pg  # 简化假设
        
        X0 = (3.08 + (m_g/(np.pi*(d/2)**2) * mu_g**0.25 * 
                    (m_gf/(np.pi*d))**(-1.25) * x)**0.8)**1.25 - X
        
        eta_gf = 1 / (1 + c_pg/c_pgf * (0.325 * (X + X0)**0.8 - 1))

        logger.debug(f"气膜冷却效率计算结果: η_gf={eta_gf:.3f}")

        return eta_gf

    def solve_heat_balance(self, T_gas, P_chamber, c_star, m_dot_coolant,
                         T_coolant_in, P_coolant_in, geometry_params,material_properties, operating_conditions, film_cooling_params=None):
        """
        求解推力室壁面热平衡方程 - 基于论文公式(1)的三层热阻模型
        
        参数:
        T_gas: 燃气温度 [K]，燃烧产物温度
        P_chamber: 燃烧室压力 [Pa]，燃烧室工作压力
        c_star: 特征速度 [m/s]，推进剂特征速度
        m_dot_coolant: 冷却剂质量流量 [kg/s]，冷却剂流量
        T_coolant_in: 冷却剂入口温度 [K]，冷却剂进口温度
        P_coolant_in: 冷却剂入口压力 [Pa]，冷却剂进口压力
        geometry_params: 几何参数字典，包含当地几何参数
        material_properties: 材料属性字典，包含材料热物性
        operating_conditions: 运行条件字典，包含流动参数
        film_cooling_params: dict - 膜冷却参数，包含：
            - film_type: str - 膜冷却类型 ('liquid', 'gas', 'none')
            - film_flow_rate: float - 膜冷却剂流量 [kg/s]
            - film_start_position: float - 膜冷却起始位置 [m]
            - film_temperature: float - 膜冷却剂初始温度 [K]
        
        返回:
        tuple: (燃气侧壁温 [K], 冷却剂侧壁温 [K])
        """
        # 解包几何参数
        logger.debug("开始求解壁面热平衡方程...")

        A_local = geometry_params['local_area']
        A_throat = geometry_params['throat_area']
        d_local = geometry_params['local_diameter']
        x_position = geometry_params.get('axial_position', 0)
        
        # 解包运行条件
        Ma = operating_conditions['mach_number']
        flow_area, b_channel, radius = self.calculate_flow_area(x_position, geometry_params.get('total_length', 0.257))
        v_c = self.calculate_coolant_velocity(m_dot_coolant, T_coolant_in, P_coolant_in, flow_area)
        d_h = operating_conditions.get('hydraulic_diameter', 0.01)

        # 更新运行条件中的流速
        operating_conditions_local = operating_conditions.copy()
        operating_conditions_local['coolant_velocity'] = v_c

        logger.debug(f"位置 {x_position:.3f}m: 流速={v_c:.2f}m/s, 流通面积={flow_area:.6f}m²")
        
        # 解包几何参数 - 移除直接访问thermal_conductivity的代码
        delta_wall = self.geometry['wall_thickness']['inner']
        
        def heat_balance_equations(vars):
            """
            热平衡方程组 - 基于论文公式(1)
            三个热流必须相等: q_gas = q_wall = q_coolant
            q_gas = h_g * (T_aw - T_wg) - 燃气侧热流密度
            q_wall = lambda_wall/delta_wall * (T_wg - T_wc) - 壁面热流密度
            q_coolant = k_R * h_c * (T_wc - T_coolant_in) - 冷却剂侧热流密度
            通过求解以下两个方程组实现热平衡:
            residual1 = q_gas - q_wall
            residual2 = q_wall - q_coolant
            """
            T_wg, T_wc = vars  # 燃气侧壁温和冷却剂侧壁温
            logger.debug(f"求解器调用: T_wg={T_wg:.1f}K, T_wc={T_wc:.1f}K")
            
            # 计算平均壁温下的热导率（用于壁面热传导）
            T_avg = (T_wg + T_wc) / 2
            lambda_wall = self.calculate_thermal_conductivity(T_avg, material_properties)
            
            # 膜冷却修正
            if film_cooling_params and film_cooling_params['film_type'] != 'none':
                # 获取燃气物性
                d_throat = self.geometry['diameter']['throat']
                eps = (d_local / d_throat) ** 2

                gas_props = self.fluid_props.get_combustion_gas_properties(T_gas, P_chamber, eps=eps)
                mu_g = gas_props['viscosity']
                
                if film_cooling_params['film_type'] == 'liquid':
                    # 液膜冷却计算 - 论文公式(11)-(12)
                    Re_lf = (film_cooling_params['film_flow_rate'] * d_h / 
                            (mu_g * A_local))
                    eta_lf = self.liquid_film_cooling_efficiency(Re_lf)
                    
                    # 液膜热平衡修正
                    T_lf = film_cooling_params['film_temperature']
                    h_g = self.gas_side_heat_transfer_coefficient(
                        P_chamber, c_star, T_gas, T_wg, Ma, d_local, A_local, A_throat
                    )
                    
                    # 液膜与燃气换热
                    q_g_lf = h_g * (T_gas - T_lf)
                    # 液膜与壁面换热
                    h_lf = self.coolant_side_heat_transfer_coefficient(
                        T_lf, P_chamber, T_wg, v_c, d_h
                    )[0]
                    q_lf_w = h_lf * (T_lf - T_wg)
                    
                    # 修正热流
                    q_gas = (q_g_lf - q_lf_w) * eta_lf
                    
                else:  # 气膜冷却
                    # 气膜冷却计算 - 论文公式(13)-(15)
                    X = x_position / d_local
                    eta_gf = self.gas_film_cooling_efficiency(
                        X, m_dot_coolant, film_cooling_params['film_flow_rate'],
                        d_local, x_position, mu_g
                    )
                    
                    # 绝热壁温修正 - 论文公式(14)
                    T_gf0 = self.T_CRITICAL_METHANE  # 气膜初始温度(甲烷临界温度)
                    T_aw = T_gas - eta_gf * (T_gas - T_gf0)
                    
                    h_g = self.gas_side_heat_transfer_coefficient(
                        P_chamber, c_star, T_gas, T_wg, Ma, d_local, A_local, A_throat
                    )
                    q_gas = h_g * (T_aw - T_wg)
            else:
                # 无膜冷却，使用原始计算方法
                h_g = self.gas_side_heat_transfer_coefficient(
                    P_chamber, c_star, T_gas, T_wg, Ma, d_local, A_local, A_throat
                )
                k_g = 1.2
                T_aw = T_gas * (1 + (k_g - 1) / 2 * Ma**2)
                q_gas = h_g * (T_aw - T_wg)
            
            # 壁面热传导
            q_wall = lambda_wall / delta_wall * (T_wg - T_wc)
            
            # 冷却剂侧对流换热
            h_c, _ = self.coolant_side_heat_transfer_coefficient(
                T_coolant_in, P_coolant_in, T_wc, v_c, d_h
            )
            b_channel_local = geometry_params.get('local_channel_width', 0.003)
            k_R = self.fin_correction_factor(h_c, lambda_wall, b_channel_local, T_wc)
            q_coolant = k_R * h_c * (T_wc - T_coolant_in)
            
            ref_flux = max(abs(q_gas), 1e3)
            residual1 = (q_gas - q_wall) / ref_flux
            residual2 = (q_wall - q_coolant) / ref_flux

            logger.debug(f"热流: q_gas={q_gas:.2f}, q_wall={q_wall:.2f}, q_coolant={q_coolant:.2f}")
            logger.debug(f"残差: residual1={residual1:.6f}, residual2={residual2:.6f}")
            
            return [residual1, residual2]

        
        # 初始猜测值 (基于典型发动机壁温范围)
        T_wc_guess = max(T_coolant_in, 100.0)  # 确保不低于冷却剂入口温度
        T_wg_guess = min(T_gas, max(500.0, T_wc_guess + 200.0))  # 合理温差
        logger.debug(f"初始猜测: T_wg={T_wg_guess}K, T_wc={T_wc_guess}K")
        initial_guess = [T_wg_guess, T_wc_guess]
        
        # 使用fsolve求解非线性方程组
        try:
            sol, infodict, ier, mesg = fsolve(heat_balance_equations, initial_guess, full_output=True)
            T_wg_sol, T_wc_sol = sol
            logger.debug(f"求解结果: T_wg={T_wg_sol:.2f}K, T_wc={T_wc_sol:.2f}K, ier={ier}, mesg={mesg}")

            if ier != 1:
                logger.warning(f"热平衡fsolve未收敛: {mesg}. 返回初值。T_wg={T_wg_sol}, T_wc={T_wc_sol}")
                return initial_guess[0], initial_guess[1]
            # 检查解的有效性
            if T_wg_sol < 1.0 or T_wc_sol < 1.0:
                logger.warning(f"热平衡求解得到非物理温度: T_wg={T_wg_sol}, T_wc={T_wc_sol}. 返回初值。")
                return initial_guess[0], initial_guess[1]
            logger.debug(f"热平衡求解成功: T_wg={T_wg_sol:.2f}K, T_wc={T_wc_sol:.2f}K")
            return T_wg_sol, T_wc_sol
        except Exception as e:
            logger.error(f"热平衡求解失败: {e}")
            # 返回保守估计值
            return initial_guess[0], initial_guess[1]

    def calculate_coolant_temperature_rise(self, m_dot_coolant, T_in, P_in, 
                                         q_total, A_heat_transfer):
        """
        计算冷却剂在吸收热量后的温度升高值 - 基于能量守恒方程
        
        参数:
        m_dot_coolant: float - 冷却剂质量流量 [kg/s]
        T_in: float - 冷却剂入口温度 [K]
        P_in: float - 冷却剂入口压力 [Pa]
        q_total: float - 总热流量 [W]
        A_heat_transfer: float - 传热面积 [m²]
        
        返回:
        float: 冷却剂温升值 [K]
        """
        logger.debug(f"计算冷却剂温升: m_dot={m_dot_coolant}kg/s, T_in={T_in}K, P_in={P_in}Pa, q_total={q_total}W, A_heat_transfer={A_heat_transfer}m²")

        # 获取冷却剂物性 (使用REFPROP精确计算)
        coolant_props = self.fluid_props.get_methane_properties(T_in, P_in)
        cp_initial = coolant_props['specific_heat']  # 初始比热 [J/(kg·K)]

        logger.debug(f"冷却剂初始比热: cp={cp_initial} J/(kg·K)")
        
        # 初步温升估计 (假设比热恒定)
        delta_T_initial = q_total / (m_dot_coolant * cp_initial)
        T_out_initial = T_in + delta_T_initial

        logger.debug(f"初步温升估计: ΔT_initial={delta_T_initial:.2f}K, T_out_initial={T_out_initial:.2f}K")
        
        # 使用平均温度重新计算物性 (提高精度)
        T_avg = (T_in + T_out_initial) / 2
        coolant_props_avg = self.fluid_props.get_methane_properties(T_avg, P_in)
        cp_avg = coolant_props_avg['specific_heat']

        logger.debug(f"平均温度物性: T_avg={T_avg:.1f}K, cp_avg={cp_avg:.0f}J/(kg·K)")
        
        # 最终温升计算
        delta_T = q_total / (m_dot_coolant * cp_avg)

        logger.debug(f"最终温升计算结果:")
        logger.debug(f"- 温升: ΔT={delta_T:.2f}K")
        logger.debug(f"- 出口温度: T_out={T_in + delta_T:.1f}K")
        logger.debug(f"- 精度提升: 使用平均温度计算，比热变化{abs(cp_avg-cp_initial)/cp_initial*100:.1f}%")
        
        return delta_T

    def calculate_axial_distribution(self, T_gas, P_chamber, c_star, 
                                   m_dot_coolant, T_coolant_in, P_coolant_in,
                                   material_properties, operating_conditions,
                                   num_segments, mixture_ratio=3.5, film_cooling_params=None):
        """
        计算推力室轴向温度、压力和热流分布
        
        参数:
        T_gas: 燃气温度 [K]
        P_chamber: 燃烧室压力 [Pa]
        c_star: 特征速度 [m/s]
        m_dot_coolant: 冷却剂质量流量 [kg/s]
        T_coolant_in: 冷却剂入口温度 [K]
        P_coolant_in: 冷却剂入口压力 [Pa]
        material_properties: 材料属性字典
        operating_conditions: 运行条件字典
        num_segments: 轴向分段数量
        film_cooling_params: dict - 膜冷却参数，包含：
            - film_type: str - 膜冷却类型 ('liquid', 'gas', 'none')
            - film_flow_rate: float - 膜冷却剂流量 [kg/s]
            - film_start_position: float - 膜冷却起始位置 [m]
            - film_temperature: float - 膜冷却剂初始温度 [K]
        mixture_ratio: float - 混合比 (O/F)
        
        返回:
        list: 包含每个轴向位置结果的字典列表
        """
        if not self._geometry_configured:
            raise ValueError("请先使用set_geometric_parameters()配置几何参数")
        
        # 获取推力室总长度
        L_total = self.geometry['length']['chamber']
        logger.debug(f"推力室总长度: {L_total:.3f} m")
        
        # 计算轴向分段
        segment_length = L_total / num_segments
        logger.debug(f"轴向分段长度: {segment_length:.4f} m, 分段数: {num_segments}")
        axial_results = []
        
        # 初始化冷却剂参数
        T_coolant_current = T_coolant_in
        P_coolant_current = P_coolant_in
        logger.debug(f"冷却剂入口温度: {T_coolant_in} K, 压力: {P_coolant_in} Pa")

        coolant_properties_data = []
        gas_properties_data = []

        T0 = T_gas  # 滞止温度（燃烧室入口温度）

        # 重置警告标志
        self.fluid_props.reset_warning_flags()
        
        for i in range(num_segments):
            # 计算当前轴向位置
            x_position = i * segment_length
            logger.debug(f"=== 计算段 {i+1}/{num_segments} === 位置: {x_position:.3f}m")
        
            # 判断是否在膜冷却区域
            film_cooling_active = False
            current_film_params = None
            
            if film_cooling_params and x_position >= film_cooling_params['start_position']:
                film_cooling_active = True
                current_film_params = film_cooling_params.copy()
                
                # 根据位置确定膜冷却类型
                if x_position < film_cooling_params['liquid_film_end']:
                    current_film_params['film_type'] = 'liquid'
                else:
                    current_film_params['film_type'] = 'gas'
            
            # 计算当地几何参数
            d_local, A_local = self.calculate_local_geometry(x_position, L_total)
            logger.debug(f"当地几何: 直径={d_local:.4f}m, 面积={A_local:.6f}m²")

            flow_area, b_channel_local, local_radius = self.calculate_flow_area(x_position, L_total)
            logger.debug(f"流通参数: 面积={flow_area:.6f}m², 通道宽={b_channel_local:.4f}m, 半径={local_radius:.3f}m")

            v_current = self.calculate_coolant_velocity(
                m_dot_coolant, T_coolant_current, P_coolant_current, flow_area
            )
            logger.debug(f"冷却剂流速: {v_current:.2f} m/s")

            local_radius = d_local / 2  # d_local 是当地直径
            circumference = 2 * math.pi * local_radius
            number_of_fins = self.geometry['cooling_channels']['number_of_fins']
            t_fin = self.geometry['cooling_channels']['fin_thickness']
            arc_per_fin = circumference / number_of_fins
            b_channel_local = arc_per_fin - t_fin

            # 验证计算结果的合理性
            if b_channel_local <= 0:
                logger.warning(f"位置 {x_position:.3f}m 处计算出的冷却通道宽度无效: {b_channel_local:.6f}m，使用最小值 1e-6m")
                b_channel_local = 1e-6  # 设置最小正值避免计算错误
            
            # 计算当地马赫数 (简化模型)
            if x_position < L_total * 0.3:
                Ma_local = 0.1  # 燃烧室段低速
            elif x_position < L_total * 0.6:
                Ma_local = 0.5 + (x_position - L_total*0.3)/(L_total*0.3) * 1.5  # 收敛段加速
            else:
                Ma_local = 2.0 + (x_position - L_total*0.6)/(L_total*0.4) * 1.0  # 扩张段加速
            logger.debug(f"当地马赫数: {Ma_local:.3f}")
            
            # 更新运行条件中的当地参数
            operating_conditions_local = operating_conditions.copy()
            operating_conditions_local['coolant_velocity'] = v_current
            operating_conditions_local['mach_number'] = Ma_local

            gas_props_stag = self.fluid_props.get_combustion_gas_properties(T0, P_chamber, mixture_ratio)
            gamma_g = gas_props_stag.get('specific_heat_ratio', 1.2)

            T_gas_local = 0

            # 等熵关系: T/T0 = 1 / (1 + (γ-1)/2 * Ma²)
            if Ma_local > 0:
                T_gas_local = T0 / (1 + (gamma_g - 1) / 2 * Ma_local**2)
            else:
                T_gas_local = T0  # 马赫数为0时使用滞止温度

            # 最低温度 / 滞止温度限制
            T_gas_local = max(T_gas_local, 500)
            T_gas_local = min(T_gas_local, T0)

            # 记录流速变化用于分析
            if i > 0:
                delta_v = v_current - axial_results[-1]['coolant_velocity']
            else:
                delta_v = 0.0

            logger.debug("开始求解热平衡方程...")
            
            # 求解当前段的热平衡
            geometry_params = {
                'local_area': A_local,
                'throat_area': np.pi * (self.geometry['diameter']['throat']/2)**2,
                'local_diameter': d_local,
                'local_channel_width': b_channel_local,
                'axial_position': x_position
            }
            
            T_wg, T_wc = self.solve_heat_balance(
                T_gas_local, P_chamber, c_star, m_dot_coolant, 
                T_coolant_current, P_coolant_current,
                geometry_params, material_properties, operating_conditions_local,
                current_film_params
            )

            logger.debug(f"求解结果: T_wg={T_wg:.2f}K, T_wc={T_wc:.2f}K")
            
            # 计算当地热流密度
            h_g = self.gas_side_heat_transfer_coefficient(
                P_chamber, c_star, T_gas_local, T_wg, Ma_local, 
                d_local, A_local, geometry_params['throat_area']
            )
            # 绝热壁温T_aw按论文公式计算（无r）
            k_g = 1.2
            T_aw = T_gas_local * (1 + (k_g - 1) / 2 * Ma_local**2)
            q_heat_flux = h_g * (T_aw - T_wg)
            
            # 计算冷却剂温升
            # 假设每个段的热流为当地热流密度乘以当地面积
            q_segment = q_heat_flux * A_local
            delta_T_segment = self.calculate_coolant_temperature_rise(
                m_dot_coolant, T_coolant_current, P_coolant_current,
                q_segment, A_local
            )
            
            # 计算压降
            # 获取冷却剂物性用于压降计算
            coolant_props = self.fluid_props.get_methane_properties(
                T_coolant_current, P_coolant_current
            )
            rho_c = coolant_props['density']
            v_c = operating_conditions['coolant_velocity']
            
            # 估算密度和速度变化
            delta_rho = 50  # 简化假设 [kg/m³]
            delta_v = 0.1 * v_c  # 简化假设
            
            delta_p_segment = self.pressure_drop_calculation(
                segment_length, operating_conditions['hydraulic_diameter'],
                rho_c, v_c, delta_rho, delta_v,
                T_coolant_current, P_coolant_current
            )
            
            # 更新冷却剂状态
            T_coolant_current += delta_T_segment
            P_coolant_current -= delta_p_segment
            
            coolant_props_segment = {
                'axial_position': x_position,
                'density': coolant_props['density'],
                'viscosity': coolant_props['viscosity'],
                'conductivity': coolant_props['conductivity'],
                'specific_heat': coolant_props['specific_heat'],
                'prandtl': coolant_props['prandtl']
            }
            coolant_properties_data.append(coolant_props_segment)
            
            # 基于半径比计算面积比：eps = (r_local / r_throat)^2 = (d_local / d_throat)^2
            d_throat = self.geometry['diameter']['throat']
            eps = (d_local / d_throat) ** 2

            # 传入正确的面积比参数
            gas_props = self.fluid_props.get_combustion_gas_properties(
                T_gas_local, P_chamber, mixture_ratio, eps  # 添加eps参数
            )
            gas_props_segment = {
                'axial_position': x_position,
                'density': gas_props['density'],
                'viscosity': gas_props['viscosity'],
                'conductivity': gas_props['conductivity'],
                'specific_heat': gas_props['specific_heat'],
                'prandtl': gas_props['prandtl']
            }
            gas_properties_data.append(gas_props_segment)

            # 存储当前段结果
            segment_result = {
                'axial_position': x_position,
                'local_diameter': d_local,
                'local_area': A_local,
                'flow_area': flow_area,
                'mach_number': Ma_local,
                'gas_side_wall_temp': T_wg,
                'coolant_side_wall_temp': T_wc,
                'heat_flux': q_heat_flux,
                'coolant_temperature': T_coolant_current,
                'coolant_pressure': P_coolant_current,
                'temperature_rise': delta_T_segment,
                'pressure_drop': delta_p_segment,
                'local_channel_width': b_channel_local,
                'coolant_velocity': v_current,
                'velocity_change': delta_v,
                'coolant_density': coolant_props['density'],
                'mass_flow_rate': m_dot_coolant,
                'coolant_properties': coolant_props,
                'gas_properties': gas_props,
                'gas_temperature': T_gas_local,  # 新增：存储当地燃气温度
                'stagnation_temperature': T0,    # 新增：存储滞止温度参考
            }
            
            axial_results.append(segment_result)
            
            # 每10段打印进度
            if i % 10 == 0:
                logger.info(f"已完成轴向位置 {x_position:.2f}m 的计算")
        
        return axial_results

    def analyze_cooling_performance(self, axial_results):
        """
        分析发动机冷却系统的整体性能指标和统计量
        
        参数:
        axial_results: list - calculate_axial_distribution()函数的输出结果，包含各轴向段的热力学数据
        
        返回:
        dict: 包含冷却性能关键指标的字典，包括：
            - max_wall_temperature: 最高壁面温度 [K]
            - max_heat_flux: 最大热流密度 [W/m²]
            - average_wall_temperature: 平均壁面温度 [K]
            - total_heat_load: 总热负荷 [W]
            - max_temp_position: 最高温度位置 [m]
            - total_pressure_drop: 总压降 [Pa]
            - coolant_outlet_temperature: 冷却剂出口温度 [K]
            - coolant_outlet_pressure: 冷却剂出口压力 [Pa]
        """
        logger.info("开始分析冷却性能指标...")

        if not axial_results:
            logger.warning("轴向结果为空，无法进行性能分析")
            return {}
        
        # 提取数据
        positions = [seg['axial_position'] for seg in axial_results]
        T_wg = [seg['gas_side_wall_temp'] for seg in axial_results]
        q_flux = [seg['heat_flux'] for seg in axial_results]

        logger.debug(f"性能分析数据: 点数={len(axial_results)}, 位置范围=[{min(positions):.3f}, {max(positions):.3f}]m")
        
        # 计算关键指标
        max_wall_temp = max(T_wg)
        max_heat_flux = max(q_flux)
        avg_wall_temp = sum(T_wg) / len(T_wg)
        total_heat_load = sum([seg['heat_flux'] * seg['local_area'] for seg in axial_results])
        
        # 找到最高温度位置
        max_temp_index = np.argmax(T_wg)
        max_temp_position = positions[max_temp_index]

        logger.debug(f"温度分析: 最高壁温={max_wall_temp:.1f}K (位置={max_temp_position:.3f}m)")
        logger.debug(f"热流分析: 最大热流={max_heat_flux/1e6:.2f}MW/m², 平均壁温={avg_wall_temp:.1f}K")
        logger.debug(f"热负荷分析: 总热负荷={total_heat_load/1e6:.2f}MW")
        
        # 计算压降总和
        total_pressure_drop = sum([seg['pressure_drop'] for seg in axial_results])
        logger.debug(f"压降分析: 总压降={total_pressure_drop/1e3:.2f}kPa")
        
        performance = {
            'max_wall_temperature': max_wall_temp,
            'max_heat_flux': max_heat_flux,
            'average_wall_temperature': avg_wall_temp,
            'total_heat_load': total_heat_load,
            'max_temp_position': max_temp_position,
            'total_pressure_drop': total_pressure_drop,
            'coolant_outlet_temperature': axial_results[-1]['coolant_temperature'],
            'coolant_outlet_pressure': axial_results[-1]['coolant_pressure']
        }

        # 质量流量守恒验证
        logger.debug("开始验证质量流量守恒...")
        mass_flow_errors = []
        for i, seg in enumerate(axial_results):
            # 计算当地质量流量: ṁ_calc = ρ·v·A
            rho = seg.get('coolant_density', 0)
            v = seg.get('coolant_velocity', 0)
            A_flow = seg.get('flow_area', 0)
            m_dot_calc = rho * v * A_flow
            
            # 理论质量流量（应恒定）
            m_dot_theoretical = axial_results[0]['mass_flow_rate']  # 入口质量流量
            
            if m_dot_theoretical > 0:
                error = abs(m_dot_calc - m_dot_theoretical) / m_dot_theoretical * 100
                mass_flow_errors.append(error)
        
        if mass_flow_errors:
            max_error = max(mass_flow_errors)
            avg_error = sum(mass_flow_errors) / len(mass_flow_errors)
            performance['mass_flow_conservation'] = {
                'max_error_percent': max_error,
                'average_error_percent': avg_error,
                'is_conserved': avg_error < 1.0  # 1%误差阈值
            }

            logger.debug(f"质量流量守恒: 平均误差={avg_error:.2f}%, 最大误差={max_error:.2f}%, 守恒={avg_error < 1.0}")
            
            if avg_error > 1.0:
                logger.warning(f"质量流量守恒误差较大: 平均误差={avg_error:.2f}%")
        
        return performance

    def evaluate_design_parameters(self, T_gas, P_chamber, c_star, 
                                 m_dot_coolant, T_coolant_in, P_coolant_in,
                                 material_properties, operating_conditions):
        """
        评估关键设计参数对发动机冷却性能的敏感性影响
        
        参数:
        T_gas: float - 燃气温度 [K]
        P_chamber: float - 燃烧室压力 [Pa]
        c_star: float - 特征速度 [m/s]
        m_dot_coolant: float - 冷却剂质量流量 [kg/s]
        T_coolant_in: float - 冷却剂入口温度 [K]
        P_coolant_in: float - 冷却剂入口压力 [Pa]
        material_properties: dict - 材料属性字典，包含热导率等参数
        operating_conditions: dict - 运行条件字典，包含流速、水力直径等参数
        
        返回:
        dict: 包含三个敏感性分析结果的字典：
            - flow_rate_analysis: 冷却剂流量影响分析
            - pressure_analysis: 燃烧室压力影响分析  
            - wall_thickness_analysis: 壁厚影响分析
        """
        # 存储不同参数下的结果
        sensitivity_results = {}
        
        # 1. 冷却剂流量影响分析
        flow_rates = [m_dot_coolant * 0.8, m_dot_coolant, m_dot_coolant * 1.2]
        flow_results = []
        
        for flow_rate in flow_rates:
            axial_results = self.calculate_axial_distribution(
                T_gas, P_chamber, c_star, flow_rate, 
                T_coolant_in, P_coolant_in,
                material_properties, operating_conditions,
                num_segments=30  # 减少分段数以提高计算效率
            )
            
            performance = self.analyze_cooling_performance(axial_results)
            flow_results.append({
                'flow_rate': flow_rate,
                'max_wall_temp': performance['max_wall_temperature'],
                'outlet_temp': performance['coolant_outlet_temperature']
            })
        
        sensitivity_results['flow_rate_analysis'] = flow_results
        
        # 2. 室压影响分析
        chamber_pressures = [P_chamber * 0.8, P_chamber, P_chamber * 1.2]
        pressure_results = []
        
        for pressure in chamber_pressures:
            axial_results = self.calculate_axial_distribution(
                T_gas, pressure, c_star, m_dot_coolant,
                T_coolant_in, P_coolant_in,
                material_properties, operating_conditions,
                num_segments=30
            )
            
            performance = self.analyze_cooling_performance(axial_results)
            pressure_results.append({
                'chamber_pressure': pressure,
                'max_wall_temp': performance['max_wall_temperature'],
                'max_heat_flux': performance['max_heat_flux']
            })
        
        sensitivity_results['pressure_analysis'] = pressure_results
        
        # 3. 壁厚影响分析
        wall_thicknesses = [
            self.geometry['wall_thickness']['inner'] * 0.5,
            self.geometry['wall_thickness']['inner'],
            self.geometry['wall_thickness']['inner'] * 1.5
        ]
        wall_results = []
        
        for thickness in wall_thicknesses:
            # 临时修改壁厚
            original_thickness = self.geometry['wall_thickness']['inner']
            self.geometry['wall_thickness']['inner'] = thickness
            
            axial_results = self.calculate_axial_distribution(
                T_gas, P_chamber, c_star, m_dot_coolant,
                T_coolant_in, P_coolant_in,
                material_properties, operating_conditions,
                num_segments=30
            )
            
            performance = self.analyze_cooling_performance(axial_results)
            wall_results.append({
                'wall_thickness': thickness,
                'max_wall_temp': performance['max_wall_temperature'],
                'avg_wall_temp': performance['average_wall_temperature']
            })
            
            # 恢复原始壁厚
            self.geometry['wall_thickness']['inner'] = original_thickness
        
        sensitivity_results['wall_thickness_analysis'] = wall_results
        
        return sensitivity_results


class MethaneEnginePlotGenerator:
    """
    液氧/甲烷发动机传热分析绘图生成器
    
    功能：
    - 自动生成发动机几何形状和热力学参数图表
    - 支持多种物性参数的可视化分析
    - 提供轴向分布数据的专业绘图功能
    - 统一管理图表输出目录和格式
    
    主要图表类型：
    - 发动机几何形状轮廓图
    - 温度分布和梯度分析图
    - 冷却剂流动参数分布图
    - 冷却剂物性参数分布图
    - 燃气物性参数分布图
    """
    
    def __init__(self, output_dir="MethaneEngineHeatTransferPlots"):
        """
        初始化绘图生成器
        
        参数:
        output_dir: str - 图表输出目录路径，默认为"MethaneEngineHeatTransferPlots"
        
        功能：
        - 创建图表输出目录
        - 设置图表保存路径
        - 初始化绘图参数
        """
        self.output_dir = output_dir
        self.csv_dir = os.path.join(output_dir, "CSV_Data")
        self._setup_plotting_environment()
        self._create_output_directory()

    def _setup_plotting_environment(self):
        """设置绘图环境"""
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass
        
    def _create_output_directory(self):
        """
        创建图表输出目录
        
        功能：
        - 检查目录是否存在
        - 自动创建不存在的目录
        - 记录目录创建状态
        
        返回:
        None - 通过日志输出创建结果
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"创建图表输出目录: {self.output_dir}")
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)
            logger.info(f"创建CSV数据输出目录: {self.csv_dir}")

    def save_plot_with_font_fix(self, filename, dpi=300, bbox_inches='tight'):
        """
        带字体修复的图表保存函数 - 增强ASCII支持
        
        参数:
        filename: 文件名
        dpi: 分辨率
        bbox_inches: 边界框设置
        """
        try:
            # 保存前确保使用ASCII兼容设置
            original_settings = {
                'unicode_minus': plt.rcParams.get('axes.unicode_minus', True),
                'mathtext_fontset': plt.rcParams.get('mathtext.fontset', 'stix')
            }
            
            # 强制使用ASCII安全设置
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['mathtext.fontset'] = 'stix'  # 使用STIX避免Unicode问题
            
            # 方法1: 使用agg后端保存（最佳兼容性）
            plt.savefig(os.path.join(self.output_dir, filename), 
                    dpi=dpi, 
                    bbox_inches=bbox_inches,
                    backend='agg',  # 使用agg后端
                    facecolor='white',  # 确保背景色
                    edgecolor='none')   # 无边框
            
            logger.debug(f"图表保存成功: {filename}")
            
            # 恢复原始设置
            plt.rcParams['axes.unicode_minus'] = original_settings['unicode_minus']
            plt.rcParams['mathtext.fontset'] = original_settings['mathtext_fontset']
            
        except Exception as e:
            logger.warning(f"标准保存方法失败: {e}，尝试备用方法")
            
            try:
                # 方法2: 简化设置再次尝试
                plt.rcParams['axes.unicode_minus'] = False
                
                plt.savefig(os.path.join(self.output_dir, filename), 
                        dpi=max(dpi, 150),  # 降低分辨率确保成功
                        bbox_inches=bbox_inches,
                        facecolor='white')
                
            except Exception as e2:
                logger.error(f"备用保存方法也失败: {e2}")
                # 最后尝试：最低质量保存
                try:
                    plt.savefig(os.path.join(self.output_dir, filename), 
                            dpi=100, 
                            bbox_inches=bbox_inches,
                            quality=85, 
                            optimize=True)
                except Exception as e3:
                    logger.critical(f"所有保存方法均失败: {e3}")

    def save_data_to_csv(self, data_dict, filename):
        """
        将数据保存为CSV文件
        
        参数:
        data_dict: dict - 包含列名和数据的字典
        filename: str - 输出文件名
        """
        try:
            import pandas as pd
            
            # 确保数据长度一致
            max_length = max(len(v) for v in data_dict.values() if hasattr(v, '__len__'))
            
            # 填充数据使其长度一致
            formatted_data = {}
            for key, values in data_dict.items():
                if hasattr(values, '__len__'):
                    if len(values) < max_length:
                        # 填充NaN使长度一致
                        formatted_data[key] = list(values) + [np.nan] * (max_length - len(values))
                    else:
                        formatted_data[key] = values
                else:
                    # 标量值重复填充
                    formatted_data[key] = [values] * max_length
            
            df = pd.DataFrame(formatted_data)
            csv_filename = filename.replace('.png', '.csv')
            df.to_csv(os.path.join(self.csv_dir, csv_filename), index=False, encoding='utf-8-sig')
            logger.info(f"数据已保存为CSV文件: {csv_filename}")
            
        except Exception as e:
            logger.warning(f"CSV文件保存失败: {e}")
    
    def plot_engine_shape(self, axial_results, geometry_data, filename="engine_shape.png"):
        """
        绘制发动机几何形状轮廓图
        
        参数:
        axial_results: list - 轴向分布计算结果，包含各位置几何数据
        geometry_data: dict - 发动机几何参数字典
        filename: str - 输出文件名，默认为"engine_shape.png"
        
        返回:
        None - 图表保存至指定文件
        
        功能：
        - 绘制发动机内壁轮廓线
        - 标记喉部关键位置
        - 显示对称轮廓和填充区域
        - 设置等比例坐标轴
        """
        plt.figure(figsize=(12, 8))
        
        # 提取数据
        positions = [seg['axial_position'] for seg in axial_results]
        diameters = [seg['local_diameter'] for seg in axial_results]
        radii = [d/2 for d in diameters]
        
        # 绘制发动机轮廓
        plt.plot(positions, radii, 'b-', linewidth=2, label='发动机内壁')
        plt.plot(positions, [-r for r in radii], 'b-', linewidth=2)
        plt.fill_between(positions, radii, [-r for r in radii], alpha=0.3, color='lightblue')
        
        # 标记关键位置
        throat_pos = positions[np.argmin(diameters)]
        throat_diam = min(diameters)
        plt.plot([throat_pos, throat_pos], [-throat_diam/2, throat_diam/2], 
                'r--', linewidth=2, label='喉部位置')
        
        plt.xlabel('轴向位置 (m)')
        plt.ylabel('半径 (m)')
        plt.title('发动机几何形状轮廓')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.tight_layout()
        self.save_plot_with_font_fix(filename, dpi=300, bbox_inches='tight')
        plt.close()

        data_dict = {
            'axial_position_m': positions,
            'local_diameter_m': diameters
        }

        self.save_data_to_csv(data_dict, filename)

    def plot_temperature_distribution(self, axial_results, filename="temperature_distribution.png"):
        """
        绘制发动机轴向温度分布图
        
        参数:
        axial_results: list - 轴向分布计算结果，包含温度数据
        filename: str - 输出文件名，默认为"temperature_distribution.png"
        
        返回:
        None - 图表保存至指定文件
        
        功能：
        - 显示燃气侧壁温、冷却剂侧壁温和冷却剂温度
        - 计算并显示壁温轴向梯度
        - 使用双子图布局分别显示温度和梯度
        - 添加网格线和图例说明
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 提取温度数据
        positions = [seg['axial_position'] for seg in axial_results]
        T_wg = [seg['gas_side_wall_temp'] for seg in axial_results]
        T_wc = [seg['coolant_side_wall_temp'] for seg in axial_results]
        T_coolant = [seg['coolant_temperature'] for seg in axial_results]

        # 提取燃气温度数据
        T_gas = [seg.get('gas_temperature', seg.get('stagnation_temperature', 3000)) for seg in axial_results]  # 兼容旧数据
        Ma_numbers = [seg.get('mach_number', 0) for seg in axial_results]
        
        # 1. 主温度图
        ax1.plot(positions, T_gas, 'c--', linewidth=3, label='燃气温度', alpha=0.9)
        ax1.plot(positions, T_wg, 'r-', linewidth=2, label='燃气侧壁温', alpha=0.8)
        ax1.plot(positions, T_wc, 'g-', linewidth=2, label='冷却剂侧壁温', alpha=0.8)
        ax1.plot(positions, T_coolant, 'b-', linewidth=2, label='冷却剂温度', alpha=0.8)
        ax1.set_xlabel('轴向位置 (m)')
        ax1.set_ylabel('温度 (K)')
        ax1.set_title('发动机轴向温度分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # 标记关键位置（喉部等）
        throat_index = np.argmin([seg['local_diameter'] for seg in axial_results])
        ax1.axvline(x=positions[throat_index], color='gray', linestyle='--', alpha=0.7, label='喉部位置')
        
        # 2. 温度梯度图
        dT_dx = np.gradient(T_wg, positions)
        ax2.plot(positions, dT_dx, color='purple', linestyle='-', linewidth=2, label='壁温梯度')
        ax2.set_xlabel('轴向位置 (m)')
        ax2.set_ylabel('温度梯度 (K/m)')
        ax2.set_title('壁温轴向梯度分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 温度比分析（展示燃气温度变化规律）
        T_ratio = [T_gas[i]/T_gas[0] for i in range(len(T_gas))]  # 当地温度/滞止温度
        theoretical_ratio = [1/(1 + 0.1 * Ma**2) for Ma in Ma_numbers]  # 理论等熵关系
        
        ax3.plot(positions, T_ratio, 'b-', linewidth=2, label='实际温度比 T/T₀')
        ax3.plot(positions, theoretical_ratio, 'r--', linewidth=2, label='理论等熵关系')
        ax3.set_xlabel('轴向位置 (m)')
        ax3.set_ylabel('温度比 T/T₀')
        ax3.set_title('燃气温度比分布（vs 理论等熵关系）')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 马赫数与温度关联分析
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(positions, Ma_numbers, 'g-', linewidth=2, label='马赫数')
        line2 = ax4_twin.plot(positions, T_gas, 'm-', linewidth=2, label='燃气温度')
        ax4.set_xlabel('轴向位置 (m)')
        ax4.set_ylabel('马赫数', color='g')
        ax4_twin.set_ylabel('燃气温度 (K)', color='m')
        ax4.set_title('马赫数与燃气温度关联分析')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_plot_with_font_fix(filename, dpi=300, bbox_inches='tight')
        plt.close()

        data_dict = {
            'axial_position_m': positions,
            'gas_temperature_K': T_gas,
            'gas_side_wall_temp_K': T_wg,
            'coolant_side_wall_temp_K': T_wc,
            'coolant_temperature_K': T_coolant,
            'dT_wg_dx_K_per_m': dT_dx,
            'local_mach_number': Ma_numbers
        }
        
        self.save_data_to_csv(data_dict, filename)

    def plot_gas_temperature_analysis(self, axial_results, filename="gas_temperature_analysis.png"):
        """
        绘制燃气温度分析图表

        参数:
        axial_results: list - 轴向分布计算结果，包含燃气温度数据
        filename: str - 输出文件名，默认为"gas_temperature_analysis.png"

        返回:
        None - 图表保存至指定文件

        功能：
        - 分析燃气温度与马赫数关系
        - 研究几何直径与燃气温度的关联
        - 计算燃气温度变化率并标记最大降温率位置
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        positions = [seg['axial_position'] for seg in axial_results]
        T_gas = [seg.get('gas_temperature', 3000) for seg in axial_results]
        Ma_numbers = [seg.get('mach_number', 0) for seg in axial_results]
        diameters = [seg['local_diameter'] for seg in axial_results]
        
        # 1. 燃气温度与马赫数关系
        ax1.plot(positions, T_gas, 'r-', linewidth=3, label='燃气温度')
        ax1.set_xlabel('轴向位置 (m)')
        ax1.set_ylabel('燃气温度 (K)', color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        ax1.set_ylim(bottom=0)
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(positions, Ma_numbers, 'b--', linewidth=2, label='马赫数')
        ax1_twin.set_ylabel('马赫数', color='b')
        ax1_twin.tick_params(axis='y', labelcolor='b')
        ax1_twin.set_ylim(bottom=0)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax1.set_title('燃气温度与马赫数轴向分布')
        ax1.grid(True, alpha=0.3)
        
        # 2. 直径与温度关系
        ax2.plot(positions, diameters, 'g-', linewidth=2, label='当地直径')
        ax2.set_xlabel('轴向位置 (m)')
        ax2.set_ylabel('直径 (m)', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(positions, T_gas, 'orange', linewidth=2, label='燃气温度')
        ax2_twin.set_ylabel('燃气温度 (K)', color='orange')
        ax2_twin.tick_params(axis='y', labelcolor='orange')
        
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax2.set_title('几何直径与燃气温度关系')
        ax2.grid(True, alpha=0.3)
        
        # 3. 温度变化率分析
        dT_dx = np.gradient(T_gas, positions)
        ax3.plot(positions, dT_dx, 'purple', linewidth=2, label='燃气温度梯度')
        ax3.set_xlabel('轴向位置 (m)')
        ax3.set_ylabel('温度梯度 (K/m)')
        ax3.set_title('燃气温度轴向梯度分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 标记最大降温率点
        min_grad_index = np.argmin(dT_dx)  # 最大负梯度（降温最快）
        ax3.axvline(x=positions[min_grad_index], color='red', linestyle=':', alpha=0.7)
        ax3.text(positions[min_grad_index], dT_dx[min_grad_index], 
                f'最大降温率: {dT_dx[min_grad_index]:.1f} K/m', 
                ha='center', va='bottom', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # 4. 理论vs实际温度比较
        T0 = T_gas[0]  # 滞止温度
        theoretical_T = [T0 / (1 + 0.1 * Ma**2) for Ma in Ma_numbers]  # 简化理论计算
        
        ax4.plot(positions, T_gas, 'b-', linewidth=2, label='实际燃气温度')
        ax4.plot(positions, theoretical_T, 'r--', linewidth=2, label='理论等熵温度')
        ax4.fill_between(positions, T_gas, theoretical_T, alpha=0.2, color='gray', label='偏差区域')
        ax4.set_xlabel('轴向位置 (m)')
        ax4.set_ylabel('温度 (K)')
        ax4.set_title('实际vs理论燃气温度分布')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 计算平均偏差
        avg_deviation = np.mean([abs(T_gas[i] - theoretical_T[i]) for i in range(len(T_gas))])
        ax4.text(0.02, 0.98, f'平均偏差: {avg_deviation:.1f} K', 
                transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        self.save_plot_with_font_fix(filename, dpi=300, bbox_inches='tight')
        plt.close()

        data_dict = {
            'axial_position_m': positions,
            'T_gas_K': T_gas,
            'mach_number': Ma_numbers,
            'dT_dx_K_per_m': dT_dx
        }

        self.save_data_to_csv(data_dict, filename)

    def plot_coolant_flow_parameters(self, axial_results, filename="coolant_flow_parameters.png"):
        """
        绘制冷却剂流动参数分布图
        
        参数:
        axial_results: list - 轴向分布计算结果，包含流动参数数据
        filename: str - 输出文件名，默认为"coolant_flow_parameters.png"
        
        返回:
        None - 图表保存至指定文件
        
        功能：
        - 显示冷却剂流速、压力、温升和压降分布
        - 计算累计压力损失
        - 使用2x2子图布局显示多参数
        - 自动处理单位转换和坐标轴标签
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 提取数据
        positions = [seg['axial_position'] for seg in axial_results]
        velocities = [seg.get('coolant_velocity', 0) for seg in axial_results]
        pressures = [seg['coolant_pressure'] for seg in axial_results]
        temperature_rises = [seg.get('temperature_rise', 0) for seg in axial_results]
        pressure_drops = [seg.get('pressure_drop', 0) for seg in axial_results]
        
        # 累计压力降
        cumulative_pressure_drop = np.cumsum(pressure_drops)
        
        # 1. 冷却剂流速分布
        ax1.plot(positions, velocities, 'b-', linewidth=2, label='冷却剂流速')
        ax1.set_xlabel('轴向位置 (m)')
        ax1.set_ylabel('流速 (m/s)')
        ax1.set_title('冷却剂流速轴向分布')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 冷却剂压力分布
        ax2.plot(positions, [p/1e6 for p in pressures], 'r-', linewidth=2, label='冷却剂压力')
        ax2.set_xlabel('轴向位置 (m)')
        ax2.set_ylabel('压力 (MPa)')
        ax2.set_title('冷却剂压力轴向分布')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. 冷却剂温升分布
        ax3.plot(positions, temperature_rises, 'g-', linewidth=2, label='段温升')
        ax3.set_xlabel('轴向位置 (m)')
        ax3.set_ylabel('温升 (K)')
        ax3.set_title('冷却剂分段温升')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. 累计压力降
        ax4.plot(positions, cumulative_pressure_drop/1e6, color='purple', linewidth=2, label='累计压降')
        ax4.set_xlabel('轴向位置 (m)')
        ax4.set_ylabel('累计压降 (MPa)')
        ax4.set_title('冷却剂累计压力损失')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        self.save_plot_with_font_fix(filename, dpi=300, bbox_inches='tight')
        plt.close()

        data_dict = {
            'axial_position_m': positions,
            'coolant_velocity_m_per_s': velocities,
            'coolant_pressure_Pa': pressures,
            'temperature_rise_K': temperature_rises,
            'pressure_drop_Pa': pressure_drops,
            'cumulative_pressure_drop_Pa': cumulative_pressure_drop
        }

        self.save_data_to_csv(data_dict, filename)

    def plot_coolant_properties(self, axial_results, filename="coolant_properties.png"):
        """
        绘制冷却剂物性参数分布图
        
        参数:
        axial_results: list - 轴向分布计算结果，包含冷却剂物性数据
        filename: str - 输出文件名，默认为"coolant_properties.png"
        
        返回:
        None - 图表保存至指定文件
        
        功能：
        - 显示密度、粘度、热导率等关键物性参数
        - 使用双Y轴显示比热和普朗特数
        - 对粘度等参数使用对数坐标
        - 提供完整的物性变化趋势分析
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 提取数据
        positions = [seg['axial_position'] for seg in axial_results]
        
        # 从冷却剂物性数据中提取参数
        densities = []
        viscosities = []
        conductivities = []
        specific_heats = []
        prandtls = []
        
        for seg in axial_results:
            props = seg.get('coolant_properties', {})
            densities.append(props.get('density', 0))
            viscosities.append(props.get('viscosity', 0))
            conductivities.append(props.get('conductivity', 0))
            specific_heats.append(props.get('specific_heat', 0))
            prandtls.append(props.get('prandtl', 0))
        
        # 1. 密度分布
        ax1.plot(positions, densities, 'b-', linewidth=2, label='密度')
        ax1.set_xlabel('轴向位置 (m)')
        ax1.set_ylabel('密度 (kg/m³)')
        ax1.set_title('冷却剂密度轴向分布')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 粘度分布
        ax2.semilogy(positions, viscosities, 'r-', linewidth=2, label='动力粘度')
        ax2.set_xlabel('轴向位置 (m)')
        ax2.set_ylabel('粘度 (Pa·s)')
        ax2.set_title('冷却剂粘度轴向分布(对数坐标)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. 热导率分布
        ax3.plot(positions, conductivities, 'g-', linewidth=2, label='热导率')
        ax3.set_xlabel('轴向位置 (m)')
        ax3.set_ylabel('热导率 (W/(m·K))')
        ax3.set_title('冷却剂热导率轴向分布')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. 比热和普朗特数
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(positions, specific_heats, 'b-', linewidth=2, label='比热')
        line2 = ax4_twin.plot(positions, prandtls, 'r--', linewidth=2, label='普朗特数')
        ax4.set_xlabel('轴向位置 (m)')
        ax4.set_ylabel('比热 (J/(kg·K))', color='b')
        ax4_twin.set_ylabel('普朗特数', color='r')
        ax4.set_title('冷却剂比热和普朗特数分布')
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_plot_with_font_fix(filename, dpi=300, bbox_inches='tight')
        plt.close()

        data_dict = {
            'axial_position_m': positions,
            'coolant_density_kg_per_m3': densities,
            'coolant_viscosity_Pa_s': viscosities,
            'coolant_conductivity_W_per_mK': conductivities,
            'coolant_specific_heat_J_per_kgK': specific_heats,
            'coolant_prandtl_number': prandtls
        }

        self.save_data_to_csv(data_dict, filename)

    def plot_gas_properties(self, axial_results, filename="gas_properties.png"):
        """
        绘制燃气物性参数分布图
        
        参数:
        axial_results: list - 轴向分布计算结果，包含燃气物性数据
        filename: str - 输出文件名，默认为"gas_properties.png"
        
        返回:
        None - 图表保存至指定文件
        
        功能：
        - 显示燃气密度、粘度、热导率等物性参数
        - 对燃气物性使用对数坐标显示
        - 使用双Y轴显示比热和普朗特数
        - 提供燃气物性变化的综合分析
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 提取数据
        positions = [seg['axial_position'] for seg in axial_results]
        
        # 从燃气物性数据中提取参数
        densities = []
        viscosities = []
        conductivities = []
        specific_heats = []
        prandtls = []
        
        for seg in axial_results:
            props = seg.get('gas_properties', {})
            densities.append(props.get('density', 0))
            viscosities.append(props.get('viscosity', 0))
            conductivities.append(props.get('conductivity', 0))
            specific_heats.append(props.get('specific_heat', 0))
            prandtls.append(props.get('prandtl', 0))
        
        # 1. 燃气密度分布
        ax1.semilogy(positions, densities, 'b-', linewidth=2, label='燃气密度')
        ax1.set_xlabel('轴向位置 (m)')
        ax1.set_ylabel('密度 (kg/m³)')
        ax1.set_title('燃气密度轴向分布(对数坐标)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 燃气粘度分布
        ax2.semilogy(positions, viscosities, 'r-', linewidth=2, label='燃气粘度')
        ax2.set_xlabel('轴向位置 (m)')
        ax2.set_ylabel('粘度 (Pa·s)')
        ax2.set_title('燃气粘度轴向分布(对数坐标)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. 燃气热导率分布
        ax3.semilogy(positions, conductivities, 'g-', linewidth=2, label='热导率')
        ax3.set_xlabel('轴向位置 (m)')
        ax3.set_ylabel('热导率 (W/(m·K))')
        ax3.set_title('燃气热导率轴向分布(对数坐标)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. 燃气比热和普朗特数
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(positions, specific_heats, 'b-', linewidth=2, label='比热')
        line2 = ax4_twin.plot(positions, prandtls, 'r--', linewidth=2, label='普朗特数')
        ax4.set_xlabel('轴向位置 (m)')
        ax4.set_ylabel('比热 (J/(kg·K))', color='b')
        ax4_twin.set_ylabel('普朗特数', color='r')
        ax4.set_title('燃气比热和普朗特数分布')
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_plot_with_font_fix(filename, dpi=300, bbox_inches='tight')
        plt.close()

        data_dict = {
            'axial_position_m': positions,
            'gas_density_kg_per_m3': densities,
            'gas_viscosity_Pa_s': viscosities,
            'gas_conductivity_W_per_mK': conductivities,
            'gas_specific_heat_J_per_kgK': specific_heats,
            'gas_prandtl_number': prandtls
        }
    
    def plot_velocity_analysis_comparison(self, axial_results, filename="velocity_analysis_comparison.png"):
        """
        绘制流速分析对比图 - 显示新旧计算方法的差异
        
        参数:
        axial_results: list - 轴向分布计算结果
        filename: str - 输出文件名
        
        返回:
        None - 生成对比分析图表
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        positions = [seg['axial_position'] for seg in axial_results]
        velocities = [seg.get('coolant_velocity', 0) for seg in axial_results]
        densities = [seg.get('coolant_density', 0) for seg in axial_results]
        flow_areas = [seg.get('flow_area', 0) for seg in axial_results]
        
        # 1. 流速与密度、流通面积的关联分析
        ax1.plot(positions, velocities, 'b-', linewidth=3, label='流速', alpha=0.8)
        ax1.set_xlabel('轴向位置 (m)', fontsize=12)
        ax1.set_ylabel('流速 (m/s)', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_ylim(bottom=0)
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(positions, densities, 'r--', linewidth=2, label='密度', alpha=0.7)
        ax1_twin.set_ylabel('密度 (kg/m³)', color='r', fontsize=12)
        ax1_twin.tick_params(axis='y', labelcolor='r')
        
        ax1_twin2 = ax1.twinx()
        ax1_twin2.spines['right'].set_position(('outward', 60))
        ax1_twin2.plot(positions, [A*10000 for A in flow_areas], 'g:', 
                    linewidth=2, label='流通面积 (cm²)', alpha=0.7)
        ax1_twin2.set_ylabel('流通面积 (cm²)', color='g', fontsize=12)
        ax1_twin2.tick_params(axis='y', labelcolor='g')
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        lines3, labels3 = ax1_twin2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
        ax1.set_title('流速与密度、流通面积关联分析', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. 流速变化率分析
        velocity_gradient = np.gradient(velocities, positions)
        acceleration = np.gradient(velocity_gradient, positions)
        
        ax2.plot(positions, velocity_gradient, 'b-', linewidth=2, label='流速梯度')
        ax2.plot(positions, acceleration, 'r--', linewidth=2, label='加速度')
        ax2.set_xlabel('轴向位置 (m)', fontsize=12)
        ax2.set_ylabel('变化率 (m/s²)', fontsize=12)
        ax2.legend()
        ax2.set_title('流速变化率分析', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 标记最大加速度点
        max_accel_index = np.argmax(np.abs(acceleration))
        ax2.scatter(positions[max_accel_index], acceleration[max_accel_index], 
                color='red', s=100, zorder=5)
        ax2.annotate(f'最大加速度: {acceleration[max_accel_index]:.2f} m/s²', 
                    xy=(positions[max_accel_index], acceleration[max_accel_index]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # 3. 区域流速统计分析
        num_regions = 4
        region_size = len(positions) // num_regions
        region_velocities = []
        region_labels = []
        
        for i in range(num_regions):
            start_idx = i * region_size
            end_idx = start_idx + region_size if i < num_regions - 1 else len(positions)
            region_vels = velocities[start_idx:end_idx]
            region_velocities.append(region_vels)
            region_labels.append(f'区域 {i+1}\n({positions[start_idx]:.2f}-{positions[end_idx-1]:.2f}m)')
        
        # 箱线图显示区域流速分布
        box_plot = ax3.boxplot(region_velocities, labels=region_labels, patch_artist=True)
        
        # 美化箱线图
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_xlabel('轴向区域', fontsize=12)
        ax3.set_ylabel('流速 (m/s)', fontsize=12)
        ax3.set_title('区域流速统计分析', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. 物理合理性验证
        # 计算马赫数验证（流速不应超过当地音速）
        speeds_of_sound = []
        mach_numbers = []
        
        for i, seg in enumerate(axial_results):
            T = seg.get('coolant_temperature', 300)
            # 简化音速计算: a = √(γRT)，对于甲烷γ≈1.3, R≈518 J/kg/K
            a = np.sqrt(1.3 * 518 * T) if T > 0 else 400
            speeds_of_sound.append(a)
            mach_numbers.append(velocities[i] / a if a > 0 else 0)
        
        ax4.plot(positions, velocities, 'b-', linewidth=2, label='流速')
        ax4.plot(positions, speeds_of_sound, 'r--', linewidth=2, label='当地音速')
        ax4.fill_between(positions, speeds_of_sound, alpha=0.2, color='red', label='音速界限')
        ax4.set_xlabel('轴向位置 (m)', fontsize=12)
        ax4.set_ylabel('速度 (m/s)', fontsize=12)
        ax4.legend()
        ax4.set_title('流速物理合理性验证 (vs 当地音速)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 检查是否超音速
        if max(mach_numbers) > 0.3:  # 保守阈值
            ax4.text(0.02, 0.98, f'最大马赫数: {max(mach_numbers):.3f}', 
                    transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        else:
            ax4.text(0.02, 0.98, f'最大马赫数: {max(mach_numbers):.3f} (亚音速)', 
                    transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        self.save_plot_with_font_fix(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"流速分析对比图表已生成: {filename}")

        data_dict = {
            'axial_position_m': positions,
            'coolant_velocity_m_per_s': velocities,
            'coolant_density_kg_per_m3': densities,
            'flow_area_m2': flow_areas,
            'velocity_gradient_m_per_s2': velocity_gradient,
            'acceleration_m_per_s2': acceleration,
            'speeds_of_sound': speeds_of_sound,
            'mach_number': mach_numbers
        }

        self.save_data_to_csv(data_dict, filename)


def validate_geometry_calculation(self):
    """
    验证几何数据插值计算的准确性和可靠性
    
    功能:
    - 比较几何插值结果与原始数据点的差异
    - 计算平均误差和最大误差
    - 输出几何数据质量评估报告
    
    返回:
    None - 通过日志输出验证结果
    """
    if not self.geometry_loader.shape_data:
        logger.warning("无几何数据可供验证")
        return
    
    points = self.geometry_loader.shape_data['points']
    logger.info("开始几何数据验证分析...")
    logger.info(f"几何数据点总数: {len(points)}个")
    
    # 比较插值结果与原始数据
    errors = []
    successful_validations = 0
    
    for i, (x_mm, r_mm) in enumerate(points):
        x_m = x_mm / 1000
        calculated_diameter = self.geometry_loader.get_diameter_at_position(x_m)
        actual_diameter = 2 * r_mm / 1000
        
        if calculated_diameter is not None and actual_diameter > 0:
            error = abs(calculated_diameter - actual_diameter)
            errors.append(error)
            
            # 详细验证输出（每10个点输出一次）
            if i % 10 == 0:
                logger.debug(f"点{i}: x={x_mm}mm, 实际直径={actual_diameter*1000:.3f}mm, "
                           f"计算直径={calculated_diameter*1000:.3f}mm, "
                           f"误差={error*1000:.3f}mm")
            
            successful_validations += 1
        else:
            logger.warning(f"点{i}验证失败: x={x_mm}mm, 计算直径={calculated_diameter}, 实际直径={actual_diameter}")
    
    if errors:
        avg_error = np.mean(errors) * 1000  # 转换为mm
        max_error = np.max(errors) * 1000
        std_error = np.std(errors) * 1000
        
        logger.info(f"几何插值精度验证:")
        logger.info(f"• 验证点数: {successful_validations}/{len(points)}")
        logger.info(f"• 平均误差: {avg_error:.3f} mm")
        logger.info(f"• 最大误差: {max_error:.3f} mm") 
        logger.info(f"• 误差标准差: {std_error:.3f} mm")

        # 误差分布分析
        error_percentiles = np.percentile(errors * 1000, [25, 50, 75, 90, 95])
        logger.info(f"• 误差百分位数: 25%={error_percentiles[0]:.3f}mm, 50%={error_percentiles[1]:.3f}mm, "
                   f"75%={error_percentiles[2]:.3f}mm, 90%={error_percentiles[3]:.3f}mm, "
                   f"95%={error_percentiles[4]:.3f}mm")
        
        # 与文档3的基准数据交叉验证
        if avg_error > 0.1:  # 0.1mm阈值
            logger.warning("几何插值误差较大，建议检查数据质量")
        else:
            logger.info("✓ 几何插值精度满足要求")
            
        return {
            'validation_points': successful_validations,
            'average_error_mm': avg_error,
            'max_error_mm': max_error,
            'std_error_mm': std_error
        }
    else:
        logger.warning("无法计算几何插值误差")
        return None


def main():
    """
    液氧/甲烷发动机再生冷却传热分析主程序
    
    功能流程:
    1. 初始化计算环境(CEA、REFPROP、几何数据)
    2. 配置发动机几何和冷却系统参数
    3. 进行CEA燃烧性能综合分析
    4. 计算轴向温度、压力、热流分布
    5. 分析冷却性能和安全裕度
    6. 生成综合性能报告
    
    返回:
    dict: 包含完整分析结果的字典，包括:
        - geometry_validation: 几何数据验证结果
        - combustion_analysis: 燃烧性能分析
        - engine_conditions: 发动机运行条件
        - cooling_performance: 冷却性能指标
        - safety_assessment: 安全裕度评估
    """
    logger.info("液氧/甲烷发动机再生冷却传热计算（集成真实几何数据）")
    logger.info("=" * 60)

    setup_chinese_font()
    
    try:
        # 1. 创建发动机计算实例（启用CEA集成和真实几何数据）
        logger.info("1. 初始化发动机计算环境...")
        shape_file = params['file_paths']['engine_shape_file']
        
        # 创建实例时传入验证后的路径
        engine = LOX_MethaneEngineHeatTransfer(
            refprop_path=params['file_paths']['refprop_path'],
            use_cea=True,
            engine_shape_file=shape_file,
        )

        # 几何文件路径验证
        if not os.path.exists(shape_file):
            logger.error(f"几何文件不存在: {shape_file}")
            logger.info("将使用程序内置的简化几何模型")
            shape_file = None
        else:
            # 验证几何文件格式
            validation = engine.geometry_loader.validate_geometry_file(shape_file)
            if not validation['format_correct']:
                logger.warning("几何文件格式可能不正确，将尝试强制加载")
            
            # 尝试加载几何文件
            engine.geometry_loader.load_geometry_from_file(shape_file)
            
            # 检查是否加载成功
            if not engine.geometry_loader.shape_data:
                logger.error("几何文件加载失败，将使用简化几何模型")
                shape_file = None
            else:
                logger.info("几何文件加载成功，将使用真实几何数据")
        
        # 2. 验证几何数据加载情况
        logger.info("2. 验证几何数据加载...")
        if hasattr(engine, 'validate_geometry_calculation'):
            engine.validate_geometry_calculation()
        
        # 几何数据验证
        if engine.geometry_loader.shape_data:
            points_count = len(engine.geometry_loader.shape_data['points'])
            logger.info(f"几何数据验证:")
            logger.info(f"• 数据点数量: {points_count}")
            
            if points_count > 0:
                # 获取关键几何参数
                total_length = engine.geometry_loader.get_total_length()
                throat_diameter = engine.geometry.get('diameter', {}).get('throat', '未知')
                logger.info(f"• 发动机总长度: {total_length:.3f} m")
                # 修正：安全处理喉部直径输出
                if isinstance(throat_diameter, (int, float)):
                    logger.info(f"• 喉部直径: {throat_diameter:.3f} m")
                else:
                    logger.info(f"• 喉部直径: {throat_diameter}")
        
        # 3. 设置冷却通道几何参数
        logger.info("3. 配置冷却系统几何参数...")
        geo = params['engine_geometry']
        engine.set_geometric_parameters(
            d_throat=geo['d_throat'],
            L_chamber=geo['L_chamber'],
            delta_wall=geo['delta_wall'],
            number_of_fins=geo['number_of_fins'],
            t_fin=geo['t_fin'],
            h_fin=geo['h_fin'],
            delta_fin=geo['delta_fin']
        )
        
        # 4. 燃烧性能综合分析
        logger.info("4. 进行CEA燃烧性能分析...")
        combustion_params = params['combustion_analysis']
        combustion_analysis = engine.analyze_combustion_performance(
            Pc_MPa=combustion_params['Pc_MPa'],
            mixture_ratios=combustion_params['mixture_ratios'],
            eps_values=combustion_params['eps_values']
        )
        
        # 5. 材料属性配置
        material_properties = {
            'thermal_conductivity_points': [tuple(point) for point in params['material_properties']['thermal_conductivity_points']]
        }
        
        # 6. 运行条件设置
        op_cond = params['operating_conditions']
        operating_conditions = {
            'coolant_velocity': op_cond['coolant_velocity'],    # 冷却剂流速 [m/s]
            'hydraulic_diameter': op_cond['hydraulic_diameter'] # 水力直径 [m]
        }
        
        # 7. 基于真实几何和CEA的发动机工况设置
        logger.info("5. 设置基于真实几何的发动机工况...")
        if combustion_analysis and combustion_analysis['mixture_ratio_analysis']:
            optimal_mr = max(combustion_analysis['mixture_ratio_analysis'], 
                           key=lambda x: x['specific_impulse'])
            
            eng_cond = params['engine_conditions']
            engine_conditions = {
                'T_gas': eng_cond['T_gas'],
                'P_chamber': eng_cond['P_chamber_MPa'] * 1e6,  # MPa转换为Pa
                'c_star': eng_cond['c_star'],
                'm_dot_coolant': eng_cond['m_dot_coolant'],
                'T_coolant_in': eng_cond['T_coolant_in'],
                'P_coolant_in': eng_cond['P_coolant_in_MPa'] * 1e6,  # MPa转换为Pa
                'mixture_ratio': eng_cond['mixture_ratio']
            }
            logger.info(f"使用最优混合比: O/F={optimal_mr['mixture_ratio']:.1f}")
        else:
            engine_conditions = {
                'T_gas': 3400.0,
                'P_chamber': 10e6,
                'c_star': 1750,
                'm_dot_coolant': 4.5,
                'T_coolant_in': 110.0,
                'P_coolant_in': 18e6,
                'mixture_ratio': 3.4
            }
        
        film_cooling_params = {
            'start_position': 0.05,  # 膜冷却起始位置 [m]
            'liquid_film_end': 0.15,  # 液膜结束位置 [m]
            'film_flow_rate': engine_conditions['m_dot_coolant'] * 0.1,  # 膜冷却流量
            'film_temperature': engine_conditions['T_coolant_in']
        }
        
        # 8. 物性计算验证
        logger.info("6. 验证物性计算模块...")
        methane_props = engine.fluid_props.get_methane_properties(
            engine_conditions['T_coolant_in'], 
            engine_conditions['P_coolant_in']
        )
        logger.info(f"甲烷物性验证 - 密度: {methane_props['density']:.1f} kg/m³, "
                   f"比热: {methane_props['specific_heat']:.0f} J/(kg·K)")
        
        # 9. 轴向分布计算
        logger.info("7. 计算基于真实几何的轴向分布...")
        axial_results = engine.calculate_axial_distribution(
            engine_conditions['T_gas'],
            engine_conditions['P_chamber'],
            engine_conditions['c_star'],
            engine_conditions['m_dot_coolant'],
            engine_conditions['T_coolant_in'],
            engine_conditions['P_coolant_in'],
            material_properties,
            operating_conditions,
            num_segments=params['calculation_settings']['num_segments'],
            mixture_ratio=engine_conditions['mixture_ratio'],
            film_cooling_params=film_cooling_params
        )
        
        # 10. 详细结果输出
        logger.info("轴向分布关键位置结果:")
        if axial_results:
            key_indices = [0, len(axial_results)//4, len(axial_results)//2, 
                         3*len(axial_results)//4, len(axial_results)-1]
            
            for idx in key_indices:
                if idx < len(axial_results):
                    seg = axial_results[idx]
                    mach_display = seg.get('mach_number', 0)
                    logger.info(f"位置 {seg['axial_position']:.3f}m: "
                               f"直径={seg['local_diameter']:.3f}m, "
                               f"壁温={seg['gas_side_wall_temp']:.0f}K, "
                               f"马赫数={mach_display:.2f}")
        
        # 11. 冷却性能综合分析
        logger.info("8. 分析冷却性能...")
        performance = engine.analyze_cooling_performance(axial_results)
        
        # 性能指标输出
        logger.info("冷却性能指标:")
        logger.info(f"• 最高壁温: {performance['max_wall_temperature']:.1f} K")
        logger.info(f"• 最高热流: {performance['max_heat_flux']/1e6:.2f} MW/m²")
        logger.info(f"• 冷却剂温升: {performance['coolant_outlet_temperature']-engine_conditions['T_coolant_in']:.1f} K")
        logger.info(f"• 系统压降: {performance['total_pressure_drop']/1e6:.3f} MPa")
        
        # 12. 安全裕度评估
        logger.info("9. 进行安全裕度评估...")
        max_allowable_temp = params['calculation_settings']['max_allowable_temp']
        
        if performance['max_wall_temperature'] < max_allowable_temp:
            safety_margin = (max_allowable_temp - performance['max_wall_temperature']) / max_allowable_temp * 100
            logger.info(f"✓ 安全裕度: {safety_margin:.1f}% (安全)")
        else:
            logger.warning(f"✗✗ 超温风险: {performance['max_wall_temperature']-max_allowable_temp:.1f}K")
            safety_margin = -1  # 表示存在风险
        
        # 13. 生成综合报告
        final_report = {
            'geometry_validation': {
                'points_count': len(engine.geometry_loader.shape_data['points']) if engine.geometry_loader.shape_data else 0,
                'total_length': engine.geometry_loader.get_total_length() if engine.geometry_loader.shape_data else 0
            },
            'combustion_analysis': combustion_analysis,
            'engine_conditions': engine_conditions,
            'cooling_performance': performance,
            'safety_assessment': {
                'max_allowable_temp': max_allowable_temp,
                'safety_margin': safety_margin
            }
        }
        
        logger.info("=" * 60)
        logger.info("计算完成! 成功集成真实几何数据与传热分析")
        logger.info("=" * 60)
        
        # 修正：安全处理最终结果摘要输出
        logger.info("最终结果摘要:")
        logger.info(f"• 几何数据点: {final_report['geometry_validation']['points_count']}个")
        
        # 安全地输出喉部直径
        throat_diameter = engine.geometry.get('diameter', {}).get('throat', '未知')
        if isinstance(throat_diameter, (int, float)):
            logger.info(f"• 喉部直径: {throat_diameter:.3f} m")
        else:
            logger.info(f"• 喉部直径: {throat_diameter}")
        
        # 安全地输出其他数值
        if performance:
            max_wall_temp = performance.get('max_wall_temperature', 0)
            if isinstance(max_wall_temp, (int, float)):
                logger.info(f"• 最高壁温: {max_wall_temp:.1f}K")
            else:
                logger.info(f"• 最高壁温: {max_wall_temp}")
        
        safety_info = final_report.get('safety_assessment', {})
        safety_margin_val = safety_info.get('safety_margin', 0)
        if isinstance(safety_margin_val, (int, float)) and safety_margin_val >= 0:
            logger.info(f"• 安全裕度: {safety_margin_val:.1f}%")
        else:
            logger.info("• 安全裕度: 存在超温风险")

        # 14. 生成图表
        logger.info("10. 生成分析图表...")
        
        # 创建绘图生成器
        plotter = MethaneEnginePlotGenerator()
        
        # 生成各类图表
        plotter.plot_engine_shape(axial_results, engine.geometry)
        plotter.plot_temperature_distribution(axial_results)
        plotter.plot_coolant_flow_parameters(axial_results)
        plotter.plot_coolant_properties(axial_results)
        plotter.plot_gas_properties(axial_results)
        plotter.plot_velocity_analysis_comparison(axial_results)
        plotter.plot_gas_temperature_analysis(axial_results)
        
        logger.info(f"所有图表已保存至: {plotter.output_dir}")
        
        return final_report
        
    except Exception as e:
        logger.error(f"主函数执行失败: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        logger.error("程序终止")
        return None

# 程序入口点
if __name__ == "__main__":
    try:
        results = main()
        
        if results:
            # 输出关键结果摘要
            logger.info("最终结果摘要:")
            logger.info(f"• 几何数据点: {results['geometry_validation']['points_count']}个")
            logger.info(f"• 最优混合比: O/F={results['engine_conditions']['mixture_ratio']:.1f}")
            logger.info(f"• 最高壁温: {results['cooling_performance']['max_wall_temperature']:.1f}K")
            logger.info(f"• 安全裕度: {results['safety_assessment']['safety_margin']:.1f}%")
        else:
            logger.error("计算未完成，无结果输出")
            
    except KeyboardInterrupt:
        logger.info("用户中断程序执行")
    except Exception as e:
        logger.error(f"程序执行异常: {e}")