#!/usr/bin/env python3
"""计算甲烷在给定温度和压力下的密度和运动粘度。

默认：T=200 K, P=5.8 MPa
使用 CoolProp 的 PropsSI 接口。
"""
import sys
try:
    from CoolProp.CoolProp import PropsSI
except Exception as e:
    print("CoolProp 未安装或导入失败：", e)
    print("请先运行: pip install CoolProp")
    sys.exit(1)


def compute_methane(T=200.0, P=5.8e6):
    fluid = 'Methane'
    rho = PropsSI('D', 'T', T, 'P', P, fluid)
    mu = PropsSI('VISCOSITY', 'T', T, 'P', P, fluid)  # 动力粘度，单位 Pa·s
    nu = mu / rho  # 运动粘度，单位 m^2/s
    return rho, mu, nu


def main():
    # 默认参数
    T = 200.0
    P = 5.8e6
    # 可通过命令行传入 T P（可选）
    if len(sys.argv) >= 3:
        try:
            T = float(sys.argv[1])
            P = float(sys.argv[2])
        except ValueError:
            print("命令行参数解析错误，使用默认值 T=200 K, P=5.8e6 Pa")

    rho, mu, nu = compute_methane(T, P)
    print(f"流体: Methane")
    print(f"温度 T = {T} K")
    print(f"压力 P = {P} Pa")
    print(f"密度 rho = {rho:.6f} kg/m^3")
    print(f"动力粘度 mu = {mu:.6e} Pa·s")
    print(f"运动粘度 nu = {nu:.6e} m^2/s")


if __name__ == '__main__':
    main()
