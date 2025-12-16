#!/usr/bin/env python3
"""
运行调试的入口脚本
"""

import subprocess
import sys

def run_debug():
    """运行调试过程"""
    
    print("=== 发动机传热分析调试 ===")
    print("1. 运行单元测试")
    print("2. 运行集成测试") 
    print("3. 运行完整调试")
    print("4. 运行特定测试")
    
    choice = input("请选择调试模式 (1-4): ").strip()
    
    if choice == "1":
        # 运行单元测试
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "test_engine_analysis.py", 
            "-v", "--tb=short", "-x"
        ])
    elif choice == "2":
        # 运行集成测试
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "test_engine_analysis.py::TestIntegration",
            "-v", "-s"
        ])
    elif choice == "3":
        # 运行完整调试
        result = subprocess.run([
            sys.executable, "debug_engine.py"
        ])
    elif choice == "4":
        # 运行特定测试
        test_name = input("请输入测试名称或模式: ")
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "test_engine_analysis.py",
            f"-k {test_name}",
            "-v", "-s"
        ])
    else:
        print("无效选择")
        return False
    
    return result.returncode == 0

if __name__ == "__main__":
    success = run_debug()
    print(f"\n调试完成: {'成功' if success else '失败'}")
    sys.exit(0 if success else 1)