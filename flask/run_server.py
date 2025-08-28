#!/usr/bin/env python3
"""
Flask Web应用启动脚本
TAS MCR-ALS 分析平台
"""

import os
import sys

def setup_environment():
    """设置环境变量"""
    # 添加项目根目录到Python路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # 添加父目录到Python路径（用于导入mcr和data模块）
    parent_dir = os.path.dirname(project_root)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

def run_app():
    """运行Flask应用"""
    try:
        # 检查Flask是否可用
        try:
            import flask
            print(f"Flask版本: {flask.__version__}")
        except ImportError:
            print("❌ Flask未安装!")
            print("请使用虚拟环境并安装Flask:")
            print("  1. 激活虚拟环境: venv\\Scripts\\activate")
            print("  2. 安装Flask: pip install Flask")
            sys.exit(1)
        
        from app import app
        print("=" * 60)
        print("🚀 TAS MCR-ALS 分析平台启动中...")
        print("=" * 60)
        print(f"📡 访问地址: http://localhost:5000")
        print(f"⏹️  停止服务: Ctrl+C")
        print("=" * 60)
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=True
        )
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保安装了所有依赖包: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    setup_environment()
    run_app()
