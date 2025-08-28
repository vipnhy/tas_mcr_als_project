@echo off
REM TAS MCR-ALS 分析工具 - Windows 批处理脚本
REM 使用方法: 双击运行此文件，然后按提示输入参数

echo ================================
echo     TAS MCR-ALS 分析工具
echo ================================
echo.

cd /d "d:\TAS\tas_mcr_als_project"

echo 当前工作目录: %CD%
echo.

echo 选择运行模式:
echo 1. 交互式模式 (推荐新用户)
echo 2. 使用示例配置文件
echo 3. 自定义命令行参数
echo 4. 查看帮助信息
echo.

set /p choice="请选择模式 (1-4): "

if "%choice%"=="1" (
    echo.
    echo 启动交互式模式...
    "D:\TAS\tas_mcr_als_project\venv\Scripts\python.exe" run_main.py
) else if "%choice%"=="2" (
    echo.
    echo 使用示例配置文件运行...
    "D:\TAS\tas_mcr_als_project\venv\Scripts\python.exe" run_main.py --config config_example.json --save_plots --save_results
) else if "%choice%"=="3" (
    echo.
    set /p filepath="输入数据文件路径: "
    set /p ncomp="输入组分数量 [默认3]: "
    if "%ncomp%"=="" set ncomp=3
    
    echo.
    echo 运行分析...
    "D:\TAS\tas_mcr_als_project\venv\Scripts\python.exe" run_main.py --file_path "%filepath%" --n_components %ncomp% --save_plots --save_results
) else if "%choice%"=="4" (
    echo.
    echo 显示帮助信息...
    "D:\TAS\tas_mcr_als_project\venv\Scripts\python.exe" run_main.py --help
) else (
    echo 无效选择，请重新运行脚本
)

echo.
echo 按任意键退出...
pause >nul
