@echo off
REM BCSR 批量实验快速启动脚本 (Windows)

echo ====================================
echo  BCSR 持续学习批量实验
echo ====================================
echo.

REM 检查Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到 Python，请先安装 Python 3.7+
    pause
    exit /b 1
)

REM 创建必要目录
if not exist "data" mkdir data
if not exist "summary" mkdir summary
if not exist "checkpoints" mkdir checkpoints
if not exist "logs" mkdir logs
if not exist "data\loss_data" mkdir data\loss_data

echo 目录已创建
echo.

REM 询问用户运行模式
echo 请选择运行模式:
echo 1. 串行运行 (安全，适合单GPU)
echo 2. 并行运行 (推荐，需要多GPU)
echo 3. 只运行 BCSR 方法
echo 4. 只测试单个配置
echo.
echo 注意: coreset 方法需要 NTK，未实现
echo.
set /p choice="请输入选择 (1-4): "

if "%choice%"=="1" (
    echo.
    echo 开始串行运行所有实验...
    python run_experiments.py --parallel 1
) else if "%choice%"=="2" (
    echo.
    set /p parallel="请输入并行数量 (建议=GPU数量): "
    echo 开始并行运行所有实验 (并行度=%parallel%)...
    python run_experiments.py --parallel %parallel%
) else if "%choice%"=="3" (
    echo.
    echo 只运行 BCSR 方法...
    python run_experiments.py --methods bcsr --parallel 1
) else if "%choice%"=="4" (
    echo.
    set /p method="请输入方法 (bcsr/uniform): "
    set /p dataset="请输入数据集 (cifar-bs-50/imb-cifar-bs-50/noise-cifar-bs-50): "
    set /p seed="请输入种子 (0/1/2): "
    echo.
    echo 运行配置: %method% @ %dataset% (seed=%seed%)
    python main.py --select_type %method% --dataset %dataset% --seed %seed%
) else (
    echo 无效选择
    pause
    exit /b 1
)

echo.
echo ====================================
echo  实验完成！
echo ====================================
echo.
echo 查看结果汇总:
echo   python summarize_results.py
echo.
echo 结果文件位置:
echo   - 日志: logs\
echo   - 结果: summary\
echo   - Loss数据: data\loss_data\
echo.
pause
