#!/bin/bash
# BCSR 批量实验快速启动脚本 (Linux/Mac)

set -e  # 遇到错误立即退出

echo "===================================="
echo " BCSR 持续学习批量实验"
echo "===================================="
echo

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3，请先安装 Python 3.7+"
    exit 1
fi

# 创建必要目录
mkdir -p data summary checkpoints logs data/loss_data

echo "目录已创建"
echo

# 询问用户运行模式
echo "请选择运行模式:"
echo "1. 串行运行 (安全，适合单GPU)"
echo "2. 并行运行 (推荐，需要多GPU)"
echo "3. 只运行 BCSR 方法"
echo "4. 只测试单个配置"
echo
echo "注意: coreset 方法需要 NTK，未实现"
echo
read -p "请输入选择 (1-4): " choice

case $choice in
    1)
        echo
        echo "开始串行运行所有实验..."
        python3 run_experiments.py --parallel 1
        ;;
    2)
        echo
        read -p "请输入并行数量 (建议=GPU数量): " parallel
        echo "开始并行运行所有实验 (并行度=$parallel)..."
        python3 run_experiments.py --parallel $parallel
        ;;
    3)
        echo
        echo "只运行 BCSR 方法..."
        python3 run_experiments.py --methods bcsr --parallel 1
        ;;
    4)
        echo
        read -p "请输入方法 (bcsr/uniform): " method
        read -p "请输入数据集 (cifar-bs-50/imb-cifar-bs-50/noise-cifar-bs-50): " dataset
        read -p "请输入种子 (0/1/2): " seed
        echo
        echo "运行配置: $method @ $dataset (seed=$seed)"
        python3 main.py --select_type $method --dataset $dataset --seed $seed
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo
echo "===================================="
echo " 实验完成！"
echo "===================================="
echo
echo "查看结果汇总:"
echo "   python3 summarize_results.py"
echo
echo "结果文件位置:"
echo "   - 日志: logs/"
echo "   - 结果: summary/"
echo "   - Loss数据: data/loss_data/"
echo
