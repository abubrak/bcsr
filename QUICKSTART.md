# 🚀 快速开始

## 一键运行（推荐）

### Windows
```bash
run_all.bat
```

### Linux/Mac
```bash
chmod +x run_all.sh
./run_all.sh
```

然后按照提示选择运行模式即可。

## 手动运行

### 1. 运行所有实验（27组）
```bash
python run_experiments.py --parallel 2
```

### 2. 运行特定实验
```bash
# 只运行BCSR方法
python run_experiments.py --methods bcsr

# 只运行不平衡数据集
python run_experiments.py --datasets imb-cifar-bs-50

# 单个实验
python main.py --select_type bcsr --dataset cifar-bs-50 --seed 0
```

### 3. 查看结果
```bash
python summarize_results.py
```

## 📁 实验文件

- `run_experiments.py` - 批量实验脚本
- `summarize_results.py` - 结果汇总脚本
- `EXPERIMENTS.md` - 详细文档
- `run_all.bat/sh` - 快速启动脚本

> **注意：** 只有 `bcsr`（论文方法）和 `uniform`（baseline）两种方法可用。`coreset` 方法（NTK-based）未实现。

详细说明请查看 [EXPERIMENTS.md](EXPERIMENTS.md)
