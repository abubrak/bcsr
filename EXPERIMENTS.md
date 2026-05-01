# BCSR 批量实验指南

本目录包含批量运行 BCSR 持续学习实验的脚本。

> **注意：** `coreset` 方法（基于 NTK）未在此代码库中实现，实验包含 `bcsr`（论文方法）、`herding`、`gradmatch`、`gss`（三种经典 baseline）和 `uniform`（随机采样 baseline）。

## 📁 文件说明

- `run_experiments.py` - 批量运行脚本
- `summarize_results.py` - 结果汇总脚本
- `EXPERIMENTS.md` - 本文件

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖（建议使用虚拟环境）
pip install -r requirements.txt

# 创建必要目录
mkdir -p data summary checkpoints data/loss_data logs
```

### 2. 运行完整实验矩阵

```bash
# 串行运行（最安全，适合单GPU）
python run_experiments.py --parallel 1

# 并行运行（推荐，根据GPU数量调整）
python run_experiments.py --parallel 2

# 自定义配置
python run_experiments.py --methods bcsr --datasets cifar-bs-50 --seeds 0 1 2
```

### 3. 查看结果

```bash
# 生成汇总表格
python summarize_results.py
```

## 📊 实验矩阵

| 方法 | 数据集 | 种子 | 说明 |
|------|--------|------|------|
| bcsr | cifar-bs-50 | 0,1,2 | 论文方法，平衡数据 |
| bcsr | imb-cifar-bs-50 | 0,1,2 | 论文方法，不平衡数据 |
| bcsr | noise-cifar-bs-50 | 0,1,2 | 论文方法，噪声标签 |
| herding | cifar-bs-50 | 0,1,2 | K-centers 聚类方法 |
| herding | imb-cifar-bs-50 | 0,1,2 | K-centers，不平衡数据 |
| herding | noise-cifar-bs-50 | 0,1,2 | K-centers，噪声标签 |
| gradmatch | cifar-bs-50 | 0,1,2 | 梯度匹配方法 |
| gradmatch | imb-cifar-bs-50 | 0,1,2 | 梯度匹配，不平衡数据 |
| gradmatch | noise-cifar-bs-50 | 0,1,2 | 梯度匹配，噪声标签 |
| gss | cifar-bs-50 | 0,1,2 | 梯度样本选择 |
| gss | imb-cifar-bs-50 | 0,1,2 | 梯度样本选择，不平衡数据 |
| gss | noise-cifar-bs-50 | 0,1,2 | 梯度样本选择，噪声标签 |
| uniform | cifar-bs-50 | 0,1,2 | 随机采样（baseline） |
| uniform | imb-cifar-bs-50 | 0,1,2 | 随机采样，不平衡数据 |
| uniform | noise-cifar-bs-50 | 0,1,2 | 随机采样，噪声标签 |

**总计：45 组实验** (5方法 × 3数据集 × 3种子)

## 🛠️ 参数说明

### run_experiments.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--methods` | all | 选择方法 [bcsr, herding, gradmatch, gss, uniform, all] |
| `--datasets` | all | 选择数据集 [cifar-bs-50, imb-cifar-bs-50, noise-cifar-bs-50, all] |
| `--seeds` | 0 1 2 | 随机种子列表 |
| `--parallel` | 1 | 并行运行数量（建议=GPU数量） |

### 超参数配置

脚本会自动为不同数据集设置正确的超参数（对应论文 Table 1-3）：

| 数据集 | lr_decay | ref_hyp | lr_proxy_model | lr_weight |
|--------|----------|---------|----------------|-----------|
| cifar-bs-50 | 0.9 | 0.5 | 5.0 | 5.0 |
| imb-cifar-bs-50 | 0.875 | 0.1 | 10.0 | 10.0 |
| noise-cifar-bs-50 | 0.875 | 0.1 | 10.0 | 10.0 |

## 📈 输出文件

### 日志文件 (`logs/`)

```
logs/
├── cifar-bs-50-bcsr-seed0.log         # 单个实验的完整日志
├── cifar-bs-50-bcsr-seed1.log
├── ...
└── summary-20240430-123456.json        # 脚本运行汇总
```

### 结果文件 (`summary/`)

```
summary/
├── 2024-04-30-12-34-56-cifar-bs-50-bcsr_seqlr0p15_seqep1_seqbs50  # 实验结果JSON
├── 2024-04-30-12-38-12-imb-cifar-bs-50-bcsr_seqlr0p15_seqep1_seqbs50
└── ...
```

每个JSON文件包含：
- `AVG ACC`: 平均准确率
- `Forgetting`: 遗忘度
- `per-task accuracy`: 每个任务的准确率
- `acc_avg`: 累计平均准确率
- `time`: 运行时间（小时）

### Loss 数据 (`data/loss_data/`)

外层优化损失曲线，用于绘制 convergence 图。

### 模型检查点 (`checkpoints/`)

每个任务的模型参数。

## 📝 汇总结果示例

运行 `python summarize_results.py` 后会输出：

```
================================================================================
平均准确率 (Average Accuracy %)
================================================================================
Method              cifar-bs-50          imb-cifar-bs-50       noise-cifar-bs-50
--------------------------------------------------------------------------------
bcsr                65.23 ± 1.45         52.18 ± 2.31          58.76 ± 1.89
herding             48.32 ± 1.23         41.56 ± 1.89          44.78 ± 2.01
gradmatch           52.18 ± 1.67         44.23 ± 2.12          47.89 ± 1.95
gss                 50.45 ± 1.34         43.12 ± 1.76          46.23 ± 2.08
uniform             42.15 ± 0.98         38.72 ± 1.45          40.23 ± 1.12

================================================================================
遗忘度 (Forgetting %，越低越好)
================================================================================
Method              cifar-bs-50          imb-cifar-bs-50       noise-cifar-bs-50
--------------------------------------------------------------------------------
bcsr                12.34 ± 1.23         18.45 ± 2.12         15.67 ± 1.89
herding             19.45 ± 1.67         22.78 ± 2.34          20.56 ± 2.12
gradmatch           16.78 ± 1.89         20.12 ± 2.45          18.34 ± 2.01
gss                 17.23 ± 1.56         21.45 ± 2.23          19.78 ± 1.95
uniform             24.56 ± 1.78         28.34 ± 2.56         26.78 ± 2.23

================================================================================
实验完整性检查
================================================================================

预期实验数量: 45
已发现结果: 45
缺失数量: 0

✓ 所有实验已完成！
```

## 🐛 故障排除

### 问题1：CUDA out of memory

**解决方案**：
- 减小 `--parallel` 参数
- 或者修改 `main.py` 中的 `--stream_size` 参数

### 问题2：comet_ml 导入错误

**解决方案**：
```bash
pip install comet-ml
```
或修改 `main.py` 第95行，设置 `disabled=True`（已设置）。

### 问题3：依赖版本冲突

**解决方案**：
```bash
# 使用虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

pip install torch==1.13.1 torchvision==0.14.1
pip install -r requirements.txt
```

## 📖 论文对应

本脚本对应论文以下表格：

- **Table 1**: Balanced CIFAR-100 → `cifar-bs-50`
- **Table 2**: Imbalanced CIFAR-100 → `imb-cifar-bs-50`
- **Table 3**: Label Noise CIFAR-100 → `noise-cifar-bs-50`

## ⏱️ 预计运行时间

| 方法 | 单个实验 | 45组实验（串行） | 45组实验（2并行） |
|------|----------|------------------|-------------------|
| bcsr | ~2小时 | ~90小时 | ~45小时 |
| herding | ~1.5小时 | ~67.5小时 | ~34小时 |
| gradmatch | ~2.5小时 | ~112.5小时 | ~56小时 |
| gss | ~2小时 | ~90小时 | ~45小时 |
| uniform | ~1小时 | ~45小时 | ~22.5小时 |

*时间仅供参考，实际取决于硬件配置*

## 🔧 高级用法

### 只运行特定方法

```bash
python run_experiments.py --methods bcsr
```

### 只运行不平衡数据集实验

```bash
python run_experiments.py --datasets imb-cifar-bs-50
```

### 使用更多随机种子（更稳定的结果）

```bash
python run_experiments.py --seeds 0 1 2 3 4 --parallel 2
```

### 直接运行单个实验

```bash
python main.py --select_type bcsr --dataset cifar-bs-50 --seed 0
```
