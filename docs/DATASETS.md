# 数据集文档

本文档描述了项目中支持的所有数据集及其变体。

## 数据集概览

| 数据集 | 类别数 | 每任务类别数 | 任务数 | 输入形状 | 平衡版本 | 不平衡版本 | 噪声版本 |
|--------|--------|--------------|--------|----------|----------|------------|----------|
| CIFAR-100 | 100 | 5 | 20 | (3, 32, 32) | ✅ | ✅ | ✅ |
| CIFAR-10 | 10 | 2 | 5 | (3, 32, 32) | ✅ | ✅ | ✅ |
| MNIST | 10 | 2 | 5 | (1, 28, 28) | ✅ | ✅ | ✅ |

## 数据集命名规范

**格式**: `[variant]-[dataset]-[type]-[size]`

- **variant**: `imb` (不平衡) 或 `noise` (噪声) 或无（平衡）
- **dataset**: `cifar` (CIFAR-100), `cifar10`, `mnist`
- **type**: `bs` (split benchmark)
- **size**: 每任务样本数 (默认 50)

**示例:**
- `cifar-bs-50`: 平衡 CIFAR-100
- `imb-cifar10-bs-50`: 不平衡 CIFAR-10
- `noise-mnist-bs-50`: 噪声 MNIST

## CIFAR-100

### 基本信息
- **类别数**: 100
- **每任务类别数**: 5
- **任务数**: 20
- **输入形状**: (3, 32, 32) - RGB 32×32 图像
- **训练集**: 50,000 图像
- **测试集**: 10,000 图像

### 变体
1. **cifar-bs-50**: 平衡版本，每类 500 样本
2. **imb-cifar-bs-50**: 不平衡版本，指数分布（imb_factor=1/10）
3. **noise-cifar-bs-50**: 噪声版本，20% 标签噪声

## CIFAR-10

### 基本信息
- **类别数**: 10
- **每任务类别数**: 2
- **任务数**: 5
- **输入形状**: (3, 32, 32) - RGB 32×32 图像
- **训练集**: 50,000 图像
- **测试集**: 10,000 图像

### 变体
1. **cifar10-bs-50**: 平衡版本，每类 5,000 样本
2. **imb-cifar10-bs-50**: 不平衡版本，指数分布（imb_factor=1/10）
3. **noise-cifar10-bs-50**: 噪声版本，20% 标签噪声

### 使用场景
- **快速实验**: 比 CIFAR-100 收敛更快
- **方法验证**: 适合快速验证新方法
- **资源受限**: 显存占用更小

## MNIST

### 基本信息
- **类别数**: 10
- **每任务类别数**: 2
- **任务数**: 5
- **输入形状**: (1, 28, 28) - 灰度 28×28 图像
- **训练集**: 60,000 图像
- **测试集**: 10,000 图像

### 变体
1. **mnist-bs-50**: 平衡版本，每类约 6,700 样本
2. **imb-mnist-bs-50**: 不平衡版本，指数分布（imb_factor=1/10）
3. **noise-mnist-bs-50**: 噪声版本，20% 标签噪声

### 使用场景
- **超参数调优**: 最快收敛，适合快速迭代
- **算法开发**: 适合调试和原型开发
- **基准测试**: 经典持续学习基准

## 数据加载示例

### Python API

```python
from core.data_utils import get_all_loaders

# 加载平衡 CIFAR-10
loaders = get_all_loaders(
    seed=0,
    dataset='cifar10-bs-50',
    num_tasks=5,
    bs_inter=10,
    bs_intra=50,
    num_examples=100
)

# 访问任务 1 的数据
task_1_train = loaders['sequential'][1]['train']
task_1_val = loaders['sequential'][1]['val']
```

### 命令行

```bash
# 在 CIFAR-10 上运行实验
python main.py --select_type bcsr --dataset cifar10-bs-50 --seed 0

# 在不平衡 MNIST 上运行实验
python main.py --select_type herding --dataset imb-mnist-bs-50 --seed 0

# 批量运行所有数据集
python run_experiments.py --datasets all --seeds 0 1 2
```

## 不平衡数据集详情

### 不平衡因子
所有不平衡数据集使用指数分布：

```
n_samples_i = n_max * (imb_factor)^(i / (n_classes - 1))
```

- **n_max**: 最大类样本数
- **imb_factor**: 不平衡因子 (默认 1/10)
- **i**: 类别索引 (按类别频率排序)

### 类别排序
数据集使用特定类别排序以创建长尾分布：
- **CIFAR-100**: [25, 11, 23, 76, 12, ...]
- **CIFAR-10**: [3, 1, 8, 4, 6, 0, 9, 2, 5, 7]
- **MNIST**: [1, 3, 0, 7, 9, 5, 2, 8, 4, 6]

## 噪声数据集详情

### 标签噪声
- **噪声类型**: 统一标签翻转
- **噪声率**: 20% (默认)
- **翻转范围**: 仅在任务类别内翻转

示例：任务 1 包含类别 [0, 1]，噪声标签只会随机分配为 0 或 1。

## 数据集统计

### 样本数量对比

| 数据集 | 平衡版每类 | 不平衡版范围 | 总训练集 |
|--------|-----------|-------------|----------|
| CIFAR-100 | 500 | 5-500 | 50,000 |
| CIFAR-10 | 5,000 | 50-5,000 | 50,000 |
| MNIST | 6,700 | 67-6,700 | 60,000 |

### 难度排名

从易到难（基于经验）：
1. **mnist-bs-50**: 最简单
2. **cifar10-bs-50**: 较简单
3. **cifar-bs-50**: 中等
4. **imb-mnist-bs-50**: 中等偏难
5. **noise-mnist-bs-50**: 中等偏难
6. **imb-cifar10-bs-50**: 较难
7. **noise-cifar10-bs-50**: 较难
8. **imb-cifar-bs-50**: 困难
9. **noise-cifar-bs-50**: 最困难

## 参考论文

- **CIFAR-100**: "Learning Multiple Layers of Features from Tiny Images" (Krizhevsky, 2009)
- **CIFAR-10**: 同上
- **MNIST**: "Gradient-Based Learning Applied to Document Recognition" (LeCun et al., 1998)
- **持续学习基准**: "Three scenarios for continual learning" (van de Ven et al., 2022)

## 更新日志

- 2026-05-01: 添加 CIFAR-10 和 MNIST 数据集支持
