# 更新日志

## 2024-04-30 - 移除未实现的 coreset 方法

### 修改内容

已从所有实验脚本中移除 `coreset` 方法，因为该方法需要 Neural Tangent Kernel (NTK) 支持，而此代码库未实现该功能。

### 修改的文件

| 文件 | 修改内容 |
|------|----------|
| `run_experiments.py` | ✅ 从 `METHODS` 列表中移除 'coreset'，更新实验矩阵为 2×3×3=18 组实验 |
| `summarize_results.py` | ✅ 从 `methods` 列表中移除 'coreset' |
| `main.py` | ✅ 添加检查，如果选择 coreset 则报错并给出友好提示 |
| `EXPERIMENTS.md` | ✅ 更新文档，移除所有 coreset 相关内容，添加说明 |
| `QUICKSTART.md` | ✅ 添加注意事项说明 |
| `run_all.bat` | ✅ 更新批处理脚本提示，移除 coreset 选项 |
| `run_all.sh` | ✅ 更新 shell 脚本提示，移除 coreset 选项 |

### 现在支持的实验方法

| 方法 | 说明 | 状态 |
|------|------|------|
| `bcsr` | 论文提出的方法（基于正则化的双层优化） | ✅ 可用 |
| `uniform` | 随机均匀采样（baseline） | ✅ 可用 |
| `coreset` | NTK-based 双层优化方法 | ❌ 未实现 |

### 实验矩阵变化

**之前：** 3方法 × 3数据集 × 3种子 = **27 组实验**
- bcsr, coreset, uniform
- cifar-bs-50, imb-cifar-bs-50, noise-cifar-bs-50
- seeds: 0, 1, 2

**现在：** 2方法 × 3数据集 × 3种子 = **18 组实验**
- bcsr, uniform
- cifar-bs-50, imb-cifar-bs-50, noise-cifar-bs-50
- seeds: 0, 1, 2

### 运行命令（无需修改）

```bash
# 运行所有实验
python run_experiments.py --parallel 2

# 只运行 BCSR 方法
python run_experiments.py --methods bcsr

# 单个实验
python main.py --select_type bcsr --dataset cifar-bs-50 --seed 0
python main.py --select_type uniform --dataset cifar-bs-50 --seed 0

# 如果尝试运行 coreset，会得到友好的错误提示
python main.py --select_type coreset --dataset cifar-bs-50 --seed 0
# 输出：错误: 'coreset' 方法未实现...
```

### 兼容性说明

- ✅ 所有现有实验结果不受影响
- ✅ 汇总脚本仍然可以正常工作
- ✅ 如果用户尝试运行 coreset，会得到清晰的错误提示

---

## 2024-04-30 - 修复 NumPy/PyTorch 兼容性问题

详见 [BUGFIX_NUMPY_TENSOR_COMPATIBILITY.md](BUGFIX_NUMPY_TENSOR_COMPATIBILITY.md)
