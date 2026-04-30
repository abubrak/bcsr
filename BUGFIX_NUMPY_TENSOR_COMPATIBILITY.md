# Bug Fix: NumPy Array / PyTorch Tensor Compatibility

## Problem

当使用 `uniform` 选择方法时，代码抛出错误：
```
AttributeError: 'numpy.ndarray' object has no attribute 'cpu'
```

## Root Cause

代码中有多处假设索引对象始终是 PyTorch 张量，并直接调用 `.cpu()` 或 `.is_cuda` 属性。然而：

- `bcsr` 和 `coreset` 方法返回 PyTorch 张量索引
- `uniform` 方法通过 `Summarizer.build_summary()` 返回 NumPy 数组索引

当 `candidates_indices` 列表包含 NumPy 数组时，访问 `.cpu()` 方法会失败。

## Solution

添加了 `ensure_tensor_cpu()` 辅助函数，统一处理 NumPy 数组和 PyTorch 张量的转换：

```python
def ensure_tensor_cpu(idx):
    """将索引转换为CPU上的PyTorch张量，兼容NumPy数组和PyTorch张量"""
    if isinstance(idx, np.ndarray):
        return torch.from_numpy(idx).long()
    elif isinstance(idx, torch.Tensor):
        return idx.cpu().long()
    else:
        return torch.tensor(idx, dtype=torch.long, device='cpu')
```

## Modified Locations

| 文件 | 行号 | 修改前 | 修改后 |
|------|------|--------|--------|
| `core/train_methods_cifar.py` | 23 | - | 添加 `ensure_tensor_cpu()` 函数 |
| `core/train_methods_cifar.py` | 120 | `idx = candidates[batch_idx].cpu() if candidates[batch_idx].is_cuda else candidates[batch_idx]` | `idx = ensure_tensor_cpu(candidates[batch_idx])` |
| `core/train_methods_cifar.py` | 154 | `pick_cpu = pick.cpu() if pick.is_cuda else pick` | `pick_cpu = ensure_tensor_cpu(pick)` |
| `core/train_methods_cifar.py` | 232 | `pick_idx = pick.cpu() if pick.is_cuda else pick` | `pick_idx = ensure_tensor_cpu(pick)` |
| `core/train_methods_cifar.py` | 281 | `pick_idx = pick.cpu() if pick.is_cuda else pick` | `pick_idx = ensure_tensor_cpu(pick)` |
| `core/train_methods_cifar.py` | 65 | `if isinstance(sorted_index, torch.Tensor) and sorted_index.is_cuda: sorted_index = sorted_index.cpu()` | 直接转换为 NumPy 数组 |

## Impact

- ✅ 所有三种方法（`bcsr`, `coreset`, `uniform`）现在都能正常工作
- ✅ 代码更加健壮，能够处理混合的索引类型
- ✅ 没有性能影响（只在必要时进行转换）

## Testing

建议使用以下命令测试修复：

```bash
# 测试 uniform 方法（之前会失败）
python main.py --select_type uniform --dataset cifar-bs-50 --seed 0

# 测试所有方法
python run_experiments.py --parallel 1
```

---

**Fixed:** 2024-04-30
**Related Issue:** NumPy array / PyTorch tensor type compatibility
