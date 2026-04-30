# PyTorch 2.x 升级和NTK移除指南

本文档说明了 Bilevel-Coreset-Selection 项目从 NTK 依赖迁移到纯 PyTorch 2.x 实现的变更。

## 主要变更

### 1. 依赖更新

**移除的依赖:**
- `jax==0.2.3`
- `jaxlib==0.1.55`
- `neural-tangents==0.3.4`

**新增/升级的依赖:**
- `torch>=2.0.0` (从 1.7.1 升级)
- `torchvision>=0.15.0` (从 0.8.2 升级)
- `numpy>=1.21.0` (从 1.17.2 升级)
- 其他依赖包版本相应升级

### 2. 代码结构变更

#### 删除的文件:
- `core/ntk_generator.py` - NTK核函数生成器

#### 新增的文件:
- `core/pytorch_proxy_model.py` - PyTorch代理模型实现

#### 修改的文件:
- `core/bilevel_coreset.py` - 移除NTK核函数依赖
- `core/bcsr_coreset.py` - PyTorch 2.x兼容性更新
- `core/bcsr_training.py` - 设备处理更新
- `core/train_methods_cifar.py` - 移除NTK导入
- `core/data_utils.py` - 清理NTK导入
- `core/__init__.py` - 移除NTK导出
- `main.py` - 添加PyTorch版本检查

### 3. API变更

#### BCSR_Coreset使用方式（保持兼容）

**之前的代码（需要NTK）:**
```python
from core.bcsr_coreset import BCSR_Coreset
from core.models import ResNet18

# 需要NTK核函数
config = {'n_classes': 100, 'dropout': 0.2}
proxy_model = ResNet18(config=config)
bc = BCSR_Coreset(proxy_model, lr_proxy_model=5.0, beta=0.1)
```

**现在的代码（纯PyTorch）:**
```python
from core.bcsr_coreset import BCSR_Coreset
from core.models import ResNet18

# 无需NTK，直接使用PyTorch模型
config = {'n_classes': 100, 'dropout': 0.2}
proxy_model = ResNet18(config=config)
bc = BCSR_Coreset(proxy_model, lr_proxy_model=5.0, beta=0.1)
```

API保持不变，但内部实现完全使用PyTorch，无需NTK。

#### 新增：PyTorch代理模型

**新增的API:**
```python
from core.pytorch_proxy_model import create_proxy_model

# 创建MLP代理模型
mlp_model = create_proxy_model(
    model_type='mlp',
    input_shape=(784,),
    num_classes=10,
    hidden_dim=256,
    num_layers=2
)

# 创建ResNet代理模型
resnet_model = create_proxy_model(
    model_type='resnet',
    input_shape=(3, 32, 32),
    num_classes=100
)
```

### 4. 设备兼容性

**PyTorch 2.x 增强:**
- 更好的CUDA支持
- 自动混合精度训练（AMP）
- torch.compile优化

**设备切换示例:**
```python
# 自动检测设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# BCSR自动适配设备
bc = BCSR_Coreset(
    proxy_model=model,
    device=device  # 'cuda' 或 'cpu'
)
```

### 5. 性能优化

**PyTorch 2.x 性能提升:**
1. **torch.compile**: 可选的模型编译加速
   ```python
   if hasattr(torch, 'compile'):
       model = torch.compile(model)
   ```

2. **更好的CUDA内核**: 透明加速，无需代码更改

3. **混合精度训练**: 减少内存占用，提高速度
   ```python
   from torch.cuda.amp import autocast, GradScaler
   ```

## 升级步骤

### 对于现有用户:

1. **更新依赖:**
   ```bash
   pip install -r requirements.txt
   ```

2. **验证安装:**
   ```bash
   python test_pytorch2_upgrade.py
   ```

3. **运行现有代码:**
   API保持兼容，现有代码应该可以直接运行。

### 对于新用户:

1. **安装PyTorch 2.x:**
   ```bash
   pip install torch>=2.0.0 torchvision>=0.15.0
   ```

2. **克隆并安装项目:**
   ```bash
   git clone <repository>
   cd Bilevel-Coreset-Selection-via-Regularization-main
   pip install -r requirements.txt
   ```

3. **运行测试:**
   ```bash
   python test_pytorch2_upgrade.py
   ```

## 常见问题

### Q: 为什么移除NTK？

A: BCSR方法实际上不需要使用NTK核函数。通过直接使用PyTorch模型进行双层优化，我们可以：
1. 简化依赖，不需要JAX和neural-tangents
2. 更好地利用PyTorch生态系统
3. 在Colab等环境中更容易部署
4. 获得PyTorch 2.x的性能优势

### Q: 现有的代码需要修改吗？

A: 大部分情况下不需要。BCSR的API保持不变，只是内部实现从NTK切换到了纯PyTorch。

### Q: 性能会受影响吗？

A: 不会。实际上，PyTorch 2.x通常比旧版本有更好的性能。移除NTK依赖也减少了运行时开销。

### Q: 如何使用PyTorch 2.x的新特性？

A: PyTorch 2.x的新特性（如torch.compile）是可选的。现有代码无需修改即可运行，也可以选择性地使用新特性进行优化。

### Q: 在Colab中如何使用？

A: 确保Colab运行时使用GPU，然后安装更新后的依赖：
```python
!pip install torch>=2.0.0 torchvision>=0.15.0
!pip install -r requirements.txt
```

## 技术细节

### BCSR实现变更

**之前（NTK方法）:**
1. 使用neural-tangents计算NTK核矩阵
2. 在核空间中进行双层优化
3. 需要JAX进行自动微分

**现在（PyTorch方法）:**
1. 直接使用PyTorch模型
2. 在模型参数空间进行双层优化
3. 完全使用PyTorch自动微分

### 双层优化实现

BCSR的核心双层优化框架保持不变：

```python
# 内层优化：在选定样本上训练模型
inner_loss = weighted_loss(model(X_selected), y_selected, weights)
inner_params = optimize(inner_loss, model.parameters())

# 外层优化：优化样本权重
outer_loss = loss(model(X_all), y_all)
weight_grad = implicit_gradient(inner_loss, outer_loss, weights)
weights -= lr * weight_grad
```

只是从核空间转移到了参数空间。

## 测试和验证

运行完整的测试套件:
```bash
python test_pytorch2_upgrade.py
```

测试包括:
- PyTorch 2.x版本检查
- 核心模块导入
- 代理模型创建
- BCSR功能测试
- BilevelCoreset功能测试
- CPU/GPU设备切换
- torch.compile功能
- 完整训练流程

## 参考资源

- [PyTorch 2.0发布说明](https://pytorch.org/get-started/pytorch-2.0/)
- [BCSR论文](https://arxiv.org/...)
- [项目仓库](https://github.com/...)

## 支持

如有问题，请：
1. 检查本文档的"常见问题"部分
2. 运行测试脚本诊断问题
3. 提交issue到项目仓库

---

最后更新: 2026-04-30
