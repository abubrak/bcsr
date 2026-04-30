# Google Colab使用指南

本指南说明如何在Google Colab中使用升级后的BCSR项目。

## 快速开始

### 1. 设置Colab环境

在Colab notebook中运行:

```python
# 检查GPU
!nvidia-smi

# 克隆项目
!git clone https://github.com/your-repo/Bilevel-Coreset-Selection-via-Regularization-main.git
%cd Bilevel-Coreset-Selection-via-Regularization-main

# 安装依赖
!pip install torch>=2.0.0 torchvision>=0.15.0
!pip install -r requirements.txt

# 验证安装
!python test_pytorch2_upgrade.py
```

### 2. 基本使用

#### 运行BCSR选择:

```python
import torch
from core.bcsr_coreset import BCSR_Coreset
from core.models import ResNet18
from core.data_utils import get_all_loaders

# 检查设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 创建代理模型
config = {'n_classes': 5, 'dropout': 0.1}
proxy_model = ResNet18(config=config).to(device)

# 创建BCSR选择器
bc = BCSR_Coreset(
    proxy_model=proxy_model,
    lr_proxy_model=2.0,
    beta=0.1,
    out_dim=5,
    max_outer_it=2,  # Colab上减少迭代次数
    max_inner_it=1,
    device=device
)

# 加载数据
loaders = get_all_loaders(
    seed=0,
    dataset='cifar-bs-50',
    num_tasks=2,
    bs_inter=10,
    bs_intra=50,
    num_examples=200
)

# 选择coreset
data = next(iter(loaders['sequential'][1]['train']))
X, y, task_id = data

selected, _ = bc.coreset_select(
    model=proxy_model,
    X=X.cpu().numpy(),
    y=y.cpu().numpy(),
    task_id=task_id,
    topk=10
)

print(f"选择了{len(selected)}个样本作为coreset")
```

### 3. Colab优化参数

针对Colab的T4 GPU (16GB显存)，建议使用以下优化参数:

```python
# 小规模快速测试
quick_test_config = {
    'num_tasks': 2,
    'stream_size': 25,
    'memory_size': 50,
    'batch_size': 10,
    'outer_iter': 2,  # 减少外层迭代
    'inner_iter': 1,  # 减少内层迭代
    'seq_epochs': 1,
}

# 中等规模测试
medium_test_config = {
    'num_tasks': 5,
    'stream_size': 50,
    'memory_size': 100,
    'batch_size': 10,
    'outer_iter': 3,
    'inner_iter': 1,
    'seq_epochs': 1,
}
```

### 4. 性能优化技巧

#### 使用混合精度训练:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### 使用torch.compile (PyTorch 2.0+):

```python
if hasattr(torch, 'compile'):
    model = torch.compile(model)
```

#### 批处理优化:

```python
# 根据GPU内存调整batch_size
if torch.cuda.get_device_properties(0).total_memory < 10e9:  # <10GB
    batch_size = 8
else:
    batch_size = 16
```

### 5. 常见问题解决

#### 内存不足 (OOM):

```python
# 解决方案1: 减少外层迭代
bc = BCSR_Coreset(..., max_outer_it=1)

# 解决方案2: 减少batch_size
loaders = get_all_loaders(..., bs_intra=25)

# 解决方案3: 使用CPU (慢但稳定)
bc = BCSR_Coreset(..., device='cpu')
```

#### 运行时间过长:

```python
# 使用预采样
import numpy as np

n_samples = len(X)
if n_samples > 2000:
    indices = np.random.choice(n_samples, 2000, replace=False)
    X = X[indices]
    y = y[indices]
```

### 6. 示例Notebook

创建一个新的Colab notebook，包含以下单元格:

**单元格1: 环境设置**
```python
# 检查GPU
!nvidia-smi

# 安装PyTorch 2.x
!pip install torch>=2.0.0 torchvision>=0.15.0 -q
```

**单元格2: 导入库**
```python
import torch
import numpy as np
from core.bcsr_coreset import BCSR_Coreset
from core.models import ResNet18
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
```

**单元格3: 快速测试**
```python
# 运行快速测试
!python test_pytorch2_upgrade.py
```

**单元格4: BCSR示例**
```python
# 运行BCSR选择示例
# (使用上面的基本使用代码)
```

### 7. 资源监控

在训练过程中监控资源使用:

```python
import psutil
import GPUtil

def print_resource_usage():
    # CPU内存
    cpu_mem = psutil.virtual_memory()
    print(f"CPU内存: {cpu_mem.percent}% 使用")

    # GPU
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.memoryUtil*100:.1f}% 内存使用")
        print(f"  温度: {gpu.temperature}°C")

# 定期调用
print_resource_usage()
```

### 8. 保存和加载模型

```python
# 保存模型
torch.save(model.state_dict(), 'model.pth')

# 加载模型
model = ResNet18(config=config)
model.load_state_dict(torch.load('model.pth'))
model.to(device)
```

### 9. 导出结果

```python
# 保存到Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 复制结果
!cp -r results /content/drive/MyDrive/bcsr_results
```

## 最佳实践

1. **从小规模开始**: 先用小数据集验证代码，然后逐步扩大规模
2. **监控资源**: 定期检查GPU/CPU使用情况
3. **保存checkpoint**: 定期保存模型和结果
4. **使用适当参数**: 根据Colab资源调整参数
5. **清理内存**: 在不同任务之间清理GPU内存
   ```python
   torch.cuda.empty_cache()
   ```

## 支持的配置

### T4 GPU (标准Colab):
- 推荐任务数: 2-5
- 推荐batch_size: 8-16
- 推荐外层迭代: 2-3

### V100 GPU (Colab Pro):
- 推荐任务数: 5-10
- 推荐batch_size: 16-32
- 推荐外层迭代: 3-5

### CPU (无GPU):
- 推荐任务数: 1-2
- 推荐batch_size: 4-8
- 推荐外层迭代: 1-2
- 注意: CPU上速度会非常慢

---

最后更新: 2026-04-30
