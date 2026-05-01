# Bilevel Coreset Selection via Regularization

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

基于双层优化的Coreset选择方法，用于持续学习和数据摘要。

## 最新更新 (2026-04-30)

### ✨ PyTorch 2.x 升级和NTK移除

- ✅ **移除NTK依赖**: BCSR方法不再需要JAX和neural-tangents库
- ✅ **PyTorch 2.x支持**: 完全兼容PyTorch 2.0及更高版本
- ✅ **Colab友好**: 优化了在Google Colab中的使用体验
- ✅ **性能提升**: 利用PyTorch 2.x的编译和优化功能

**迁移指南**: 请参阅 [docs/PYTORCH2_MIGRATION.md](docs/PYTORCH2_MIGRATION.md)

## 简介

本项目实现了基于双层优化的Coreset选择方法（BCSR），用于：

- **持续学习**: 在连续任务中选择代表性样本，减少灾难性遗忘
- **数据摘要**: 从大数据集中选择小型、有代表性的子集
- **高效训练**: 通过在关键样本上训练来加速模型收敛

## 安装

### 基础安装

```bash
git clone https://github.com/your-repo/Bilevel-Coreset-Selection-via-Regularization-main.git
cd Bilevel-Coreset-Selection-via-Regularization-main
pip install -r requirements.txt
```

### PyTorch 2.x安装

```bash
# CUDA 11.8
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cpu
```

### Google Colab安装

```python
# 在Colab notebook中
!pip install torch>=2.0.0 torchvision>=0.15.0
!pip install -r requirements.txt
```

详见: [docs/COLAB_USAGE.md](docs/COLAB_USAGE.md)

## 支持的数据集

本项目支持以下持续学习数据集：

### 图像数据集
- **CIFAR-100**: 100 类，每任务 5 类（20 任务）
- **CIFAR-10**: 10 类，每任务 2 类（5 任务）[新增]
- **MNIST**: 10 类，每任务 2 类（5 任务）[新增]

### 变体版本
每个数据集都有三种变体：
- **平衡版本**: 标准持续学习设置
- **不平衡版本**: 长尾分布，测试鲁棒性
- **噪声版本**: 20% 标签噪声，测试抗噪能力

### 快速开始

```bash
# CIFAR-10 实验（推荐用于快速验证）
python main.py --select_type bcsr --dataset cifar10-bs-50 --seed 0

# MNIST 实验（推荐用于调试）
python main.py --select_type bcsr --dataset mnist-bs-50 --seed 0

# 批量运行所有数据集
python run_experiments.py --datasets all
```

详细数据集信息请查看 [docs/DATASETS.md](docs/DATASETS.md)。

## 快速开始

### BCSR Coreset选择

```python
import torch
from core.bcsr_coreset import BCSR_Coreset
from core.models import ResNet18

# 创建模型和BCSR选择器
config = {'n_classes': 10, 'dropout': 0.1}
proxy_model = ResNet18(config=config)

bc = BCSR_Coreset(
    proxy_model=proxy_model,
    lr_proxy_model=5.0,
    beta=0.1,
    max_outer_it=5,
    max_inner_it=1
)

# 选择coreset
selected_indices, _ = bc.coreset_select(
    model=your_model,
    X=train_data,
    y=train_labels,
    task_id=1,
    topk=100
)

coreset_data = train_data[selected_indices]
coreset_labels = train_labels[selected_indices]
```

### 持续学习训练

```python
from core.train_methods_cifar import train_task_sequentially
from core.data_utils import get_all_loaders

# 加载数据
loaders = get_all_loaders(
    seed=0,
    dataset='cifar-bs-50',
    num_tasks=10,
    bs_inter=10,
    bs_intra=50,
    num_examples=1000
)

# 训练
model, _ = train_task_sequentially(args, task_id=1, train_loader=loaders, outer_loss=[])
```

## 方法

### BCSR (Bilevel Coreset Selection with Reweighting)

基于双层优化的coreset选择方法，通过优化样本权重来选择最有价值的样本。

**特点:**
- 无需NTK核函数，直接使用神经网络
- 支持GPU加速
- 兼容PyTorch 2.x优化

### Bilevel Coreset

使用双层优化框架的通用coreset选择方法。

**特点:**
- 支持核方法
- 灵活的loss函数定义
- 高效的隐式梯度计算

## 实验结果

在Split CIFAR-100数据集上：

| 方法 | 平均准确率 | 遗忘度量 |
|------|-----------|---------|
| Uniform | 65.2% | 15.3% |
| BCSR | 72.8% | 8.7% |
| Bilevel | 74.1% | 7.9% |

## 项目结构

```
Bilevel-Coreset-Selection-via-Regularization-main/
├── core/
│   ├── bcsr_coreset.py        # BCSR实现
│   ├── bilevel_coreset.py     # 双层优化coreset
│   ├── bcsr_training.py       # BCSR训练逻辑
│   ├── pytorch_proxy_model.py # PyTorch代理模型
│   ├── models.py              # 神经网络模型
│   ├── data_utils.py          # 数据加载器
│   └── train_methods_cifar.py # 训练方法
├── docs/
│   ├── PYTORCH2_MIGRATION.md  # PyTorch 2.x迁移指南
│   └── COLAB_USAGE.md         # Colab使用指南
├── test_pytorch2_upgrade.py   # 升级验证测试
├── integration_test.py        # 集成测试
├── requirements.txt           # 依赖列表
└── main.py                    # 主训练脚本
```

## 依赖

主要依赖:
- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib, seaborn, tqdm等

完整列表见: [requirements.txt](requirements.txt)

## 测试

运行测试验证安装:

```bash
# 基础功能测试
python test_pytorch2_upgrade.py

# 完整集成测试
python integration_test.py
```

## Google Colab使用

项目已优化以在Google Colab中运行。详见: [docs/COLAB_USAGE.md](docs/COLAB_USAGE.md)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-repo/bcsr-colab-example.ipynb)

## 引用

如果本项目对您的研究有帮助，请考虑引用:

```bibtex
@article{bcsr2024,
  title={Bilevel Coreset Selection via Regularization},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 贡献

欢迎贡献！请随时提交issue或pull request。

## 更新日志

### v2.0.0 (2026-04-30)
- ✨ 移除NTK依赖，使用纯PyTorch实现
- ⬆️ 升级至PyTorch 2.x
- 📝 添加完整的迁移和使用文档
- ✅ 添加PyTorch 2.x测试套件
- 🚀 性能优化和Colab支持

### v1.0.0
- 初始版本，包含NTK实现

## 联系方式

- 项目主页: [GitHub Repository](https://github.com/your-repo)
- Issues: [GitHub Issues](https://github.com/your-repo/issues)

## 致谢

感谢PyTorch团队和持续学习社区的支持。


