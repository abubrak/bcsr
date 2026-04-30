"""
PyTorch代理模型用于BCSR双层优化

替代NTK核函数方法，直接使用轻量级PyTorch模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PyTorchProxyModel(nn.Module):
    """
    轻量级代理模型，用于BCSR的样本权重学习

    相比完整模型，这个代理模型参数更少，适合高频的双层优化
    """

    def __init__(
        self,
        input_shape: tuple,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        初始化代理模型

        参数:
            input_shape: 输入形状 (C, H, W) 或 (H*W,)
            num_classes: 类别数
            hidden_dim: 隐藏层维度
            num_layers: 隐藏层数量
            dropout: Dropout概率
        """
        super().__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes

        # 计算展平后的输入维度
        if len(input_shape) == 3:  # (C, H, W)
            input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        else:  # 已经展平
            input_dim = input_shape[0]

        # 构建MLP
        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(current_dim, num_classes))

        self.network = nn.Sequential(*layers)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """使用Xavier初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量
            task_id: 任务ID（用于持续学习场景）

        返回:
            输出logits
        """
        # 展平输入如果需要
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        logits = self.network(x)

        # 如果提供了task_id，应用类别掩码
        if task_id is not None and hasattr(self, '_num_classes_per_task'):
            # 这里可以根据需要实现任务特定的掩码
            pass

        return logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取倒数第二层的特征表示

        用于核方法或特征匹配

        参数:
            x: 输入张量

        返回:
            特征表示
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # 前向传播到倒数第二层
        for i, module in enumerate(self.network):
            x = module(x)
            if i == len(self.network) - 2:  # 倒数第二层
                return x

        return x


class ResNetProxyModel(nn.Module):
    """
    轻量级ResNet代理模型，用于CIFAR等图像数据集

    相比完整ResNet18，这个版本参数更少，训练更快
    """

    def __init__(
        self,
        num_classes: int = 100,
        num_blocks: int = 1,  # 每层只有1个block vs ResNet18的2个
        base_channels: int = 32,  # 更少的通道数
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_classes = num_classes
        self.in_planes = base_channels

        # 初始卷积
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        # ResNet块
        self.layer1 = self._make_layer(base_channels, num_blocks, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, num_blocks, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, num_blocks, stride=2)
        self.layer4 = self._make_layer(base_channels * 8, num_blocks, stride=2)

        # 输出层
        self.linear = nn.Linear(base_channels * 8, num_classes)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def _make_layer(self, planes: int, num_blocks: int, stride: int):
        """创建ResNet层"""
        layers = []
        layers.append(BasicBlock(self.in_planes, planes, stride))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """前向传播"""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)

        if self.dropout is not None:
            out = self.dropout(out)

        out = self.linear(out)

        # 任务ID掩码（如果需要）
        if task_id is not None:
            # 实现任务特定的输出掩码
            pass

        return out


class BasicBlock(nn.Module):
    """ResNet基本块"""
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def create_proxy_model(
    model_type: str,
    input_shape: tuple,
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    创建代理模型的工厂函数

    参数:
        model_type: 模型类型 ('mlp' 或 'resnet')
        input_shape: 输入形状
        num_classes: 类别数
        **kwargs: 额外参数

    返回:
        代理模型
    """
    if model_type == 'mlp':
        return PyTorchProxyModel(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type == 'resnet':
        return ResNetProxyModel(
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")
