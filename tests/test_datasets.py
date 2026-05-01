"""
测试所有数据集加载器

验证新添加的 CIFAR-10 和 MNIST 数据集是否正确加载
"""

import pytest
import torch
import numpy as np
from core.data_utils import get_all_loaders


class TestCIFAR10Datasets:
    """测试 CIFAR-10 数据集"""

    def test_cifar10_balanced_loading(self):
        """测试平衡 CIFAR-10 加载"""
        loaders = get_all_loaders(
            seed=0,
            dataset='cifar10-bs-50',
            num_tasks=5,
            bs_inter=10,
            bs_intra=50,
            num_examples=100
        )

        # 验证所有任务都已加载
        assert len(loaders['sequential']) == 5
        assert len(loaders['coreset']) == 5

        # 验证任务 1 的数据形状
        train_batch = loaders['sequential'][1]['train'][0]
        assert train_batch[0].shape[1:] == torch.Size([3, 32, 32])  # CIFAR-10 图像形状

    def test_cifar10_imbalanced_loading(self):
        """测试不平衡 CIFAR-10 加载"""
        loaders = get_all_loaders(
            seed=0,
            dataset='imb-cifar10-bs-50',
            num_tasks=5,
            bs_inter=10,
            bs_intra=50,
            num_examples=100
        )

        # 验证加载成功
        assert len(loaders['sequential']) == 5

    def test_cifar10_noisy_loading(self):
        """测试噪声 CIFAR-10 加载"""
        loaders = get_all_loaders(
            seed=0,
            dataset='noise-cifar10-bs-50',
            num_tasks=5,
            bs_inter=10,
            bs_intra=50,
            num_examples=100
        )

        # 验证加载成功
        assert len(loaders['sequential']) == 5


class TestMNISTDatasets:
    """测试 MNIST 数据集"""

    def test_mnist_balanced_loading(self):
        """测试平衡 MNIST 加载"""
        loaders = get_all_loaders(
            seed=0,
            dataset='mnist-bs-50',
            num_tasks=5,
            bs_inter=10,
            bs_intra=50,
            num_examples=100
        )

        # 验证所有任务都已加载
        assert len(loaders['sequential']) == 5
        assert len(loaders['coreset']) == 5

        # 验证任务 1 的数据形状
        train_batch = loaders['sequential'][1]['train'][0]
        assert train_batch[0].shape[1] == 784  # MNIST 展平后形状

    def test_mnist_imbalanced_loading(self):
        """测试不平衡 MNIST 加载"""
        loaders = get_all_loaders(
            seed=0,
            dataset='imb-mnist-bs-50',
            num_tasks=5,
            bs_inter=10,
            bs_intra=50,
            num_examples=100
        )

        # 验证加载成功
        assert len(loaders['sequential']) == 5

    def test_mnist_noisy_loading(self):
        """测试噪声 MNIST 加载"""
        loaders = get_all_loaders(
            seed=0,
            dataset='noise-mnist-bs-50',
            num_tasks=5,
            bs_inter=10,
            bs_intra=50,
            num_examples=100
        )

        # 验证加载成功
        assert len(loaders['sequential']) == 5


class TestDatasetConsistency:
    """测试数据集一致性"""

    @pytest.mark.parametrize("dataset", [
        'cifar10-bs-50',
        'imb-cifar10-bs-50',
        'noise-cifar10-bs-50',
        'mnist-bs-50',
        'imb-mnist-bs-50',
        'noise-mnist-bs-50',
    ])
    def test_all_datasets_loadable(self, dataset):
        """参数化测试所有新数据集都可加载"""
        loaders = get_all_loaders(
            seed=0,
            dataset=dataset,
            num_tasks=5,
            bs_inter=10,
            bs_intra=50,
            num_examples=100
        )

        # 验证基本结构
        assert 'sequential' in loaders
        assert 'coreset' in loaders
        assert len(loaders['sequential']) == 5

    def test_cifar100_still_works(self):
        """确保 CIFAR-100 仍然正常工作（回归测试）"""
        loaders = get_all_loaders(
            seed=0,
            dataset='cifar-bs-50',
            num_tasks=20,
            bs_inter=10,
            bs_intra=50,
            num_examples=100
        )

        # CIFAR-100 有 20 个任务
        assert len(loaders['sequential']) == 20
