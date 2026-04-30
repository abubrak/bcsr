"""
Tests for coreset selection baseline methods
"""
import torch
import numpy as np
from core.coreset_baselines import HerdingCoreset


def test_herding_interface():
    """Test that HerdingCoreset implements required interface"""
    X = torch.randn(100, 3, 32, 32)
    y = torch.randint(0, 5, (100,))

    coreset = HerdingCoreset(device='cpu')
    selected_indices = coreset.select(X, y, num_samples=20, task_id=1)

    assert isinstance(selected_indices, np.ndarray)
    assert len(selected_indices) == 20
    assert all(0 <= idx < 100 for idx in selected_indices)


def test_herding_more_samples_than_data():
    """Test Herding when requesting more samples than available"""
    X = torch.randn(10, 3, 32, 32)
    y = torch.randint(0, 3, (10,))

    coreset = HerdingCoreset(device='cpu')
    selected_indices = coreset.select(X, y, num_samples=20, task_id=1)

    # Should return all available samples
    assert len(selected_indices) == 10


def test_herding_single_class():
    """Test Herding with single class"""
    X = torch.randn(50, 3, 32, 32)
    y = torch.zeros(50, dtype=torch.long)  # All same class

    coreset = HerdingCoreset(device='cpu')
    selected_indices = coreset.select(X, y, num_samples=10, task_id=1)

    assert len(selected_indices) == 10
