# tests/test_gradmatch.py
import torch
import torch.nn as nn
import numpy as np
from core.gradmatch_coreset import GradMatchCoreset
from core.coreset_baselines import GradMatchCoreset as GradMatchCoresetBaseline
from core.models import ResNet18

def test_gradmatch_interface():
    """Test that GradMatch implements required interface"""
    # Create simple model
    config = {'n_classes': 5, 'dropout': 0.0, 'mlp_hiddens': 256}
    model = ResNet18(config=config)

    X = torch.randn(50, 3, 32, 32)
    y = torch.randint(0, 5, (50,))

    coreset = GradMatchCoreset(device='cpu')
    selected_indices = coreset.select(
        X, y, num_samples=10, task_id=1, model=model
    )

    assert isinstance(selected_indices, np.ndarray)
    assert len(selected_indices) == 10
    assert len(set(selected_indices)) == 10  # No duplicates

def test_gradmatch_deterministic():
    """Test that GradMatch produces consistent results with same seed"""
    config = {'n_classes': 5, 'dropout': 0.0, 'mlp_hiddens': 256}
    model = ResNet18(config=config)

    X = torch.randn(30, 3, 32, 32)
    y = torch.randint(0, 5, (30,))

    torch.manual_seed(42)
    coreset1 = GradMatchCoreset(device='cpu')
    indices1 = coreset1.select(X, y, num_samples=10, task_id=1, model=model)

    torch.manual_seed(42)
    coreset2 = GradMatchCoreset(device='cpu')
    indices2 = coreset2.select(X, y, num_samples=10, task_id=1, model=model)

    assert np.array_equal(indices1, indices2)

def test_gradmatch_baseline_consistency():
    """Test that baseline version works the same as standalone version"""
    config = {'n_classes': 5, 'dropout': 0.0, 'mlp_hiddens': 256}
    model = ResNet18(config=config)

    X = torch.randn(30, 3, 32, 32)
    y = torch.randint(0, 5, (30,))

    # Test standalone version
    coreset_standalone = GradMatchCoreset(device='cpu')
    indices_standalone = coreset_standalone.select(
        X, y, num_samples=10, task_id=1, model=model
    )

    # Test baseline version
    coreset_baseline = GradMatchCoresetBaseline(device='cpu')
    indices_baseline = coreset_baseline.select(
        X, y, num_samples=10, task_id=1, model=model
    )

    assert np.array_equal(indices_standalone, indices_baseline)
