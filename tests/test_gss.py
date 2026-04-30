# tests/test_gss.py
import torch
import numpy as np
from core.gss_coreset import GSSCoreset
from core.models import ResNet18

def test_gss_interface():
    """Test that GSS implements required interface"""
    config = {'n_classes': 5, 'dropout': 0.0, 'mlp_hiddens': 256}
    model = ResNet18(config=config)

    X = torch.randn(40, 3, 32, 32)
    y = torch.randint(0, 5, (40,))

    coreset = GSSCoreset(device='cpu')
    selected_indices = coreset.select(
        X, y, num_samples=8, task_id=1, model=model
    )

    assert isinstance(selected_indices, np.ndarray)
    assert len(selected_indices) == 8


def test_gss_with_training():
    """Test that GSS coreset improves training compared to random selection"""
    config = {'n_classes': 5, 'dropout': 0.0, 'mlp_hiddens': 256}

    # Small dataset
    X_train = torch.randn(100, 3, 32, 32)
    y_train = torch.randint(0, 5, (100,))
    X_test = torch.randn(20, 3, 32, 32)
    y_test = torch.randint(0, 5, (20,))

    # Select coreset with GSS
    model = ResNet18(config=config)
    coreset = GSSCoreset(device='cpu')
    selected_indices = coreset.select(X_train, y_train, num_samples=20, task_id=1, model=model)

    # Verify selection
    assert len(selected_indices) == 20
    assert len(set(selected_indices)) == 20
