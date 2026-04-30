# core/gss_coreset.py
import torch
import torch.nn as nn
import numpy as np


class GSSCoreset:
    """
    Gradient Sample Selection (GSS) Coreset

    Selects samples based on influence functions and gradient similarity.
    Measures how much each sample influences the model's parameters.

    Reference:
        "Gradient Sample Selection for Deep Learning"
    """

    def __init__(self, device='cuda', influence_method='grad_norm'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.influence_method = influence_method

    def select(self, X, y, num_samples, task_id=None, model=None, **kwargs):
        """
        Select coreset using gradient sample selection

        Args:
            X: Data tensor (N, C, H, W)
            y: Label tensor (N,)
            num_samples: Number of samples to select
            task_id: Current task ID
            model: Neural network model (required)

        Returns:
            selected_indices: NumPy array of selected indices
        """
        if model is None:
            raise ValueError("GSS requires a model for influence computation")

        X_tensor = X if isinstance(X, torch.Tensor) else torch.from_numpy(X).float()
        y_tensor = y if isinstance(y, torch.Tensor) else torch.from_numpy(y).long()

        # Compute influence scores
        influence_scores = self._compute_influence(
            X_tensor, y_tensor, model, task_id
        )

        # Select samples with highest influence
        selected_indices = np.argsort(influence_scores)[-num_samples:]

        return selected_indices[::-1]

    def _compute_influence(self, X, y, model, task_id):
        """
        Compute influence score for each sample

        Uses gradient norm w.r.t. input as proxy for influence.
        """
        model.train()  # Use train mode for gradient computation
        influence_scores = []

        criterion = nn.CrossEntropyLoss()
        batch_size = 32

        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size].to(self.device)
            batch_y = y[i:i+batch_size].to(self.device)

            # Enable gradient computation for input
            batch_X.requires_grad = True

            # Forward pass
            logits = model(batch_X, task_id)
            loss = criterion(logits, batch_y)

            # Backward pass
            loss.backward()

            # Compute influence score (gradient norm w.r.t input)
            if batch_X.grad is not None:
                # Reshape gradient to (N, -1) and compute L2 norm
                grad_flat = batch_X.grad.reshape(batch_X.grad.size(0), -1)
                grad_norm = torch.linalg.vector_norm(grad_flat, dim=1)  # (N,)
                influence_scores.extend(grad_norm.cpu().detach().numpy())

            # Detach gradients
            batch_X = batch_X.detach()

        return np.array(influence_scores)
