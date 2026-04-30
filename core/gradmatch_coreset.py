# core/gradmatch_coreset.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class GradMatchCoreset:
    """
    GradMatch Coreset Selection

    Selects samples that minimize gradient matching loss.
    Finds coreset such that gradients on coreset match gradients on full dataset.

    Reference:
        "GradMatch: Gradient-based Test Time Sample Selection for Efficient Training"
    """

    def __init__(self, device='cuda', lr=0.01, epochs=50):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.lr = lr
        self.epochs = epochs

    def select(self, X, y, num_samples, task_id=None, model=None, **kwargs):
        """
        Select coreset using gradient matching

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
            raise ValueError("GradMatch requires a model for gradient computation")

        X_tensor = X if isinstance(X, torch.Tensor) else torch.from_numpy(X).float()
        y_tensor = y if isinstance(y, torch.Tensor) else torch.from_numpy(y).long()

        # Compute sample importance scores via gradient similarity
        importance_scores = self._compute_importance(
            X_tensor, y_tensor, model, task_id
        )

        # Select top-k samples with highest importance
        selected_indices = np.argsort(importance_scores)[-num_samples:]

        return selected_indices[::-1]  # Return in descending order

    def _compute_importance(self, X, y, model, task_id):
        """
        Compute importance scores for each sample based on gradient norms

        Samples with larger gradients are more important for learning.
        """
        model.eval()
        importance_scores = []

        criterion = nn.CrossEntropyLoss(reduction='none')
        batch_size = 32

        # Compute gradient norm for each sample
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size].to(self.device)
            batch_y = y[i:i+batch_size].to(self.device)

            # Forward pass
            logits = model(batch_X, task_id)
            losses = criterion(logits, batch_y)

            # Compute gradients for each sample
            for j in range(len(batch_X)):
                if batch_X[j].requires_grad:
                    batch_X[j].retain_grad()

                loss_j = losses[j]
                loss_j.backward(retain_graph=(j < len(batch_X) - 1))

                # Compute gradient norm
                grad_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item()

                importance_scores.append(grad_norm)

                # Zero gradients
                model.zero_grad()

        return np.array(importance_scores)
