"""
Coreset selection baseline methods

This module implements classic coreset selection algorithms that serve as
baselines for comparison with the BCSR method.
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from abc import ABC, abstractmethod


class BaseCoresetSelector(ABC):
    """Base class for coreset selection methods"""

    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'

    @abstractmethod
    def select(self, X, y, num_samples, task_id=None, model=None, **kwargs):
        """
        Select coreset indices from dataset

        Args:
            X: Data tensor (N, ...) or (N, C, H, W)
            y: Label tensor (N,)
            num_samples: Number of samples to select
            task_id: Current task ID (optional)
            model: Neural network model (optional, for gradient-based methods)
            **kwargs: Additional method-specific parameters

        Returns:
            selected_indices: NumPy array of selected indices
        """
        pass

    def _ensure_numpy(self, tensor):
        """Convert tensor to numpy array"""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return tensor

    def _ensure_tensor(self, array):
        """Convert numpy array to tensor"""
        if isinstance(array, np.ndarray):
            return torch.from_numpy(array).float()
        return array


class HerdingCoreset(BaseCoresetSelector):
    """
    Herding (K-Centers) Coreset Selection

    Selects samples closest to cluster centers in feature space.
    Uses k-means clustering on features extracted from model.

    Reference:
        "Herding" by Welling et al.
    """

    def __init__(self, device='cuda', use_model_features=True):
        super().__init__(device)
        self.use_model_features = use_model_features

    def select(self, X, y, num_samples, task_id=None, model=None, **kwargs):
        """
        Select coreset using herding/k-centers

        Args:
            X: Data tensor (N, C, H, W) for images or (N, D) for features
            y: Label tensor (N,)
            num_samples: Number of samples to select
            task_id: Current task ID
            model: Neural network for feature extraction (optional)

        Returns:
            selected_indices: NumPy array of selected indices
        """
        # Validate inputs
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")
        if len(X) == 0:
            raise ValueError("Dataset cannot be empty")

        X_np = self._ensure_numpy(X)
        y_np = self._ensure_numpy(y)

        # If image data and model provided, extract features
        if len(X_np.shape) == 4 and model is not None and self.use_model_features:
            features = self._extract_features(X, y, model, task_id)
        else:
            # Flatten image data or use features directly
            if len(X_np.shape) == 4:
                features = X_np.reshape(X_np.shape[0], -1)
            else:
                features = X_np

        # Perform per-class herding
        selected_indices = self._herding_per_class(features, y_np, num_samples)

        return selected_indices

    def _extract_features(self, X, y, model, task_id):
        """Extract features from penultimate layer of model"""
        if not hasattr(model, 'embed'):
            raise ValueError(
                f"Model {type(model).__name__} does not have 'embed()' method. "
                "HerdingCoreset requires models with feature extraction capability."
            )

        model.eval()
        features_list = []
        batch_size = 128

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size].to(self.device)
                batch_y = y[i:i+batch_size]

                # Get features from penultimate layer
                features = model.embed(batch_X)
                features_list.append(features.cpu().numpy())

        return np.vstack(features_list)

    def _herding_per_class(self, features, labels, num_samples):
        """Perform herding separately for each class"""
        unique_labels = np.unique(labels)
        n_total = len(features)

        # If requesting more samples than available, return all
        if num_samples >= n_total:
            return np.arange(n_total)

        samples_per_class = max(1, num_samples // len(unique_labels))
        selected_indices = []

        for label in unique_labels:
            class_mask = (labels == label)
            class_features = features[class_mask]
            class_indices = np.where(class_mask)[0]

            if len(class_indices) <= samples_per_class:
                # If class has fewer samples than needed, take all
                selected_indices.extend(class_indices)
            else:
                # Perform k-means to find centers
                n_clusters = min(samples_per_class, len(class_indices))
                kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
                kmeans.fit(class_features)

                # Find samples closest to each center
                centers = kmeans.cluster_centers_
                distances = kmeans.transform(class_features)

                # Select closest sample to each center
                selected_for_class = []
                for center_idx in range(n_clusters):
                    closest_idx = np.argmin(distances[:, center_idx])
                    selected_for_class.append(class_indices[closest_idx])

                selected_indices.extend(selected_for_class)

        # If we didn't get enough samples, add more from remaining classes
        if len(selected_indices) < num_samples:
            remaining_indices = set(range(n_total)) - set(selected_indices)
            selected_indices.extend(list(remaining_indices)[:num_samples - len(selected_indices)])

        return np.array(selected_indices[:num_samples])
