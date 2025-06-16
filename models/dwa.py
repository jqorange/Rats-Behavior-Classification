import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from collections import deque


class DWAOptimizer:
    """Dynamic Weight Averaging for multi-task learning"""

    def __init__(self, num_tasks=2, temp=2.0, window_size=5):
        self.num_tasks = num_tasks
        self.temp = temp
        self.window_size = window_size
        self.loss_history = [deque(maxlen=window_size) for _ in range(num_tasks)]
        self.weights = [1.0 if (i == 1 or i == 0) else 0.0 for i in range(num_tasks)]

    def update(self, losses):
        """Update task weights based on loss history"""
        for i, loss in enumerate(losses):
            self.loss_history[i].append(loss)

        if len(self.loss_history[0]) >= 2:  # Need at least 2 points to compute rate
            rates = []
            for i in range(self.num_tasks):
                if len(self.loss_history[i]) >= 2:
                    # Compute relative decrease rate
                    current = self.loss_history[i][-1]
                    previous = self.loss_history[i][-2]
                    rate = current / (previous + 1e-8)
                    rates.append(rate)
                else:
                    rates.append(1.0)

            # Convert to weights using softmax
            rates = torch.tensor(rates)
            weights = F.softmax(rates / self.temp, dim=0)
            self.weights = weights.tolist()

        return self.weights