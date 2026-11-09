import numpy as np
import torch
from torch import nn

from .body import from6


class BiGRU(nn.Module):
    def __init__(self, features=72, hidden=64, layers=2):
        super().__init__()
        self.gru = nn.GRU(features, hidden, layers, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, 21 * 6)
        nn.init.normal_(self.head.weight, std=0.005)
        self.head.bias.data.copy_(torch.tensor([1.0, 0, 0, 0, 1, 0] * 21))

    def forward(self, x):
        y = self.head(self.gru(x)[0]).reshape(*x.shape[:2], 21, 6)
        r = from6(y)
        eye = torch.eye(3, device=x.device).expand(*x.shape[:2], 1, 3, 3)
        return torch.cat([eye, r[:, :, 1:]], 2)


def windows(n, size, stride):
    if n < 1 or not 0 < stride <= size:
        raise ValueError("invalid window contract")
    return list(range(0, max(n - size, 0) + 1, stride)) + (
        [max(n - size, 0)] if max(n - size, 0) % stride else []
    )


def predict_windows(model, x, size=200, stride=100):
    """Choose most central window estimate per frame; never average Euler/6D rotations."""
    result = np.zeros((len(x), 21, 3, 3))
    priority = np.full(len(x), -np.inf)
    with torch.inference_mode():
        for start in windows(len(x), size, stride):
            part = x[start : start + size]
            length = len(part)
            padded = np.pad(part, ((0, size - length), (0, 0)), mode="edge")
            pred = model(torch.from_numpy(padded[None]).float()).cpu().numpy()[0, :length]
            center = -np.abs(np.arange(length) - (length - 1) / 2)
            choose = center > priority[start : start + length]
            result[start : start + length][choose] = pred[choose]
            priority[start : start + length][choose] = center[choose]
    return result
