import math
import torch
from torch import nn
from torch.nn import functional as F

class ProgressiveLayerwiseLoss(nn.Module):
    def __init__(self, max_power=1.5, k_power=0.05, class_weights=(0.1, 0.9)):
        super().__init__()
        if max_power <= 0 or k_power <= 0:
            raise ValueError('max_power and k_power must be > 0')
        self.max_power, self.k_power = float(max_power), float(k_power)
        self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))

    def get_power(self, epoch: int) -> float:
        return self.max_power * math.exp(-self.k_power * epoch)

    def get_weights(self, n_layers: int, epoch: int, *, device=None, dtype=torch.float32):
        if n_layers < 1:
            raise ValueError('Must have at least one hidden state')
        indices = torch.arange(1, n_layers + 1, dtype=dtype, device=device)
        return torch.softmax(self.get_power(epoch) * indices.log(), dim=0)

    def forward(self, logits_list, targets, epoch: int):
        if not logits_list:
            raise ValueError('logits_list is empty')
        weights = self.class_weights.to(dtype=logits_list[-1].dtype)
        if epoch % 2 == 0:
            return F.cross_entropy(logits_list[-1], targets, weight=weights)
        losses = torch.stack([F.cross_entropy(logits, targets, weight=weights) for logits in logits_list])
        depth_weights = self.get_weights(len(logits_list), epoch, device=losses.device, dtype=losses.dtype)
        return (depth_weights * losses).sum()
