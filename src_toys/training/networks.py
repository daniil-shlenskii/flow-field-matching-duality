import torch
import torch.nn as nn
from torch import Tensor


class ToyNet(nn.Module):
    def __init__(self, hidden_dims: list[int], data_dim: int):
        super().__init__()
        in_dims = [data_dim + 1] + hidden_dims
        out_dims = hidden_dims + [data_dim]
        layers = []
        for in_dim, out_dim in zip(in_dims, out_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.pop()
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor):
        x = torch.cat((x, t.view(-1, 1)), dim=1)
        return self.layers(x)
