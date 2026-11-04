import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.fields import Field


class SILoss:
    def __call__(
        self, net: nn.Module, field: Field, x0: Tensor, x1: Tensor, eps: float=1e-3
    ) -> Tensor:
        z = eps + torch.rand(x0.shape[0], device=x0.device) * (1 - 2 * eps)
        x = field.sample(z, x0, x1)

        v_target = field.velocity(x, z, x0, x1)
        v_pred = net(x, z)

        return F.mse_loss(v_pred, v_target)

class ExEzLoss:
    def __call__(
        self, net: nn.Module, field: Field, x0: Tensor, x1: Tensor, eps: float=1e-3
    ) -> Tensor:

        z = eps + torch.rand(x0.shape[0], device=x0.device) * (1 - 2 * eps)
        x = field.sample(z, x0, x1)
        E = field.strength(x, z, x0, x1, unnormalized=True)

        v_target= E[:, :-1] / E[:, -1:]
        v_pred = net(x, z)

        return F.mse_loss(v_pred, v_target)

class ENormELoss:
    def __call__(
        self, net: nn.Module, field: Field, x0: Tensor, x1: Tensor, eps: float=1e-3
    ) -> Tensor:

        z = eps + torch.rand(x0.shape[0], device=x0.device) * (1 - 2 * eps)
        x = field.sample(z, x0, x1)
        E = field.strength(x, z, x0, x1, unnormalized=True)

        E_pred = net(x, z)

        return F.mse_loss(E_pred, E)