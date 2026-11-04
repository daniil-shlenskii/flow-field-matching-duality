import abc
import math

import torch
from torch import Tensor


class BaseInterpolant(abc.ABC):
    @abc.abstractmethod
    def __call__(self, z: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        pass
    
    @abc.abstractmethod
    def derivative(self, z: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        pass

class BaseSigmaSchedule(abc.ABC):
    def __init__(self, scale: float = 1.0):
        self.scale = scale

    @abc.abstractmethod
    def __call__(self, z: Tensor) -> Tensor:
        pass
    
    @abc.abstractmethod
    def derivative(self, z: Tensor) -> Tensor:
        pass

    def ratio(self, z: Tensor) -> Tensor:
        sigma = self(z)
        dsigma = self.derivative(z)
        return dsigma / sigma


#####


class OneSidedLinearInterpolant:
    def __call__(self, z: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        assert z.shape[0] == x0.shape[0] == x1.shape[0]
        return x0 * z.view(-1, 1)
    
    def derivative(self, z: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        assert z.shape[0] == x0.shape[0] == x1.shape[0]
        return x0

class LinearInterpolant:
    def __call__(self, z: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        assert z.shape[0] == x0.shape[0] == x1.shape[0]
        return x0 + z.view(-1, 1) * (x1 - x0)
    
    def derivative(self, z: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        assert z.shape[0] == x0.shape[0] == x1.shape[0]
        return x1 - x0

class SineCosineInterpolant:
    def __call__(self, z: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        assert z.shape[0] == x0.shape[0] == x1.shape[0]
        z = z.view(-1, 1)
        return torch.cos(math.pi / 2 * z) * x0 + torch.sin(math.pi / 2 * z) * x1
    
    def derivative(self, z: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        assert z.shape[0] == x0.shape[0] == x1.shape[0]
        z = z.view(-1, 1)
        return (math.pi / 2) * (-torch.sin(math.pi / 2 * z) * x0 + torch.cos(math.pi / 2 * z) * x1)

class LinearSigmaSchedule(BaseSigmaSchedule):
    def __call__(self, z: Tensor) -> Tensor:
        return self.scale * (1 - z)
    
    def derivative(self, z: Tensor) -> Tensor:
        return -self.scale * torch.ones_like(z)

class QuadraticSigmaSchedule(BaseSigmaSchedule):
    def __call__(self, z: Tensor) -> Tensor:
        return self.scale * torch.sqrt(z * (1 - z))
    
    def derivative(self, z: Tensor) -> Tensor:
        return self.scale * 0.5 *(1 - 2 * z) / torch.sqrt(z * (1 - z))

class ZeroSigmaSchedule(BaseSigmaSchedule):
    def __call__(self, z: Tensor) -> Tensor:
        return torch.zeros_like(z)
    
    def derivative(self, z: Tensor) -> Tensor:
        return torch.zeros_like(z)

    def ratio(self, z: Tensor) -> Tensor:
        return torch.ones_like(z)
