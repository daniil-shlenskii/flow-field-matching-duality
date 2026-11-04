import abc

import torch
from torch import Tensor
from torch.distributions import Beta


class Field(abc.ABC):
    def __init__(self, interpolant, sigma_schedule, data_dim: int):
        self.interpolant = interpolant
        self.sigma_schedule = sigma_schedule
        self.data_dim = data_dim

    @abc.abstractmethod
    def sample_eps(self, sigma: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def unnormalizled_distribution(self, r: Tensor, sigma: Tensor) -> Tensor:
        pass

    def sample(self, z: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        xz = self.interpolant(z, x0, x1)
        sigma = self.sigma_schedule(z)
        eps = self.sample_eps(sigma)
        return xz + eps

    def magnitude(self, r: Tensor, sigma: Tensor) -> Tensor:
        return self.unnormalizled_distribution(r, sigma) / sigma**self.data_dim

    def velocity(self, x: Tensor, z: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        ratio = self.sigma_schedule.ratio(z).view(-1, 1)

        xz = self.interpolant(z, x0, x1)
        dxz = self.interpolant.derivative(z, x0, x1)

        return ratio * (x - xz) + dxz

    def strength(
        self,
        x: Tensor, # [B, data_dim] -> xt
        z: Tensor, # [B] -> time
        x0: Tensor, # [N, data_dim]
        x1: Tensor, # [N, data_dim]
        unnormalized: bool = False,
    ) -> Tensor:
        # v_z(x | x0, x1) = Idot(z,x0,x1) + ratio(z,x0,x1) * (x(z,x0,x1) - I(z,x0,x1))
        B, N, D = x.shape[0], x0.shape[0], self.data_dim

        sigma = self.sigma_schedule(z) # [B]
        ratio = self.sigma_schedule.ratio(z) # [B]

        z = z[:, None].expand(B, N).reshape(B * N) # [B*N]
        x0 = x0[None, :, :].expand(B, N, D).reshape(B * N, D) # [B*N, D]
        x1 = x1[None, :, :].expand(B, N, D).reshape(B * N, D) # [B*N, D]
        x = x[:, None, :].expand(B, N, D).reshape(B * N, D) # [B*N, D]
        sigma = sigma[:, None].expand(B, N).reshape(B * N) # [B*N]
        ratio = ratio[:, None].expand(B, N).reshape(B * N) # [B*N]

        xz = self.interpolant(z, x0, x1) # [B*N, D] -> deterministic part of interpolant
        dxz = self.interpolant.derivative(z, x0, x1) # [B*N, D] -> derivative of deterministic part of interpolant

        # noise = (xt - I(t, x0, x1)) / s(t)
        # vt(xt | x0,x1) = \dot{I}(t, x0, x1) + \dot{s}(t) * (xt - I(t, x0, x1)) / s(t)
        eps = x - xz # [B*N, D] xt = I + s noise # s noise = xt - I
        r = torch.linalg.norm(eps, dim=-1) # [B*N]

        if unnormalized:
            mag = self.unnormalizled_distribution(r, sigma) # [B*N] -> density
        else:
            mag = self.magnitude(r, sigma) # [B*N] -> density

        Ex = mag[:, None] * (eps * ratio[:, None] + dxz) # [B*N, D]
        Ez = mag # [B*N]

        Ex = Ex.view(B, N, D).sum(dim=1) # [B, D]
        Ez = Ez.view(B, N).sum(dim=1).unsqueeze(1) # [B, 1]

        return torch.cat([Ex, Ez], dim=1)


class GaussianField(Field):
    def unnormalizled_distribution(self, r: Tensor, sigma: Tensor) -> Tensor:
        return torch.exp(-r**2 / (2 * sigma**2))

    def sample_eps(self, sigma: Tensor) -> Tensor:
        return torch.randn(sigma.shape[0], self.data_dim, device=sigma.device, dtype=sigma.dtype) * sigma.view(-1, 1)

class PFGMPPField(Field):
    def __init__(self, interpolant, sigma_schedule, data_dim: int, aux_dim: float=1.0, eps: float=1e-6):
        super().__init__(interpolant, sigma_schedule, data_dim)
        self.aux_dim = aux_dim
        self.eps = eps

    def unnormalizled_distribution(self, r: Tensor, sigma: Tensor) -> Tensor:
        return (1 + r**2 / sigma**2)**(-(self.data_dim + self.aux_dim) / 2)

    def sample_eps(self, sigma: Tensor) -> Tensor:
        # Sample from inverse-beta distribution
        beta_gen = Beta(torch.FloatTensor([self.data_dim / 2.]), torch.FloatTensor([self.aux_dim / 2.]))
        beta_sample = beta_gen.sample((sigma.shape[0],)).to(sigma.device).to(sigma.dtype).reshape(-1)
        inverse_beta_sample = beta_sample / (1 - beta_sample).clamp(min=self.eps)

        # Sample from p(R)
        R = sigma * torch.sqrt(inverse_beta_sample.clamp(min=self.eps))

        # Sample angle
        gaussian = torch.randn(sigma.shape[0], self.data_dim, device=sigma.device, dtype=sigma.dtype)
        angle = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)

        return R.view(-1, 1) * angle
