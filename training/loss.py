# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

from sympy.geometry import ellipse
import torch
from torch.onnx.symbolic_opset9 import view_as
from torch_utils import persistence
import numpy as np

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------

@persistence.persistent_class
class FlowMatchingLoss:
    def __init__(self, t_min=1e-4, t_max=1.0-1e-4, scale=0.1, eps=1e-12):
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.scale = float(scale)
        self.eps = float(eps)

    def gamma_stochastic(self, t):
        return torch.sqrt(torch.clamp(t * (1.0 - t), min=self.eps))

    def deriv_gamma(self, t):
        denom = torch.sqrt(torch.clamp(t * (1.0 - t), min=self.eps))
        return (1.0 - 2.0 * t) / (2.0 * denom)

    def __call__(self, net, images, labels=None, augment_pipe=None):
        t = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        t = t * (self.t_max - self.t_min) + self.t_min

        x1 = torch.randn_like(images)
        x0, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

        eps = torch.randn_like(images)
        x_t = x1 * t + x0 * (1.0 - t) + self.scale * self.gamma_stochastic(t) * eps

        target_velocity = (x1 - x0) + self.scale * self.deriv_gamma(t) * eps
        predicted_velocity = net(x_t, t, labels, augment_labels=augment_labels)

        # print(f"target_velocity: {target_velocity.max()}, {target_velocity.min()}")

        return (predicted_velocity - target_velocity) ** 2

from typing import Literal
@persistence.persistent_class
class MultiSampleFlowMatchingInterpolantLoss:
    def __init__(self,
        t_min=1e-4,
        t_max=1.0-1e-4,
        scale=0.1,
        batch_size_for_loss=None,
        eps=1e-12,
        target_ver: Literal["Ex/Ez", "E/|E|"] = "Ex/Ez",
        leave_best_out: bool = False
    ):
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.scale = float(scale)
        self.eps = float(eps)

        self.batch_size_for_loss = batch_size_for_loss
        self.target_ver = target_ver
        self.leave_best_out = leave_best_out

        print(f"leave_best_out: {leave_best_out}")

    def gamma_stochastic(self, t):
        return torch.sqrt(torch.clamp(t * (1.0 - t), min=self.eps))

    def deriv_gamma(self, t):
        denom = torch.sqrt(torch.clamp(t * (1.0 - t), min=self.eps))
        return (1.0 - 2.0 * t) / (2.0 * denom)

    @torch.no_grad()
    def target_Ex_Ez(self, images, t, x_t, x0, x1):
        '''
        xt: [B, C, H, W]
        t: [B, 1, 1, 1]
        x0: [N, C, H, W]
        x1: [N, C, H, W]

        B = opts.batch_size_for_loss // world_size

        -------------------
        In algorithm we have x and (x0, x1) pairs and we want to compute field in x from (x0, x1) pairs
        x := xt in our case

        === Compute E_z = p(xt | x0, x1) ===
        (x0, x1) pairs are not the things, that xt is stochastic interpolation of, they and xt are not related generally
        p(xt | x0, x1) =  N(xt | (1-t)x0 + t x1, s(t)^2 I) - in xt compute distribution using (x0, x1) pairs 

        ==== compute v(xt | x0, x1) =====
        vt(xt | x0,x1) = \dot{I no noise}(t, x0, x1) + \dot{s}(t) * noise
        noise = (xt - I(t, x0, x1)) / s(t) - noisy part of interpolant
        --------------------

        Target = Sum (v * p) / Sum (p),
        can use unnormalized p, because sum is over N for fixed xt with preselected t, so Sum const = const'

        Output: [batch_size_for_loss, C, H, W] 
        '''

        B, N, D = self.batch_size_for_loss, images.shape[0], images[0].numel()  # C * H * W (dimensionality)
        C, H, W = images.shape[1], images.shape[2], images.shape[3]

        # Expanding to make B*N elements:
        N, C, H, W = x0.shape
        B = x_t.shape[0]

        # x0, x1: [N, C, H, W] → [B*N, C, H, W]
        x0 = x0[None, :, :, :, :].expand(B, N, C, H, W).reshape(B * N, C, H, W)     # [x0[0], x0[1], ..., x0[N-1], x0[0], x0[1], ...] (repeated B times)
        x1 = x1[None, :, :, :, :].expand(B, N, C, H, W).reshape(B * N, C, H, W)

        # x_t: [B, C, H, W] → [B*N, C, H, W]
        x_t = x_t[:, None, :, :, :].expand(B, N, C, H, W).reshape(B * N, C, H, W)   # [x[0], x[0], ...(N times)..., x[1], x[1], ...]: for each x we have N (x0, x1) pairs

        # t: [B, 1, 1, 1] → [B*N, 1, 1, 1]
        t = t[:, None, :, :, :].expand(B, N, 1, 1, 1).reshape(B * N, 1, 1, 1)

        mean = (1 - t) * x0 + t * x1  # [B*N, ...]
        diff_sq = torch.linalg.norm((x_t - mean).flatten(1), dim=1) ** 2 # [B*N] || [B*N, C, H, W] -- flatten -- > [B*N, D]
        # print(f"diff_sq: {diff_sq}")

        sigma = self.scale * self.gamma_stochastic(t) # [B*N]
        var = (sigma ** 2).flatten(0) # [B*N]

        gauss_arg = diff_sq / (2 * var) # [B*N]
        # print(f"gauss_arg: {gauss_arg.view(B, N)}")

        noise = (x_t - mean) / sigma # [B*N, C, H, W]
        velocity = (x1 - x0) + self.scale * self.deriv_gamma(t) * noise # [B*N, C, H, W]

        # print(f"velocity: {velocity.max()}, {velocity.min()}")


        ### ========= Less Stable version: =========
        # unnorm_distr = torch.exp(-gauss_arg) # [B*N] -> unnormalized density: Sum of this into [B] is E_z
        # Ex = unnorm_distr.reshape(-1, 1, 1, 1) * velocity # [B*N, C, H, W]
        # Ex = Ex.view(B, N, C, H, W).sum(dim=1) # [B, C, H, W]

        # Ez = unnorm_distr.view(B, N).sum(dim=1) # [B]
        # print(f"Ez: {Ez}")

        # target = Ex / Ez.reshape(-1, 1, 1, 1)   # [B, C, H, W]

        ### ========= More Stable version, mathematically equivalent: =========
        ## $$\frac{\sum_i e^{-g_i} v_i}{\sum_i e^{-g_i}} = \sum_i \text{softmax}(-g)i \cdot v_i$$

        log_weights = -gauss_arg.view(B, N)  # [B, N]

        if self.leave_best_out:
            diag_mask = torch.eye(B, N, dtype=torch.bool, device=log_weights.device)
            log_weights = log_weights.masked_fill(diag_mask, float('-inf'))

        print(f"log_weights: {log_weights}")

        weights = torch.softmax(log_weights, dim=1)  # [B, N] : dim 1: for each x we get distribution for N pairs

        non_zero_counts = (weights > 1e-7).sum(dim=1)  # [B] : count non-zero elements in each row
        print(f"Non-zero weights count: {non_zero_counts}")
        print(f"non zero weights: {weights[weights > 1e-7]}")

        velocity_reshaped = velocity.view(B, N, C, H, W)

        target = (weights[:, :, None, None, None] * velocity_reshaped).sum(dim=1)  # [B, C, H, W]
        return target

    def __call__(self, net, images, labels=None, augment_pipe=None):
        '''
        images_target: x0~p_vol to compute multi-sample Ex / Ez target
        '''
        t = torch.rand([self.batch_size_for_loss, 1, 1, 1], device=images.device)
        t = t * (self.t_max - self.t_min) + self.t_min

        # t = torch.ones_like(t) * 0.99
        # print(f"t: {t}")

        images, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

        x1 = torch.randn_like(images)  # [N, ...]
        x0 = images[:, ...]  # [N, ...]

        # x_t are B points, in which we will compute the field from N points from the planes of capacitor: x0 and x1
        B = self.batch_size_for_loss
        eps = torch.randn_like(x1[:B])  # [B, ...]
        x_t = (1 - t) * images[:B] + t * x1[:B] + self.scale * self.gamma_stochastic(t) * eps # [B, ...]
        labels = labels[:B]

        if self.target_ver == "Ex/Ez":
            target = self.target_Ex_Ez(images, t, x_t, x0, x1)
        elif self.target_ver == "E/|E|":
            target = self.target_E_normE(images, t, x_t, x0, x1)

        if augment_pipe is None:
            augment_labels = None
        else:
            augment_labels = augment_labels[:B]

        pred = net(x_t, t, labels, augment_labels=augment_labels)

        return (pred - target) ** 2

@persistence.persistent_class
class EDMInterpolantLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, D=128, N=3072, pfgmpp=False):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.D = D
        self.N = N
        self.pfgmpp = pfgmpp

    def __call__(self, net, images, labels=None, augment_pipe=None):
        x0, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        # weight = sigma ** 2 * weight

        if self.pfgmpp:
            #### One-sided interpolant: x_sigma = x0 + R * v = x0 + r * sqrt(R2) * v = x0 + sigma * sqrt(D R2) * v
            #### velocity = d x_sigma / (d sigma) = sqrt(D R2) * v, v = u / ||u||
    
            r = sigma.flatten().double() * np.sqrt(self.D).astype(np.float64)
            samples_norm = np.random.beta(a=self.N / 2., b=self.D / 2., size=images.shape[0]).astype(np.double)
            samples_norm = np.clip(samples_norm, 1e-3, 1-1e-3)

            inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
            inverse_beta = torch.from_numpy(inverse_beta).to(images.device).double()

            samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
            samples_norm = samples_norm.view(len(samples_norm), -1)

            gaussian = torch.randn(images.shape[0], self.N).to(samples_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)

            perturbation_x = (unit_gaussian * samples_norm).float()
            eps = perturbation_x.view_as(x0)

            x_sigma = x0 + eps
            # velocity_target = (np.sqrt(self.D) * (torch.sqrt(inverse_beta).unsqueeze(1) * unit_gaussian)).float().view_as(x0)
        else:
            eps = torch.randn_like(x0)
            x_sigma = x0 + eps * sigma
            # velocity_target = eps   # for EDM in velocity notation

        pred = net(x_sigma, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((pred - x0) ** 2)
        return loss

@persistence.persistent_class
class EDMMultiInterpolantLoss:
    def __init__(
        self,
        P_mean=-1.2, P_std=1.2, sigma_data=0.5, D=128, N=3072, pfgmpp=False, 
        batch_size_for_loss=None, leave_best_out=False,
        chunk_size=None
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.D = D
        self.N = N  # data dimensionalty
        self.pfgmpp = pfgmpp
        self.batch_size_for_loss = batch_size_for_loss
        self.leave_best_out = leave_best_out
        self.chunk_size = chunk_size

    @torch.no_grad()
    def target_Ex_Ez(self, sigma, x_sigma, x0):
        N, C, H, W = x0.shape   # N: number of samples to estimate multi-sample target
        B = x_sigma.shape[0]

        x0_expanded = x0[None, :, :, :, :].expand(B, N, C, H, W).reshape(B * N, C, H, W)
        x_sigma_expanded = x_sigma[:, None, :, :, :].expand(B, N, C, H, W).reshape(B * N, C, H, W)
        sigma_expanded = sigma[:, None, :, :, :].expand(B, N, 1, 1, 1).reshape(B * N, 1, 1, 1)

        diff_sq = torch.linalg.norm((x_sigma_expanded - x0_expanded).flatten(1), dim=1) ** 2    # [B*N] || [B*N, C, H, W] -- flatten -- > [B*N, D]
        if self.pfgmpp:
            print(f"PFGMPP:")
            r_sq = self.D * sigma_expanded.flatten(0) ** 2 # [B*N]
            log_weights = -(self.N + self.D) / 2 * torch.log(diff_sq + r_sq + 1e-8)
            log_weights = log_weights.view(B, N)

        else:
            ### EDM:
            ### x_sigma = x0 + sigma * eps, velocity = eps
            ### x0_target = x_sigma - sigma * velocity

            gauss_arg = diff_sq / (2 * sigma_expanded.flatten(0) ** 2)
            log_weights = -gauss_arg.view(B, N)

        velocity = (x_sigma_expanded - x0_expanded) / sigma_expanded     # [B*N, C, H, W]        

        if self.leave_best_out:
            diag_mask = torch.eye(B, N, dtype=torch.bool, device=log_weights.device)
            log_weights = log_weights.masked_fill(diag_mask, float('-inf'))

        weights = torch.softmax(log_weights, dim=1)
        # print(f"weights: {weights}")
        # non_zero_counts = (weights > 1e-7).sum(dim=1)
        # print(f"Non-zero weights count: {non_zero_counts}")
        # print(f"non zero weights: {weights[weights > 1e-7]}")
        # print(f"=" * 100)
        
        velocity_reshaped = velocity.view(B, N, C, H, W)
        velocity_multi_sample = (weights[:, :, None, None, None] * velocity_reshaped).sum(dim=1)

        x0_multi_sample_target = x_sigma - sigma * velocity_multi_sample

        return x0_multi_sample_target

    @torch.no_grad()
    def target_Ex_Ez_chunked(self, sigma, x_sigma, x0):
        N, C, H, W = x0.shape
        B = x_sigma.shape[0]

        running_max = torch.full((B,), float('-inf'), device=x_sigma.device)
        running_sum_exp = torch.zeros(B, device=x_sigma.device)
        running_weighted_vel = torch.zeros(B, C, H, W, device=x_sigma.device)

        for start in range(0, N, self.chunk_size):
            end = min(start + self.chunk_size, N)
            cs = end - start

            x0_chunk = x0[start:end]
            x0_exp = x0_chunk[None].expand(B, cs, C, H, W)
            x_sigma_exp = x_sigma[:, None].expand(B, cs, C, H, W)
            sigma_exp = sigma[:, None].expand(B, cs, 1, 1, 1)

            diff = x_sigma_exp - x0_exp
            diff_sq = torch.linalg.norm(diff.reshape(B * cs, -1), dim=1) ** 2
            diff_sq = diff_sq.view(B, cs)

            if self.pfgmpp:
                r_sq = self.D * sigma_exp.reshape(B, cs) ** 2
                log_w = -(self.N + self.D) / 2 * torch.log(diff_sq + r_sq + 1e-8)
            else:
                gauss_arg = diff_sq / (2 * sigma_exp.reshape(B, cs) ** 2)
                log_w = -gauss_arg

            vel = diff / sigma_exp

            if self.leave_best_out:
                global_indices = torch.arange(start, end, device=log_w.device)
                batch_indices = torch.arange(B, device=log_w.device)
                mask = batch_indices[:, None] == global_indices[None, :]
                log_w = log_w.masked_fill(mask, float('-inf'))

            chunk_max = log_w.max(dim=1).values
            new_max = torch.maximum(running_max, chunk_max)

            scale = torch.exp(running_max - new_max)
            running_sum_exp = running_sum_exp * scale
            running_weighted_vel = running_weighted_vel * scale[:, None, None, None]

            exp_w = torch.exp(log_w - new_max[:, None])
            running_sum_exp = running_sum_exp + exp_w.sum(dim=1)
            running_weighted_vel = running_weighted_vel + (exp_w[:, :, None, None, None] * vel).sum(dim=1)

            running_max = new_max

        velocity_multi_sample = running_weighted_vel / running_sum_exp[:, None, None, None]
        x0_multi_sample_target = x_sigma - sigma * velocity_multi_sample
        return x0_multi_sample_target

    def __call__(self, net, images, labels=None, augment_pipe=None):
        images, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        B = self.batch_size_for_loss

        rnd_normal = torch.randn([B, 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        # weight = sigma ** 2 * weight

        if self.pfgmpp:
            r = sigma.flatten().double() * np.sqrt(self.D).astype(np.float64)
            samples_norm = np.random.beta(a=self.N / 2., b=self.D / 2., size=B).astype(np.double)
            samples_norm = np.clip(samples_norm, 1e-3, 1-1e-3)

            inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
            inverse_beta = torch.from_numpy(inverse_beta).to(images.device).double()

            samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
            samples_norm = samples_norm.view(len(samples_norm), -1)

            gaussian = torch.randn(B, self.N).to(samples_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)

            perturbation_x = (unit_gaussian * samples_norm).float()
            eps = perturbation_x.view_as(images[:B])
            x_sigma = images[:B] + eps
        else:
            eps = torch.randn_like(images[:B])
            x_sigma = images[:B] + eps * sigma

        labels = labels[:B]
        if augment_pipe is not None:
            augment_labels = augment_labels[:B]
        else:
            augment_labels = None

        pred = net(x_sigma, sigma, labels, augment_labels=augment_labels)

        if self.chunk_size is None:
            target = self.target_Ex_Ez(sigma, x_sigma, images)
        else:
            target = self.target_Ex_Ez_chunked(sigma, x_sigma, images)

        loss = weight * ((pred - target) ** 2)
        return loss