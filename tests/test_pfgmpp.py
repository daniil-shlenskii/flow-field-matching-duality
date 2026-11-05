import unittest
from unittest import mock

import numpy as np
import torch

from training.loss import EDMInterpolantLoss, EDMMultiInterpolantLoss


class PFGMPPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        np.random.seed(7)

    def test_single_and_multi_forward_use_image_dimension(self):
        for resolution in (32, 64):
            images = torch.randn(4, 3, resolution, resolution)
            labels = torch.zeros(4, 0)
            for cls in (EDMInterpolantLoss, EDMMultiInterpolantLoss):
                with self.subTest(resolution=resolution, loss=cls.__name__):
                    kwargs = dict(batch_size_for_loss=2) if cls is EDMMultiInterpolantLoss else {}
                    loss = cls(pfgmpp=True, **kwargs)
                    with mock.patch('numpy.random.beta', wraps=np.random.beta) as beta:
                        result = loss(lambda x, sigma, labels, **kw: x * 0.5, images, labels)
                    self.assertEqual(beta.call_args.kwargs['a'], images[0].numel() / 2)
                    self.assertEqual(result.shape, images[:2].shape if kwargs else images.shape)
                    self.assertTrue(torch.isfinite(result).all())

    def test_targets_match_pfgmpp_posterior_at_both_resolutions(self):
        for resolution in (32, 64):
            for leave_best_out in (False, True):
                with self.subTest(resolution=resolution, leave_best_out=leave_best_out):
                    x0 = torch.randn(5, 3, resolution, resolution, dtype=torch.float64) * 0.1
                    x_sigma = torch.randn(2, 3, resolution, resolution, dtype=torch.float64) * 0.1
                    sigma = torch.full((2, 1, 1, 1), 2.0, dtype=torch.float64)
                    data_dim = x0[0].numel()
                    distances = (x_sigma[:, None] - x0[None]).flatten(2).square().sum(2)
                    log_weights = -(data_dim + 128) / 2 * (distances + 128 * sigma.flatten()[:, None] ** 2 + 1e-8).log()
                    if leave_best_out:
                        log_weights[torch.arange(2), torch.arange(2)] = -torch.inf
                    expected = torch.einsum('bn,nchw->bchw', log_weights.softmax(1), x0)
                    loss = EDMMultiInterpolantLoss(pfgmpp=True, leave_best_out=leave_best_out, chunk_size=3)
                    dense = loss.target_Ex_Ez(sigma, x_sigma, x0)
                    chunked = loss.target_Ex_Ez_chunked(sigma, x_sigma, x0)
                    torch.testing.assert_close(dense, expected, atol=1e-10, rtol=1e-8)
                    torch.testing.assert_close(chunked, expected, atol=1e-10, rtol=1e-8)

    def test_cifar_noise_matches_original_formula(self):
        images = torch.randn(2, 3, 32, 32)
        torch.manual_seed(19)
        np.random.seed(19)
        sigma = (torch.randn(2, 1, 1, 1) * 1.2 - 1.2).exp()
        beta = np.clip(np.random.beta(3072 / 2, 128 / 2, size=2).astype(np.double), 1e-3, 1-1e-3)
        inverse_beta = torch.from_numpy(beta / (1-beta+1e-8)).double()
        radius = sigma.flatten().double() * np.sqrt(128).astype(np.float64) * torch.sqrt(inverse_beta + 1e-8)
        direction = torch.randn(2, 3072)
        direction /= direction.norm(dim=1, keepdim=True)
        noisy = images + (direction * radius[:, None]).float().view_as(images)
        expected = (sigma.square() + 0.5 ** 2) / (sigma * 0.5).square() * (noisy * 0.5 - images).square()
        torch.manual_seed(19)
        np.random.seed(19)
        actual = EDMInterpolantLoss(pfgmpp=True)(lambda x, sigma, labels, **kw: x * 0.5, images)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
