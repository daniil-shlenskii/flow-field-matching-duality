def get_dataset(ds_name, aux_dim=None):
    if ds_name == "swiss":
        def make_swiss_dataset(num_samples):
            X0, _ = make_swiss_roll(num_samples // 2, noise=0.3, random_state=0)
            X1, _ = make_swiss_roll(num_samples // 2, noise=0.3, random_state=0)
            X0 = X0[:, [0, 2]]
            X1 = X1[:, [0, 2]]
            X1 = -X1
            X, y = shuffle(
                np.concatenate([X0, X1], axis=0),
                np.concatenate([np.zeros(len(X0)), np.ones(len(X1))], axis=0),
                random_state=0)
            X = (X - X.mean(axis=0)) / X.std(axis=0)
        
            return X.astype(np.float32), y
        X, Y = make_swiss_dataset(5000)
        
    elif ds_name == "gaussian":
        sigma = .1
        ds_size = 10_000
        
        X = np.random.randn(ds_size, 2) * sigma
        Y = None

    elif ds_name == "gaussians":
        factor, sigma = 1., 0.2
        mu1, sigma1 = factor * np.array([-1, -1]), sigma
        mu2, sigma2 = factor * np.array([1, 1]), sigma
        ds_size = 5_000
        
        X1 = np.random.randn(ds_size//2, len(mu1)) * sigma1 + mu1
        Y1 = np.ones(len(X1)) * 0.
        X2 = np.random.randn(ds_size//2, len(mu2)) * sigma2 + mu2
        Y2 = np.ones(len(X2)) * 1.
        
        X = np.concatenate([X1, X2]).astype(np.float32)
        Y = np.concatenate([Y1, Y2])

    elif ds_name == "gaussians2":
        factor, sigma = 1., 0.2
        mu1, sigma1 = factor * np.array([1, -1]), sigma
        mu2, sigma2 = factor * np.array([-1, 1]), sigma
        ds_size = 5_000
        
        X1 = np.random.randn(ds_size//2, len(mu1)) * sigma1 + mu1
        Y1 = np.ones(len(X1)) * 0.
        X2 = np.random.randn(ds_size//2, len(mu2)) * sigma2 + mu2
        Y2 = np.ones(len(X2)) * 1.
        
        X = np.concatenate([X1, X2]).astype(np.float32)
        Y = np.concatenate([Y1, Y2])

    elif ds_name == "circle":
        R = 1.
        sigma = 0.05
        ds_size = 5000
        angle = np.random.rand(ds_size) * 2 * np.pi
        noise = np.random.randn(ds_size, 2) * sigma
        X = np.stack([np.sin(angle), np.cos(angle)], axis=1) + noise
        Y = None
    elif ds_name == "pfgmpp":
        eps = 1e-6
        ds_size = 10000
        sigma = 1. * torch.ones(ds_size) * aux_dim**0.5

        # Sample from inverse-beta distribution
        beta_gen = Beta(torch.FloatTensor([2. / 2.]), torch.FloatTensor([aux_dim / 2.]))
        beta_sample = beta_gen.sample((sigma.shape[0],)).reshape(-1)
        inverse_beta_sample = beta_sample / (1 - beta_sample).clamp(min=eps)

        # Sample from p(R)
        R = sigma * torch.sqrt(inverse_beta_sample.clamp(min=eps))

        # Sample angle
        gaussian = torch.randn(sigma.shape[0], 2, device=sigma.device, dtype=sigma.dtype)
        angle = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)

        X = np.asarray(R.view(-1, 1) * angle)
        Y = None
    elif ds_name == "stable":
        alpha = 1.7    # stability parameter
        beta = 0    # skewness parameter
        loc = 0        # location parameter
        scale = 0.05      # scale parameter
        
        # Generate samples
        X = np.stack([
            levy_stable.rvs(alpha, beta, loc=loc, scale=scale, size=10000),
            levy_stable.rvs(alpha, beta, loc=loc, scale=scale, size=10000),
        ], axis=-1)
        Y = None

    return X.astype(np.float32), Y