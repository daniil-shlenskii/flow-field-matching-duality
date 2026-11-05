"""EDM weight concentration (Appendix E.3)."""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from training.dataset import ImageFolderDataset


@torch.no_grad()
def concentration(pool, query_count, generator):
    sigma = torch.exp(torch.randn(query_count, generator=generator) * 1.2 - 1.2)
    noise = torch.randn(pool[:query_count].shape, generator=generator)
    sigma = sigma.to(device=pool.device, dtype=pool.dtype)
    query = pool[:query_count] + noise.to(pool) * sigma[:, None]
    distance_sq = torch.cdist(query, pool).square()
    weights = torch.softmax(-distance_sq / (2 * sigma[:, None].square()), dim=1)
    return (1 - weights.square().sum(dim=1)).mean().item()


def resize_pool(images, resolution, device):
    resized = []
    for image in images:
        image = image.transpose(1, 2, 0)
        if image.shape[-1] == 1:
            image = image[..., 0]
        image = Image.fromarray(image).resize((resolution, resolution), Image.BILINEAR)
        resized.append(np.asarray(image).reshape(-1))
    return torch.as_tensor(np.stack(resized), dtype=torch.float64, device=device) / 127.5 - 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Training images as a ZIP or directory')
    parser.add_argument('--out', default='weight_concentration.csv')
    parser.add_argument('--pool-sizes', type=int, nargs='+', default=[128, 256, 512, 1024, 2048])
    parser.add_argument('--resolutions', type=int, nargs='+', default=[8, 16, 32, 64])
    parser.add_argument('--queries', type=int, default=64, help='Query images per pool and repeat')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu', help='cpu or cuda')
    args = parser.parse_args()
    if min(args.pool_sizes + args.resolutions + [args.queries, args.repeats]) < 1:
        parser.error('Pool sizes, resolutions, queries, and repeats must be positive')
    if args.queries > min(args.pool_sizes):
        parser.error('--queries must not exceed the smallest pool size')

    dataset = ImageFolderDataset(path=args.data)
    if max(args.pool_sizes) > len(dataset):
        parser.error('The largest pool must not exceed the dataset size')
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    results = {(n, r): [] for n in args.pool_sizes for r in args.resolutions}
    try:
        for n in args.pool_sizes:
            for _ in range(args.repeats):
                indices = rng.choice(len(dataset), size=n, replace=False)
                images = [dataset[int(index)][0] for index in indices]
                for resolution in args.resolutions:
                    pool = resize_pool(images, resolution, args.device)
                    results[n, resolution].append(concentration(pool, args.queries, generator))
    finally:
        dataset.close()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pool_size', 'resolution', 'gini_mean', 'gini_std',
                         'queries', 'repeats', 'seed', 'resize'])
        for (n, resolution), values in results.items():
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if args.repeats > 1 else 0.0
            writer.writerow([n, resolution, mean, std, args.queries, args.repeats,
                             args.seed, 'bilinear'])
            print(f'N={n:4d} resolution={resolution:2d}: G={mean:.6f} +/- {std:.6f}')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
