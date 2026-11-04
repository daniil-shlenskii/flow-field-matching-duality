# Unlocking the Duality between Flow and Field Matching

## Install requirements:

With conda:
```bash
conda create -n env python=3.8
conda activate env
pip install -r requirements.txt
```

## Toy Experiments:

Experiments on the Toy Dataset with Gaussian Mixtures can be found in:
  - [ToyExperimentsDuality.ipynb](./ToyExperimentsDuality.ipynb) - Experiments, that show empirical confirmation of the Duality
  - [ToyExperimentsVolume.ipynb](./ToyExperimentsVolume.ipynb) - Experiments with different p_vol

## Download data:

**CIFAR-10:** Download the [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar.html) and convert to ZIP archive:

```bash
python dataset_tool.py --source=downloads/cifar10/cifar-10-python.tar.gz \
    --dest=datasets/cifar10-32x32.zip
python fid.py ref --data=datasets/cifar10-32x32.zip --dest=fid-refs/cifar10-32x32.npz
```

**FFHQ-64**: Download the [Flickr-Faces-HQ](https://github.com/NVlabs/ffhq-dataset) dataset as 1024x1024 images and convert to ZIP archive at 64x64 resolution: 
```bash
python dataset_tool.py --source=downloads/ffhq/images1024x1024 \
    --dest=datasets/ffhq-64x64.zip --resolution=64x64
python fid.py ref --data=datasets/ffhq-64x64.zip --dest=fid-refs/ffhq-64x64.npz
```

## CIFAR / FFHQ-64 Training:

- Code to run train on FFHQ is the same as for CIFAR, for it just change: `--data=datasets/ffhq-64x64.zip --lr=2e-4 --dropout=0.05 --augment=0.15`

### Two-Sided Interpolant with $s=0.1$:

- N=1:
- B=256

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    train.py --outdir=training-runs --data=datasets/cifar10-32x32.zip --precond=fm --arch=ddpmpp \
    --snap 50 \
    --batch 256 \
    --flow-matching-scale 0.1 \
    --duration 110

```

- N=2048:
- B=256:

```bash
nproc=2
N=$((2048 * nproc))
B=256
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    train.py --outdir=training-runs --data=datasets/cifar10-32x32.zip --precond=multi-fm --arch=ddpmpp \
    --snap 50 \
    --batch $N \
    --flow-matching-scale 0.1 \
    --duration 110 \
    --multi-target-batch-size-for-loss $B

```

### EDM:

- **N=1**:
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
    --outdir=training-runs-edm-interp-x0 \
    --data=datasets/cifar10-32x32.zip \
    --precond=edm-interpolant \
    --batch=256 \
    --tick 10 \
```

- N=2048:
```bash
nproc=2
N=$((2048 * nproc))
B=256
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=2 train.py \
    --outdir=training-runs-edm-interp-x0 \
    --cond=0 \
    --data=datasets/cifar10-32x32.zip \
    --precond=edm-multi-interpolant \
    --batch=$B \
    --multi-target-batch-size-for-loss $B \
    --tick 10
```

### PFGM++

- **N=1**:
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
    --outdir=training-runs-pfgmpp-interp-x0 \
    --data=datasets/cifar10-32x32.zip \
    --precond=edm-interpolant \
    --batch=256 \
    --tick 10 \
    --pfgmpp=1 \
    --pfgmpp_aug_dim 2048
```

- N=2048:
```bash
nproc=2
N=$((2048 * nproc))
B=256
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
    --outdir=training-runs-pfgmpp-interp-x0 \
    --data=datasets/cifar10-32x32.zip \
    --precond=edm-multi-interpolant \
    --batch=$N \
    --tick 10 \
    --multi-target-batch-size-for-loss $B \
    --pfgmpp=1 \
    --pfgmpp_aug_dim 128
```

## Evaluate FID:

- Compute the reference statistics for your dataset as follows:

```.bash
python fid.py ref --data=datasets/my-dataset.zip --dest=fid-refs/my-dataset.npz
```

- use your `/path/to/train_dir` with .pkl files created during training to compute FID for

```bash
dirs=(
    /path/to/train_dir
)

for dir in "${dirs[@]}"; do
    # Extract the last part of the directory path
    dir_name=$(basename "$dir")
    csv_name="results/fid_${dir_name}.csv"

    outdir="fid-out-gens/${dir_name}"
    
    # Determine sampler based on directory name
    if [[ "$dir" == *"edm"* ]]; then
        sampler="edm"
    else
        sampler="fm"
    fi

    echo "Running FID for $dir with sampler $sampler and outdir $outdir"
    
    bash scripts/many_fid.sh $dir fid-refs/cifar10-32x32.npz "$csv_name" 50000 0 5000 $outdir $sampler
done
```