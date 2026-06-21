# NVAE from Scratch

PyTorch implementation of [NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) (Vahdat & Kautz, NeurIPS 2020), built from first principles. Trained and evaluated on MNIST, CIFAR-10, and CelebA-64.

## Setup

```bash
# install uv if needed: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync   # installs all dependencies including torch==2.5.1+cu121
```

**Requirements:** NVIDIA GPU with CUDA 12.1+. Tested on A100 80GB × 2.

## Datasets

Default data path: `dataset/` (project root). Override with `--data_path /your/path`.

**MNIST / CIFAR-10** — downloaded automatically into `dataset/` on first run.

**CelebA-64** — requires manual download from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset). Download into `dataset/`:

```bash
# requires ~/.kaggle/kaggle.json
kaggle datasets download -d jessicali9530/celeba-dataset -p dataset --unzip
```

Expected layout after unzip:

```
dataset/
  img_align_celeba/
    img_align_celeba/
      000001.jpg
      ...
  list_eval_partition.csv
```

## Project Structure

```
train.py                  training loop (AdaMax, KL annealing, AMP, DDP)
evaluation.py             IW-NLL evaluation (importance-weighted, K=200)
configs/                  one YAML per experiment
figures/                  scripts to generate visualizations
  sample.py               unconditional samples from a checkpoint
  reconstruction.py       original vs. reconstruction side-by-side
  latent_interpolation.py spherical interpolation between two images
  latent_space_exploration.py  per-group latent perturbation grid
  cherry.py               interactive cherry-picking of samples
src/
  model.py                AutoEncoder (hierarchical encoder/decoder tower)
  modules/
    architecture.py       ResidualCell{Encoder,Decoder}, Combiner, SE, NF cells
    distributions.py      Normal (residual KL), DiscMixLogistic
  utils.py                KL balancer, free bits, LR schedule, BN calibration
logs/                     training logs for all experiments
```

## Training

All commands run from the project root. Use `--amp` for bfloat16 (recommended on A100).

```bash
# MNIST
uv run python train.py --config configs/mnist.yaml --dataset mnist --run_name mnist

# CIFAR-10 1×30 (paper architecture, 400 epochs)
uv run python train.py --config configs/cifar10.yaml --dataset cifar10 --run_name cifar10 --amp

# CIFAR-10 1×30 (600 epochs)
uv run python train.py --config configs/cifar10_600ep.yaml --dataset cifar10 --run_name cifar10_600ep --amp

# CIFAR-10 3×8 multi-scale (600 epochs)
uv run python train.py --config configs/cifar10_8x3.yaml --dataset cifar10 --run_name cifar10_8x3 --amp

# CIFAR-10 3×8 + Free Bits
uv run python train.py --config configs/cifar10_8x3_freebits.yaml --dataset cifar10 --run_name cifar10_8x3_freebits --amp

# CIFAR-10 3×8 + Auxiliary Reconstruction
uv run python train.py --config configs/cifar10_8x3_aux.yaml --dataset cifar10 --run_name cifar10_8x3_aux --amp

# CelebA-64 baseline
torchrun --nproc_per_node=2 train.py --config configs/celeba64.yaml --dataset celeba64 --run_name celeba64 --amp

# CelebA-64 + Free Bits
torchrun --nproc_per_node=2 train.py --config configs/celeba64_freebits.yaml --dataset celeba64 --run_name celeba64_freebits --amp

# CelebA-64 + Auxiliary Reconstruction
torchrun --nproc_per_node=2 train.py --config configs/celeba64_aux.yaml --dataset celeba64 --run_name celeba64_aux --amp

# CelebA-64 + Progressive Scale Training
torchrun --nproc_per_node=2 train.py --config configs/celeba64_progressive.yaml --dataset celeba64 --run_name celeba64_progressive --amp
```

**Arguments:**

| Argument | Description |
|---|---|
| `--config` | YAML config file path |
| `--dataset` | `cifar10` / `mnist` / `celeba64` |
| `--run_name` | checkpoint name; saved to `runs/<run_name>_last.pt` |
| `--amp` | bfloat16 mixed precision (recommended on A100) |
| `--resume_from` | resume training from a checkpoint path |
| `--batch_size` | override batch size from config |
| `--accum_steps` | gradient accumulation steps (effective bs = bs × GPUs × accum_steps) |
| `--nf_cells` | normalizing flow blocks per latent group (0 = disabled) |
| `--base_lr` | override learning rate from config |
| `--data_path` | dataset directory (default: `dataset/`) |
| `--nproc_per_node` | number of GPUs for multi-GPU training via `torchrun` |

## Evaluation

IW-NLL with K=200 importance samples (same as reported results):

```bash
uv run python evaluation.py --ckpt runs/cifar10_last.pt --K 200
```

## Visualization

All figure scripts run from the project root.

```bash
# Unconditional samples (saves grid to samples/)
uv run python figures/sample.py --ckpt runs/celeba64_last.pt --config configs/celeba64.yaml --dataset celeba64 --t 0.7 --n 64

# Original vs. reconstruction side-by-side
uv run python figures/reconstruction.py --ckpt runs/cifar10_last.pt --config configs/cifar10.yaml --dataset cifar10

# Latent interpolation between two images
uv run python figures/latent_interpolation.py --ckpt runs/celeba64_last.pt --config configs/celeba64.yaml img1.png img2.png --steps 6

# Per-group latent space exploration (what each latent group controls)
uv run python figures/latent_space_exploration.py --ckpt runs/cifar10_8x3_last.pt --config configs/cifar10_8x3.yaml

# Interactive cherry-picking of generated samples
uv run python figures/cherry.py --ckpt runs/celeba64_last.pt --config configs/celeba64.yaml --t 0.7 --winners 9
```

## Results

IW-NLL evaluated with 200 importance samples. MNIST reported in nats; others in bpd (lower is better).

| Dataset | Model | Paper | Ours |
|---|---|---|---|
| MNIST 28×28 | 2-scale, adaptive | 78.01 | 78.57 |
| CIFAR-10 32×32 | 1×30, 400ep | 2.93 | 3.51 |
| CIFAR-10 32×32 | 1×30, 600ep | — | 3.41 |
| CIFAR-10 32×32 | 3×8, 600ep | — | 3.29 |
| CelebA-64 | 3-scale, adaptive | 2.04 | 2.63 |
| CelebA-64 | + Progressive Training | — | **2.57** |

### Addressing Coarse-Scale Posterior Collapse

Multi-scale NVAE suffers from posterior collapse at coarse latent groups — fine-resolution groups explain the data on their own, leaving coarse groups unused. Three approaches were evaluated:

| Method | CIFAR-10 coarse KL | CelebA coarse KL | CelebA bpd |
|---|---|---|---|
| Baseline (3×8) | → 0 (collapsed) | → 0 (collapsed) | 2.63 |
| + Free Bits | persistent collapse | persistent collapse | — |
| + Auxiliary Reconstruction | groups 5–7 active | coarse + middle active | 2.63 |
| + Progressive Training | — | coarse + middle active | **2.57** |

Progressive Scale Training showed the strongest result: unlocking scales coarse→fine forces coarse groups to establish useful representations before finer scales are introduced.
