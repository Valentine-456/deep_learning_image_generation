# Teammate Runbook

**Read this first.** This file is the practical guide for running training, evaluation, and the cats+dogs extension on a GPU machine. The main [README.md](README.md) covers general project setup; this runbook has the exact commands and file paths you need.

---

## 0. What is already done vs what you run

| Done in repo | You run on GPU |
|--------------|----------------|
| VAE, DCGAN, DDPM, DDPM+attention training code | Final / missing training runs if needed |
| Cats-only configs (`*_64.yaml`, etc.) | Preprocessing + training |
| Evaluation scripts (`scripts/eval_*.py`) | FID, diversity, interpolation |
| Cats+dogs fast configs (`*_cats_dogs_fast.yaml`) | Exploratory extension |
| `results/` metadata (JSON histories) | Unpack checkpoints if shared separately |

Evaluation outputs go to `evaluation/` (gitignored). Do not commit datasets or checkpoints.

---

## 1. Setup

```bash
git clone https://github.com/Valentine-456/deep_learning_image_generation.git
cd deep_learning_image_generation
git checkout development   # eval code lives here

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements-cuda.txt   # or requirements-cpu.txt
pip install lpips pytorch-fid          # evaluation extras
```

If checkpoints were shared outside git, unpack them into `results/` at the project root:

```text
results/
  20260605-160116_vae_64/best_model.pth
  20260605-165157_dcgan_64/best_model.pth
  20260614-123700_ddpm/best_model.pth
  20260614-194406_ddpm_attention/best_model.pth
  ...
```

**Checkpoint rule:** use `best_model.pth` by default. DDPM runs may also have `final_model.pth`; prefer `best_model.pth` unless you have a reason not to.

---

## 2. Data and preprocessing

Download [Dogs vs Cats](https://www.kaggle.com/competitions/dogs-vs-cats/) and place images under `data/` (e.g. `data/train/` with `cat.*.jpg` and `dog.*.jpg`).

Edit `scripts/preprocess.py` if needed:

```python
INPUT_DIR = Path("data")
OUTPUT_DIR = Path("preprocessed64")   # use preprocessed64 for 64x64 runs
IMAGE_SIZE = 64
```

```bash
python scripts/preprocess.py
```

Cats-only configs filter to `cat.*` files via `only_cats: true`. Cats+dogs configs use both classes (`only_cats: false`).

---

## 3. Training

Entrypoint (pass a config path as the only argument):

```bash
python scripts/train.py configs/<config>.yaml
```

### Main report models (cats only, 64×64)

| Model | Config | Notes |
|-------|--------|-------|
| VAE | `configs/VAE_64.yaml` | Early stopping |
| DCGAN | `configs/DCGAN_64.yaml` | Check mode collapse |
| DDPM | `configs/DDPM.yaml` | Slower; 1000 timesteps |
| DDPM + attention | `configs/DDPM_attention.yaml` | Extension variant |

```bash
python scripts/train.py configs/VAE_64.yaml
python scripts/train.py configs/DCGAN_64.yaml
python scripts/train.py configs/DDPM.yaml
python scripts/train.py configs/DDPM_attention.yaml
```

Outputs per run:

```text
results/<timestamp>_<config_name>/
  best_model.pth
  results.json
  samples/          # qualitative grids during training
```

128×128 variants: `configs/VAE.yaml`, `configs/DCGAN.yaml`, `configs/DCGAN_upsample.yaml`.

### Exploratory extension (cats + dogs, fast)

Use after preprocessing the **full** train set into `preprocessed64/`:

```bash
python scripts/train.py configs/VAE_64_cats_dogs_fast.yaml
python scripts/train.py configs/DCGAN_64_cats_dogs_fast.yaml
python scripts/train.py configs/DDPM_64_cats_dogs_fast.yaml
```

These are shortened runs (fewer epochs / smaller models / 200 DDPM steps). Start with VAE or DCGAN for quickest results. Compare sample grids against the cats-only runs.

---

## 4. Evaluation (required for the report)

Install once: `pip install lpips pytorch-fid`

Use the **same `--num-samples`** for real images and every model.

### 4a. FID

```bash
# Export real test images once
python scripts/eval_fid.py \
  --export-real \
  --config configs/VAE_64.yaml \
  --num-samples 5000

# Generate per model (repeat for each checkpoint)
python scripts/eval_fid.py \
  --config configs/VAE_64.yaml \
  --checkpoint results/20260605-160116_vae_64/best_model.pth \
  --model-name vae \
  --num-samples 5000 \
  --seed 42

python scripts/eval_fid.py \
  --config configs/DCGAN_64.yaml \
  --checkpoint results/20260605-165157_dcgan_64/best_model.pth \
  --model-name dcgan \
  --num-samples 5000 \
  --seed 42

python scripts/eval_fid.py \
  --config configs/DDPM.yaml \
  --checkpoint results/<ddpm_run>/best_model.pth \
  --model-name ddpm \
  --num-samples 5000 \
  --seed 42

python scripts/eval_fid.py \
  --config configs/DDPM_attention.yaml \
  --checkpoint results/<ddpm_attention_run>/best_model.pth \
  --model-name ddpm_attention \
  --num-samples 5000 \
  --seed 42

# Compute FID
python -m pytorch_fid evaluation/real evaluation/vae
python -m pytorch_fid evaluation/real evaluation/dcgan
python -m pytorch_fid evaluation/real evaluation/ddpm
python -m pytorch_fid evaluation/real evaluation/ddpm_attention
```

Shortcut: `--run-dir results/<run_name>` loads config from `results.json` and `best_model.pth` automatically.

Outputs:

```text
evaluation/
  real/
  vae/  dcgan/  ddpm/  ddpm_attention/
  fid_summary.json
```

### 4b. Mode collapse / diversity

Best run on DCGAN; reusable for other models with `.sample()`.

```bash
python scripts/eval_diversity.py \
  --config configs/DCGAN_64.yaml \
  --checkpoint results/20260605-165157_dcgan_64/best_model.pth \
  --model-name dcgan \
  --num-samples 64 \
  --num-lpips-pairs 500 \
  --seed 42
```

Outputs: `evaluation/diversity/dcgan/grid_64.jpg`, `nearest_neighbors.jpg`, `lpips_diversity.json`, `summary.json`.

### 4c. Latent interpolation (10 images: 2 endpoints + 8 between)

Works for **DCGAN** and **VAE**. DDPM is not implemented (no DDIM sampler).

```bash
python scripts/eval_interpolate.py \
  --config configs/DCGAN_64.yaml \
  --checkpoint results/20260605-165157_dcgan_64/best_model.pth \
  --model-name dcgan \
  --seed 42

python scripts/eval_interpolate.py \
  --config configs/VAE_64.yaml \
  --checkpoint results/20260605-160116_vae_64/best_model.pth \
  --model-name vae \
  --seed 42
```

Outputs: `evaluation/interpolation/<model>/interpolation.jpg`, `z1.pt`, `z2.pt`.

---

## 5. What to put in the report

| Requirement | Source |
|-------------|--------|
| FID table (quantitative) | `python -m pytorch_fid` scores across `evaluation/*/` |
| Sample grids (qualitative) | `evaluation/*/sample_grid.jpg` and `results/*/samples/` |
| Hyperparameter discussion | Compare runs in `results/*/results.json` (64 vs 128, upsample, attention, etc.) |
| Mode collapse | `evaluation/diversity/` NN grids + LPIPS JSON |
| Latent interpolation | `evaluation/interpolation/` figure + short discussion |
| Limited compute choices | 64×64, reduced epochs in fast configs, early stopping (VAE) |
| Cats vs cats+dogs | Compare cats-only `*_64.yaml` samples vs `*_cats_dogs_fast.yaml` |

---

## 6. Quick reference

```text
scripts/train.py              Training entrypoint
scripts/preprocess.py         Resize/crop to fixed size
scripts/eval_fid.py           FID folder preparation
scripts/eval_diversity.py     Diversity / mode-collapse metrics
scripts/eval_interpolate.py   Latent interpolation (DCGAN, VAE)
src/evaluation/               Shared eval helpers
configs/                      All experiment YAML files
results/                      Checkpoints + training logs
evaluation/                   Generated eval outputs (local only)
```

### Help flags

```bash
python scripts/eval_fid.py --help
python scripts/eval_diversity.py --help
python scripts/eval_interpolate.py --help
```

### Known limitations

- DDPM latent interpolation: not implemented (needs DDIM or fixed-noise sampling).
- README.md has some outdated module names (`src.train_vae`, etc.); use `scripts/train.py` instead.
- `evaluation/` and `results/` are gitignored; share checkpoints and eval outputs via drive/archive if needed.

---

## 7. Suggested order of work

1. Preprocess data → confirm `preprocessed64/train/` exists.
2. Confirm checkpoints in `results/` (train missing ones if needed).
3. Run FID pipeline for all four main models.
4. Run diversity on DCGAN.
5. Run interpolation on DCGAN (and VAE if time allows).
6. Run one cats+dogs fast model (DCGAN recommended).
7. Collect figures + FID numbers into the report.
