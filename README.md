# Cat Image Generation with Generative Models

This project trains and compares generative image models for creating cat images using PyTorch. Planned approaches include VAE, GAN/DCGAN, and diffusion models, with a stronger focus on diffusion and latent-space generation.

---

## 1. Clone the Repository

```bash
git clone https://github.com/Valentine-456/deep_learning_image_generation
cd deep_learning_image_generation
```

---

## 2. Create Python Virtual Environment

Python 3.10 is recommended.

### Mac / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install Dependencies

If your machine has CUDA GPU run this:

```bash
pip install -r requirements-cuda.txt
```

Otherwise run this:

```bash
pip install -r requirements-cpu.txt
```

If you want to automatically lint and format code before commits, run this:

```bash
pre-commit install
```

---

## 4. Download Image Dataset

For the cats-and-dogs extension, download the dataset from:

[https://www.kaggle.com/competitions/dogs-vs-cats/](https://www.kaggle.com/competitions/dogs-vs-cats/)

IMPORTANT:

* Do NOT commit datasets to GitHub.
* The `data/` folder is ignored by git.
* Images have different resolutions, so preprocessing should resize or crop them to a fixed size such as `64x64` or `128x128`.

---

## 5. Smoke Test Dataset Loading

After placing the dataset correctly, run:

```bash
python -m src.train --smoke_test
```

Expected output example:

```text
Using device: cuda
Batch image shape: torch.Size([64, 3, 128, 128])
Batch labels shape: torch.Size([64])
```

If you see similar output, the dataset pipeline is working correctly.

## 6. Train a Baseline Model

To run a real training experiment, use the relevant training entrypoint. Example baseline commands:

```bash
python -m src.train_vae --epochs 1
python -m src.train_gan --epochs 1
python -m src.train_diffusion --epochs 1
```

You can also use the shared YAML config for a reproducible baseline:

```bash
python -m src.train_vae --config configs/vae.yaml
python -m src.train_gan --config configs/gan.yaml
python -m src.train_diffusion --config configs/diffusion.yaml
```

Useful flags for the report experiments:

* `--image_size 64|128`
* `--batch_size 32|64`
* `--latent_dim 64|128|256`
* `--timesteps 100|500|1000` for diffusion
* `--dataset cats|cats_dogs`
* `--seed 42` for reproducibility

---

## Project Structure

```text
src/        - source code
configs/    - experiment configuration files
scripts/    - helper scripts
results/    - experiment summary tables and metrics
data/       - datasets (not tracked by git)
runs/       - training outputs, checkpoints, and generated samples (not tracked by git)
```

---

## Next Steps

After confirming dataset loading works:

1. Train a VAE baseline on resized cat images.
2. Train a DCGAN baseline and check for mode collapse.
3. Implement a diffusion model, preferably with a U-Net denoiser.
4. Extend the diffusion approach toward latent-space generation with an encoder-decoder.
5. Sweep hyperparameters such as learning rate, latent dimension, image size, and diffusion timesteps.
6. Compare generated samples using FID, visual inspection, diversity, and latent interpolation.
