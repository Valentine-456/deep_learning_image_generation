from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import ImageGenerationDataset, split_image_paths
from src.models import build_model
from src.trainers import DCGANTrainer, VAETrainer
from src.utils import save_json, seed_everything


DEFAULT_CONFIG_PATH = Path("configs/VAE.yaml")


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH

    with config_path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    data_cfg = cfg["data"]
    training_cfg = cfg["training"]

    seed_everything(training_cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"config={config_path}  device={device}")

    data_root = Path(data_cfg["cache_dir"])
    train_paths, val_paths, test_paths = split_image_paths(
        data_root,
        seed=training_cfg.get("seed", 42),
        train_fraction=data_cfg.get("train_fraction", 0.8),
        val_fraction=data_cfg.get("val_fraction", 0.1),
        test_fraction=data_cfg.get("test_fraction"),
        only_cats=data_cfg.get("only_cats", False),
    )

    image_size = data_cfg["image_size"]
    train_ds = ImageGenerationDataset(
        train_paths,
        image_size=image_size,
        augment=data_cfg.get("augmentation", False),
    )
    val_ds = ImageGenerationDataset(val_paths, image_size=image_size)
    test_ds = ImageGenerationDataset(test_paths, image_size=image_size)

    loader_kwargs = {
        "batch_size": training_cfg["batch_size"],
        "num_workers": training_cfg["workers"],
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"params={n_params:,}  "
        f"train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}"
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}_{config_path.stem.lower()}"
    out_dir = Path(training_cfg["out_dir"]) / run_name
    sample_dir = out_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / "best_model.pth"

    if cfg["model"]["type"].lower() == "dcgan":
        train_dcgan(model, train_loader, training_cfg, device, out_dir, sample_dir, cfg)
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        factor=0.5,
    )

    fixed_images = next(iter(val_loader))
    fixed_images = fixed_images[: training_cfg.get("num_sample_images", 16)]

    best_val_loss = float("inf")
    patience = training_cfg.get("early_stopping_patience", 7)
    no_improve = 0
    history: list[dict[str, float | int]] = []
    kl_weight = training_cfg.get("kl_weight", 1.0)
    sample_every = training_cfg.get("sample_every", 5)

    for epoch in range(1, training_cfg["epochs"] + 1):
        t0 = time.time()
        train_metrics = VAETrainer.run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            kl_weight=kl_weight,
        )
        val_metrics = VAETrainer.run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            train=False,
            kl_weight=kl_weight,
        )
        scheduler.step(val_metrics["loss"])

        print(
            f"epoch {epoch:3d}/{training_cfg['epochs']}  "
            f"train loss={train_metrics['loss']:.4f} "
            f"recon={train_metrics['reconstruction_loss']:.4f} "
            f"kl={train_metrics['kl_loss']:.4f}  "
            f"val loss={val_metrics['loss']:.4f} "
            f"recon={val_metrics['reconstruction_loss']:.4f} "
            f"kl={val_metrics['kl_loss']:.4f}  "
            f"{time.time() - t0:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_reconstruction_loss": train_metrics["reconstruction_loss"],
                "train_kl_loss": train_metrics["kl_loss"],
                "valid_loss": val_metrics["loss"],
                "valid_reconstruction_loss": val_metrics["reconstruction_loss"],
                "valid_kl_loss": val_metrics["kl_loss"],
            }
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            no_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            no_improve += 1

        if epoch == 1 or epoch % sample_every == 0:
            VAETrainer.save_samples(model, fixed_images, sample_dir, epoch, device)

        if no_improve >= patience:
            print(f"Early stopping: val loss did not improve for {patience} epochs.")
            break

    print("\nEvaluating on test set")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_metrics = VAETrainer.run_epoch(
        model,
        test_loader,
        optimizer=None,
        device=device,
        train=False,
        kl_weight=kl_weight,
    )
    print(
        f"test loss={test_metrics['loss']:.4f} "
        f"recon={test_metrics['reconstruction_loss']:.4f} "
        f"kl={test_metrics['kl_loss']:.4f}"
    )

    save_json(
        out_dir / "results.json",
        {
            "config": cfg,
            "device": str(device),
            "run_name": run_name,
            "best_val_loss": best_val_loss,
            "history": history,
            "test_metrics": test_metrics,
        },
    )
    print(f"\nSaved => {out_dir}/")


def train_dcgan(
    model: nn.Module,
    train_loader: DataLoader,
    training_cfg: dict,
    device: torch.device,
    out_dir: Path,
    sample_dir: Path,
    cfg: dict,
) -> None:
    generator_optimizer = torch.optim.Adam(
        model.generator.parameters(),
        lr=training_cfg["generator_lr"],
        betas=(training_cfg["beta1"], training_cfg["beta2"]),
    )
    discriminator_optimizer = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=training_cfg["discriminator_lr"],
        betas=(training_cfg["beta1"], training_cfg["beta2"]),
    )
    criterion = nn.BCEWithLogitsLoss()
    fixed_noise = torch.randn(
        training_cfg.get("num_sample_images", 16),
        model.generator.latent_dim,
        1,
        1,
        device=device,
    )

    history = []
    sample_every = training_cfg.get("sample_every", 5)

    for epoch in range(1, training_cfg["epochs"] + 1):
        t0 = time.time()
        metrics = DCGANTrainer.run_epoch(
            model,
            train_loader,
            generator_optimizer,
            discriminator_optimizer,
            criterion,
            device,
        )
        print(
            f"epoch {epoch:3d}/{training_cfg['epochs']}  "
            f"g_loss={metrics['generator_loss']:.4f}  "
            f"d_loss={metrics['discriminator_loss']:.4f}  "
            f"real={metrics['real_score']:.4f}  "
            f"fake={metrics['fake_score']:.4f}  "
            f"{time.time() - t0:.1f}s"
        )
        history.append({"epoch": epoch, **metrics})

        if epoch == 1 or epoch % sample_every == 0:
            DCGANTrainer.save_samples(model, fixed_noise, sample_dir, epoch, device)

    torch.save(
        {
            "generator": model.generator.state_dict(),
            "discriminator": model.discriminator.state_dict(),
        },
        out_dir / "best_model.pth",
    )
    save_json(
        out_dir / "results.json",
        {
            "config": cfg,
            "device": str(device),
            "history": history,
        },
    )
    print(f"\nSaved => {out_dir}/")


if __name__ == "__main__":
    main()
