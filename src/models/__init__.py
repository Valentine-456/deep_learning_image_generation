from .DCGAN import DCGAN, DCGANDiscriminator, DCGANGenerator
from .VAE import VAEImageGenerator


def build_model(cfg: dict):
    model_cfg = cfg["model"]
    model_type = model_cfg["type"].lower()

    if model_type == "vae":
        return VAEImageGenerator(
            in_channels=model_cfg.get("in_channels", 3),
            latent_dim=model_cfg.get("latent_dim", 128),
            image_size=model_cfg.get("image_size", cfg["data"]["image_size"]),
            hidden_dims=model_cfg.get("hidden_dims", [32, 64, 128, 256]),
        )

    if model_type == "dcgan":
        return DCGAN(
            in_channels=model_cfg.get("in_channels", 3),
            latent_dim=model_cfg.get("latent_dim", 128),
            image_size=model_cfg.get("image_size", cfg["data"]["image_size"]),
            generator_features=model_cfg.get("generator_features", 64),
            discriminator_features=model_cfg.get("discriminator_features", 64),
        )

    raise ValueError(f"Unknown model type: {model_cfg['type']}")


__all__ = [
    "DCGAN",
    "DCGANDiscriminator",
    "DCGANGenerator",
    "VAEImageGenerator",
    "build_model",
]
