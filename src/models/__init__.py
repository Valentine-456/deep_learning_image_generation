from .DCGAN import DCGAN, DCGANDiscriminator, DCGANGenerator, UpsampleConvGenerator
from .DDPM import DDPMUNet
from .DDPMscheduler import DDPMScheduler
from .diffusion import Diffusion
from .attention_blocks import AttentionDownBlock, AttentionUpBlock, SpatialAttentionBlock
from .diffusion_attention import AttentionDDPMUNet, AttentionDiffusion
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
            generator_type=model_cfg.get("generator_type", "transpose"),
        )

    if model_type in {"ddpm", "ddpm_attention"}:
        diffusion_cfg = cfg.get("diffusion", {})
        diffusion_class = AttentionDiffusion if model_type == "ddpm_attention" else Diffusion
        model_args = {
            "in_channels": model_cfg.get("in_channels", 3),
            "image_size": model_cfg.get("image_size", cfg["data"]["image_size"]),
            "base_channels": model_cfg.get("base_channels", 64),
            "channel_multipliers": model_cfg.get("channel_multipliers", [1, 2, 4, 8]),
            "time_dim": model_cfg.get("time_dim", 256),
            "timesteps": diffusion_cfg.get("timesteps", 1000),
            "beta_start": diffusion_cfg.get("beta_start", 0.0001),
            "beta_end": diffusion_cfg.get("beta_end", 0.02),
        }
        if model_type == "ddpm_attention":
            model_args["attention_heads"] = model_cfg.get("attention_heads", 4)
            model_args["attention_resolution"] = model_cfg.get("attention_resolution", 16)
        return diffusion_class(**model_args)

    raise ValueError(f"Unknown model type: {model_cfg['type']}")


__all__ = [
    "AttentionDDPMUNet",
    "AttentionDiffusion",
    "AttentionDownBlock",
    "AttentionUpBlock",
    "DCGAN",
    "DCGANDiscriminator",
    "DCGANGenerator",
    "DDPMScheduler",
    "DDPMUNet",
    "Diffusion",
    "SpatialAttentionBlock",
    "UpsampleConvGenerator",
    "VAEImageGenerator",
    "build_model",
]
