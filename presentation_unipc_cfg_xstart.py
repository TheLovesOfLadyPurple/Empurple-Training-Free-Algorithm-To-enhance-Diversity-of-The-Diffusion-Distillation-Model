import argparse
import os
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast

from diffusers import StableDiffusionPipeline

from sampler import UniPCSampler


def model_closure(pipe):
    def model_fn(x, t, c):
        return pipe.unet(x, t, encoder_hidden_states=c).sample

    return model_fn


def encode_prompt(prompt_batch, text_encoder, tokenizer):
    with torch.no_grad():
        text_inputs = tokenizer(
            prompt_batch,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0]
    return prompt_embeds


def feature_tensor_to_uint8_image(
    feature: torch.Tensor,
    use_fixed_range: bool = True,
    fixed_min_value: float = -1.0,
    fixed_max_value: float = 1.0,
) -> np.ndarray:
    feature = feature.detach().float().cpu()
    if feature.ndim != 3:
        raise ValueError(f"Expected feature tensor with 3 dims (C,H,W), got {feature.shape}")

    if use_fixed_range:
        min_value = fixed_min_value
        max_value = fixed_max_value
    else:
        min_value = float(feature.min().item())
        max_value = float(feature.max().item())

    if max_value <= min_value:
        max_value = min_value + 1e-6

    feature = torch.clamp(feature, min=min_value, max=max_value)
    feature = (feature - min_value) / (max_value - min_value)
    feature = (feature * 255.0).round().to(torch.uint8)

    channels = feature.shape[0]
    if channels == 1:
        feature = feature.repeat(3, 1, 1)
    elif channels == 2:
        feature = torch.cat([feature, feature[:1]], dim=0)
    elif channels > 3:
        feature = feature[:3]

    return rearrange(feature.numpy(), "c h w -> h w c")


def decode_latent_feature_to_uint8_image(feature: torch.Tensor, vae) -> np.ndarray:
    if feature.ndim != 3:
        raise ValueError(f"Expected feature tensor with 3 dims (C,H,W), got {feature.shape}")

    latent = feature.unsqueeze(0).to(device=vae.device, dtype=vae.dtype)
    decoded = vae.decode(latent / vae.config.scaling_factor).sample
    decoded = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
    image = 255.0 * rearrange(decoded[0].detach().float().cpu().numpy(), "c h w -> h w c")
    return image.round().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="./presentation_cfg_xstart_unipc", help="output root directory")
    parser.add_argument("--model_id", type=str, default="sd-legacy/stable-diffusion-v1-5", help="diffusers model id")
    parser.add_argument("--prompt", type=str, default="an astronaut riding a horse in the mars.", help="positive prompt")
    parser.add_argument("--negative_prompt", type=str, default="", help="negative prompt")
    parser.add_argument("--H", type=int, default=512, help="image height")
    parser.add_argument("--W", type=int, default=512, help="image width")
    parser.add_argument("--n_samples", type=int, default=1, help="number of samples")
    parser.add_argument("--stop_steps", type=int, default=8, help="UniPC steps")
    parser.add_argument("--scale", type=float, default=5.5, help="classifier-free guidance scale")
    parser.add_argument("--use_free_net", action="store_true", default=True, help="enable freeU in mid steps")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--precision", type=str, choices=["full", "autocast"], default="autocast", help="inference precision")

    parser.add_argument("--save_cfg_xstart_vis", action="store_true", default=True, help="save cfg_xstart latent visualization")
    parser.add_argument("--no_save_cfg_xstart_vis", dest="save_cfg_xstart_vis", action="store_false", help="disable cfg_xstart latent visualization")
    parser.add_argument("--save_cfg_xstart_decode", action="store_true", default=True, help="save cfg_xstart decoded images")
    parser.add_argument("--no_save_cfg_xstart_decode", dest="save_cfg_xstart_decode", action="store_false", help="disable cfg_xstart decoded images")

    parser.add_argument("--cfg_xstart_fixed_range", action="store_true", default=True, help="use fixed latent range for visualization")
    parser.add_argument("--cfg_xstart_auto_range", dest="cfg_xstart_fixed_range", action="store_false", help="use per-feature auto range")
    parser.add_argument("--cfg_xstart_min_value", type=float, default=-1.0, help="fixed minimum for latent visualization")
    parser.add_argument("--cfg_xstart_max_value", type=float, default=1.0, help="fixed maximum for latent visualization")

    opt = parser.parse_args()

    if opt.cfg_xstart_fixed_range and opt.cfg_xstart_max_value <= opt.cfg_xstart_min_value:
        raise ValueError("--cfg_xstart_max_value must be greater than --cfg_xstart_min_value.")

    seed_everything(opt.seed)
    torch.manual_seed(opt.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    pipe = StableDiffusionPipeline.from_pretrained('sd-legacy/stable-diffusion-v1-5')
    pipe.to(device=device, torch_dtype=dtype)

    sampler = UniPCSampler(pipe
                           , model_closure=model_closure
                           , steps=opt.stop_steps
                           , guidance_scale=opt.scale
                           , is_high_resoulution=False)

    os.makedirs(opt.outdir, exist_ok=True)
    vae_scale_factor = getattr(pipe, "vae_scale_factor", 2 ** (len(pipe.vae.config.block_out_channels) - 1))
    if opt.H % vae_scale_factor != 0 or opt.W % vae_scale_factor != 0:
        raise ValueError(
            f"Image size must be divisible by the VAE scale factor {vae_scale_factor}, got H={opt.H}, W={opt.W}."
        )
    latent_shape = (
        opt.n_samples,
        pipe.unet.config.in_channels,
        opt.H // vae_scale_factor,
        opt.W // vae_scale_factor,
    )

    run_tag = f"pure-noise-steps-{opt.stop_steps}-scale-{opt.scale}"
    cfg_xstart_vis_mode = "saturation" if opt.cfg_xstart_fixed_range else "auto-range"
    cfg_xstart_vis_path = os.path.join(opt.outdir, f"cfg_xstart_latent-{run_tag}-{cfg_xstart_vis_mode}")
    cfg_xstart_decoded_path = os.path.join(opt.outdir, f"cfg_xstart_decoded-{run_tag}")

    if opt.save_cfg_xstart_vis:
        os.makedirs(cfg_xstart_vis_path, exist_ok=True)
    if opt.save_cfg_xstart_decode:
        os.makedirs(cfg_xstart_decoded_path, exist_ok=True)

    if opt.precision == "autocast" and device == "cuda":
        precision_scope = lambda: autocast("cuda")
    else:
        precision_scope = nullcontext

    with torch.no_grad():
        with precision_scope():
            noise = torch.randn(latent_shape, device=device, dtype=dtype)

            prompts = [opt.prompt] * noise.shape[0]
            c = encode_prompt(prompts, pipe.text_encoder, pipe.tokenizer)

            neg_prompts = [opt.negative_prompt] * noise.shape[0]
            uc = encode_prompt(neg_prompts, pipe.text_encoder, pipe.tokenizer)

            grather_feature_dict = {
                "cond_noise": None,
                "uncond_noise": None,
                "cond_xstart": None,
                "uncond_xstart": None,
                "intermediate_x": None,
                "tmp_t": [],
                "cfg_xstart": [],
            }

            # sampler.sample(
            #     conditioning=c,
            #     batch_size=noise.shape[0],
            #     shape=list(noise.shape[1:]),
            #     unconditional_conditioning=uc,
            #     x_T=noise,
            #     start_free_u_step=4 if opt.use_free_net else -1,
            #     use_corrector=True,
            #     grather_feature_dict=grather_feature_dict,
            # )
            sampler.sample(
                            conditioning=c,
                            batch_size=opt.n_samples,
                            shape=latent_shape[1:],
                            unconditional_conditioning=uc,
                            x_T=noise,
                            start_free_u_step=4,
                            use_corrector=True,
                            grather_feature_dict=grather_feature_dict
                        )

            cfg_latent_count = 0
            cfg_decoded_count = 0
            for step_idx, cfg_xstart_batch in enumerate(grather_feature_dict["cfg_xstart"]):
                if step_idx >= len(grather_feature_dict["tmp_t"]):
                    continue
                time_tag = int(grather_feature_dict["tmp_t"][step_idx][0])

                for sample_idx, feature in enumerate(cfg_xstart_batch):
                    if opt.save_cfg_xstart_vis:
                        vis_image = feature_tensor_to_uint8_image(
                            feature,
                            use_fixed_range=opt.cfg_xstart_fixed_range,
                            fixed_min_value=opt.cfg_xstart_min_value,
                            fixed_max_value=opt.cfg_xstart_max_value,
                        )
                        vis_name = f"{cfg_latent_count:05}_{time_tag:05}_{sample_idx:02}.png"
                        Image.fromarray(vis_image).save(os.path.join(cfg_xstart_vis_path, vis_name))
                        cfg_latent_count += 1

                    if opt.save_cfg_xstart_decode:
                        decoded_image = decode_latent_feature_to_uint8_image(feature, pipe.vae)
                        decoded_name = f"{cfg_decoded_count:05}_{time_tag:05}_{sample_idx:02}.png"
                        Image.fromarray(decoded_image).save(os.path.join(cfg_xstart_decoded_path, decoded_name))
                        cfg_decoded_count += 1

    print(f"Done. Solver input latent shape: {latent_shape}")
    print(f"Output root: {opt.outdir}")


if __name__ == "__main__":
    main()
