import argparse
import json
import os
import random
import re
import time

import numpy as np
import torch
from PIL import Image

from diffusers import LCMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from huggingface_hub import hf_hub_download


ORIGINAL_DMD2_TIMESTEPS = [999, 749, 499, 249]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def predicted_origin(model_output, timesteps, sample, alphas, sigmas, prediction_type="epsilon"):
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")
    return pred_x_0


class LCMSDESolverXL:
    def __init__(
        self,
        inner_model,
        alpha_cumprods,
        timesteps=1000,
        ddim_timesteps=4,
        start_timestep=None,
        timesteps_override=None,
    ):
        self.sqrt_alpha_cumprods = torch.sqrt(alpha_cumprods)
        self.sigmas = torch.sqrt(1 - alpha_cumprods)
        if timesteps_override is not None:
            ddim_schedule = np.array(timesteps_override, dtype=np.int64)
        else:
            step_ratio = timesteps // ddim_timesteps if start_timestep is None else start_timestep // ddim_timesteps
            ddim_schedule = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
            ddim_schedule = np.flip(ddim_schedule).copy()
        self.ddim_timesteps = torch.from_numpy(ddim_schedule).long()
        self.inner_model = inner_model
        self.to(inner_model.device)

    def scalings_for_boundary_conditions(self, timestep, sigma_data=0.5):
        scaled_timestep = timestep * self.inner_model.scheduler.config.timestep_scaling
        c_skip = sigma_data**2 / ((scaled_timestep / 0.1) ** 2 + sigma_data**2)
        c_out = (scaled_timestep / 0.1) / ((scaled_timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
        return c_skip, c_out

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.sqrt_alpha_cumprods = self.sqrt_alpha_cumprods.to(device)
        self.sigmas = self.sigmas.to(device)
        return self

    def solve(self, sample, prompt_embeds, cond_kwargs):
        tmp_hidden_z = sample
        for index, timestep in enumerate(self.ddim_timesteps):
            sample = tmp_hidden_z
            timestep = timestep * sample.new_ones([sample.shape[0]])
            noise_pred = self.inner_model.unet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds.to(device=sample.device, dtype=sample.dtype),
                added_cond_kwargs=cond_kwargs,
            ).sample
            tmp_hidden_z = predicted_origin(
                model_output=noise_pred,
                timesteps=timestep.long(),
                sample=tmp_hidden_z,
                alphas=self.sqrt_alpha_cumprods,
                sigmas=self.sigmas,
                prediction_type=self.inner_model.scheduler.config.prediction_type,
            )
            c_skip, c_out = self.scalings_for_boundary_conditions(timestep)
            tmp_hidden_z = tmp_hidden_z * c_out + c_skip * sample
            if index == len(self.ddim_timesteps) - 1:
                break
            noise = torch.randn_like(tmp_hidden_z)
            tmp_hidden_z = self.inner_model.scheduler.add_noise(
                original_samples=tmp_hidden_z,
                noise=noise,
                timesteps=self.ddim_timesteps[index + 1].unsqueeze(0),
            )
        return tmp_hidden_z

    def schedule_as_list(self):
        return self.ddim_timesteps.detach().cpu().tolist()


def prepare_sdxl_pipeline_step_parameter(pipe, prompt, device, width=1024, height=1024):
    (
        prompt_embeds,
        _negative_prompt_embeds,
        pooled_prompt_embeds,
        _negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=[prompt],
        negative_prompt="",
        device=device,
        do_classifier_free_guidance=False,
    )

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = pooled_prompt_embeds.to(device)
    original_size = (width, height)
    crops_coords_top_left = (0, 0)
    target_size = (width, height)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    if pipe.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim
    passed_add_embed_dim = (
        pipe.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
    )
    expected_add_embed_dim = pipe.unet.add_embedding.linear_1.in_features
    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            "Model expects an added time embedding vector of length "
            f"{expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created."
        )
    add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype, device=device)
    cond_kwargs = {
        "text_embeds": add_text_embeds,
        "time_ids": add_time_ids,
    }
    return prompt_embeds, cond_kwargs


def build_dmd2_pipeline(dtype, device):
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    repo_name = "tianweiy/DMD2"
    ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"

    unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to(device, dtype)
    unet.load_state_dict(torch.load(hf_hub_download(repo_name, ckpt_name), map_location=device))

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_id,
        unet=unet,
        torch_dtype=dtype,
    ).to(device)
    pipe.scheduler = LCMScheduler.from_pretrained(
        base_model_id,
        subfolder="scheduler",
        timestep_spacing="trailing",
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def auto_resolve_latent_dir(mode):
    base = "./gen_img_train_xl_coco2014_unipc"
    if mode == "intermediate":
        return os.path.join(base, "intermediate-12-5.5")
    return os.path.join(base, "final_x0-12-5.5")


def collect_latent_files(latent_dir, latent_timestep):
    suffix = f"_{latent_timestep}.pth"
    latent_files = [
        os.path.join(latent_dir, name)
        for name in sorted(os.listdir(latent_dir))
        if name.endswith(suffix)
    ]
    if not latent_files:
        raise FileNotFoundError(
            f"No latent files ending with {suffix} were found in {latent_dir}."
        )
    return latent_files


def load_latent_tensor(latent_path, device, dtype):
    latent = torch.load(latent_path, map_location="cpu", weights_only=True)
    if latent.ndim == 3 and latent.shape[-1] == 4:
        latent = latent.permute(2, 0, 1).unsqueeze(0)
    elif latent.ndim == 3 and latent.shape[0] == 4:
        latent = latent.unsqueeze(0)
    elif latent.ndim == 4 and latent.shape[0] == 1:
        pass
    else:
        raise ValueError(f"Unsupported latent shape {tuple(latent.shape)} in {latent_path}")
    return latent.to(device=device, dtype=dtype)


def decode_to_image(pipe, latent):
    decoded = pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample
    decoded = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
    image = decoded[0].detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((255.0 * image).astype(np.uint8))


def save_image_grid(images, output_path):
    if not images:
        return
    columns = int(np.ceil(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / columns))
    width, height = images[0].size
    grid = Image.new("RGB", (columns * width, rows * height))
    for index, image in enumerate(images):
        x_offset = (index % columns) * width
        y_offset = (index // columns) * height
        grid.paste(image, (x_offset, y_offset))
    grid.save(output_path)


def sanitize_prompt(prompt):
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", prompt.strip()).strip("-").lower()
    if not slug:
        slug = "prompt"
    return slug[:80]


def prompt_from_cli_or_input(cli_prompt):
    prompt = cli_prompt.strip() if cli_prompt is not None else input("Enter prompt: ").strip()
    if not prompt:
        raise ValueError("Prompt must not be empty.")
    return prompt


def save_metadata(output_path, payload):
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate 9 DMD2 baseline images and 9 random-latent augment images for one prompt."
    )
    parser.add_argument("--prompt", type=str, default="a woman smile", help="Prompt text. If omitted, the script asks interactively.")
    parser.add_argument("--outdir", type=str, default="./gen_img_compare_dmd2_random_latent_xl")
    parser.add_argument("--latent_dir", type=str, default=None)
    parser.add_argument("--latent_timestep", type=str, default="00852")
    parser.add_argument("--mode", type=str, choices=["intermediate", "final_x0"], default="intermediate")
    parser.add_argument("--noise_timestep", type=int, default=852)
    parser.add_argument("--num_images", type=int, default=9)
    parser.add_argument("--H", type=int, default=1024)
    parser.add_argument("--W", type=int, default=1024)
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--f", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    opt = parser.parse_args()

    if opt.num_images < 1:
        raise ValueError("--num_images must be >= 1")
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA.")

    prompt = prompt_from_cli_or_input(opt.prompt)
    latent_dir = opt.latent_dir or auto_resolve_latent_dir(opt.mode)
    if not os.path.isdir(latent_dir):
        raise FileNotFoundError(f"Latent directory not found: {latent_dir}")

    seed_everything(opt.seed)

    dtype = torch.float32
    device = "cuda"

    pipe = build_dmd2_pipeline(dtype=dtype, device=device)
    alpha_schedule = pipe.scheduler.alphas_cumprod.to(device=device, dtype=dtype)
    augment_solver = LCMSDESolverXL(
        inner_model=pipe,
        alpha_cumprods=alpha_schedule,
        timesteps=pipe.scheduler.config.num_train_timesteps,
        ddim_timesteps=4,
        start_timestep=opt.noise_timestep,
    )

    prompt_embeds, cond_kwargs = prepare_sdxl_pipeline_step_parameter(
        pipe=pipe,
        prompt=prompt,
        device=device,
        width=opt.W,
        height=opt.H,
    )
    latent_files = collect_latent_files(latent_dir=latent_dir, latent_timestep=opt.latent_timestep)

    run_name = f"{time.strftime('%Y%m%d-%H%M%S')}-{sanitize_prompt(prompt)}"
    run_dir = os.path.join(opt.outdir, run_name)
    original_dir = os.path.join(run_dir, "original_dmd2")
    augment_dir = os.path.join(run_dir, "augment_random_latent")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(augment_dir, exist_ok=True)

    original_images = []
    augment_images = []
    augment_sources = []
    shape = (1, opt.C, opt.H // opt.f, opt.W // opt.f)

    with torch.inference_mode():
        for index in range(opt.num_images):
            image = pipe(
                prompt=prompt,
                num_inference_steps=4,
                guidance_scale=0,
                timesteps=ORIGINAL_DMD2_TIMESTEPS,
                output_type="pil",
            ).images[0]
            image.save(os.path.join(original_dir, f"{index:02d}.png"))
            original_images.append(image)

        for index in range(opt.num_images):
            latent_path = random.choice(latent_files)
            start_latent = load_latent_tensor(latent_path=latent_path, device=device, dtype=dtype)
            if opt.mode == "final_x0":
                noise = torch.randn_like(start_latent)
                timesteps = torch.full((1,), opt.noise_timestep, device=device, dtype=torch.long)
                start_latent = pipe.scheduler.add_noise(start_latent, noise, timesteps)
            final_latent = augment_solver.solve(
                sample=start_latent,
                prompt_embeds=prompt_embeds,
                cond_kwargs=cond_kwargs,
            )
            image = decode_to_image(pipe=pipe, latent=final_latent)
            image.save(os.path.join(augment_dir, f"{index:02d}.png"))
            augment_images.append(image)
            augment_sources.append(os.path.basename(latent_path))

    save_image_grid(original_images, os.path.join(run_dir, "original_dmd2_grid.png"))
    save_image_grid(augment_images, os.path.join(run_dir, "augment_random_latent_grid.png"))
    save_metadata(
        os.path.join(run_dir, "metadata.json"),
        {
            "prompt": prompt,
            "mode": opt.mode,
            "num_images_per_method": opt.num_images,
            "batch_size": 1,
            "seed": opt.seed,
            "latent_dir": latent_dir,
            "latent_timestep": opt.latent_timestep,
            "noise_timestep": opt.noise_timestep,
            "original_dmd2_timesteps": ORIGINAL_DMD2_TIMESTEPS,
            "augment_timesteps": augment_solver.schedule_as_list(),
            "augment_source_latents": augment_sources,
        },
    )

    print(f"Prompt: {prompt}")
    print(f"Original DMD2 images: {original_dir}")
    print(f"Augment images: {augment_dir}")
    print(f"Comparison grids: {run_dir}")


if __name__ == "__main__":
    main()