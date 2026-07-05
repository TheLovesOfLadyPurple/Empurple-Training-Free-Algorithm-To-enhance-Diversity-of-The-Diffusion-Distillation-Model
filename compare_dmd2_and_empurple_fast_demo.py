import argparse
import gc
import json
import os
import random
import re
import shutil
import time

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO

from diffusers import AutoencoderKL, LCMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from huggingface_hub import hf_hub_download

from sampler import UniPCSampler


ORIGINAL_DMD2_TIMESTEPS = [999, 749, 499, 249]
SOURCE_GENERATION_SCALE = 5.5
SOURCE_GENERATION_STOP_STEPS = 8
SOURCE_GENERATION_START_FREE_U_STEP = 6
SOURCE_VAE_REPO_ID = "madebyollin/sdxl-vae-fp16-fix"
SOURCE_VAE_FILENAME = "sdxl_vae.safetensors"
SCRIPT_DTYPE = torch.float32
DEFAULT_REFERENCE_INSTANCES_FILE = "./instances_train2014.json"
DEFAULT_REFERENCE_CAPTIONS_FILE = "./captions_train2014.json"


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


def prepare_sdxl_pipeline_step_parameter(
    pipe,
    prompts,
    need_cfg,
    device,
    negative_prompts,
    W=1024,
    H=1024,
):
    if isinstance(prompts, str):
        prompts = [prompts]
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompts,
        negative_prompt=negative_prompts,
        device=device,
        do_classifier_free_guidance=need_cfg,
    )

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = pooled_prompt_embeds.to(device)
    original_size = (W, H)
    crops_coords_top_left = (0, 0)
    target_size = (W, H)
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
    batch_size = prompt_embeds.shape[0]
    add_time_ids = add_time_ids.repeat(batch_size, 1)
    negative_add_time_ids = add_time_ids.clone()

    if need_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    cond_kwargs = {
        "text_embeds": add_text_embeds,
        "time_ids": add_time_ids,
    }
    return prompt_embeds, cond_kwargs


def model_closure(pipe):
    def model_fn(x, t, c):
        prompt_embeds = c[0]
        cond_kwargs = c[1] if len(c) > 1 else None
        return pipe.unet(
            x,
            t,
            encoder_hidden_states=prompt_embeds.to(device=x.device, dtype=x.dtype),
            added_cond_kwargs=cond_kwargs,
        ).sample

    return model_fn


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


def build_unipc_source_pipeline(dtype, device):
    downloaded_path = hf_hub_download(repo_id=SOURCE_VAE_REPO_ID, filename=SOURCE_VAE_FILENAME, cache_dir=".")
    vae = AutoencoderKL.from_single_file(downloaded_path, torch_dtype=dtype).to(device)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        vae=vae,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def auto_resolve_latent_dir(outdir):
    return os.path.join(outdir, "_local_middle_latent_cache")


def load_reference_coco_prompts(num_images, instances_file, captions_file, seed):
    coco_annotation = COCO(annotation_file=instances_file)
    coco_caption = COCO(annotation_file=captions_file)
    real_img_ids = coco_annotation.getImgIds()
    rng = random.Random(seed)
    rng.shuffle(real_img_ids)
    real_img_ids = real_img_ids[:10000]

    caption_ids = coco_caption.getAnnIds(imgIds=real_img_ids)
    captions = coco_caption.loadAnns(caption_ids)
    captions = [caption for index, caption in enumerate(captions) if index % 5 == 0]
    prompts = [caption["caption"] for caption in captions][:num_images]
    if len(prompts) < num_images:
        raise ValueError(
            f"Expected at least {num_images} COCO captions from {captions_file}, found {len(prompts)}."
        )
    return prompts


def resolve_prompt_cache_dir(latent_root, num_images, seed, instances_file, captions_file):
    instances_tag = sanitize_prompt(os.path.splitext(os.path.basename(instances_file))[0])
    captions_tag = sanitize_prompt(os.path.splitext(os.path.basename(captions_file))[0])
    cache_key = (
        f"coco-reference-{instances_tag}-{captions_tag}"
        f"-n{num_images}"
        f"-seed{seed}"
        f"-unipc{SOURCE_GENERATION_STOP_STEPS}"
        f"-scale{SOURCE_GENERATION_SCALE}"
    )
    return os.path.join(latent_root, cache_key)


def load_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_cache_files(directory, suffix):
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if name.endswith(suffix)
    ]


def collect_latent_files(latent_dir, latent_timestep):
    suffix = f"_{latent_timestep}.pth"
    latent_files = collect_cache_files(latent_dir, suffix)
    if not latent_files:
        raise FileNotFoundError(
            f"No latent files ending with {suffix} were found in {latent_dir}."
        )
    return latent_files


def is_prompt_cache_ready(cache_dir, num_images, latent_timestep, mode, instances_file, captions_file):
    metadata_path = os.path.join(cache_dir, "metadata.json")
    source_image_dir = os.path.join(cache_dir, "source_images")
    intermediate_dir = os.path.join(cache_dir, "intermediate")
    final_x0_dir = os.path.join(cache_dir, "final_x0")

    if not os.path.isdir(cache_dir) or not os.path.isfile(metadata_path):
        return False

    metadata = load_metadata(metadata_path)
    if metadata.get("dtype") != str(SCRIPT_DTYPE):
        return False
    if metadata.get("reference_instances_file") != instances_file:
        return False
    if metadata.get("reference_caption_file") != captions_file:
        return False
    if len(metadata.get("source_prompts", [])) < num_images:
        return False

    source_images = collect_cache_files(source_image_dir, ".png")
    if len(source_images) < num_images:
        return False

    if mode == "intermediate":
        latent_files = collect_cache_files(intermediate_dir, f"_{latent_timestep}.pth")
    else:
        latent_files = collect_cache_files(final_x0_dir, ".pth")
    return len(latent_files) >= num_images


def persist_source_generation_cache(cache_dir, reference_prompts, opt):
    source_image_dir = os.path.join(cache_dir, "source_images")
    intermediate_dir = os.path.join(cache_dir, "intermediate")
    final_x0_dir = os.path.join(cache_dir, "final_x0")
    os.makedirs(source_image_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)
    os.makedirs(final_x0_dir, exist_ok=True)

    source_dtype = SCRIPT_DTYPE
    source_device = "cuda"
    source_pipe = build_unipc_source_pipeline(dtype=source_dtype, device=source_device)
    sampler = UniPCSampler(
        source_pipe,
        model_closure=model_closure,
        steps=SOURCE_GENERATION_STOP_STEPS,
        guidance_scale=SOURCE_GENERATION_SCALE,
        denoise_to_zero=False,
        ultilize_vae_in_fp16=False,
    )
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    available_timesteps = set()
    generated_images = []

    try:
        with torch.inference_mode():
            for index, reference_prompt in enumerate(reference_prompts):
                start_code = torch.randn([1, *shape], device=source_device, dtype=source_dtype)
                grather_feature_dict = {
                            'cond_noise': [] ,
                            'uncond_noise': [],
                            'cond_xstart': [] ,
                            'uncond_xstart': [],
                            'intermediate_x': [] ,
                            'tmp_t':[],
                            'cfg_xstart': []
                        }
                samples, _ = sampler.sample(
                            conditioning=reference_prompt,
                    batch_size=1,
                    shape=shape,
                    unconditional_conditioning="",
                    x_T=start_code,
                    start_free_u_step=SOURCE_GENERATION_START_FREE_U_STEP,
                    xl_preprocess_closure=prepare_sdxl_pipeline_step_parameter,
                    use_corrector=True,
                    grather_feature_dict=grather_feature_dict,
                )

                decoded = source_pipe.vae.decode(samples / source_pipe.vae.config.scaling_factor).sample
                decoded = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
                image = decoded[0].detach().cpu().permute(1, 2, 0).numpy()
                image = Image.fromarray((255.0 * image).astype(np.uint8))
                image.save(os.path.join(source_image_dir, f"{index:02d}.png"))
                generated_images.append(image)

                torch.save(samples[0].detach().cpu().permute(1, 2, 0), os.path.join(final_x0_dir, f"{index:02d}.pth"))

                for latent_batch, timestep_batch in zip(
                    grather_feature_dict["intermediate_x"],
                    grather_feature_dict["tmp_t"],
                ):
                    if latent_batch.ndim != 4:
                        continue
                    time_tag = int(timestep_batch[0])
                    available_timesteps.add(time_tag)
                    for batch_offset, latent in enumerate(latent_batch):
                        file_index = index + batch_offset
                        if file_index >= opt.num_images:
                            break
                        torch.save(
                            latent.detach().cpu().permute(1, 2, 0),
                            os.path.join(intermediate_dir, f"{file_index:02d}_{time_tag:05}.pth"),
                        )

        save_image_grid(generated_images, os.path.join(cache_dir, "source_generation_grid.png"))
        save_metadata(
            os.path.join(cache_dir, "metadata.json"),
            {
                "num_images": len(reference_prompts),
                "seed": opt.seed,
                "generator": "coco_data_gen_xl_reference_unipc",
                "dtype": str(source_dtype),
                "reference_instances_file": opt.reference_instances_file,
                "reference_caption_file": opt.reference_caption_file,
                "source_prompts": reference_prompts,
                "guidance_scale": SOURCE_GENERATION_SCALE,
                "stop_steps": SOURCE_GENERATION_STOP_STEPS,
                "start_free_u_step": SOURCE_GENERATION_START_FREE_U_STEP,
                "available_timesteps": sorted(available_timesteps),
            },
        )
    finally:
        del sampler
        del source_pipe
        gc.collect()
        torch.cuda.empty_cache()


def prepare_prompt_cache(cache_root, reference_prompts, opt):
    cache_dir = resolve_prompt_cache_dir(
        cache_root,
        opt.num_images,
        opt.seed,
        opt.reference_instances_file,
        opt.reference_caption_file,
    )
    if not is_prompt_cache_ready(
        cache_dir,
        opt.num_images,
        opt.latent_timestep,
        opt.mode,
        opt.reference_instances_file,
        opt.reference_caption_file,
    ):
        print(f"Generating local latent cache in {cache_dir}")
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        persist_source_generation_cache(cache_dir, reference_prompts, opt)
    else:
        print(f"Reusing local latent cache from {cache_dir}")
    return cache_dir


def collect_demo_latent_files(cache_dir, mode, latent_timestep, num_images):
    if mode == "intermediate":
        latent_dir = os.path.join(cache_dir, "intermediate")
        latent_files = collect_latent_files(latent_dir, latent_timestep)
    else:
        latent_dir = os.path.join(cache_dir, "final_x0")
        latent_files = collect_cache_files(latent_dir, ".pth")
        if not latent_files:
            raise FileNotFoundError(f"No final_x0 latents were found in {latent_dir}.")

    if len(latent_files) < num_images:
        raise FileNotFoundError(
            f"Expected at least {num_images} cached demo latents in {latent_dir}, found {len(latent_files)}."
        )
    return latent_files[:num_images]


def load_images_from_dir(directory, limit=None):
    image_paths = collect_cache_files(directory, ".png")
    if limit is not None:
        image_paths = image_paths[:limit]
    return [Image.open(path).convert("RGB") for path in image_paths]


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
        description="Generate 9 DMD2 baseline images and 9 augment images from cached middle latents for one prompt."
    )
    parser.add_argument("--prompt", type=str, default="a woman smile", help="Prompt text. If omitted, the script asks interactively.")
    parser.add_argument("--outdir", type=str, default="./gen_img_compare_dmd2_random_latent_xl")
    parser.add_argument(
        "--latent_dir",
        type=str,
        default=None,
        help="Root folder for locally cached middle latents generated from the UniPC reference pass.",
    )
    parser.add_argument("--latent_timestep", type=str, default="00852")
    parser.add_argument("--mode", type=str, choices=["intermediate", "final_x0"], default="intermediate")
    parser.add_argument("--noise_timestep", type=int, default=852)
    parser.add_argument("--num_images", type=int, default=9)
    parser.add_argument("--H", type=int, default=1024)
    parser.add_argument("--W", type=int, default=1024)
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--f", type=int, default=8)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--reference_instances_file", type=str, default=DEFAULT_REFERENCE_INSTANCES_FILE)
    parser.add_argument("--reference_caption_file", type=str, default=DEFAULT_REFERENCE_CAPTIONS_FILE)
    opt = parser.parse_args()

    if opt.num_images < 1:
        raise ValueError("--num_images must be >= 1")
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA.")

    prompt = prompt_from_cli_or_input(opt.prompt)
    latent_root = opt.latent_dir or auto_resolve_latent_dir(opt.outdir)
    os.makedirs(latent_root, exist_ok=True)

    seed_everything(opt.seed)
    reference_prompts = load_reference_coco_prompts(
        num_images=opt.num_images,
        instances_file=opt.reference_instances_file,
        captions_file=opt.reference_caption_file,
        seed=opt.seed,
    )

    cache_dir = prepare_prompt_cache(latent_root, reference_prompts, opt)
    cached_latent_files = collect_demo_latent_files(
        cache_dir=cache_dir,
        mode=opt.mode,
        latent_timestep=opt.latent_timestep,
        num_images=opt.num_images,
    )
    cached_source_images = load_images_from_dir(os.path.join(cache_dir, "source_images"), limit=opt.num_images)

    dtype = SCRIPT_DTYPE
    device = "cuda"

    seed_everything(opt.seed)

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
        prompts=prompt,
        need_cfg=False,
        device=device,
        negative_prompts="",
        W=opt.W,
        H=opt.H,
    )

    run_name = f"{time.strftime('%Y%m%d-%H%M%S')}-{sanitize_prompt(prompt)}"
    run_dir = os.path.join(opt.outdir, run_name)
    original_dir = os.path.join(run_dir, "original_dmd2")
    augment_dir = os.path.join(run_dir, "augment_cached_middle_latent")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(augment_dir, exist_ok=True)

    original_images = []
    augment_images = []
    augment_sources = []

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

        for index, latent_path in enumerate(cached_latent_files):
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

    save_image_grid(cached_source_images, os.path.join(run_dir, "reference_source_grid.png"))
    save_image_grid(original_images, os.path.join(run_dir, "original_dmd2_grid.png"))
    save_image_grid(augment_images, os.path.join(run_dir, "augment_cached_middle_latent_grid.png"))
    save_metadata(
        os.path.join(run_dir, "metadata.json"),
        {
            "prompt": prompt,
            "mode": opt.mode,
            "num_images_per_method": opt.num_images,
            "batch_size": 1,
            "seed": opt.seed,
            "latent_dir": cache_dir,
            "latent_timestep": opt.latent_timestep,
            "noise_timestep": opt.noise_timestep,
            "original_dmd2_timesteps": ORIGINAL_DMD2_TIMESTEPS,
            "augment_timesteps": augment_solver.schedule_as_list(),
            "augment_source_latents": augment_sources,
            "reference_source_prompts": reference_prompts,
            "reference_source_metadata": load_metadata(os.path.join(cache_dir, "metadata.json")),
        },
    )

    print(f"Prompt: {prompt}")
    print(f"Reference source cache: {cache_dir}")
    print(f"Original DMD2 images: {original_dir}")
    print(f"Augment images: {augment_dir}")
    print(f"Comparison grids: {run_dir}")


if __name__ == "__main__":
    main()