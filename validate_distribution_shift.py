import argparse, os
import multiprocessing as mp
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import accelerate
from collections import defaultdict
from diffusers import (
    StableDiffusionPipeline,
    LatentConsistencyModelPipeline,
    StableDiffusionPix2PixZeroPipeline,
    AutoencoderKL
)
import functools
import random
from pycocotools.coco import COCO
from typing import Union, Tuple, Optional
import glob
from torchvision import transforms as tvt

# ---------------------------------------------------------------------------
# Utility functions (from lcm_distill_aug.py)
# ---------------------------------------------------------------------------

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


def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])
    with torch.no_grad():
        text_inputs = tokenizer(
            captions, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0]
    return prompt_embeds


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

# ---------------------------------------------------------------------------
# LCM SDE Solver (from lcm_distill_aug.py)
# ---------------------------------------------------------------------------


class DDIMInverseSolver:
    def __init__(self
                 , inner_model: Union[LatentConsistencyModelPipeline, StableDiffusionPipeline]
                 , alpha_cumprods
                 , alphas
                 , timesteps=1000
                 , ddim_timesteps=4
                 , start_timestep=800):
        self.sqrt_alpha_cumprods = torch.sqrt(alpha_cumprods)
        self.alphas = alphas
        self.sigmas = torch.sqrt(1 - alpha_cumprods)
        self.lda = torch.log(self.alphas / self.sigmas)
        step_ratio = (timesteps-start_timestep) // ddim_timesteps #timesteps // ddim_timesteps if start_timestep is None else start_timestep // ddim_timesteps
        self.ddim_timesteps = (np.arange(0, ddim_timesteps + 1) * step_ratio + start_timestep).round().astype(np.int64) - 1
        # self.ddim_timesteps = np.flip(self.ddim_timesteps).copy()
        self.ddim_timesteps[0] = self.ddim_timesteps[0] + 1
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.inner_model = inner_model
        self.to(inner_model.device)
        

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.sqrt_alpha_cumprods = self.sqrt_alpha_cumprods.to(device)
        self.sigmas = self.sigmas.to(device)
        return self

    def solve(self, sample, prompt_embeds, scale=7.5, is_lcm_model=False):
        tmp_hidden_z = sample
        intermediate_state = {}
        intermediate_state["intermediate_z"] = []
        intermediate_state["pred_x0"] = []
        for idx, timestep in enumerate(self.ddim_timesteps):
            sample = tmp_hidden_z
            intermediate_state["intermediate_z"].append(sample)
            timestep = timestep * sample.new_ones([sample.shape[0]])
            if isinstance(self.inner_model, LatentConsistencyModelPipeline):
                w_embedding = self.inner_model.get_guidance_scale_embedding(
                    torch.tensor([scale]),
                    embedding_dim=self.inner_model.unet.config["time_cond_proj_dim"],
                    dtype=torch.float32
                ).to(self.inner_model.device)
                noise_pred = self.inner_model.unet(
                    sample=sample, timestep=timestep,
                    timestep_cond=w_embedding,
                    encoder_hidden_states=prompt_embeds.float()
                ).sample
            else:
                noise_pred = self.inner_model.unet(
                    sample=sample, timestep=timestep,
                    encoder_hidden_states=prompt_embeds.float()
                ).sample
            pred_x0 = predicted_origin(
                model_output=noise_pred, timesteps=timestep.long(), sample=tmp_hidden_z,
                alphas=self.sqrt_alpha_cumprods, sigmas=self.sigmas,
                prediction_type=self.inner_model.scheduler.config.prediction_type
            )
            intermediate_state["pred_x0"].append(pred_x0)
            tmp_hidden_z = self.sqrt_alpha_cumprods[self.ddim_timesteps[idx + 1]] * pred_x0 + torch.sqrt(1 - self.sqrt_alpha_cumprods[self.ddim_timesteps[idx + 1]]**2) * noise_pred

            if idx == len(self.ddim_timesteps) - 2:
                break
            
        return intermediate_state, tmp_hidden_z


def load_and_encode_image_via_batch(image_files, batch_size=50, pipe=None):
    """Calculate CLIP score for images and prompts using batching."""
    total_samples = 0
    print(f"Processing {len(image_files)} images in batches of {batch_size}...")
    noisy_latents = []
    with torch.no_grad():
    
        for batch_start in range(0, len(image_files), batch_size):
            batch_end = min(batch_start + batch_size, len(image_files))
        # Load batch of images
            batch_images = load_image_batch(image_files, batch_start, batch_size)
            latents = img_to_latents(batch_images, pipe.vae)
            tmp_t = (torch.ones(latents.shape[0], device=latents.device) * 800).long()
            x = pipe.scheduler.add_noise(latents, torch.randn_like(latents), tmp_t)
            noisy_latents += list(x.unbind(dim=0))
        
        # Accumulate weighted score
            batch_size_actual = batch_end - batch_start
            total_samples += batch_size_actual
        
            print(f"Processed batch {batch_start//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size}")
        
        # Clear memory
            del batch_images
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return noisy_latents

def load_image_batch(image_files, batch_start, batch_size):
    """Load a batch of images from file paths."""
    batch_end = min(batch_start + batch_size, len(image_files))
    images = []
    
    for i in range(batch_start, batch_end):
        img = Image.open(image_files[i]).convert('RGB')
        img_tensor = tvt.ToTensor()(img)
        images.append(img_tensor)
    img_tensor_with_batch = torch.stack(images, dim=0)  # Shape: [B, C, H, W]
    return img_tensor_with_batch

def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = x.to(dtype=torch.float32, device=vae.device)  # Ensure the input is in float32 and on the same device as the VAE
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * vae.config.scaling_factor #0.18215
    return latents

def get_image_files(image_folder):
    """Get sorted list of image files."""
    return sorted(glob.glob(os.path.join(image_folder, "*.png")))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch pre-stored latents via prompt cosine similarity and generate images."
    )
    parser.add_argument("--mode", type=str, choices=["intermediate", "final_x0"], default="intermediate", 
                        help="'intermediate': use stored z_t directly; "
                             "'final_x0': load predicted x0 and add noise before solving")
    parser.add_argument("--outdir", type=str, default="./gen_img_val_fetch_latent_ood")
    parser.add_argument("--prompt_dir", type=str,
                        default="./gen_img_val_v15_coco2014_unipc_low_new_c/prompt-12-5.5")
    parser.add_argument("--latent_dir", type=str, default=None,
                        help="Override latent directory (auto-set from --mode if omitted)")
    parser.add_argument("--latent_timestep", type=str, default="00857",
                        help="Timestep suffix for latent files (00585 / 00719 / 00857)")
    parser.add_argument("--noise_timestep", type=int, default=800,
                        help="Timestep used for adding noise in final_x0 mode")
    parser.add_argument("--ddim_steps", type=int, default=12)
    parser.add_argument("--scale", type=float, default=8.0)
    parser.add_argument("--n_samples", type=int, default=1, help="Batch size")
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--f", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_encode_type", type=str, default="Flash",
                        choices=["Flash", "LCM", "SD"])
    parser.add_argument("--from-instances-file", type=str, default="./instances_val2014.json")
    parser.add_argument("--from-caption-file", type=str, default="./captions_val2014.json")
    parser.add_argument("--img_folder", type=str, default="./gen_img_val_fetch_latent_val/samples-final_x0-ts00857-step-12-5.5-Flash-org")
    parser.add_argument("--precision", type=str, default="autocast", choices=["full", "autocast"])
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--force_org", type=bool, default=True)
    parser.add_argument("--is_abalation", type=bool, default=True)
    parser.add_argument("--is_test", type=bool, default=False)


    opt = parser.parse_args()

    seed_everything(opt.seed)

    DTYPE = torch.float32
    device = "cuda"

    # ------------------------------------------------------------------
    # 1. Load pipeline
    # ------------------------------------------------------------------

    repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    

    pipe = StableDiffusionPipeline.from_pretrained(repo_id, safety_checker=None, torch_dtype=DTYPE)
    pipe.to(device)


    noise_scheduler = pipe.scheduler
    alpha_schedule = noise_scheduler.alphas_cumprod.to(device=device, dtype=DTYPE)
    
    alphas = noise_scheduler.alphas.to(device=device, dtype=DTYPE)
    solver = DDIMInverseSolver(
        inner_model=pipe,
        alpha_cumprods=alpha_schedule,
        alphas=alphas,
        timesteps=noise_scheduler.config.num_train_timesteps,
        ddim_timesteps=4,
        start_timestep=opt.noise_timestep,
    )

    # ------------------------------------------------------------------
    # 2. Text-encoding helper
    # ------------------------------------------------------------------
    def compute_embeddings(prompt_batch, proportion_empty_prompts,
                           text_encoder, tokenizer, is_train=True):
        prompt_embeds = encode_prompt(prompt_batch, text_encoder, tokenizer,
                                      proportion_empty_prompts, is_train)
        return {"prompt_embeds": prompt_embeds}

    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        proportion_empty_prompts=0,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
    )

    # ------------------------------------------------------------------
    # 4. Load COCO captions as query prompts
    # ------------------------------------------------------------------
    print(f"Reading prompts from {opt.from_caption_file}")
    coco_annotation = COCO(annotation_file=getattr(opt, 'from_instances_file'))
    coco_caption = COCO(annotation_file=getattr(opt, 'from_caption_file'))
    img_ids = coco_annotation.getImgIds()
    random.shuffle(img_ids)
    img_ids = img_ids[:10000]

    caption_ids = coco_caption.getAnnIds(imgIds=img_ids)
    captions = coco_caption.loadAnns(caption_ids)
    captions = [cap for i, cap in enumerate(captions) if i % 5 == 0]
    data = [cap['caption'] for cap in captions][:10000]
    print(f"Total query prompts: {len(data)}")

    batch_size = opt.n_samples
    grouped = [list(t) for t in chunk(data, batch_size)]

    # ------------------------------------------------------------------
    # 5. Prepare output directory
    # ------------------------------------------------------------------
    os.makedirs(opt.outdir, exist_ok=True)
    folder_name = f"samples-{opt.use_encode_type}"
    test_output_folder_name = folder_name + "-test-decode"
    
    sample_path = os.path.join(opt.outdir, folder_name)
    decode_path = os.path.join(opt.outdir, test_output_folder_name)
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(decode_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    img_folder = getattr(opt, 'img_folder', None)
    image_files = get_image_files(img_folder)
    latent_noisy_list = load_and_encode_image_via_batch(image_files, batch_size=10, pipe=pipe)
    grouped_noisy = [list(t) for t in chunk(latent_noisy_list, batch_size)]

    # ------------------------------------------------------------------
    # 6. Generation loop
    # ------------------------------------------------------------------
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    idx = 0
    with torch.no_grad():
        with precision_scope("cuda"):
            tic = time.time()
            for n in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(grouped, desc="data"):
                    if isinstance(prompts, str):
                        prompts = [prompts]
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    curr_noisy_lantents = grouped_noisy[idx]
                    x = torch.stack(curr_noisy_lantents, dim=0).to(device=device, dtype=DTYPE)
                    # Encode query prompt
                    encoded = compute_embeddings_fn(prompts)
                    c = encoded["prompt_embeds"].to(device=device, dtype=DTYPE)
                    del encoded

                    # Solve
                    guide_distill, samples_ddim = solver.solve(
                        sample=x, prompt_embeds=c, scale=1.0 # totally dosen't matter, since we only use cfg on LCM
                    )
                    idx += 1
                    if True:
                        cache_tmp_count = base_count
                        for secidx,img in enumerate(samples_ddim):
                            img = img.permute(1,2,0)
                            torch.save(img,os.path.join(sample_path, f"{base_count:05}.pth"))
                            base_count += 1
                        if opt.is_test:
                            base_count = cache_tmp_count
                            out = pipe(
                                prompt=prompts,
                                num_inference_steps=50,
                                guidance_scale=7.5,
                                latents=samples_ddim
                            ).images
                            for x_sample in out:
                            # x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                x_sample.save(os.path.join(decode_path, f"{base_count:05}.png"))
                                base_count += 1
                        # base_count = cache_tmp_count
                        # for img in c:
                        #     torch.save(img,os.path.join(prompt_save_path, f"{base_count:05}.pth"))
                        #     base_count += 1



            toc = time.time()

    print(f"Your samples are ready: {sample_path}")
    print(f"Total time: {toc - tic:.2f}s")


if __name__ == "__main__":
    main()
