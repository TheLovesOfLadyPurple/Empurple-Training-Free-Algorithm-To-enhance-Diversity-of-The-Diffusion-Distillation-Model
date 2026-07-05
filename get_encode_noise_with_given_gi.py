import argparse, os, sys, glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import accelerate
import torchsde
import pandas as pd
import diffusers
from pycocotools.coco import COCO
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    LCMScheduler,
    DDIMScheduler,
    DDIMInverseScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DiffusionPipeline,
    LatentConsistencyModelPipeline,
)
from huggingface_hub import login
import shutil
import functools
import random
from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
import json
import subprocess
import os
from typing import Union, Tuple, Optional
from torchvision import transforms as tvt


def load_image(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    pil_img = Image.open(imgname).convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return tvt.ToTensor()(pil_img)[None, ...]  # add batch dimension


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = x.to(dtype=torch.float16,device=vae.device)  # Ensure the input is in float16 and on the same device as the VAE
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * vae.config.scaling_factor #0.18215
    return latents

def latents_to_img(x: torch.Tensor, vae: AutoencoderKL, scaling_factor = 0.18215):
    x = x.to(dtype=torch.float16,device=vae.device)   # Ensure the input is in float16
    x = (x + 1.) / 2
    posterior = vae.decode(x * scaling_factor).sample()
    return posterior

# New helper to load a list-of-dicts preference JSON
# JSON schema: [ { 'human_preference': [int], 'prompt': str, 'file_path': [str] }, ... ]
def load_preference_json(json_path: str) -> list[dict]:
    """Load records from a JSON file formatted as a list of preference dicts."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# New helper to extract just the prompts from the preference JSON
# Returns a flat list of all 'prompt' values

def extract_prompts_from_pref_json(json_path: str) -> list[str]:
    """Load a JSON of preference records and return only the prompts."""
    records = load_preference_json(json_path)
    return [rec['prompt'] for rec in records]

# Example usage:
# prompts = extract_prompts_from_pref_json("path/to/preference.json")
# print(prompts)



# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0]

    return prompt_embeds

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def convert_caption_json_to_str(json):
    caption = json["caption"]
    return caption

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./gen_img_val_pure_noise_encode"
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps_encode",
        type=int,
        default=4,
        help="number of ddim sampling steps for encode",
    )
    parser.add_argument(
        "--ddim_steps_decode",
        type=int,
        default=4,
        help="number of ddim sampling steps for decode",
    )
    parser.add_argument(
        "--iDDD_stop_steps",
        type=int,
        default=5,
        help="number of iDDD sampling steps",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-instances-file",
        type=str,
        default='./instances_train2014.json',
        help="if specified, load instances from this file",
    )
    parser.add_argument(
        "--from-caption-file",
        type=str,
        default='./captions_train2014.json',
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--npnet-checkpoint",
        type=str,
        default='./HPSFilterFix.pth',
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--naf-opt",
        type=str,
        default= 'options/test/improved-DDD/LCMXABWithPromptNAFVal.yml', #'options/test/improved-DDD/LCMXABWithPromptNAFVal-ReTrain4.yml',#'options/test/improved-DDD/LCMXABWithPromptNAFVal.yml',
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--use_encode_net_type",
        type=str,
        default= 'LCM', 
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--use_decode_net_type",
        type=str,
        default= 'LCM', 
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--force_not_use_inverse",
        action='store_true',
        default=False,
        help="do not use the inverse network for decode.",
    )
    parser.add_argument(
        "--use_free_net",
        action='store_true',
        default=False,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--force_not_use_NPNet",
        action='store_true',
        default=False,
        help="do not use the NPNet for inference.",
    )
    parser.add_argument(
        "--use_retrain",
        action='store_true',
        default=True,
        help="use the retrained network for inference.",
    )
    parser.add_argument(
        "--use_raw_golden_noise",
        action='store_true',
        default=False,
        help="use the raw golden noise for inference.",
    )
    parser.add_argument(
        "--use_org_model",
        action='store_true',
        default=True,
        help="use the org network for inference.",
    )
    parser.add_argument(
        "--inner_lcm_step",
        action='store_true',
        default=4,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--use_8full_trcik",
        action='store_true',
        default=True,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    # login("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") # login to HuggingFace Hub
    opt = parser.parse_args()

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    seed_everything(opt.seed)
    seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes])
    torch.manual_seed(seeds[accelerator.process_index].item())
    
    seed_everything(opt.seed)

    dtype = torch.float32  # torch.float16 works as well, but pictures seem to be a bit worse
    device = "cuda" 

    adapter_id = "jasperai/flash-sd"
    if opt.use_encode_net_type != 'LCM' and opt.use_encode_net_type != 'SDXL-turbo':
        repo_id_encode = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    elif opt.use_encode_net_type == 'LCM':
        repo_id_encode = "SimianLuo/LCM_Dreamshaper_v7"
    elif opt.use_encode_net_type == 'SDXL-turbo':
        repo_id_encode = "stabilityai/sdxl-turbo"
    inverse_scheduler = DDIMInverseScheduler.from_pretrained(repo_id_encode, subfolder='scheduler')
    
    pipe_encode = StableDiffusionPipeline.from_pretrained(repo_id_encode,
                                                   scheduler=inverse_scheduler,
                                                   safety_checker=None,
                                                   torch_dtype=dtype)
    
    
    # if opt.use_decode_net_type != 'LCM' and opt.use_decode_net_type != 'SDXL-turbo':
    #     repo_id_decode = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    # elif opt.use_decode_net_type == 'LCM':
    #     repo_id_decode = "SimianLuo/LCM_Dreamshaper_v7"
    # elif opt.use_decode_net_type == 'SDXL-turbo':
    #     repo_id_decode = "stabilityai/sdxl-turbo"
    # decode_scheduler = DDIMScheduler.from_pretrained(repo_id_decode, subfolder='scheduler')
    # pipe_decode = StableDiffusionPipeline.from_pretrained(repo_id_decode,
    #                                                scheduler=decode_scheduler,
    #                                                safety_checker=None,
    #                                                torch_dtype=dtype)


    pipe_encode.to(device)
    # pipe_decode.to(device)
    if opt.use_encode_net_type == 'Flash':
        pipe_encode.load_lora_weights(adapter_id)
        pipe_encode.fuse_lora()
    # if opt.use_decode_net_type == 'Flash':
    #     pipe_decode.load_lora_weights(adapter_id)
    #     pipe_decode.fuse_lora()
        
    vae = pipe_encode.vae

    # input_img = load_image(imgname).to(device=device, dtype=dtype)
    # latents = img_to_latents(input_img, vae)
    
    def compute_embeddings(prompt_batch, proportion_empty_prompts, text_encoder, tokenizer, is_train=True):
        prompt_embeds = encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train)
        return {"prompt_embeds": prompt_embeds}
    
    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        proportion_empty_prompts=0,
        text_encoder=pipe_encode.text_encoder,
        tokenizer=pipe_encode.tokenizer,
    )


    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = 1#opt.n_samples
    latent_dir = './gen_img_val_v15_coco2014_unipc_low_new_c/final_x0-12-5.5'
    latent_paths = []
    
    prompt_dir = './gen_img_val_v15_coco2014_unipc_low_new_c/prompt-12-5.5'
    prompt_paths = []

    # Collect .pth files, keeping only the first suffix per prefix (e.g., 00000_00585 from 00000_00585/00000_00719/00000_00857)
    seen_prefixes = set()
    for filename in sorted(os.listdir(latent_dir)):
        if not filename.endswith('.pth'):
            continue
        prefix = filename.split('_')[1]  # e.g., '00000'
        if prefix == '00585.pth':
            seen_prefixes.add(prefix)
            latent_paths.append(os.path.join(latent_dir, filename))
            prompt_paths.append(os.path.join(prompt_dir, filename))
    
    data = list(range(len(latent_paths)))
    def normalize_tag(value: str) -> str:
        return str(value).strip().lower().replace("/", "-").replace("\\", "-").replace(" ", "")

    inverse_tag = "noinvlatent" if opt.force_not_use_inverse else "useinvlatent"
    folder_name = (
        f"samples-pure_noise-enc_{normalize_tag(opt.use_encode_net_type)}"
        f"-dec_{normalize_tag(opt.use_decode_net_type)}"
        f"-encstep_{opt.ddim_steps_encode}"
        f"-decstep_{opt.ddim_steps_decode}"
        f"-{inverse_tag}"
    )
    prompt_folder_name = folder_name + "_prompt"
    sample_path = os.path.join(outpath, folder_name)
    prompt_save_path = os.path.join(outpath, prompt_folder_name)
    
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(prompt_save_path, exist_ok=True)
    
    def compute_embeddings(prompt_batch, proportion_empty_prompts, text_encoder, tokenizer, is_train=True):
        prompt_embeds = encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train)
        return {"prompt_embeds": prompt_embeds}
    
    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        proportion_empty_prompts=0,
        text_encoder=pipe_encode.text_encoder,
        tokenizer=pipe_encode.tokenizer,
    )

    
    base_count = len(os.listdir(sample_path))

    if len(latent_paths) == 0:
        raise RuntimeError(f"No .pth latent files found in {latent_dir}. Please check the directory.")
    
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            tic = time.time()
            all_samples = list()
            for n in trange(opt.n_iter, desc="Sampling", disable =not accelerator.is_main_process):
                for idx in tqdm(data, desc="data", disable=not accelerator.is_main_process):
                    pipe_encode.scheduler = inverse_scheduler
                    torch.cuda.empty_cache()
                    
                    latent_path = latent_paths[idx]
                    prompt_path = prompt_paths[idx]

                    c = torch.load(prompt_path, map_location=device).to(dtype=dtype, device=device)
                    # if the embedding requires batch dimension
                    if c.dim() == 2:
                        c = c.unsqueeze(0)
                        
                    # Load pre-generated SD latents (stored as HWC, convert back to CHW and add batch dim)
                    latents = torch.load(latent_path, map_location=device)
                    latents = latents.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype, device=device)
                    
                    inv_latents, _ = pipe_encode(prompt_embeds=c, negative_prompt="", guidance_scale= 1.0,
                          width=latents.shape[-1], height=latents.shape[-2],
                          output_type='latent', return_dict=False,
                          num_inference_steps=opt.ddim_steps_encode, latents=latents)
                    
                    if not opt.skip_save:
                        cache_tmp_count = base_count
                        for secidx,img in enumerate(inv_latents):
                            img = img.permute(1,2,0)
                            torch.save(img,os.path.join(sample_path, f"{base_count:05}.pth"))
                            base_count += 1
                        base_count = cache_tmp_count
                        for img in c:
                            torch.save(img,os.path.join(prompt_save_path, f"{base_count:05}.pth"))
                            base_count += 1
                        

            toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()