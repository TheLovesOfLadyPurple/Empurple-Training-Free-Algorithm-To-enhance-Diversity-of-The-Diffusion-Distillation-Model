import argparse, os, sys, glob
import math
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
    AutoencoderKL
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


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def convert_caption_json_to_str(json):
    caption = json["caption"]
    return caption

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latent_dir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./gen_img_val_fetch_latent_ood/samples-Flash"
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
    parser.add_argument(
        "--use_random_gaussian",
        action="store_true",
        help="skip cached latent loads and use fresh Gaussian noise with shape 64x64x4",
        default=True
    )
    # login("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") # login to HuggingFace Hub
    opt = parser.parse_args()

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    seed_everything(opt.seed)
    seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes])
    torch.manual_seed(seeds[accelerator.process_index].item())
    
    seed_everything(opt.seed)

    latent_dir = opt.latent_dir
    latent_paths = []
    

    # Collect .pth files, keeping only the first suffix per prefix (e.g., 00000_00585 from 00000_00585/00000_00719/00000_00857)
    seen_prefixes = set()
    for filename in sorted(os.listdir(latent_dir)):
        if not filename.endswith('.pth'):
            continue
        latent_paths.append(os.path.join(latent_dir, filename))
        
    
    data = list(range(len(latent_paths)))
    
    cur_num = 0
    satifiying_count = 0
    means = 0.0
    epsilon = 2e-2  # increased typical-set tolerance (needs to correspond to ~3 standard deviations of log-prob)
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        # with precision_scope("cuda"):
            tic = time.time()
            all_samples = list()
            # for n in trange(1, desc="Sampling", disable =not accelerator.is_main_process):
            for idx in tqdm(data, desc="data", disable=not accelerator.is_main_process):
                torch.cuda.empty_cache()

                if opt.use_random_gaussian:
                    noise = torch.randn((64, 64, 4), dtype=torch.float32, device=device)
                else:
                    latent_path = latent_paths[idx]
                    noise = torch.load(latent_path, map_location=device).to(dtype=torch.float32, device=device)
                num_dimensions = noise.numel()
                
                log_gaussian_normalizer = 0.5 * math.log(2 * math.pi)
                typical_entropy = 0.5 * math.log(2 * math.pi * math.e)
                log_probability_on_each_element = -0.5 * (noise ** 2) - log_gaussian_normalizer
                log_probability_of_noise = log_probability_on_each_element.sum().item()
                means += log_probability_of_noise
                
                low_bound = -num_dimensions * (typical_entropy + epsilon)
                upper_bound = -num_dimensions * (typical_entropy - epsilon)
                within_bounds = low_bound < log_probability_of_noise < upper_bound
                satifiying_count += int(within_bounds)

                # means += torch.mean(noise)  # mean over batch and spatial dimensions
                cur_num += 1
                        

            out = means / cur_num
            ratio = satifiying_count / cur_num
            print(f"Mean log probability across all samples: {out}")
            print(f"Ratio of samples satisfying the typical set condition: {ratio}")
            toc = time.time()

            print(f"Number of samples satisfying the typical set condition: {satifiying_count}")

if __name__ == "__main__":
    main()