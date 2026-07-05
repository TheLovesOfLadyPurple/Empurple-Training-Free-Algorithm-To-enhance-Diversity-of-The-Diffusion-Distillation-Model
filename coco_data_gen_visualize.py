import argparse, os
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
from pycocotools.coco import COCO
from diffusers import (
    StableDiffusionPipeline,
)

import functools
import random
from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import json
import subprocess
import os
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
from sampler import UniPCSampler

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

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

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


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


def model_closure(pipe):
    def model_fn(x, t, c):
        return pipe.unet(x, t, encoder_hidden_states=c).sample

    return model_fn


def feature_tensor_to_uint8_image(
    feature: torch.Tensor,
    use_fixed_range: bool = True,
    fixed_min_value: float = -1.0,
    fixed_max_value: float = 1.0,
) -> np.ndarray:
    """Convert a latent feature tensor (C,H,W) to an uint8 RGB image."""
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

    return rearrange(feature.numpy(), 'c h w -> h w c')


def decode_latent_feature_to_uint8_image(feature: torch.Tensor, vae) -> np.ndarray:
    """Decode a latent feature (C,H,W) with VAE and convert to uint8 RGB image."""
    if feature.ndim != 3:
        raise ValueError(f"Expected feature tensor with 3 dims (C,H,W), got {feature.shape}")

    latent = feature.unsqueeze(0).to(device=vae.device, dtype=vae.dtype)
    decoded = vae.decode(latent / vae.config.scaling_factor).sample
    decoded = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
    image = 255.0 * rearrange(decoded[0].detach().float().cpu().numpy(), 'c h w -> h w c')
    return image.round().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./gen_img_val_v15_coco2014_unipc_low_new_c_visualize"
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=12,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--stop_steps",
        type=int,
        default=8,
        help="number of stop sampling steps",
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
        default=5.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        default='./instances_train2014.json',
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--npnet-checkpoint",
        type=str,
        default='./HPSFilterFix.pth',
        help="if specified, load prompts from this file",
    )
    
    parser.add_argument(
        "--use_free_net",
        action='store_true',
        default=True,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--force_not_use_ct",
        action='store_true',
        default=False,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--force_not_use_NPNet",
        action='store_true',
        default=True,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--use_retrain",
        action='store_true',
        default=True,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--use_raw_golden_noise",
        action='store_true',
        default=False,
        help="use the free network for inference.",
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
    parser.add_argument(
        "--save_cfg_xstart_vis",
        action='store_true',
        default=True,
        help="save visualization images for cached cfg_xstart features.",
    )
    parser.add_argument(
        "--no_save_cfg_xstart_vis",
        dest='save_cfg_xstart_vis',
        action='store_false',
        help="disable saving visualization images for cfg_xstart.",
    )
    parser.add_argument(
        "--cfg_xstart_fixed_range",
        action='store_true',
        default=False,
        help="use fixed range [cfg_xstart_min_value, cfg_xstart_max_value] for visualization.",
    )
    parser.add_argument(
        "--cfg_xstart_auto_range",
        dest='cfg_xstart_fixed_range',
        action='store_false',
        help="use per-feature dynamic min/max range for visualization.",
    )
    parser.add_argument(
        "--cfg_xstart_min_value",
        type=float,
        default=-1.0,
        help="fixed minimum value used when mapping cfg_xstart features to 0-255.",
    )
    parser.add_argument(
        "--cfg_xstart_max_value",
        type=float,
        default=1.0,
        help="fixed maximum value used when mapping cfg_xstart features to 0-255.",
    )
    parser.add_argument(
        "--save_cfg_xstart_vae_decode",
        action='store_true',
        default=True,
        help="save VAE-decoded images from cached cfg_xstart latent features.",
    )
    
    opt = parser.parse_args()

    if opt.cfg_xstart_fixed_range and opt.cfg_xstart_max_value <= opt.cfg_xstart_min_value:
        raise ValueError("--cfg_xstart_max_value must be greater than --cfg_xstart_min_value.")

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    seed_everything(opt.seed)
    seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes])
    torch.manual_seed(seeds[accelerator.process_index].item())
    
    seed_everything(opt.seed)

    DTYPE = torch.float32  # torch.float16 works as well, but pictures seem to be a bit worse
    device = "cuda" 
    pipe = StableDiffusionPipeline.from_pretrained('sd-legacy/stable-diffusion-v1-5')
    
    pipe.to(device=device, torch_dtype=DTYPE)
    
    sampler = UniPCSampler(pipe
                           , model_closure=model_closure
                           , steps=opt.stop_steps
                           , guidance_scale=opt.scale
                           , is_high_resoulution=False)
    
    def compute_embeddings(prompt_batch, proportion_empty_prompts, text_encoder, tokenizer, is_train=True):
        prompt_embeds = encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train)
        return {"prompt_embeds": prompt_embeds}
    
    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        proportion_empty_prompts=0,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
    )


    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        coco_annotation_file_path = opt.from_file
        coco_caption_file_path = './captions_train2014.json'
        coco_annotation = COCO(annotation_file=coco_annotation_file_path)
        coco_caption = COCO(annotation_file=coco_caption_file_path)
        query_names = [] #['cup','broccoli','dining table','toaster','carrot','toilet','sink','fork','hot dog','knife','pizza','spoon','donut','clock','bowl','cake','vase','banana','scissors','couch','apple','sandwich','potted plant','microwave','orange','bed','oven']
        unselect_names = [] # ['person','airplane','bird','mouse','cat','dog','horse','clock']

        # 获取包含指定类别的图像ID
        query_ids = []
        img_ids = coco_annotation.getImgIds()
        # for query_name in query_names:
        # query_ids += coco_annotation.getCatIds(catNms=query_names)
        # for query_id in query_ids:
        #     img_ids += coco_annotation.getImgIds(catIds=query_id)

        # 获取包含不需要类别的图像ID
        unselect_id = []
        unselect_img_ids = []
        for unselect_name in unselect_names:
            unselect_id += coco_annotation.getCatIds(catNms=[unselect_name])
            unselect_img_ids += coco_annotation.getImgIds(catIds=unselect_id)

        # 过滤掉包含不需要类别的图像ID
        real_img_ids = [item for item in img_ids if item not in unselect_img_ids]
        random.shuffle(real_img_ids)
        
        real_img_ids = real_img_ids[0:50]

        # 获取这些图像的caption ID
        caption_ids = coco_caption.getAnnIds(imgIds=real_img_ids)

        # 获取并显示这些图像的captions
        captions = coco_caption.loadAnns(caption_ids)
        tmp_caption = []
        for idx,caption in enumerate(captions):
            if idx % 5 != 0:
                continue
            tmp_caption.append(caption)
        captions = tmp_caption
        
        data = list(map(lambda x: x['caption'], captions))
        data = data[(0):50]
        images = coco_caption.loadImgs(ids=real_img_ids)
        folder_name = 'E:\\txt2img-samples\\scls_coco_img_val_random'
        img_path = 'D:\\research_project\\archive(2)\\coco2014\\images\\val2014'
        # if not os.path.exists(folder_name):
        #     os.makedirs(name=folder_name,exist_ok=True)
        #     img_file_name = [ img['file_name'] for img in images ]
        #     for filename in os.listdir(path=img_path):
        #         if filename in img_file_name:
        #             shutil.copy(os.path.join(img_path, filename), folder_name)

    grouped = [list(t) for t in chunk(data, batch_size)]
    data = grouped
    if opt.stop_steps !=-1:
        folder_name = f"samples-customed-{opt.stop_steps}-unipc"
        if  opt.use_retrain:
            folder_name += "-retrain"
        if opt.use_free_net:
            folder_name += "-free"
        if opt.force_not_use_NPNet:
            folder_name += "-notNPNet"
        if opt.force_not_use_ct:
            folder_name += "-noneCT"
        if opt.use_raw_golden_noise:
            folder_name += "-rawGoldenNoise"
        if opt.use_8full_trcik:
            folder_name += "-full-trick"
        
        folder_name +=f"-{opt.scale}"
        sample_path = os.path.join(outpath, folder_name)
    elif opt.stop_steps == -1:
        folder_name = f"samples-org-{opt.ddim_steps}"
        if opt.use_free_net:
            folder_name += "-free"
        if opt.force_not_use_NPNet:
            folder_name += "-notNPNet"
        if opt.use_raw_golden_noise:
            folder_name += "-rawGoldenNoise"
        sample_path = os.path.join(outpath, folder_name)
    cfg_xstart_vis_mode = 'saturation' if opt.cfg_xstart_fixed_range else 'auto-range'
    cfg_xstart_vis_path = os.path.join(outpath, f'cfg_xstart_vis-{opt.ddim_steps}-{opt.scale}-{cfg_xstart_vis_mode}')
    cfg_xstart_decoded_path = os.path.join(outpath, f'cfg_xstart_decoded-{opt.ddim_steps}-{opt.scale}')
    os.makedirs(sample_path, exist_ok=True)
    if opt.save_cfg_xstart_vis:
        os.makedirs(cfg_xstart_vis_path, exist_ok=True)
    if opt.save_cfg_xstart_vae_decode:
        os.makedirs(cfg_xstart_decoded_path, exist_ok=True)


    base_count = len(os.listdir(sample_path))
    cfg_xstart_vis_count = len(os.listdir(cfg_xstart_vis_path)) if opt.save_cfg_xstart_vis else 0
    cfg_xstart_decoded_count = len(os.listdir(cfg_xstart_decoded_path)) if opt.save_cfg_xstart_vae_decode else 0
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            tic = time.time()
            all_samples = list()
            for n in trange(opt.n_iter, desc="Sampling", disable =not accelerator.is_main_process):
                for prompts in tqdm(data, desc="data", disable=not accelerator.is_main_process):
                    # torch.cuda.empty_cache()
                    intermediate_photos = list()
                    # prompts = prompts[0]
                            
                    # if isinstance(prompts, tuple) or isinstance(prompts, str):
                    #     prompts = list(prompts)
                    if isinstance(prompts, str):
                        prompts = prompts #+ 'high quality, best quality, masterpiece, 4K, highres, extremely detailed, ultra-detailed'
                        prompts = (prompts,)
                    if isinstance(prompts, tuple) or isinstance(prompts, str):
                        prompts = list(prompts)
                    encoded_text = compute_embeddings_fn(prompts)
                    uc = None
                    if opt.scale != 1.0:
                        uc = compute_embeddings_fn(batch_size * [""])
                    uc = uc.pop("prompt_embeds") if uc is not None else None
                    c =  encoded_text.pop("prompt_embeds")
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    
                    
                    x = torch.randn([opt.n_samples, *shape], device=device) 
                    
                    extra_args = {'cond': c, 'uncond': uc, 'cond_scale': opt.scale}
                    noise_training_list = {}

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
                        conditioning=c,
                        batch_size=opt.n_samples,
                        shape=shape,
                        unconditional_conditioning=uc,
                        x_T=x,
                        start_free_u_step=4 if opt.use_free_net else -1,
                        use_corrector=True,
                        grather_feature_dict=grather_feature_dict
                    )
                    
                        
                    x_samples_ddim = pipe.vae.decode(samples / pipe.vae.config.scaling_factor).sample
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    if not opt.skip_save:
                        for x_sample in x_samples_ddim:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:05}.png"))
                            base_count += 1

                    if opt.save_cfg_xstart_vis or opt.save_cfg_xstart_vae_decode:
                        for idx, cfg_xstart_batch in enumerate(grather_feature_dict['cfg_xstart']):
                            time_tag = int(grather_feature_dict['tmp_t'][idx][0])
                            for sample_idx, feature in enumerate(cfg_xstart_batch):
                                if opt.save_cfg_xstart_vis:
                                    vis_image = feature_tensor_to_uint8_image(
                                        feature,
                                        use_fixed_range=opt.cfg_xstart_fixed_range,
                                        fixed_min_value=opt.cfg_xstart_min_value,
                                        fixed_max_value=opt.cfg_xstart_max_value,
                                    )
                                    vis_name = f"{cfg_xstart_vis_count:05}_{time_tag:05}_{sample_idx:02}.png"
                                    Image.fromarray(vis_image).save(os.path.join(cfg_xstart_vis_path, vis_name))
                                    cfg_xstart_vis_count += 1

                                if opt.save_cfg_xstart_vae_decode:
                                    decoded_image = decode_latent_feature_to_uint8_image(feature, pipe.vae)
                                    decoded_name = f"{cfg_xstart_decoded_count:05}_{time_tag:05}_{sample_idx:02}.png"
                                    Image.fromarray(decoded_image).save(os.path.join(cfg_xstart_decoded_path, decoded_name))
                                    cfg_xstart_decoded_count += 1

            toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()