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
    AutoencoderKL,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    LatentConsistencyModelPipeline,
    StableDiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel
)
import functools
import random
from pycocotools.coco import COCO
from typing import Union, Tuple, Optional
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from huggingface_hub import login, hf_hub_download

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



def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def pth_writer_worker(save_queue):
    while True:
        item = save_queue.get()
        if item is None:
            break
        tensor, file_path = item
        torch.save(tensor, file_path)


# ---------------------------------------------------------------------------
# LCM SDE Solver (from lcm_distill_aug.py)
# ---------------------------------------------------------------------------

class LCMSDESolverXL:
    def __init__(self
                 , inner_model: StableDiffusionXLPipeline #, StableDiffusionPipeline]
                 , alpha_cumprods
                 , timesteps=1000
                 , ddim_timesteps=4
                 , start_timestep=None
                 , is_abalation=False
                 , inference_steps_setting = None):
        self.sqrt_alpha_cumprods = torch.sqrt(alpha_cumprods)
        self.sigmas = torch.sqrt(1 - alpha_cumprods)
        step_ratio = timesteps // ddim_timesteps if start_timestep is None else start_timestep // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        if is_abalation:
            self.ddim_timesteps = np.append(self.ddim_timesteps, 999)
        self.ddim_timesteps = np.flip(self.ddim_timesteps).copy()
        if inference_steps_setting is not None:
            self.ddim_timesteps = np.array(inference_steps_setting)
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
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

    def solve(self, sample, prompt_embeds, cond_kwargs, scale=7.5, is_lcm_model=False):
        
        tmp_hidden_z = sample
        intermediate_state = {}
        intermediate_state["intermediate_z"] = []
        intermediate_state["pred_x0"] = []
        for idx, timestep in enumerate(self.ddim_timesteps):
            sample = tmp_hidden_z
            intermediate_state["intermediate_z"].append(sample)
            timestep = timestep * sample.new_ones([sample.shape[0]])
            if is_lcm_model:
                w_embedding = self.inner_model.get_guidance_scale_embedding(
                    torch.tensor([scale]),
                    embedding_dim=self.inner_model.unet.config["time_cond_proj_dim"],
                    dtype=torch.float32
                ).to(self.inner_model.device)
                noise_pred = self.inner_model.unet(
                    sample=sample, timestep=timestep,
                    timestep_cond=w_embedding, 
                    encoder_hidden_states=prompt_embeds.to(device=sample.device, dtype=sample.dtype), 
                    added_cond_kwargs=cond_kwargs
                ).sample
            else:
                noise_pred = self.inner_model.unet(
                    sample=sample, timestep=timestep,
                    encoder_hidden_states=prompt_embeds.to(device=sample.device, dtype=sample.dtype), 
                    added_cond_kwargs=cond_kwargs
                ).sample
            tmp_hidden_z = predicted_origin(
                model_output=noise_pred, timesteps=timestep.long(), sample=tmp_hidden_z,
                alphas=self.sqrt_alpha_cumprods, sigmas=self.sigmas,
                prediction_type=self.inner_model.scheduler.config.prediction_type
            )
            intermediate_state["pred_x0"].append(tmp_hidden_z)
            c_skip, c_out = self.scalings_for_boundary_conditions(timestep)
            tmp_hidden_z = tmp_hidden_z * c_out + c_skip * sample
            if idx == len(self.ddim_timesteps) - 1:
                break
            noise = torch.randn_like(tmp_hidden_z)
            tmp_hidden_z = self.inner_model.scheduler.add_noise(
                original_samples=tmp_hidden_z
                , noise=noise
                , timesteps=self.ddim_timesteps[idx + 1].unsqueeze(0)
            )
        return intermediate_state, tmp_hidden_z


# ---------------------------------------------------------------------------
# Prompt database & cosine similarity matching
# ---------------------------------------------------------------------------

def load_prompt_database(prompt_dir, ref_timestep="00857"):
    """Load stored prompt embeddings, one per unique sample ID.

    Returns:
        sample_ids: sorted list of sample-ID strings
        embeddings: tensor [N, 77, 768]
    """
    files = os.listdir(prompt_dir)
    sample_files = {}
    # fallback_files = {}
    for f in files:
        if not f.endswith('.pth'):
            continue
        base = f[:-4]  # strip .pth
        parts = base.split('_')
        if len(parts) != 2:
            continue
        sample_id, ts = parts
        if ts == ref_timestep:
            sample_files[sample_id] = f
        # elif sample_id not in fallback_files:
        #     fallback_files[sample_id] = f

    # Use fallback for any sample missing the ref timestep
    # for sid, fb in fallback_files.items():
    #     if sid not in sample_files:
    #         sample_files[sid] = fb

    sample_ids = sorted(sample_files.keys())
    print(f"Loading {len(sample_ids)} prompt embeddings from {prompt_dir} ...")
    embeddings = []
    for sid in tqdm(sample_ids, desc="Loading prompt embeddings"):
        emb = torch.load(os.path.join(prompt_dir, sample_files[sid]), map_location='cpu',
                         weights_only=True)
        embeddings.append(emb)
    embeddings = torch.stack(embeddings, dim=0)  # [N, 77, 768]
    return sample_ids, embeddings


def prepare_sdxl_pipeline_step_parameter(pipe, prompts, need_cfg, device, negative_prompts, W = 1024, H = 1024):
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
    # timesteps = pipe.scheduler.timesteps
    
    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = pooled_prompt_embeds.to(device)
    original_size = (W, H)
    crops_coords_top_left = (0, 0)
    target_size = (W, H)
    text_encoder_projection_dim = None
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
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )
    add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype, device=device)
    batch_size = prompt_embeds.shape[0]
    add_time_ids = add_time_ids.repeat(batch_size, 1)
    negative_add_time_ids = add_time_ids.clone()

    if need_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
    ret_dict = {
        "text_embeds": add_text_embeds,
        "time_ids": add_time_ids
    }
    return prompt_embeds, ret_dict



def cosine_similarity_match(query_embeds, stored_embeds):
    """Return indices into stored_embeds for each query via softmax-multinomial.

    Args:
        query_embeds:  [B, 77, 768]
        stored_embeds: [N, 77, 768]
    Returns:
        idx: [B] long tensor
    """
    query_vec = torch.nn.functional.normalize(query_embeds.mean(dim=1), dim=-1)
    stored_vec = torch.nn.functional.normalize(stored_embeds.mean(dim=1), dim=-1)
    similarity = torch.matmul(query_vec, stored_vec.T)
    similarity = similarity.masked_fill(similarity > 0.99, 0.0)
    probs = torch.nn.functional.softmax(similarity, dim=-1)
    idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return idx


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
    parser.add_argument("--outdir", type=str, default="./gen_img_val_fetch_latent_val_2014_xl")
    parser.add_argument("--prompt_dir", type=str,
                        default="./gen_img_train_xl_coco2014_unipc/prompt-12-5.5")
    parser.add_argument("--latent_dir", type=str, default=None,
                        help="Override latent directory (auto-set from --mode if omitted)")
    parser.add_argument("--latent_timestep", type=str, default="00852",
                        help="Timestep suffix for latent files (00572 / 00710 / 00852)")
    parser.add_argument("--noise_timestep", type=int, default=852,
                        help="Timestep used for adding noise in final_x0 mode")
    parser.add_argument("--ddim_steps", type=int, default=12)
    parser.add_argument("--scale", type=float, default=8.0)
    parser.add_argument("--n_samples", type=int, default=1, help="Batch size")
    parser.add_argument("--H", type=int, default=1024)
    parser.add_argument("--W", type=int, default=1024)
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--f", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_decode_net_type", type=str, default="DMD2",
                        choices=["Flash", "LCM","SDXL-lightning","DMD2"],)
    parser.add_argument("--from-instances-file", type=str, default="./instances_val2014.json")
    parser.add_argument("--from-caption-file", type=str, default="./captions_val2014.json")
    parser.add_argument("--precision", type=str, default="autocast", choices=["full", "autocast"])
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--force_org", type=bool, default=True)
    parser.add_argument("--is_abalation", type=bool, default=False)

    opt = parser.parse_args()

    # Auto-resolve latent directory
    if opt.latent_dir is None:
        base = "./gen_img_train_xl_coco2014_unipc"
        if opt.mode == "intermediate":
            opt.latent_dir = os.path.join(base, "intermediate-12-5.5")
        else:
            opt.latent_dir = os.path.join(base, "final_x0-12-5.5")

    seed_everything(opt.seed)

    DTYPE = torch.float32
    device = "cuda"

    # ------------------------------------------------------------------
    # 1. Load pipeline
    # ------------------------------------------------------------------
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=DTYPE)
    vae.to('cuda')
    inference_step = 4
    adapter_id = "jasperai/flash-sdxl"
    timestep = None
    if opt.use_decode_net_type == "Flash":
        repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
    elif opt.use_decode_net_type == "DMD2":
        repo_id = "tianweiy/DMD2"
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        repo_name = "tianweiy/DMD2"
        ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"
        unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to("cuda", DTYPE)
        unet.load_state_dict(torch.load(hf_hub_download(repo_name, ckpt_name), map_location="cuda"))
        pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, unet=unet, torch_dtype=DTYPE).to("cuda")
        inference_step = 4
        if not opt.is_abalation and opt.force_org:
            timestep = [999, 749, 499, 249,0]
    elif opt.use_decode_net_type == "SDXL-lightning":
        repo_id = "ByteDance/SDXL-Lightning"
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", DTYPE)
        unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
        pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=DTYPE).to("cuda")
        inference_step = 4
    elif opt.use_decode_net_type == "LCM":
        repo_id = "latent-consistency/lcm-sdxl"
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        unet = UNet2DConditionModel.from_pretrained("latent-consistency/lcm-sdxl", torch_dtype=DTYPE)
        pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=DTYPE).to("cuda")
        # pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    if opt.use_decode_net_type != "SDXL-lightning" and opt.use_decode_net_type != "DMD2" and opt.use_decode_net_type != "LCM":
        pipe = StableDiffusionXLPipeline.from_pretrained(repo_id, vae=vae, torch_dtype=DTYPE)
    pipe.scheduler = LCMScheduler.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="scheduler",
        timestep_spacing="trailing",
    )
    if opt.is_abalation and opt.force_org:
        opt.force_org = True

    pipe.to(device)

    if opt.use_decode_net_type == "Flash":
        pipe.load_lora_weights(adapter_id)
        pipe.fuse_lora()

    noise_scheduler = pipe.scheduler
    alpha_schedule = noise_scheduler.alphas_cumprod.to(device=device, dtype=DTYPE)
    solver = LCMSDESolverXL(
        inner_model=pipe,
        alpha_cumprods=alpha_schedule,
        timesteps=noise_scheduler.config.num_train_timesteps,
        ddim_timesteps=inference_step,
        start_timestep=opt.noise_timestep if not opt.force_org or opt.is_abalation else None,
        is_abalation=opt.is_abalation,
        inference_steps_setting=timestep
    )


    # ------------------------------------------------------------------
    # 3. Load stored prompt-embedding database
    # ------------------------------------------------------------------
    if not opt.force_org:
        sample_ids, stored_embeds = load_prompt_database(opt.prompt_dir,
                                                     ref_timestep=opt.latent_timestep)
        stored_embeds = stored_embeds.to(device=device, dtype=DTYPE)
        print(f"Prompt database: {len(sample_ids)} samples, embeddings {stored_embeds.shape}")

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
    folder_name = f"samples-{opt.mode}-ts{opt.latent_timestep}-step-{opt.ddim_steps}-{opt.scale}-{opt.use_decode_net_type}"
    if opt.is_abalation:
        folder_name += "-ablation"
        opt.force_org = True
    if opt.force_org:
        folder_name += "-org"
    sample_path = os.path.join(opt.outdir, folder_name)
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    # ------------------------------------------------------------------
    # 6. Generation loop
    # ------------------------------------------------------------------
    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    with torch.no_grad():
        # with precision_scope("cuda"):
            tic = time.time()
            for n in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(grouped, desc="data"):
                    if isinstance(prompts, str):
                        prompts = [prompts]
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    # Encode query prompt
                    prompt_embeds, cond_kwargs = prepare_sdxl_pipeline_step_parameter(pipe=pipe
                                                                                      ,prompts = prompts
                                                                                      , need_cfg=False
                                                                                      , device=pipe.device
                                                                                      ,negative_prompts="")
                    

                    if not opt.force_org:
                        # Cosine-similarity matching against stored embeddings
                        idx = cosine_similarity_match(prompt_embeds, stored_embeds)

                        # Load latents for the matched samples
                        latents_list = []
                        for i in idx.cpu().tolist():
                            sid = sample_ids[int(i)]
                            latent_file = os.path.join(
                                opt.latent_dir, f"{sid}_{opt.latent_timestep}.pth"
                            )
                            latent = torch.load(latent_file, map_location='cpu',
                                            weights_only=True)
                        # stored shape [64, 64, 4] → [1, 4, 64, 64]
                            latent = latent.permute(2, 0, 1).unsqueeze(0)
                            latents_list.append(latent)

                        x = torch.cat(latents_list, dim=0).to(device=device, dtype=DTYPE)

                        # ---- Mode-specific latent preparation ----
                        if opt.mode == "final_x0":
                        # x0 prediction → add noise to create a noisy starting latent
                            noise = torch.randn_like(x)
                            tmp_t = (torch.ones(x.shape[0], device=device) * opt.noise_timestep).long()
                            x = pipe.scheduler.add_noise(x, noise, tmp_t)
                        # For "intermediate" mode, x is already a noisy latent — use directly.
                    else:
                        shape = [ opt.C, opt.H // opt.f, opt.W // opt.f]
                        x = torch.randn([opt.n_samples, *shape],dtype=DTYPE, device=device)

                    # Solve
                    guide_distill, samples_ddim = solver.solve(
                        sample=x
                        , prompt_embeds=prompt_embeds
                        , cond_kwargs=cond_kwargs
                        , scale=opt.scale
                        , is_lcm_model=True if opt.use_decode_net_type == "LCM" else False
                    )

                    # Decode latent → image
                    x_samples = pipe.vae.decode(
                        samples_ddim / pipe.vae.config.scaling_factor
                    ).sample
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples:
                        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, f"{base_count:05}.png"))
                        base_count += 1
                    # if not opt.force_org:
                    #     del c, idx, latents_list, x, guide_distill, samples_ddim, x_samples

            toc = time.time()

    print(f"Your samples are ready: {sample_path}")
    print(f"Total time: {toc - tic:.2f}s")


if __name__ == "__main__":
    main()
