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
from diffusers import (
    StableDiffusionPipeline,
    LatentConsistencyModelPipeline,
)
import functools
import random
from pycocotools.coco import COCO
from typing import Union, Tuple, Optional


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

class LCMSDESolver:
    def __init__(self, inner_model: Union[LatentConsistencyModelPipeline, StableDiffusionPipeline],
                 alpha_cumprods, timesteps=1000, ddim_timesteps=4, start_timestep=None):
        self.sqrt_alpha_cumprods = torch.sqrt(alpha_cumprods)
        self.sigmas = torch.sqrt(1 - alpha_cumprods)
        step_ratio = timesteps // ddim_timesteps if start_timestep is None else start_timestep // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_timesteps = np.flip(self.ddim_timesteps).copy()
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.inner_model = inner_model
        self.to(inner_model.device)

    def scalings_for_boundary_conditions(self, timestep, sigma_data=0.5, timestep_scaling=10.0):
        c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
        c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
        return c_skip, c_out

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.sqrt_alpha_cumprods = self.sqrt_alpha_cumprods.to(device)
        self.sigmas = self.sigmas.to(device)
        return self

    def solve(self, sample, prompt_embeds, scale=7.5):
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
                original_samples=tmp_hidden_z, noise=noise, timesteps=self.ddim_timesteps[idx + 1]
            )
        return intermediate_state, tmp_hidden_z


# ---------------------------------------------------------------------------
# Prompt database & cosine similarity matching
# ---------------------------------------------------------------------------

def load_prompt_database(prompt_dir):
    """Load stored prompt embeddings from a folder of simple-numbered .pth files.

    Expects files named like XXXXX.pth (e.g. 00000.pth, 00001.pth, ...).
    Each .pth contains a tensor of shape [77, 768] (CLIP text embedding).

    Returns:
        sample_ids: sorted list of sample-ID strings (e.g. ['00000', '00001', ...])
        embeddings: tensor [N, 77, 768]
    """
    files = [f for f in os.listdir(prompt_dir) if f.endswith('.pth')]
    # Extract the numeric sample ID (filename without .pth)
    sample_ids = sorted([f[:-4] for f in files])
    print(f"Loading {len(sample_ids)} prompt embeddings from {prompt_dir} ...")
    embeddings = []
    for sid in tqdm(sample_ids, desc="Loading prompt embeddings"):
        emb = torch.load(os.path.join(prompt_dir, f"{sid}.pth"), map_location='cpu',
                         weights_only=True)
        embeddings.append(emb)
    embeddings = torch.stack(embeddings, dim=0)  # [N, 77, 768]
    return sample_ids, embeddings


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
        description="Fetch pre-stored latents via prompt cosine similarity and generate images. "
                    "Reads prompt embeddings from a prompt folder, matches query captions by "
                    "cosine similarity, then loads the corresponding latent from the latent folder."
    )
    parser.add_argument("--mode", type=str, choices=["intermediate", "final_x0"],
                        default="intermediate",
                        help="'intermediate': use stored latent directly; "
                             "'final_x0': load predicted x0 and add noise before solving")
    parser.add_argument("--outdir", type=str, default="./gen_img_val_fetch_pure_noise")
    parser.add_argument("--prompt_dir", type=str,
                        default="./gen_img_val_other/samples-pure_noise-enc_flash-dec_lcm-encstep_4-decstep_4-useinvlatent_prompt",
                        help="Folder containing stored prompt embeddings (XXXXX.pth, each [77, 768])")
    parser.add_argument("--latent_dir", type=str,
                        default="./gen_img_val_other/samples-pure_noise-enc_flash-dec_lcm-encstep_4-decstep_4-useinvlatent",
                        help="Folder containing stored latents (XXXXX.pth, each [64, 64, 4]). "
                             "Must share the same sample IDs as the prompt folder.")
    parser.add_argument("--noise_timestep", type=int, default=857,
                        help="Timestep used for adding noise in final_x0 mode")
    parser.add_argument("--ddim_steps", type=int, default=12)
    parser.add_argument("--inner_lcm_step", type=int, default=4,
                        help="Number of inner LCM solver steps")
    parser.add_argument("--scale", type=float, default=5.5)
    parser.add_argument("--n_samples", type=int, default=1, help="Batch size")
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--f", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_decode_net_type", type=str, default="Flash",
                        choices=["Flash", "LCM"])
    parser.add_argument("--from-instances-file", type=str, default="./instances_val2014.json")
    parser.add_argument("--from-caption-file", type=str, default="./captions_val2014.json")
    parser.add_argument("--precision", type=str, default="autocast", choices=["full", "autocast"])
    parser.add_argument("--n_iter", type=int, default=1)

    opt = parser.parse_args()

    seed_everything(opt.seed)

    DTYPE = torch.float32
    device = "cuda"

    # ------------------------------------------------------------------
    # 1. Load pipeline
    # ------------------------------------------------------------------
    adapter_id = "jasperai/flash-sd"
    if opt.use_decode_net_type == "Flash":
        repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    elif opt.use_decode_net_type == "LCM":
        repo_id = "SimianLuo/LCM_Dreamshaper_v7"

    pipe = StableDiffusionPipeline.from_pretrained(repo_id, safety_checker=None, torch_dtype=DTYPE)
    pipe.to(device)

    if opt.use_decode_net_type == "Flash":
        pipe.load_lora_weights(adapter_id)
        pipe.fuse_lora()

    noise_scheduler = pipe.scheduler
    alpha_schedule = noise_scheduler.alphas_cumprod.to(device=device, dtype=DTYPE)
    solver = LCMSDESolver(
        inner_model=pipe,
        alpha_cumprods=alpha_schedule,
        timesteps=noise_scheduler.config.num_train_timesteps,
        ddim_timesteps=opt.inner_lcm_step,
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
    # 3. Load stored prompt-embedding database
    # ------------------------------------------------------------------
    sample_ids, stored_embeds = load_prompt_database(opt.prompt_dir)
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
    data = [cap['caption'] for cap in captions]
    print(f"Total query prompts: {len(data)}")

    batch_size = opt.n_samples
    grouped = [list(t) for t in chunk(data, batch_size)]

    # ------------------------------------------------------------------
    # 5. Prepare output directory
    # ------------------------------------------------------------------
    os.makedirs(opt.outdir, exist_ok=True)
    latent_dir_name = os.path.basename(os.path.normpath(opt.latent_dir))
    folder_name = f"samples-{opt.mode}-{latent_dir_name}-step-{opt.ddim_steps}-{opt.scale}"
    sample_path = os.path.join(opt.outdir, folder_name)
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    # ------------------------------------------------------------------
    # 6. Generation loop
    # ------------------------------------------------------------------
    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    with torch.no_grad():
        with precision_scope("cuda"):
            tic = time.time()
            for n in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(grouped, desc="data"):
                    if isinstance(prompts, str):
                        prompts = [prompts]
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    # Encode query prompt
                    encoded = compute_embeddings_fn(prompts)
                    c = encoded["prompt_embeds"].to(device=device, dtype=DTYPE)
                    del encoded

                    # Cosine-similarity matching against stored embeddings
                    idx = cosine_similarity_match(c, stored_embeds)

                    # Load latents for the matched samples
                    latents_list = []
                    for i in idx.cpu().tolist():
                        sid = sample_ids[int(i)]
                        latent_file = os.path.join(opt.latent_dir, f"{sid}.pth")
                        latent = torch.load(latent_file, map_location='cpu',
                                            weights_only=True)
                        # stored shape [64, 64, 4] → [1, 4, 64, 64]
                        latent = latent.permute(2, 0, 1).unsqueeze(0)
                        latents_list.append(latent)

                    x = torch.cat(latents_list, dim=0).to(device=device, dtype=DTYPE)

                    # # ---- Mode-specific latent preparation ----
                    # if opt.mode == "final_x0":
                    #     # x0 prediction → add noise to create a noisy starting latent
                    #     noise = torch.randn_like(x)
                    #     tmp_t = (torch.ones(x.shape[0], device=device) * opt.noise_timestep).long()
                    #     x = pipe.scheduler.add_noise(x, noise, tmp_t)
                    # # For "intermediate" mode, x is already a noisy latent — use directly.

                    # Solve
                    guide_distill, samples_ddim = solver.solve(
                        sample=x, prompt_embeds=c, scale=opt.scale
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

                    del c, idx, latents_list, x, guide_distill, samples_ddim, x_samples

            toc = time.time()

    print(f"Your samples are ready: {sample_path}")
    print(f"Total time: {toc - tic:.2f}s")


if __name__ == "__main__":
    main()
