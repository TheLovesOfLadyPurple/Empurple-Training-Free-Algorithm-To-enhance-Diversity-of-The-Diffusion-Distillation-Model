# coding=utf-8
"""Estimate the empirical sample-sample covariance of hat_z1 from a Stable Diffusion measure model."""

import argparse
import logging
import math
import os
import random
from itertools import islice

import numpy as np
import torch
from diffusers import DDPMScheduler, StableDiffusionPipeline
from pycocotools.coco import COCO

DEFAULT_COMPUTE_EIGENVALUES = True
DEFAULT_MAX_EXACT_EIGEN_SAMPLES = 10000


def parse_bool(value):
  if isinstance(value, bool):
    return value

  normalized = value.strip().lower()
  if normalized in {"1", "true", "t", "yes", "y"}:
    return True
  if normalized in {"0", "false", "f", "no", "n"}:
    return False
  raise argparse.ArgumentTypeError("Expected a boolean value: true/false")


def parse_args():
  parser = argparse.ArgumentParser(
    description="Estimate the empirical sample-sample covariance of hat_z1 from a Stable Diffusion measure model."
  )
  parser.add_argument(
    "--measure_model",
      type=str,
      default="sd-legacy/stable-diffusion-v1-5",
    help="Stable Diffusion measure model ID or a local diffusers checkpoint path.",
  )
  parser.add_argument(
      "--output_path",
      type=str,
      default="./hat_z1_covariance.pt",
      help="Path to save covariance statistics.",
  )
  parser.add_argument(
      "--num_samples",
      type=int,
      default=1000,
      help="Number of sampling rounds. Total returned samples = num_samples * batch_size.",
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=10,
      help="Number of Gaussian noise samples processed at once.",
  )
  parser.add_argument("--seed", type=int, default=42, help="Random seed.")
  parser.add_argument(
      "--from_instances_file",
      type=str,
      default="./instances_train2014.json",
      help="COCO instances annotation file used to sample image IDs for caption guidance.",
  )
  parser.add_argument(
      "--from_caption_file",
      type=str,
      default="./captions_train2014.json",
      help="COCO captions annotation file used to guide hat_z1 sampling.",
  )
  parser.add_argument(
      "--num_channels",
      type=int,
      default=4,
      help="Latent channel count used for hat_z1 sampling.",
  )
  parser.add_argument(
      "--image_size",
      type=int,
      default=64,
      help="Latent spatial size used for hat_z1 sampling.",
  )
  parser.add_argument(
      "--local_files_only",
      type=parse_bool,
      default=False,
      help="Load Stable Diffusion weights from local files/cache only. Use true or false.",
  )
  parser.add_argument(
      "--covariance_dtype",
      type=str,
      default="float32",
      choices=["float32", "float64"],
      help="Dtype used for covariance and spectrum computation.",
  )
  parser.add_argument(
      "--compute_device",
      type=str,
      default="auto",
      choices=["auto", "cpu", "cuda"],
      help="Device used for covariance computation after samples are generated.",
  )
  parser.add_argument(
      "--compute_eigenvalues",
      type=parse_bool,
      default=DEFAULT_COMPUTE_EIGENVALUES,
      help="Whether to compute covariance eigenvalue diagnostics from the centered samples. Use true or false.",
  )
  parser.add_argument(
      "--cfg_scale",
      type=float,
      default=5.5,
      help="Classifier-free guidance scale used for caption-guided hat_z1 sampling.",
  )
  parser.add_argument(
      "--max_exact_eigen_samples",
      type=int,
      default=DEFAULT_MAX_EXACT_EIGEN_SAMPLES,
      help="Maximum sample count allowed for exact eigendecomposition. Increase this limit if you want an exact solve for more samples.",
  )
  return parser.parse_args()


def validate_args(args):
  if args.num_samples <= 0:
    raise ValueError("num_samples must be positive")
  if args.batch_size <= 0:
    raise ValueError("batch_size must be positive")
  if args.num_channels <= 0:
    raise ValueError("num_channels must be positive")
  if args.image_size <= 0:
    raise ValueError("image_size must be positive")
  if args.cfg_scale < 0:
    raise ValueError("cfg_scale must be non-negative")
  if args.max_exact_eigen_samples <= 0:
    raise ValueError("max_exact_eigen_samples must be positive")


def resolve_dtype(name):
  if name == "float64":
    return torch.float64
  return torch.float32


def resolve_compute_device(compute_device_name):
  if compute_device_name == "auto":
    if torch.cuda.is_available():
      return torch.device("cuda:0")
    return torch.device("cpu")
  if compute_device_name == "cuda":
    if not torch.cuda.is_available():
      raise RuntimeError("compute_device=cuda requested, but CUDA is unavailable")
    return torch.device("cuda:0")
  return torch.device("cpu")


def load_measure_pipeline(model_name_or_path, device, dtype, local_files_only):
  try:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
    )
  except Exception as exc:
    raise RuntimeError(
        "Failed to load the Stable Diffusion measure model. Pass a local path or pre-cache the model."
    ) from exc

  pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
  pipe.unet.to(device=device, dtype=dtype)
  pipe.unet.requires_grad_(False)
  pipe.unet.eval()
  pipe.text_encoder.to(device=device, dtype=dtype)
  pipe.text_encoder.requires_grad_(False)
  pipe.text_encoder.eval()

  if getattr(pipe, 'vae', None) is not None:
    pipe.vae.to('cpu')
  if getattr(pipe, 'image_encoder', None) is not None:
    pipe.image_encoder.to('cpu')
  if getattr(pipe, 'safety_checker', None) is not None:
    pipe.safety_checker = None

  return pipe


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
        captions,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    prompt_embeds = text_encoder(text_inputs.input_ids.to(text_encoder.device))[0]

  return prompt_embeds


def chunk(it, size):
  it = iter(it)
  return iter(lambda: tuple(islice(it, size)), ())


def load_coco_prompt_batches(instances_file, caption_file, batch_size, num_batches):
  logging.info("Reading prompts from %s", caption_file)
  coco_annotation = COCO(annotation_file=instances_file)
  coco_caption = COCO(annotation_file=caption_file)

  requested_prompt_count = num_batches * batch_size
  img_ids = coco_annotation.getImgIds()
  random.shuffle(img_ids)
  img_ids = img_ids[:requested_prompt_count]

  caption_ids = coco_caption.getAnnIds(imgIds=img_ids)
  captions = coco_caption.loadAnns(caption_ids)
  captions = [caption for index, caption in enumerate(captions) if index % 5 == 0]
  prompt_texts = [caption["caption"] for caption in captions][:requested_prompt_count]

  if len(prompt_texts) < requested_prompt_count:
    raise ValueError(
        f"Needed {requested_prompt_count} COCO captions, but only found {len(prompt_texts)}."
    )

  prompt_batches = [list(batch) for batch in chunk(prompt_texts, batch_size)]
  if len(prompt_batches) != num_batches:
    raise ValueError(
        f"Expected {num_batches} prompt batches, but built {len(prompt_batches)} from COCO captions."
    )

  logging.info("Loaded %d caption-guided prompt batches from COCO Train 2014", len(prompt_batches))
  return prompt_batches


def get_uncond_prompt_embeds(pipe, batch_size, device, dtype, cache):
  prompt_embeds = cache.get(batch_size)
  if prompt_embeds is None:
    with torch.no_grad():
      text_inputs = pipe.tokenizer(
          [""] * batch_size,
          padding="max_length",
          max_length=pipe.tokenizer.model_max_length,
          truncation=True,
          return_tensors="pt",
      )
      prompt_embeds = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    cache[batch_size] = prompt_embeds
  return prompt_embeds


def sample_hat_z1_batch(pipe, prompt_batch, batch_shape, device, dtype, cfg_scale, uncond_prompt_cache):
  noise = torch.randn(batch_shape, device=device, dtype=dtype)
  cond_prompt_embeds = encode_prompt(
      prompt_batch,
      pipe.text_encoder,
      pipe.tokenizer,
      proportion_empty_prompts=0,
      is_train=False,
  ).to(device=device, dtype=dtype)
  uncond_prompt_embeds = get_uncond_prompt_embeds(
      pipe, batch_shape[0], device, dtype, uncond_prompt_cache
  )
  alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device=device, dtype=dtype)
  measure_step = alphas_cumprod.shape[0] - 1
  measure_labels = torch.full(
      (batch_shape[0],), measure_step, device=device, dtype=torch.long
  )
  with torch.no_grad():
    batched_noise = torch.cat([noise, noise], dim=0)
    batched_labels = torch.cat([measure_labels, measure_labels], dim=0)
    batched_prompt_embeds = torch.cat([cond_prompt_embeds, uncond_prompt_embeds], dim=0)
    batched_pred_noise = pipe.unet(
      batched_noise, batched_labels, encoder_hidden_states=batched_prompt_embeds
    ).sample
    cond_pred_noise, uncond_pred_noise = batched_pred_noise.chunk(2, dim=0)
  pred_noise = uncond_pred_noise + cfg_scale * (cond_pred_noise - uncond_pred_noise)
  sqrt_alpha = torch.sqrt(alphas_cumprod[measure_labels])[:, None, None, None]
  sqrt_one_minus_alpha = torch.sqrt(1.0 - alphas_cumprod[measure_labels])[:, None, None, None]
  hat_z1 = (noise - pred_noise * sqrt_one_minus_alpha) / sqrt_alpha
  return hat_z1


def collect_hat_z1_samples(
    pipe,
    prompt_batches,
    sample_shape,
    num_samples,
    batch_size,
    device,
    dtype,
    cfg_scale,
):
  flat_dim = int(np.prod(sample_shape))
  total_num_samples = num_samples * batch_size
  samples = torch.empty((total_num_samples, flat_dim), dtype=torch.float32)
  uncond_prompt_cache = {}
  offset = 0

  if len(prompt_batches) < num_samples:
    raise ValueError(
        f"Expected at least {num_samples} prompt batches, but only found {len(prompt_batches)}."
    )

  for prompt_batch in prompt_batches[:num_samples]:
    if len(prompt_batch) != batch_size:
      raise ValueError(
          f"Prompt batch size mismatch: expected {batch_size}, got {len(prompt_batch)}"
      )

    hat_z1 = sample_hat_z1_batch(
        pipe,
        prompt_batch,
        (batch_size, *sample_shape),
        device,
        dtype,
        cfg_scale,
        uncond_prompt_cache,
    ).flatten(start_dim=1).cpu()
    samples[offset:offset + batch_size] = hat_z1
    offset += batch_size
    logging.info("Collected %d/%d hat_z1 samples", offset, total_num_samples)

  return samples


def compute_effective_rank_from_eigenvalues(eigenvalues):
  positive = eigenvalues[eigenvalues > 0]
  if positive.numel() == 0:
    return 0.0
  weights = positive / positive.sum()
  entropy = -(weights * torch.log(weights)).sum()
  return float(torch.exp(entropy).detach().cpu())


def compute_covariance_statistics(
    samples,
    compute_device,
    covariance_dtype,
    compute_eigenvalues=DEFAULT_COMPUTE_EIGENVALUES,
    max_exact_eigen_samples=DEFAULT_MAX_EXACT_EIGEN_SAMPLES,
):
  num_samples, flat_dim = samples.shape
  denom = max(flat_dim - 1, 1)
  work_samples = samples.to(device=compute_device, dtype=covariance_dtype)
  sample_means = work_samples.mean(dim=1, keepdim=True)
  centered = work_samples - sample_means
  covariance = centered.matmul(centered.T) / denom
  covariance_shape = tuple(covariance.shape)
  sample_gram_shape = (int(num_samples), int(num_samples))

  result = {
      "sample_means": sample_means.squeeze(1).detach().cpu(),
      "covariance": covariance.detach().cpu(),
      "trace": float(torch.trace(covariance).detach().cpu()),
      "flat_dim": int(flat_dim),
      "num_samples": int(num_samples),
      "covariance_kind": "sample_sample",
      "covariance_shape": covariance_shape,
      "sample_gram_shape": sample_gram_shape,
  }

  if compute_eigenvalues:
    if num_samples > max_exact_eigen_samples:
      raise ValueError(
          f"Exact eigendecomposition requested for {num_samples} samples, which exceeds max_exact_eigen_samples={max_exact_eigen_samples}. Increase --max_exact_eigen_samples to continue."
      )

    eigenvalues = torch.linalg.eigvalsh(covariance)
    eigenvalues = torch.flip(eigenvalues.clamp_min(0), dims=[0])
    result["eigenvalues"] = eigenvalues.detach().cpu()

    eigenvalues = result["eigenvalues"]
    result["positive_eigenvalue_count"] = int((eigenvalues > 0).sum().item())
    if eigenvalues.numel() > 0:
      result["eigenvalue_mean"] = float(eigenvalues.mean().detach().cpu())
      result["effective_rank"] = compute_effective_rank_from_eigenvalues(eigenvalues)
    else:
      result["eigenvalue_mean"] = float("nan")
      result["effective_rank"] = 0.0

    top_k = min(int(math.ceil(result["effective_rank"])), int(eigenvalues.numel()))
    result["top_eigenvalue_count"] = top_k
    result["top_eigenvalues"] = eigenvalues[:top_k].detach().cpu()
    result["sum_top_eigenvalues"] = float(result["top_eigenvalues"].sum().detach().cpu())

  return result


def save_results(output_path, results, sample_shape, measure_model, num_samples, batch_size):
  output_dir = os.path.dirname(output_path)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  payload = dict(results)
  payload["sample_shape"] = torch.tensor(sample_shape, dtype=torch.int64)
  payload["measure_model"] = measure_model
  payload["num_sampling_rounds_requested"] = num_samples
  payload["batch_size"] = batch_size
  payload["num_samples_requested"] = num_samples * batch_size
  torch.save(payload, output_path)


def main():
  args = parse_args()
  validate_args(args)

  logging.basicConfig(
      level=logging.INFO,
      format='%(levelname)s - %(filename)s - %(asctime)s - %(message)s',
  )
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  random.seed(args.seed)

  sample_shape = (
    args.num_channels,
    args.image_size,
    args.image_size,
  )
  measure_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
  covariance_device = resolve_compute_device(args.compute_device)
  covariance_dtype = resolve_dtype(args.covariance_dtype)
  prompt_batches = load_coco_prompt_batches(
      instances_file=args.from_instances_file,
      caption_file=args.from_caption_file,
      batch_size=args.batch_size,
      num_batches=args.num_samples,
  )

  pipe = load_measure_pipeline(
    args.measure_model,
      device=measure_device,
      dtype=torch.float32,
    local_files_only=args.local_files_only,
  )
  logging.info("Loaded Stable Diffusion measure model from %s", args.measure_model)
  logging.info("Using classifier-free guidance scale %.1f for hat_z1 sampling", args.cfg_scale)

  samples = collect_hat_z1_samples(
      pipe,
      prompt_batches=prompt_batches,
      sample_shape=sample_shape,
    num_samples=args.num_samples,
    batch_size=args.batch_size,
      device=measure_device,
      dtype=torch.float32,
      cfg_scale=args.cfg_scale,
  )

  stats = compute_covariance_statistics(
      samples,
      compute_device=covariance_device,
      covariance_dtype=covariance_dtype,
    compute_eigenvalues=args.compute_eigenvalues,
    max_exact_eigen_samples=args.max_exact_eigen_samples,
  )

  save_results(
    args.output_path,
    stats,
    sample_shape,
    measure_model=args.measure_model,
    num_samples=args.num_samples,
    batch_size=args.batch_size,
  )
  logging.info("Saved covariance statistics to %s", args.output_path)
  logging.info("Trace(cov): %.6f", stats["trace"])
  logging.info(
      "Sample covariance shape: %s x %s",
      stats["covariance_shape"][0],
      stats["covariance_shape"][1],
  )
  logging.info(
      "Sample Gram shape: %s x %s",
      stats["sample_gram_shape"][0],
      stats["sample_gram_shape"][1],
  )
  if "eigenvalues" in stats:
    logging.info(
        "Top eigenvalues (effective-rank count=%d): %s",
        stats["top_eigenvalue_count"],
        stats["top_eigenvalues"].tolist(),
    )
    logging.info("Sum of top eigenvalues: %.6f", stats["sum_top_eigenvalues"])
    logging.info("Eigenvalue mean: %.6f", stats["eigenvalue_mean"])
    logging.info("Positive eigenvalue count: %d", stats["positive_eigenvalue_count"])
    logging.info("Effective rank: %.6f", stats["effective_rank"])


if __name__ == '__main__':
  main()