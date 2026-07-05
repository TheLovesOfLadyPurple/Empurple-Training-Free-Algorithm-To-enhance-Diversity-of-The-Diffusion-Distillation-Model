# coding=utf-8
"""Compute sample-sample covariance for image latents encoded with a Stable Diffusion VAE."""

import argparse
import logging
import math
import os

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image, ImageOps


DEFAULT_COMPUTE_EIGENVALUES = True
DEFAULT_MAX_EXACT_EIGEN_SAMPLES = 10000

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
RESAMPLING_BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


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
      description="Compute sample-sample covariance for image latents encoded with a Stable Diffusion VAE."
  )
  parser.add_argument(
      "--image_folder",
      type=str,
      default='./gen_img_val_v15_coco2014_unipc_low_new_c/samples-customed-8-unipc-retrain-free-notNPNet-full-trick-5.5',
      help="Folder containing images to encode before covariance computation.",
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
      default="./image_folder_covariance.pt",
      help="Path to save covariance statistics.",
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=1,
      help="Number of images processed at once for VAE encoding.",
  )
  parser.add_argument(
      "--image_size",
      type=int,
      default=512,
      help="Square image size used before VAE encoding.",
  )
  parser.add_argument(
      "--max_files",
      type=int,
      default=0,
      help="Optional limit on the number of images to encode. 0 means all images.",
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
      help="Device used for covariance computation after samples are encoded.",
  )
  parser.add_argument(
      "--compute_eigenvalues",
      type=parse_bool,
      default=DEFAULT_COMPUTE_EIGENVALUES,
      help="Whether to compute covariance eigenvalue diagnostics. Use true or false.",
  )
  parser.add_argument(
      "--max_exact_eigen_samples",
      type=int,
      default=DEFAULT_MAX_EXACT_EIGEN_SAMPLES,
      help="Maximum sample count allowed for exact eigendecomposition. Increase this limit if you want an exact solve for more samples.",
  )
  return parser.parse_args()


def validate_args(args):
  if args.batch_size <= 0:
    raise ValueError("batch_size must be positive")
  if args.image_size <= 0:
    raise ValueError("image_size must be positive")
  if args.max_files < 0:
    raise ValueError("max_files must be non-negative")
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


def list_image_files(folder_path, max_files):
  if not os.path.isdir(folder_path):
    raise ValueError(f"image_folder does not exist or is not a directory: {folder_path}")

  image_paths = []
  for root, _, filenames in os.walk(folder_path):
    for filename in sorted(filenames):
      extension = os.path.splitext(filename)[1].lower()
      if extension in IMAGE_EXTENSIONS:
        image_paths.append(os.path.join(root, filename))

  image_paths.sort()
  if max_files > 0:
    image_paths = image_paths[:max_files]
  if not image_paths:
    raise ValueError(f"No images found under {folder_path}")
  return image_paths


def load_vae(model_name_or_path, device, dtype, local_files_only):
  attempts = (
      {"subfolder": "vae"},
      {},
  )
  last_error = None

  for extra_kwargs in attempts:
    try:
      vae = AutoencoderKL.from_pretrained(
          model_name_or_path,
          local_files_only=local_files_only,
          **extra_kwargs,
      )
      vae.to(device=device, dtype=dtype)
      vae.requires_grad_(False)
      vae.eval()
      return vae
    except Exception as exc:
      last_error = exc

  raise RuntimeError(
      "Failed to load the Stable Diffusion VAE. Pass a local path or pre-cache the model."
  ) from last_error


def load_image_tensor(image_path, image_size):
  image = Image.open(image_path).convert("RGB")
  image = ImageOps.fit(image, (image_size, image_size), method=RESAMPLING_BICUBIC)
  image = np.asarray(image, dtype=np.float32) / 255.0
  image = torch.from_numpy(image).permute(2, 0, 1)
  image = image * 2.0 - 1.0
  return image


def encode_image_batch(vae, batch_paths, image_size, device, dtype):
  pixel_values = torch.stack(
      [load_image_tensor(image_path, image_size) for image_path in batch_paths], dim=0
  )
  pixel_values = pixel_values.to(device=device, dtype=dtype)
  with torch.no_grad():
    latent_dist = vae.encode(pixel_values).latent_dist
    latents = latent_dist.mean * vae.config.scaling_factor
  return latents


def batched(items, batch_size):
  for start in range(0, len(items), batch_size):
    yield items[start:start + batch_size]


def collect_encoded_samples(vae, image_paths, image_size, batch_size, device, dtype):
  if batch_size <= 0:
    raise ValueError("batch_size must be positive")

  samples = []
  sample_shape = None

  for batch_paths in batched(image_paths, batch_size):
    latents = encode_image_batch(vae, batch_paths, image_size, device, dtype).detach().cpu()
    if sample_shape is None:
      sample_shape = tuple(latents.shape[1:])
    elif tuple(latents.shape[1:]) != sample_shape:
      raise ValueError(
          f"Latent shape mismatch: expected {sample_shape}, got {tuple(latents.shape[1:])}"
      )

    samples.append(latents.flatten(start_dim=1).to(torch.float32))
    loaded = sum(batch.shape[0] for batch in samples)
    if loaded == latents.shape[0] or loaded % 100 == 0 or loaded == len(image_paths):
      logging.info("Encoded %d/%d images", loaded, len(image_paths))

  return torch.cat(samples, dim=0), sample_shape


def compute_effective_rank_from_eigenvalues(eigenvalues):
  positive = eigenvalues[eigenvalues > 0]
  if positive.numel() == 0:
    return 0.0
  weights = positive / positive.sum()
  entropy = -(weights * torch.log(weights)).sum()
  return float(torch.exp(entropy).detach().cpu())


def compute_sample_covariance_statistics(
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

  result = {
      "covariance": covariance.detach().cpu(),
      "covariance_kind": "sample_sample",
      "covariance_shape": tuple(covariance.shape),
      "sample_means": sample_means.squeeze(1).detach().cpu(),
      "trace": float(torch.trace(covariance).detach().cpu()),
      "flat_dim": int(flat_dim),
      "num_samples": int(num_samples),
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


def save_results(
    output_path,
    results,
    image_paths,
    sample_shape,
    image_folder,
    measure_model,
    image_size,
    batch_size,
):
  output_dir = os.path.dirname(output_path)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)

  payload = dict(results)
  payload["image_folder"] = image_folder
  payload["measure_model"] = measure_model
  payload["image_size"] = int(image_size)
  payload["batch_size"] = int(batch_size)
  payload["sample_shape"] = torch.tensor(sample_shape, dtype=torch.int64)
  payload["image_paths"] = image_paths
  torch.save(payload, output_path)


def main():
  args = parse_args()
  validate_args(args)

  logging.basicConfig(
      level=logging.INFO,
      format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
  )

  encode_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  compute_device = resolve_compute_device(args.compute_device)
  covariance_dtype = resolve_dtype(args.covariance_dtype)
  encode_dtype = torch.float32

  image_paths = list_image_files(args.image_folder, args.max_files)
  logging.info("Found %d images under %s", len(image_paths), args.image_folder)

  vae = load_vae(
      args.measure_model,
      device=encode_device,
      dtype=encode_dtype,
      local_files_only=args.local_files_only,
  )
  logging.info("Loaded Stable Diffusion VAE from %s", args.measure_model)

  samples, sample_shape = collect_encoded_samples(
      vae,
      image_paths=image_paths,
      image_size=args.image_size,
      batch_size=args.batch_size,
      device=encode_device,
      dtype=encode_dtype,
  )
  logging.info("Encoded latent shape per image: %s", sample_shape)

  stats = compute_sample_covariance_statistics(
      samples,
      compute_device=compute_device,
      covariance_dtype=covariance_dtype,
      compute_eigenvalues=args.compute_eigenvalues,
      max_exact_eigen_samples=args.max_exact_eigen_samples,
  )

  save_results(
      args.output_path,
      stats,
      image_paths,
      sample_shape,
      image_folder=args.image_folder,
      measure_model=args.measure_model,
      image_size=args.image_size,
      batch_size=args.batch_size,
  )
  logging.info("Saved covariance statistics to %s", args.output_path)
  logging.info("Sample covariance shape: %s x %s", *stats["covariance_shape"])
  logging.info("Trace(cov): %.6f", stats["trace"])

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


if __name__ == "__main__":
  main()