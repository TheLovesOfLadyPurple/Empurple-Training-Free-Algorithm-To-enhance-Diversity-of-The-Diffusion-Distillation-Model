# EMPURPLE (Enhance Model Painting Using Recycled and Proper Latents is Easy)

<p align="center">
  <strong>Training-free feature reuse for better few-step diffusion inference.</strong><br/>
  EMPURPLE preserves informative middle-step features so distilled samplers can avoid unnecessary out-of-distribution drift.
</p>

<p align="center">
  <a href="https://colab.research.google.com/drive/1DVw41J_7yzhBxRhhNeOsCFxz-ROZWHLw?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" />
  </a>
</p>

## Overview

Diffusion models achieve impressive image-generation quality but remain expensive at inference time. Diffusion distillation reduces sampling steps, yet many distilled models, including SDXL-Lightning and distribution matching distillation methods, suffer from degraded Fr\'echet Inception Distance (FID). We analyze this phenomenon through a PAC-style generalization bound. Our analysis suggests that aggressive early-step redirection of the velocity field makes the distillation target harder to learn, enlarging the train-test gap. As a result, early-step output distributions differ between training and inference, causing distribution mismatch in the intermediate noisy latent used as next-step inputs. We empirically validate this mechanism by showing reduced diversity in both intermediate features and final outputs. To address this issue, we propose EMPURPLE, a simple training-free method that recycles intermediate latents sampled from the original model. EMPURPLE is model-agnostic and improves FID by 7\% to 20\% across DMD2, Hyper-SD, FlashSD, and SDXL-Lightning.

In the provided demos, EMPURPLE can improve FID by up to 20% without introducing an extra training stage.

## Algorithm Intuition

The figure below summarizes the motivation behind EMPURPLE. The upper path shows the desired behavior: We preserve useful features and keep the inference problem easy. The lower path shows the origianl distillation algorithm, where random noise destroys details of the predict image and try to train a neural network to cope with a more difficult task in a meaningless way.

I want to briefly connects PAC-style analyze to a classic religious debate. ``Probatio diabolica'' argues that: finding a demon can prove the existence of a demon, but failing to find a demon in daily life is not evidence of the nonexistence of the demon. Analogously, a law like $F=ma$ can fit all daily observations yet fail in an unseen corner case.  Fail to find the demon in the classic mechanism dosn't mean the demon not exists. In PAC terms, many hypotheses can explain the observed data; the question is why we should trust a particular one to generalize. The usual answer is simplicity through constraint. By restricting the hypothesis class (e.g., discouraging overly complex functions), we trade expressiveness for robustness. Actually, probatio diabolica also just discuss a very abstract problem, and it unintentionally relate to the classic mechanism.   

A similar, romantic accident appears in the song \emph{Empurple}, which uses a specific purple (\#664f8c) to create a melody (# 66 4 f 8 c) via the drum, a low-frequency musical instrument, and then it adds music with various frequency to further enhance the melody. Our method similarly reuses blurred color blocks and adds high-frequency detail later to generate a detailed image. 

<p align="center">
  <img src="./Empurple_FinalVer_01.png" alt="EMPURPLE algorithm overview" width="100%" />
</p>

## Visual Comparison

The following grids come from [gen_img_compare_dmd2_random_latent_xl/20260705-103647-a-woman-smile](./gen_img_compare_dmd2_random_latent_xl/20260705-103647-a-woman-smile). The left column is the original DMD2 result, while the right column shows the EMPURPLE result produced with cached middle-latent augmentation.  Both of it use the same guidance prompt: a woman smile.  The result is from os: window 11, and the linux version can be view in the colab: https://colab.research.google.com/drive/1DVw41J_7yzhBxRhhNeOsCFxz-ROZWHLw?usp=sharing

| Original DMD2 | EMPURPLE with cached middle latent |
| --- | --- |
| ![Original DMD2 result grid](./gen_img_compare_dmd2_random_latent_xl/20260705-103647-a-woman-smile/original_dmd2_grid.png) | ![EMPURPLE result grid](./gen_img_compare_dmd2_random_latent_xl/20260705-103647-a-woman-smile/augment_cached_middle_latent_grid.png) |

The comparison highlights the intended effect of EMPURPLE: keep the realism benefits of few-step sampling while recovering richer variation and better feature preservation.

## Installation

This project uses diffusers, so the environment can usually be installed directly with pip.

```bash
conda create --name empurple python=3.9
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
```

To choose a matching PyTorch build, check: https://pytorch.org/get-started/locally/

## Quick Start

For the simplest way to try EMPURPLE, open the ready-to-run Colab demo:

[Open In Colab](https://colab.research.google.com/drive/1DVw41J_7yzhBxRhhNeOsCFxz-ROZWHLw?usp=sharing)

Or Run the fast demo locally:

```bash
python compare_dmd2_and_empurple_fast_demo.py
```

Before running this COCO-based experiments, download `instances_train2014.json` and `captions_train2014.json` from the official COCO website.

If you already have enough cached latent samples, use:

```bash
python compare_dmd2_random_latent_xl.py
```

## Data Generation And Sampling

- Run `coco_data_gen_xl.py` to build the cached features.
- Run `lcm_fetch_latent_xl.py` or `lcm_fetch_latent_xl_random.py` to generate improved results.
- For SDXL-Lightning and Hyper-SD, use `ddim_fetch_latent_xl.py`.
- For SD 1.5, the process is similar.

`lcm_fetch_latent_xl_random.py` and `lcm_fetch_latent_xl.py` would normally be combined into one script, but they are kept separate here to reduce unintended randomness and make the reported results easier to reproduce.

## Reproducing The Tables

### Table 1

1. Run `coco_data_gen.py` to prepare the SD 1.5 cached data.
2. Run `covariance_pth_folder.py`.
3. Run `covariance_hat_z1.py`.

### Table 2

1. Run `lcm_fetch_latent_cached_version.py`.
2. Run `ddim_inverse_latent_typical_set_distill_latest.py`.
3. Run `ddim_inverse_latent_typical_set.py`.
4. Run `cal_noise_l1_norm.py`, `cal_noise_mean.py`, and `cal_typical_ratio.py`.

The latter script encodes middle noisy features from the original diffusion model, while the distillation script encodes middle noisy features from the distilled model.

### Table 3

1. Run `coco_data_gen_val.py`.
2. Make sure you have already generated images with the distilled model.
3. Run `validate_distribution_shift.py` to encode those $z_0$ values.
4. Run `cal_noise_l1_norm.py`, `cal_noise_mean.py`, and `cal_typical_ratio.py`.

## Ablation

`lcm_fetch_latent_xl.py` includes the ablation-study mode used for the analysis in this repository.