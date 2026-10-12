# Different approach: native video diffusion (Wan 2.2 Animate)

This section answers one question with numbers: can this computer run
Wan 2.2 Animate? The numbers come from
`faceswap_wan_feasibility_budget.json` in this folder, written by
`scripts/video/wan_budget.py`.

## What Wan 2.2 Animate is

Wan 2.2 Animate is an open video model from the Wan-AI team. It takes a driving
video and a reference image and replicates the movement and the expression.
It generates the whole clip at once, so attention runs across all frames. The
flicker of a frame-by-frame pipeline cannot occur. The weights are Apache-2.0
and ungated.

The current pipeline renders each frame alone. Nothing attends across time,
and the face boils between frames. Wan 2.2 Animate would fix that at the
source. The release holds one set of denoising weights in one file per
precision. There is no separate high-noise and low-noise expert pair to add.
The open question is only whether this computer can run it.

## The exact weights one run needs

Required for one run: a diffusion model, a text encoder, one VAE, the relight
LoRA and the preprocessing checkpoints. Sizes are GiB. Byte counts are exact.

| File | Bytes | GiB |
|---|---|---|
| Diffusion model, bf16 | 34549787368 | 32.18 |
| Diffusion model, int8 | 18413068672 | 17.15 |
| Relight LoRA, bf16 | 1436673432 | 1.34 |
| Text encoder, fp8 | 6735906897 | 6.27 |
| Text encoder, fp16 | 11366399385 | 10.59 |
| VAE, wan_2.1 (used by the Animate workflow) | 253815318 | 0.24 |
| VAE, wan2.2 | 1409400960 | 1.31 |
| Preprocess checkpoint, SAM2 tiny | 155906050 | 0.15 |
| Preprocess checkpoint, SAM2 small | 184309650 | 0.17 |
| Preprocess checkpoint, SAM2 base plus | 323493298 | 0.3 |
| Preprocess checkpoint, SAM2 large | 897952466 | 0.84 |

Third-party GGUF quantizations of the same model:

| Quantization | Bytes | GiB |
|---|---|---|
| Q2_K | 6457431872 | 6.01 |
| Q3_K_S | 7969675072 | 7.42 |
| Q3_K_M | 8630769472 | 8.04 |
| Q4_0 | 10402699072 | 9.69 |
| Q4_K_S | 10592753472 | 9.87 |
| Q4_K_M | 11496331072 | 10.71 |
| Q5_0 | 12526065472 | 11.67 |
| Q5_K_S | 12349118272 | 11.5 |
| Q5_K_M | 13003659072 | 12.11 |
| Q6_K | 14605195072 | 13.6 |
| Q8_0 | 18719217472 | 17.43 |

Sources: the Hugging Face repository listings, fetched 11 October 2026. The
safetensors files are the Comfy-Org repackaged release of the official Wan-AI
weights. The SAM2 checkpoints are from the Wan-AI repository. The GGUF files
are a third-party conversion. The GGUF files are Apache-2.0 in licence, but
they are not an official release.

## The machine's measured limit

Read live during the budget run.

| Limit | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM total | 4.0 GiB |
| VRAM free at the run | 3.87 GiB |
| Compute capability | 8.6, Ampere |
| FP8 tensor cores | none on this part |
| RAM total | 7.38 GiB |
| RAM available | 5.93 GiB |
| Disk free | 920.97 GiB |
| PCIe link | gen 4 x8, 15.75 GB/s theoretical |

The card has no FP8 tensor cores. The fp8 checkpoint still loads. It gives no
FP8 compute speedup on this part.

For contrast, the project runs FLUX.2-klein-4B today: 2.6 GB of weights on
the GPU and a peak of 3.6 GiB of the 4.0 GiB card. That figure is from
`.agent/STATE.md`.

## The measured requirement and the arithmetic

The minimum resident set is the diffusion model, plus the text encoder, plus
one VAE, plus the relight LoRA, plus one preprocessing checkpoint.

| Configuration | Weights, GiB | Times the card | Times available RAM | Estimated peak, GiB |
|---|---|---|---|---|
| Smallest possible: third-party Q2_K | 14.01 | 3.5 | 2.36 | 15.21 |
| Official int8 | 25.14 | 6.29 | 4.24 | 26.34 |
| Official bf16 | 40.17 | 10.04 | 6.77 | 41.37 |

The weight column is measured: the sums are 15039733569, 26995370369 and
43132089065 bytes. The peak column is estimated: the weights plus 1.2 GiB of
runtime memory. That 1.2 GiB is the measured floor that the project's own
FLUX run showed above its weights. A video model adds more.

## Verdict

No configuration fits this machine.

- The smallest possible set is 14.01 GiB of weights. That is 3.5 times the
  4.0 GiB card. It uses a third-party Q2_K quantization, the lowest published
  precision, and even it does not fit.
- With the estimated runtime floor the set is 15.21 GiB, 3.8 times the card.
- The official int8 set is 25.14 GiB, 6.29 times the card.
- The official bf16 set is 40.17 GiB, 10.04 times the card.
- The smallest set is also larger than the available RAM (5.93 GiB).

Honest limit: this analysis is a budget calculation from measured file sizes
and live machine readings, not a measured run; no Wan weight was downloaded
because the weights cannot fit.

## Hardware that would be needed

Recommendations by arithmetic, not measurements.

| Configuration | Smallest card | Times this card | Recommended card | Times this card |
|---|---|---|---|---|
| Third-party Q2_K | 16 GiB | 4.0 | 24 GiB | 6.0 |
| Official int8 | 32 GiB | 8.0 | 48 GiB | 12.0 |
| Official bf16 | 48 GiB | 12.0 | 80 GiB | 20.0 |

The RAM-plus-offload route also fails here. Layer-by-layer offload keeps the
weights in system RAM. The smallest set (14.01 GiB) is larger than the total
RAM (7.38 GiB). The weights would stream from disk. Disk space is not the
problem (920.97 GiB free). Time is: one full pass over the smallest set takes
at least 0.95 s at the PCIe rate with zero compute, and every step moves
weights. No offload run was measured.

## What the report should say

The report should state that native video diffusion is the correct fix for
the flicker, and that this machine cannot run it. Quote the budget: the
smallest configuration, and it uses a third-party quantization, is 3.5 times
the card. State that the analysis is a budget calculation from measured file
sizes and live machine readings, not a measured run. State that no Wan
weights were downloaded, because they cannot fit.
