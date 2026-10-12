"""Load FLUX.2-klein-4B (GGUF Q4_K_M) on a 4 GB card and generate one test image.

The text encoder is Qwen3-4B. Its bf16 weights are 8.05 GB, larger than free
RAM, so it is loaded from an NF4 (bitsandbytes) copy and kept on the CPU. The
transformer is GGUF Q4_K_M, 2.60 GB, and lives on the GPU.

Run this script with the fluxenv interpreter, not the main project venv, because
diffusers is not installed in the main venv.
"""

import argparse
import gc
import os
import sys
import time

import psutil
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

REPO = os.path.join(PROJECT_ROOT, "flux_models", "flux2-klein-4b-gguf")
ENCODER = os.path.join(PROJECT_ROOT, "flux_models", "qwen3-4b-bnb4")
OUT = os.path.join(PROJECT_ROOT, "out", "flux", "test.png")


def ram_gb():
    return psutil.Process().memory_info().rss / 1024**3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--skip-text-encoder", action="store_true")
    args = ap.parse_args()

    from diffusers import (
        AutoencoderKLFlux2,
        FlowMatchEulerDiscreteScheduler,
        Flux2KleinPipeline,
        Flux2Transformer2DModel,
        GGUFQuantizationConfig,
    )
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    device = "cuda"
    print(f"[setup] vram_total={torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GiB", flush=True)
    print(f"[setup] ram_rss_at_start={ram_gb():.2f} GiB", flush=True)

    t0 = time.time()
    gguf_path = os.path.join(REPO, "transformer", "diffusion_pytorch_model.gguf")
    print(f"[load] transformer GGUF Q4_K_M from {gguf_path}", flush=True)
    transformer = Flux2Transformer2DModel.from_single_file(
        gguf_path,
        config=REPO,
        subfolder="transformer",
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16,
    )
    print(f"[load] transformer ok in {time.time()-t0:.1f}s  ram={ram_gb():.2f} GiB", flush=True)

    vae = AutoencoderKLFlux2.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(REPO, subfolder="tokenizer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(REPO, subfolder="scheduler")

    if args.skip_text_encoder:
        print("[load] SKIPPING text encoder (no real prompt embeds)", flush=True)
        text_encoder = None
    else:
        t1 = time.time()
        print(f"[load] text encoder NF4 from {ENCODER} (CPU)", flush=True)
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            ENCODER,
            dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        text_encoder.eval()
        print(f"[load] text encoder ok in {time.time()-t1:.1f}s  ram={ram_gb():.2f} GiB", flush=True)

    prompt = "a plain grey sphere on a white background, studio photo"

    # Stage A: encode the prompt on the CPU with the quantised text encoder,
    # then free it. Its NF4 weights are about 2.6 GB of RAM and the transformer
    # needs that RAM back.
    if text_encoder is None:
        prompt_embeds = None
        print("[encode] skipped", flush=True)
    else:
        t_enc = time.time()
        enc_pipe = Flux2KleinPipeline(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )
        with torch.no_grad():
            prompt_embeds, _ = enc_pipe.encode_prompt(
                prompt=prompt, device=torch.device("cpu"), max_sequence_length=256
            )
        print(f"[encode] done in {time.time()-t_enc:.1f}s  "
              f"shape={tuple(prompt_embeds.shape)}  ram={ram_gb():.2f} GiB", flush=True)
        del enc_pipe
        del text_encoder
        gc.collect()

    pipe = Flux2KleinPipeline.from_pretrained(
        REPO,
        transformer=transformer,
        vae=vae,
        text_encoder=None,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )

    pipe.transformer.to(device)
    pipe.vae.to(torch.bfloat16).to(device)
    torch.cuda.reset_peak_memory_stats()
    print(f"[load] on GPU. vram_alloc={torch.cuda.memory_allocated()/1024**3:.2f} GiB  "
          f"ram={ram_gb():.2f} GiB", flush=True)

    prompt_embeds = prompt_embeds.to(device=device, dtype=torch.bfloat16)

    t2 = time.time()
    try:
        with torch.no_grad():
            result = pipe(
                prompt_embeds=prompt_embeds,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=torch.Generator(device="cpu").manual_seed(0),
            )
        image = result.images[0]
    except torch.cuda.OutOfMemoryError as exc:
        print(f"[FAIL] CUDA OOM: {str(exc).splitlines()[0]}", flush=True)
        print(f"[peak] vram={torch.cuda.max_memory_allocated()/1024**3:.2f} GiB ram={ram_gb():.2f} GiB", flush=True)
        return 2

    dt = time.time() - t2
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    image.save(args.out)

    vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[ok] saved {args.out}", flush=True)
    print(f"[ok] {args.width}x{args.height} steps={args.steps} seconds={dt:.1f} "
          f"({dt/max(args.steps,1):.1f}s/step)", flush=True)
    print(f"[peak] vram_max_allocated={vram:.2f} GiB of "
          f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GiB", flush=True)
    print(f"[peak] ram_rss={ram_gb():.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
