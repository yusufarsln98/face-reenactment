"""Prove the BFS head-swap LoRA numerically changes FLUX.2-klein output.

Generates the same image twice with one fixed prompt and seed:
  A: LoRA absent
  B: LoRA applied via pipe.load_lora_weights(...)

Then reports the mean absolute pixel difference. A difference of 0.0 means the
LoRA did not attach, which is a failure and is reported as such.

Run this script with the fluxenv interpreter, not the main project venv, because
diffusers is not installed in the main venv.
"""

import argparse
import gc
import os
import sys
import time
import warnings

import numpy as np
import psutil
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

REPO = os.path.join(PROJECT_ROOT, "flux_models", "flux2-klein-4b-gguf")
ENCODER = os.path.join(PROJECT_ROOT, "flux_models", "qwen3-4b-bnb4")
LORA = os.path.join(
    PROJECT_ROOT, "flux_models", "lora", "bfs_head_v1_flux-klein_4b.safetensors"
)
OUTDIR = os.path.join(PROJECT_ROOT, "out", "flux")

# A head-shaped subject, not a person. The LoRA swaps heads, so a helmet on a
# visible head is the clearest still-image proof that it does something.
PROMPT = "a polished chrome astronaut helmet on a workbench, studio product photo"


def ram_gb():
    return psutil.Process().memory_info().rss / 1024**3


def count_lora_layers(model, adapter_name="bfs"):
    """Count injected LoRA layers. peft names its layer class 'Linear', so test
    for the lora_A attribute rather than the class name."""
    names = []
    for name, mod in model.named_modules():
        la = getattr(mod, "lora_A", None)
        if la is not None and adapter_name in getattr(la, "keys", lambda: [])():
            names.append(name)
    return len(names), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
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

    t0 = time.time()
    transformer = Flux2Transformer2DModel.from_single_file(
        os.path.join(REPO, "transformer", "diffusion_pytorch_model.gguf"),
        config=REPO,
        subfolder="transformer",
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16,
    )
    print(f"[load] transformer ok in {time.time()-t0:.1f}s  ram={ram_gb():.2f} GiB", flush=True)

    vae = AutoencoderKLFlux2.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(REPO, subfolder="tokenizer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(REPO, subfolder="scheduler")

    t1 = time.time()
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        ENCODER, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True
    ).eval()
    print(f"[load] text encoder ok in {time.time()-t1:.1f}s  ram={ram_gb():.2f} GiB", flush=True)

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
            prompt=PROMPT, device=torch.device("cpu"), max_sequence_length=256
        )
    print(f"[encode] done in {time.time()-t_enc:.1f}s  shape={tuple(prompt_embeds.shape)}", flush=True)
    del enc_pipe, text_encoder
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
    embeds = prompt_embeds.to(device=device, dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    print(f"[load] on GPU. vram_alloc={torch.cuda.memory_allocated()/1024**3:.2f} GiB", flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    off_path = os.path.join(OUTDIR, "lora_off.png")
    on_path = os.path.join(OUTDIR, "lora_on.png")

    def generate():
        t = time.time()
        with torch.no_grad():
            res = pipe(
                prompt_embeds=embeds,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=torch.Generator(device="cpu").manual_seed(args.seed),
            )
        return res.images[0], time.time() - t

    # ---- Image A: LoRA absent ----
    img_a, dt_a = generate()
    img_a.save(off_path)
    print(f"[A] lora_off saved {off_path}  seconds={dt_a:.1f}", flush=True)

    # ---- Apply LoRA ----
    n_before, _ = count_lora_layers(pipe.transformer)
    print(f"[lora] modules with LoRA before load: {n_before}", flush=True)
    print(f"[lora] api=pipe.load_lora_weights(path, adapter_name=...)  path={LORA}", flush=True)

    t_l = time.time()
    load_ok, load_err = True, ""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            pipe.load_lora_weights(LORA, adapter_name="bfs")
            pipe.set_adapters(["bfs"])
        except Exception as exc:  # noqa: BLE001
            load_ok, load_err = False, f"{type(exc).__name__}: {exc}"
    print(f"[lora] load_lora_weights ok={load_ok} in {time.time()-t_l:.1f}s", flush=True)
    if load_err:
        print(f"[lora] ERROR {load_err}", flush=True)
    for w in caught:
        msg = str(w.message).replace("\n", " ")[:300]
        print(f"[lora] warning: {msg}", flush=True)

    n_after, lora_names = count_lora_layers(pipe.transformer)
    print(f"[lora] modules with LoRA after load: {n_after}", flush=True)
    print(f"[lora] target modules (sample): {sorted({n.rsplit('.',1)[-1] for n in lora_names})[:8]}", flush=True)
    try:
        ranks = {mod.r for _, mod in pipe.transformer.named_modules()
                 if hasattr(mod, "lora_A") and "bfs" in getattr(mod.lora_A, "keys", lambda: [])()}
        alphas = {mod.lora_alpha["bfs"] for _, mod in pipe.transformer.named_modules()
                  if hasattr(mod, "lora_alpha") and "bfs" in getattr(mod.lora_alpha, "keys", lambda: [])()}
        print(f"[lora] ranks present: {sorted(ranks)}", flush=True)
        print(f"[lora] alphas present: {sorted(alphas)}", flush=True)
        print(f"[lora] peft_config present: {getattr(pipe.transformer, 'peft_config', None) is not None}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[lora] rank unavailable: {exc}", flush=True)

    # ---- Image B: LoRA applied ----
    img_b, dt_b = generate()
    img_b.save(on_path)
    print(f"[B] lora_on saved {on_path}  seconds={dt_b:.1f}", flush=True)

    # ---- Numeric comparison ----
    a = np.asarray(img_a, dtype=np.float64)
    b = np.asarray(img_b, dtype=np.float64)
    if a.dtype != np.uint8 or b.dtype != np.uint8:
        a = a.astype(np.uint8)
        b = b.astype(np.uint8)
    mad = float(np.abs(a - b).mean())
    maxdiff = float(np.abs(a - b).max())
    changed = float((np.abs(a - b) > 0).mean() * 100.0)
    print(f"[diff] mean_abs_pixel_diff={mad:.4f}  max_abs_diff={maxdiff:.1f}  "
          f"pixels_changed={changed:.2f}%", flush=True)
    print(f"[diff] verdict={'LORA_ACTIVE' if mad > 0.5 else 'LORA_INEFFECTIVE'}", flush=True)
    print(f"[peak] vram_max_allocated={torch.cuda.max_memory_allocated()/1024**3:.2f} GiB  "
          f"ram_rss={ram_gb():.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())