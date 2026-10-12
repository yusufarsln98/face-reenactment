"""Two-image head swap with FLUX.2-klein plus the BFS head-swap LoRA.

The script extracts three frames from the source video with ffmpeg, keeps them as
Picture 1 (the base image with the body), and keeps the portrait photo as Picture 2
(the reference face). It then runs the same three frames twice:

  pass 1: LoRA absent  -> faceswap_<frame>_lora_off.png
  pass 2: LoRA present -> faceswap_<frame>_lora_on.png

The seeds are identical in both passes, so the only difference between a pair of
images is the adapter. The script prints the pixel difference of each pair, the
timings per image and per step, the peak VRAM, and the peak RAM.

Picture 1 is the base image and comes FIRST in the image list. Picture 2 is the
reference face and comes SECOND. Reversing the two produces a plausible image with
the wrong head.

The script never retries a run at a lower resolution. It reports OOM and stops with
exit status 2.

Run this script with the fluxenv interpreter, not the main project venv, because
diffusers is not installed in the main venv.
"""

import argparse
import gc
import os
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np
import psutil
import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

REPO = os.path.join(PROJECT_ROOT, "flux_models", "flux2-klein-4b-gguf")
ENCODER = os.path.join(PROJECT_ROOT, "flux_models", "qwen3-4b-bnb4")
LORA = os.path.join(
    PROJECT_ROOT, "flux_models", "lora", "bfs_head_v1_flux-klein_4b.safetensors"
)
VIDEO = os.path.join(PROJECT_ROOT, "WhatsApp Video 2026-10-04 at 18.51.45.mp4")
REFERENCE = os.path.join(PROJECT_ROOT, "IMG_20260817_111005.jpg")

DEFAULT_FRAMES = "0,288,576"
DEFAULT_OUT_PARTS = ("out", "flux")
ADAPTER_NAME = "bfs"

# The prompt text comes from the model author. Do not edit it.
PROMPT = (
    "head_swap: start with Picture 1 as the base image, keeping its lighting, "
    "environment, and background. remove the head from Picture 1 completely and "
    "replace it with the head from Picture 2, strictly preserving the hair, eye "
    "color, and nose structure of Picture 2. copy the direction of the eye, head "
    "rotation, and micro-expressions from Picture 1. Keep the body, clothing, "
    "pose, framing, and background intact."
)

_PEAK_RAM = [0.0]


class OutOfMemory(Exception):
    """A generation call hit torch.cuda.OutOfMemoryError."""


def ram_gb():
    return psutil.Process().memory_info().rss / 1024**3


def peak_ram_gb():
    """The highest RSS seen so far. psutil has no peak field on Linux, so the
    script samples it after every stage."""
    value = ram_gb()
    if value > _PEAK_RAM[0]:
        _PEAK_RAM[0] = value
    return _PEAK_RAM[0]


def true_peak_ram_gb():
    """The kernel peak RSS for this process, in GiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def vram_gb():
    return torch.cuda.memory_allocated() / 1024**3


def parse_frames(text):
    """Read a --frames value such as '0,288,576' into three integers."""
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--frames needs three comma separated integers, got {text!r}"
        )
    try:
        values = [int(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--frames needs three integers, got {text!r}"
        ) from None
    if any(v < 0 for v in values):
        raise argparse.ArgumentTypeError(f"--frames needs no negative value, got {text!r}")
    return values


def extract_frames(frames, workdir):
    """Decode the three source frames with ffmpeg. The venv has no video decoder,
    so ffmpeg is the only path. The argument is a list, because the video path
    holds spaces."""
    select = "+".join(f"eq(n\\,{n})" for n in frames)
    pattern = os.path.join(workdir, "faceswap_body_%03d.png")
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        VIDEO,
        "-vf",
        f"select={select}",
        "-vsync",
        "0",
        "-start_number",
        "0",
        pattern,
    ]
    print(f"[frames] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[frames] ffmpeg failed with exit status {proc.returncode}", flush=True)
        print(f"[frames] stderr: {proc.stderr}", flush=True)
        sys.exit(3)
    if proc.stderr.strip():
        print(f"[frames] ffmpeg stderr: {proc.stderr.strip()}", flush=True)

    paths = [os.path.join(workdir, f"faceswap_body_{i:03d}.png") for i in range(len(frames))]
    for path in paths:
        if not os.path.isfile(path):
            print(f"[frames] missing extracted frame {path}", flush=True)
            sys.exit(3)
    return paths


def load_body(path, width, height):
    """Scale the frame so it covers the target box, then centre-crop to the exact
    size. Scaling by the longest side instead would leave a portrait frame
    shorter than the target, and the crop would then squash the face flat."""
    image = Image.open(path).convert("RGB")
    src_w, src_h = image.size
    scale = max(float(width) / float(src_w), float(height) / float(src_h))
    new_w = max(width, int(round(src_w * scale)))
    new_h = max(height, int(round(src_h * scale)))
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = max(0, (new_w - width) // 2)
    top = max(0, (new_h - height) // 2)
    image = image.crop((left, top, left + width, top + height))
    if image.size != (width, height):
        raise RuntimeError(f"crop gave {image.size}, expected {(width, height)}")
    return image, (src_w, src_h)


def load_reference(ref_size):
    """Scale the portrait to a square. There is no face box crop and no colour
    change, because the model needs the whole head."""
    image = Image.open(REFERENCE).convert("RGB")
    return image.resize((ref_size, ref_size), Image.Resampling.LANCZOS)


def count_lora_modules(model, adapter_name=ADAPTER_NAME):
    """Count the transformer modules that carry a LoRA for the named adapter."""
    names = []
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        if lora_a is None:
            continue
        keys = getattr(lora_a, "keys", None)
        if callable(keys) and adapter_name in keys():
            names.append(name)
    return len(names), names


def generate(pipe, embeds, body_image, reference_image, args, seed):
    """Run one generation. An OOM is reported and the script stops. It never
    retries, never changes the dtype, and never enables slicing or offload."""
    try:
        with torch.no_grad():
            result = pipe(
                image=[body_image, reference_image],
                prompt_embeds=embeds,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                generator=torch.Generator(device="cpu").manual_seed(seed),
            )
    except torch.cuda.OutOfMemoryError:
        print(
            f"OOM width={args.width} height={args.height} ref_size={args.ref_size} "
            f"steps={args.steps} seed={seed} "
            f"peak_vram_alloc={torch.cuda.max_memory_allocated()/1024**3:.2f} GiB",
            flush=True,
        )
        raise OutOfMemory from None
    return result.images[0]


def compare(off_image, on_image):
    a = np.asarray(off_image, dtype=np.int16)
    b = np.asarray(on_image, dtype=np.int16)
    delta = np.abs(a - b)
    return float(delta.mean()), float((delta > 0).mean() * 100.0), float(delta.max())


def main():
    ap = argparse.ArgumentParser(
        description="Two-image head swap test for FLUX.2-klein with the BFS head LoRA."
    )
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--ref-size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--frames", type=parse_frames, default=DEFAULT_FRAMES)
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.abspath(
            os.path.join(
                PROJECT_ROOT, *DEFAULT_OUT_PARTS, f"faceswap_{args.width}x{args.height}"
            )
        )
    else:
        args.out_dir = os.path.abspath(args.out_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="faceswap_frames_", dir=args.out_dir)

    print("[run] FLUX.2-klein head swap test", flush=True)
    print(f"[run] out_dir={args.out_dir}", flush=True)
    print(f"[run] work_dir={workdir}", flush=True)
    print(f"[run] width={args.width} height={args.height} ref_size={args.ref_size} "
          f"steps={args.steps}", flush=True)
    print(f"[run] frames={args.frames} base_seed={args.seed} "
          f"seeds={[args.seed + i for i in range(len(args.frames))]}", flush=True)
    print(f"[run] ram={ram_gb():.2f} GiB", flush=True)

    frame_paths = extract_frames(args.frames, workdir)
    body_images = []
    for index, (frame_no, path) in enumerate(zip(args.frames, frame_paths)):
        body, source_size = load_body(path, args.width, args.height)
        body_images.append(body)
        body.save(os.path.join(args.out_dir, f"faceswap_{index}_body_input.png"))
        print(
            f"[frames] frame {frame_no} -> {path} source={source_size} "
            f"model_input={body.size}",
            flush=True,
        )
    reference_image = load_reference(args.ref_size)
    reference_path = os.path.join(args.out_dir, "faceswap_reference_input.png")
    reference_image.save(reference_path)
    print(
        f"[frames] reference -> {reference_path} model_input={reference_image.size} "
        f"(Picture 1 = body, Picture 2 = reference)",
        flush=True,
    )
    del frame_paths
    print(f"[ram] after inputs ram={ram_gb():.2f} GiB peak={peak_ram_gb():.2f} GiB", flush=True)

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

    # Encode the prompt once, on the CPU. Passing prompt_embeds to __call__ then
    # skips the text encoder, so it can be freed before the GPU work starts.
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
            prompt=PROMPT, device=torch.device("cpu"), max_sequence_length=512
        )
    print(
        f"[encode] done in {time.time()-t_enc:.1f}s  shape={tuple(prompt_embeds.shape)}  "
        f"prompt_chars={len(PROMPT)}",
        flush=True,
    )
    del enc_pipe, text_encoder
    gc.collect()
    print(f"[ram] after freeing text encoder ram={ram_gb():.2f} GiB "
          f"peak={peak_ram_gb():.2f} GiB", flush=True)

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
    print(
        f"[load] on GPU. vram_alloc={vram_gb():.2f} GiB  ram={ram_gb():.2f} GiB "
        f"peak={peak_ram_gb():.2f} GiB",
        flush=True,
    )

    # ---- Pass 1: LoRA absent. All three frames run before the adapter loads. ----
    off_images = []
    pass1_times = []
    try:
        for index, body in enumerate(body_images):
            seed = args.seed + index
            start = time.time()
            image = generate(pipe, embeds, body, reference_image, args, seed)
            elapsed = time.time() - start
            pass1_times.append(elapsed)
            off_images.append(image)
            path = os.path.join(args.out_dir, f"faceswap_{index}_lora_off.png")
            image.save(path)
            print(
                f"[pass1] frame={index} seed={seed} seconds={elapsed:.1f} "
                f"size={image.size} -> {path}",
                flush=True,
            )
    except OutOfMemory:
        return 2

    n_before, _ = count_lora_modules(pipe.transformer)
    print(f"[lora] modules with a lora_A attribute before load: {n_before}", flush=True)
    print(f"[lora] path={LORA}", flush=True)

    t_l = time.time()
    pipe.load_lora_weights(LORA, adapter_name=ADAPTER_NAME)
    pipe.set_adapters([ADAPTER_NAME])
    print(f"[lora] loaded in {time.time()-t_l:.1f}s adapter_name={ADAPTER_NAME}", flush=True)

    n_after, lora_names = count_lora_modules(pipe.transformer)
    print(f"[lora] modules with a lora_A attribute after load: {n_after}", flush=True)
    print(f"[lora] example targets: {lora_names[:4]}", flush=True)

    # ---- Pass 2: LoRA present. The same seeds as pass 1. ----
    on_images = []
    pass2_times = []
    try:
        for index, body in enumerate(body_images):
            seed = args.seed + index
            start = time.time()
            image = generate(pipe, embeds, body, reference_image, args, seed)
            elapsed = time.time() - start
            pass2_times.append(elapsed)
            on_images.append(image)
            path = os.path.join(args.out_dir, f"faceswap_{index}_lora_on.png")
            image.save(path)
            print(
                f"[pass2] frame={index} seed={seed} seconds={elapsed:.1f} "
                f"size={image.size} -> {path}",
                flush=True,
            )
    except OutOfMemory:
        return 2

    print("[diff] comparing lora_off with lora_on", flush=True)
    for index, (off_image, on_image) in enumerate(zip(off_images, on_images)):
        mad, changed_pct, max_diff = compare(off_image, on_image)
        print(
            f"[diff] frame={index} seed_pass1={args.seed + index} "
            f"seed_pass2={args.seed + index} "
            f"mean_abs_pixel_diff={mad:.4f} max_abs_diff={max_diff:.1f} "
            f"pixels_changed={changed_pct:.2f}%",
            flush=True,
        )

    n_images = len(body_images)
    per_image_1 = sum(pass1_times) / n_images if n_images else 0.0
    per_image_2 = sum(pass2_times) / n_images if n_images else 0.0
    card_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"[peak] vram_max_allocated={torch.cuda.max_memory_allocated()/1024**3:.2f} GiB  "
          f"card_total={card_total:.2f} GiB", flush=True)
    print(f"[peak] ram_peak_sampled={peak_ram_gb():.2f} GiB  "
          f"ram_rss={ram_gb():.2f} GiB  ram_kernel_peak={true_peak_ram_gb():.2f} GiB",
          flush=True)
    print(f"[time] pass1 total={sum(pass1_times):.1f}s per_image={per_image_1:.2f}s "
          f"per_step={per_image_1 / max(1, args.steps):.3f}s", flush=True)
    print(f"[time] pass2 total={sum(pass2_times):.1f}s per_image={per_image_2:.2f}s "
          f"per_step={per_image_2 / max(1, args.steps):.3f}s", flush=True)

    print("[args] resolved values", flush=True)
    print(f"[args] width={args.width}", flush=True)
    print(f"[args] height={args.height}", flush=True)
    print(f"[args] ref_size={args.ref_size}", flush=True)
    print(f"[args] steps={args.steps}", flush=True)
    print(f"[args] seed={args.seed}", flush=True)
    print(f"[args] out_dir={args.out_dir}", flush=True)
    print(f"[args] work_dir={workdir}", flush=True)
    print(f"[args] frames={args.frames}", flush=True)
    print(f"[args] seeds={[args.seed + i for i in range(n_images)]}", flush=True)
    print(f"[args] video={VIDEO}", flush=True)
    print(f"[args] reference={REFERENCE}", flush=True)
    print(f"[args] repo={REPO}", flush=True)
    print(f"[args] encoder={ENCODER}", flush=True)
    print(f"[args] lora={LORA}", flush=True)
    print(f"[args] prompt={PROMPT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())