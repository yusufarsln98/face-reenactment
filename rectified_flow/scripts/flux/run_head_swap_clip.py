"""Run of consecutive two-image head-swap frames with FLUX.2-klein.

The script extracts one run of consecutive decoded frames from the source video
with ffmpeg. It keeps each frame as Picture 1 (the base image with the body), and
it keeps the portrait photo as Picture 2 (the reference face). It runs every
frame once with the BFS head-swap LoRA present, and it writes each result.

Picture 1 is the base image and comes FIRST in the image list. Picture 2 is the
reference face and comes SECOND. Reversing the two produces a plausible image
with the wrong head.

One fixed seed drives the whole run. The first frame, the last frame, and every
frame between them use the same seed. This keeps the head swap stable across the
clip. There is no seed per frame.

For each frame the script writes two 512x512 images. The --out-prefix value
replaces the leading faceswap of each name:

  <prefix>_body_%04d.png   the model input body frame
  <prefix>_frame_%04d.png  the swapped output frame

The %04d value is the decoded frame number from the source video. The script also
writes the 256x256 reference and a metrics file.

The script picks its frame numbers from --start and --count, or from --frames,
which names exact decoded frames. The flags --lora, --out-dir, --out-prefix and
--prompt-file let one run differ from another without a code change. Without a
flag, every value stays at the default of the recorded run.

The script reports the seconds per frame, the mean seconds per frame, the peak
VRAM, the peak RSS, and the frame count. Before it loads the LoRA, it runs one
forward with no adapter, on the first body frame. That forward is a warm-up. The
script throws its image away and counts no time for it. When a generation hits
OOM, the script reports the peak VRAM and stops with exit status 2. It never
retries at a lower resolution.

Run this script with the fluxenv interpreter, not the main project venv, because
diffusers is not installed in the main venv.
"""

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import tempfile
import time

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

OUT_PARTS = ("out", "video_report", "01_raw_faceswap")
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


def extract_frames(frame_numbers, workdir):
    """Decode the requested decoded frame numbers from the source video.

    The container frame rate is 600/19, so a naive dump writes more PNG files
    than the count of decoded frames. The script selects each frame by its
    decoded frame number instead. The venv has no video decoder, so ffmpeg is
    the only path. The argument is a list, because the video path holds spaces.

    The video carries a 90 degree rotation tag. ffmpeg auto-rotates on decode, so
    the extracted frames are already portrait. The script adds no transpose.

    The script verifies the exact numbered file names that it asked for, so only
    the files of this run count. A listing of the directory is not a count,
    because a directory can hold the files of another run.
    """
    select = "+".join(f"eq(n\\,{n})" for n in frame_numbers)
    pattern = os.path.join(workdir, "raw_frame_%04d.png")
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

    # Count only the exact numbered names that this run asked for. ffmpeg
    # numbers its output from --start_number, so the file for frame i of this run
    # is raw_frame_{i:04d}.png. A listing of the directory can also hold the
    # files of another run, so a listing count is wrong. The workdir is fresh for
    # every run, so no other file can share these names.
    paths = [
        os.path.join(workdir, f"raw_frame_{i:04d}.png")
        for i in range(len(frame_numbers))
    ]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        print(
            f"[frames] ffmpeg wrote no file for {len(missing)} of "
            f"{len(frame_numbers)} requested decoded frames in {workdir}; "
            f"missing={missing}",
            flush=True,
        )
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


def write_metrics(out_dir, args, frame_numbers, frame_times, peak_vram, peak_ram):
    """Write the clip metrics file and return its path."""
    mean_seconds = sum(frame_times) / len(frame_times)
    metrics = {
        "count": len(frame_numbers),
        "timings": {
            "seconds_per_frame": [round(t, 3) for t in frame_times],
            "total_seconds": round(sum(frame_times), 3),
            "mean_seconds_per_frame": round(mean_seconds, 3),
        },
        "vram": {"peak_allocated_gib": round(peak_vram, 3)},
        "ram": {"peak_rss_gib": round(peak_ram, 3)},
        "seed": int(args.seed),
        "start": int(frame_numbers[0]),
        "frames": [int(f) for f in frame_numbers],
    }
    metrics_path = os.path.join(out_dir, f"{args.out_prefix}_clip_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics_path


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run of consecutive two-image head-swap frames for FLUX.2-klein with "
            "the BFS head LoRA."
        )
    )
    ap.add_argument(
        "--start",
        type=int,
        help="the first decoded frame number in the source video",
    )
    ap.add_argument(
        "--count",
        type=int,
        help="how many consecutive decoded frames to swap",
    )
    ap.add_argument(
        "--frames",
        default=None,
        help=(
            "a comma-separated list of exact decoded frame numbers, for example "
            "175,222,270; it replaces --start and --count"
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="one fixed seed for the whole run (default: 0)",
    )
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--ref-size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument(
        "--lora",
        default=LORA,
        help="the head-swap LoRA file (default: the recorded BFS head v1 LoRA)",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help=(
            "the output directory, from the project root "
            "(default: out/video_report/01_raw_faceswap)"
        ),
    )
    ap.add_argument(
        "--out-prefix",
        default="faceswap",
        help="the name prefix of every output file (default: faceswap)",
    )
    ap.add_argument(
        "--prompt-file",
        default=None,
        help="a plain text file that holds the prompt (default: the built-in prompt)",
    )
    args = ap.parse_args()

    if args.frames is None:
        if args.start is None or args.count is None:
            ap.error("--start and --count are required unless --frames is given")
        if args.start < 0:
            ap.error("--start must not be negative")
        if args.count < 1:
            ap.error("--count must be at least 1")
        frame_numbers = [args.start + i for i in range(args.count)]
    else:
        try:
            frame_numbers = [
                int(part) for part in args.frames.split(",") if part.strip()
            ]
        except ValueError:
            ap.error("--frames must hold comma-separated decoded frame numbers")
        if not frame_numbers:
            ap.error("--frames must name at least one decoded frame number")
        if min(frame_numbers) < 0:
            ap.error("--frames must not hold a negative frame number")
    if "faceswap" not in args.out_prefix:
        ap.error("--out-prefix must contain faceswap")

    # The built-in prompt stays the default, character for character. Only
    # --prompt-file replaces it.
    prompt = PROMPT
    if args.prompt_file:
        prompt_path = os.path.abspath(os.path.join(PROJECT_ROOT, args.prompt_file))
        with open(prompt_path, encoding="utf-8") as handle:
            prompt = handle.read().strip()
        print(
            f"[prompt] file={prompt_path} chars={len(prompt)}",
            flush=True,
        )

    lora_path = os.path.abspath(os.path.join(PROJECT_ROOT, args.lora))

    if args.out_dir:
        out_dir = os.path.abspath(os.path.join(PROJECT_ROOT, args.out_dir))
    else:
        out_dir = os.path.abspath(os.path.join(PROJECT_ROOT, *OUT_PARTS))
    os.makedirs(out_dir, exist_ok=True)
    # One fresh workdir per run. The name holds the process id and the frame
    # range, and mkdtemp adds a random suffix, so the name cannot collide with a
    # workdir of an earlier run. ffmpeg writes raw_frame_%04d.png inside it, so
    # its output can never share a name with a file that is already on disk.
    workdir = tempfile.mkdtemp(
        prefix=(
            f"faceswap_frames_{os.getpid()}_"
            f"{frame_numbers[0]}_{len(frame_numbers)}_"
        ),
        dir=out_dir,
    )

    print("[run] FLUX.2-klein head swap clip run", flush=True)
    print(f"[run] out_dir={out_dir}", flush=True)
    print(f"[run] work_dir={workdir}", flush=True)
    print(
        f"[run] start={frame_numbers[0]} count={len(frame_numbers)} seed={args.seed} "
        f"width={args.width} height={args.height} ref_size={args.ref_size} "
        f"steps={args.steps}",
        flush=True,
    )
    print(f"[run] out_prefix={args.out_prefix}", flush=True)
    print(f"[run] lora={lora_path}", flush=True)
    print(f"[run] frame_numbers={frame_numbers}", flush=True)
    print(f"[run] ram={ram_gb():.2f} GiB", flush=True)

    frame_paths = extract_frames(frame_numbers, workdir)

    body_inputs = []
    for index, frame_no in enumerate(frame_numbers):
        body, source_size = load_body(frame_paths[index], args.width, args.height)
        body_path = os.path.join(
            out_dir, f"{args.out_prefix}_body_{frame_no:04d}.png"
        )
        body.save(body_path)
        body_inputs.append((frame_no, body))
        print(
            f"[frames] n={frame_no} -> {frame_paths[index]} source={source_size} "
            f"model_input={body.size} body_png={body_path}",
            flush=True,
        )
    del frame_paths

    reference_image = load_reference(args.ref_size)
    reference_path = os.path.join(out_dir, f"{args.out_prefix}_reference.png")
    reference_image.save(reference_path)
    print(
        f"[frames] reference -> {reference_path} model_input={reference_image.size} "
        f"(Picture 1 = body, Picture 2 = reference)",
        flush=True,
    )
    print(
        f"[ram] after inputs ram={ram_gb():.2f} GiB peak={peak_ram_gb():.2f} GiB",
        flush=True,
    )

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
    print(
        f"[load] transformer ok in {time.time()-t0:.1f}s  ram={ram_gb():.2f} GiB",
        flush=True,
    )

    vae = AutoencoderKLFlux2.from_pretrained(
        REPO, subfolder="vae", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(REPO, subfolder="tokenizer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        REPO, subfolder="scheduler"
    )

    t1 = time.time()
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        ENCODER, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True
    ).eval()
    print(
        f"[load] text encoder ok in {time.time()-t1:.1f}s  ram={ram_gb():.2f} GiB",
        flush=True,
    )

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
            prompt=prompt, device=torch.device("cpu"), max_sequence_length=512
        )
    print(
        f"[encode] done in {time.time()-t_enc:.1f}s  "
        f"shape={tuple(prompt_embeds.shape)}  prompt_chars={len(prompt)}",
        flush=True,
    )
    del enc_pipe, text_encoder
    gc.collect()
    print(
        f"[ram] after freeing text encoder ram={ram_gb():.2f} GiB "
        f"peak={peak_ram_gb():.2f} GiB",
        flush=True,
    )

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
    print(
        f"[load] on GPU. vram_alloc={vram_gb():.2f} GiB  ram={ram_gb():.2f} GiB "
        f"peak={peak_ram_gb():.2f} GiB",
        flush=True,
    )

    # One forward with no adapter loaded, on the first body frame. This is the
    # warm-up. The verified script runs the same pass before it loads the LoRA,
    # and one pass brings the first call down from about 47 s to about 16 s. The
    # image is thrown away. It is not saved, it is not a frame, and it adds no
    # entry to the frame list or to the metrics.
    warm_frame_no, warm_body = body_inputs[0]
    warm_start = time.time()
    try:
        warm_image = generate(
            pipe, embeds, warm_body, reference_image, args, args.seed
        )
    except OutOfMemory:
        return 2
    print(
        f"[warmup] one no-adapter forward n={warm_frame_no} seed={args.seed} "
        f"seconds={time.time()-warm_start:.1f} (warm-up, not a frame)",
        flush=True,
    )
    del warm_image

    # Load the BFS head-swap LoRA. One adapter, present for every frame in the
    # run. There is no LoRA-absent pass in this script.
    n_before, _ = count_lora_modules(pipe.transformer)
    t_l = time.time()
    pipe.load_lora_weights(lora_path, adapter_name=ADAPTER_NAME)
    pipe.set_adapters([ADAPTER_NAME])
    n_after, _ = count_lora_modules(pipe.transformer)
    print(
        f"[lora] loaded in {time.time()-t_l:.1f}s adapter_name={ADAPTER_NAME} "
        f"modules_before={n_before} modules_after={n_after}",
        flush=True,
    )
    print(f"[lora] path={lora_path}", flush=True)
    print(
        f"[lora] vram_alloc={vram_gb():.2f} GiB  ram={ram_gb():.2f} GiB",
        flush=True,
    )

    torch.cuda.reset_peak_memory_stats()

    frame_times = []
    try:
        for frame_no, body in body_inputs:
            start = time.time()
            image = generate(pipe, embeds, body, reference_image, args, args.seed)
            elapsed = time.time() - start
            frame_times.append(elapsed)
            frame_path = os.path.join(
                out_dir, f"{args.out_prefix}_frame_{frame_no:04d}.png"
            )
            image.save(frame_path)
            print(
                f"[frame] n={frame_no} seed={args.seed} seconds={elapsed:.1f} "
                f"size={image.size} -> {frame_path}",
                flush=True,
            )
    except OutOfMemory:
        return 2

    mean_seconds = sum(frame_times) / len(frame_times)
    peak_vram_value = torch.cuda.max_memory_allocated() / 1024**3
    peak_ram_value = true_peak_ram_gb()
    card_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print("[summary]", flush=True)
    print(f"[summary] total_frames={len(frame_numbers)}", flush=True)
    print(f"[summary] mean_seconds_per_frame={mean_seconds:.2f}", flush=True)
    print(f"[summary] total_generate_seconds={sum(frame_times):.1f}", flush=True)
    print(
        f"[summary] peak_vram_allocated={peak_vram_value:.2f} GiB  "
        f"card_total={card_total:.2f} GiB",
        flush=True,
    )
    print(f"[summary] peak_rss_kernel={peak_ram_value:.2f} GiB", flush=True)
    print(f"[summary] out_dir={out_dir}", flush=True)

    metrics_path = write_metrics(
        out_dir, args, frame_numbers, frame_times, peak_vram_value, peak_ram_value
    )
    print(f"[summary] metrics -> {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
