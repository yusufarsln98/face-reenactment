"""Draw the two generation figures of the report as PNG diagrams.

`make_generation_figures.py` writes two presentation-grade figures for the
generation stage.  The wording follows section 3 of `final_report/REPORT.md`.

1. `faceswap_30_generation_architecture.png` - four columns, left to right:
   inputs, encoders, denoiser for the GPU, decode.
2. `faceswap_31_denoising_loop.png` - one generation as four denoising steps
   with conditioning that stays constant.

Every number in the figures is a measured value:

* body frame 512x512 -> latent 64x64x32 -> patchify 2x2 -> 128 channels
  -> about 1024 tokens;
* portrait 256x256 -> latent 32x32x32 -> about 256 tokens;
* prompt embeddings (1, 512, 7680);
* 4 denoising steps, one fixed seed, 16.3 s per frame, 96 frames;
* peak VRAM 3.60 of 4.00 GiB.

The figures need no GPU.  matplotlib and PIL are enough.  Run from the
project root:

    .venv/bin/python scripts/video/make_generation_figures.py
"""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "final_report"
ARCH_PATH = REPORT_DIR / "faceswap_30_generation_architecture.png"
LOOP_PATH = REPORT_DIR / "faceswap_31_denoising_loop.png"

DPI = 150
ARCH_SIZE_IN = (16.0, 9.0)
LOOP_SIZE_IN = (16.0, 7.0)
ARCH_ASPECT = 16.0 / 9.0
LOOP_ASPECT = 16.0 / 7.0

# Flat palette, one colour pair (face, edge) per class of box.
CPU = ("#dbe7f6", "#1b4f8a")
GPU = ("#ffe6c9", "#9a4b00")
DATA = ("#ffffff", "#3a3a3a")
NOTE = ("#f2f2f2", "#4d4d4d")
BAND = ("#e7eef6", "#31506e")
CALLOUT = ("#fff7e0", "#8a6a12")
INK = "#111111"
MUTED = "#454545"
ARROW = "#2b2b2b"

LABEL_WRAP = 34


def rounded(ax, x, y, w, h, colours, lw=1.8, radius=0.09, zorder=2):
    """Add one rounded box with a flat fill and a solid edge."""
    face, edge = colours
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def text(ax, x, y, body, size=11.0, bold=False, colour=INK, ha="center", va="center", linespacing=1.3, zorder=6):
    """Add one text block."""
    return ax.text(
        x,
        y,
        body,
        fontsize=size,
        fontweight="bold" if bold else "normal",
        color=colour,
        ha=ha,
        va=va,
        linespacing=linespacing,
        zorder=zorder,
    )


def arrow(ax, start, end, lw=1.8, scale=18, colour=ARROW, zorder=3):
    """Add one straight arrow between two points."""
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=scale,
        linewidth=lw,
        color=colour,
        shrinkA=0,
        shrinkB=0,
        connectionstyle="arc3,rad=0",
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def wrap(body, width=LABEL_WRAP):
    """Wrap a long label onto several lines."""
    return "\n".join(textwrap.wrap(body, width=width)) or body


def draw_architecture():
    """Column diagram: inputs, encoders, denoiser on the GPU, decode."""
    fig = plt.figure(figsize=ARCH_SIZE_IN, dpi=DPI, facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, 16.0)
    ax.set_ylim(0.0, 9.0)
    ax.set_axis_off()

    # Title, subtitle, legend.
    text(ax, 8.0, 8.76, "Generation: architecture and data flow", size=22, bold=True)
    text(
        ax,
        8.0,
        8.38,
        "Two-image head swap: FLUX.2-klein-4B plus a head-swap LoRA adapter on a 4 GB GPU",
        size=12.5,
        colour=MUTED,
    )
    rounded(ax, 12.30, 7.80, 3.50, 0.46, DATA, lw=1.2, radius=0.06)
    legend_x = 12.50
    for legend_name, colours in (("CPU stage", CPU), ("GPU stage", GPU), ("data", DATA)):
        rounded(ax, legend_x, 7.93, 0.26, 0.20, colours, lw=1.4, radius=0.045)
        text(ax, legend_x + 0.34, 8.03, legend_name, size=9.5, ha="left")
        legend_x += 1.16

    # Column headers.
    text(ax, 1.775, 7.58, "Inputs", size=14, bold=True)
    text(ax, 5.200, 7.58, "Encoders", size=14, bold=True)
    text(ax, 10.150, 7.58, "Denoiser on the GPU", size=14, bold=True)
    text(ax, 14.300, 7.58, "Decode", size=14, bold=True)

    # Column geometry.
    col_a_x, col_a_w = 0.20, 3.15
    col_b_x, col_b_w = 3.85, 2.70
    col_c_x, col_c_w = 8.05, 4.20
    col_d_x, col_d_w = 12.80, 3.00
    row = (6.375, 4.575, 2.775)
    box_h = 1.55

    # Column A: the three inputs.
    for centre, title, subtitle in (
        (row[0], "Portrait 256x256", "supplies identity"),
        (row[1], "Body frame 512x512", "supplies pose, lighting,\nbackground"),
        (row[2], "Instruction text: 409 chars", "fixed text that names\nPicture 1 (body) and\nPicture 2 (portrait)"),
    ):
        rounded(ax, col_a_x, centre - box_h / 2.0, col_a_w, box_h, DATA)
        text(ax, col_a_x + col_a_w / 2.0, centre + 0.38, title, size=13.0, bold=True)
        text(ax, col_a_x + col_a_w / 2.0, centre - 0.22, subtitle, size=10.0, colour=MUTED)

    # Column B, top: the VAE encoder, which covers both pictures.
    b1_y0, b1_y1 = row[1] - box_h / 2.0, row[0] + box_h / 2.0
    rounded(ax, col_b_x, b1_y0, col_b_w, b1_y1 - b1_y0, GPU)
    text(ax, col_b_x + col_b_w / 2.0, 5.98, "VAE encoder", size=13.0, bold=True)
    text(
        ax,
        col_b_x + col_b_w / 2.0,
        5.32,
        "pixels -> coarse latent grids\nfar cheaper to process than\npixels; the same VAE encodes\nboth pictures and decodes the\nresult back to an image",
        size=9.8,
        colour=MUTED,
        linespacing=1.45,
    )

    # Column B, bottom: the text encoder, on the CPU.
    b2_y0, b2_y1 = row[2] - box_h / 2.0, row[2] + box_h / 2.0
    rounded(ax, col_b_x, b2_y0, col_b_w, b2_y1 - b2_y0, CPU)
    text(ax, col_b_x + col_b_w / 2.0, 3.22, "Text encoder: Qwen3-4B", size=12.0, bold=True)
    text(
        ax,
        col_b_x + col_b_w / 2.0,
        2.72,
        "runs on the CPU, 4-bit\n409-char prompt in about 80 s\nonce per clip; freed before\ndenoising starts",
        size=9.8,
        colour=MUTED,
        linespacing=1.45,
    )

    # Column C: the transformer plus the LoRA adapter.
    c_y0, c_y1 = row[2] - box_h / 2.0, row[0] + box_h / 2.0
    rounded(ax, col_c_x, c_y0, col_c_w, c_y1 - c_y0, GPU, lw=2.2)
    c_mid = col_c_x + col_c_w / 2.0
    text(ax, c_mid, 6.88, "Denoising transformer", size=14.0, bold=True)
    text(ax, c_mid, 6.50, "FLUX.2-klein-4B, 4-bit GGUF, 2.60 GB", size=10.5, colour=MUTED)
    text(ax, c_mid, 6.22, "+ head-swap LoRA adapter, 80 adapted weights", size=10.5, colour=MUTED)

    rounded(ax, 8.55, 4.55, 3.20, 1.20, DATA, lw=1.4, radius=0.07)
    text(ax, c_mid, 5.42, "4 denoising steps", size=13.0, bold=True)
    text(ax, c_mid, 5.02, "FlowMatchEulerDiscreteScheduler", size=9.0, colour=MUTED)
    text(ax, c_mid, 4.75, "one fixed seed for the whole clip", size=9.0, colour=MUTED)
    text(ax, c_mid, 4.26, "conditioning enters every step", size=10.0, colour=MUTED)

    # Column D: the decoder and the output frame.
    rounded(ax, col_d_x, 4.55, col_d_w, 1.80, GPU)
    d_mid = col_d_x + col_d_w / 2.0
    text(ax, d_mid, 6.00, "VAE decoder", size=13.0, bold=True)
    text(
        ax,
        d_mid,
        5.42,
        "latent grid -> pixels\nthe same VAE as the encoder\nturns the latent into an image",
        size=10.0,
        colour=MUTED,
        linespacing=1.45,
    )
    rounded(ax, col_d_x, 2.55, col_d_w, 1.60, DATA, lw=2.6)
    text(ax, d_mid, 3.70, "512x512 RGB frame", size=14.0, bold=True)
    text(ax, d_mid, 3.22, "one frame per generation", size=10.5, colour=MUTED)

    # Arrows, column A to column B.
    for centre in row:
        arrow(ax, (col_a_x + col_a_w, centre), (col_b_x, centre))

    # Arrows, column B to column C, with the two latent shapes.
    arrow(ax, (col_b_x + col_b_w, 6.40), (col_c_x, 6.40))
    text(ax, 7.30, 6.85, "portrait latent\n32x32x32\nabout 256 tokens", size=9.5, colour=MUTED)
    arrow(ax, (col_b_x + col_b_w, 4.60), (col_c_x, 4.60))
    text(ax, 7.30, 4.00, "body latent\n64x64x32\npatchify 2x2\n-> 128 channels\nabout 1024 tokens", size=9.5, colour=MUTED)
    arrow(ax, (col_b_x + col_b_w, 2.86), (col_c_x, 3.06))
    text(ax, 7.30, 3.22, "prompt embeddings\n(1, 512, 7680)", size=9.5, colour=MUTED)

    # Noise into the top of the transformer, then the decode path.
    arrow(ax, (11.50, 7.54), (11.50, c_y1))
    text(ax, 11.30, 7.34, "noise in, one fixed seed", size=9.5, colour=MUTED, ha="right")
    arrow(ax, (col_c_x + col_c_w, 5.45), (col_d_x, 5.45))
    text(ax, 12.525, 5.64, "latent", size=9.0, colour=MUTED)
    arrow(ax, (d_mid, 4.55), (d_mid, 4.15))
    text(ax, d_mid + 0.15, 4.35, "decode", size=9.0, colour=MUTED, ha="left")

    # Footnote strip.
    rounded(ax, 0.20, 0.60, 15.60, 0.70, CALLOUT, lw=1.5, radius=0.07)
    text(
        ax,
        8.0,
        0.95,
        "16.3 s per frame   |   peak VRAM 3.60 of 4.00 GiB   |   96 frames per clip",
        size=14.5,
        bold=True,
    )

    fig.savefig(ARCH_PATH, dpi=DPI, facecolor="white")
    plt.close(fig)


def draw_denoising_loop():
    """Chain diagram: noise, four steps, decode, frame, and the conditioning."""
    fig = plt.figure(figsize=LOOP_SIZE_IN, dpi=DPI, facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, 16.0)
    ax.set_ylim(0.0, 7.0)
    ax.set_axis_off()

    text(ax, 8.0, 6.68, "One generation: the denoising loop", size=22, bold=True)
    text(
        ax,
        8.0,
        6.30,
        "Four steps from noise to a 512x512 RGB frame, with conditioning that never changes",
        size=12.5,
        colour=MUTED,
    )

    # The fixed seed feeds the noise only.
    rounded(ax, 0.35, 5.50, 2.10, 0.60, NOTE, lw=1.5, radius=0.07)
    text(ax, 1.40, 5.85, "fixed seed", size=12.0, bold=True)
    text(ax, 1.40, 5.62, "one seed for the clip", size=9.0, colour=MUTED)
    arrow(ax, (1.40, 5.50), (1.40, 5.15))

    # The chain of boxes.
    chain_y0, chain_y1 = 4.15, 5.15
    box_w, gap = 1.77, 0.515
    chain = (
        ("noise", "t = 1.00", NOTE),
        ("step 1", "t = 0.75", GPU),
        ("step 2", "t = 0.50", GPU),
        ("step 3", "t = 0.25", GPU),
        ("final latent", "t = 0.00", DATA),
        ("VAE decode", "latent -> pixels", GPU),
        ("frame", "512x512 RGB", DATA),
    )
    chain_x = []
    for index, (title, subtitle, colours) in enumerate(chain):
        x0 = 0.26 + index * (box_w + gap)
        chain_x.append(x0)
        rounded(ax, x0, chain_y0, box_w, chain_y1 - chain_y0, colours, lw=2.0 if index == len(chain) - 1 else 1.8)
        centre = x0 + box_w / 2.0
        text(ax, centre, 4.83, title, size=13.0, bold=True)
        text(ax, centre, 4.40, subtitle, size=10.0, colour=MUTED)

    # The chain arrows, with the Euler step labels between the steps.
    for index in range(len(chain) - 1):
        right = chain_x[index] + box_w
        left = chain_x[index + 1]
        arrow(ax, (right, 4.65), (left, 4.65))
        if index <= 3:
            text(ax, (right + left) / 2.0, 5.30, "Euler step", size=9.5, colour=MUTED)

    # The conditioning band, read by every step.
    rounded(ax, 0.26, 1.95, 15.48, 1.15, BAND, lw=1.8)
    text(ax, 0.50, 2.86, "conditioning, constant at every step", size=12.5, bold=True, ha="left")
    item_w, item_gap = 4.83, 0.30
    for index, item in enumerate(
        (
            "prompt embeddings (1, 512, 7680)",
            "portrait tokens about 256",
            "body tokens about 1024",
        )
    ):
        x0 = 0.45 + index * (item_w + item_gap)
        rounded(ax, x0, 2.05, item_w, 0.65, DATA, lw=1.4, radius=0.06)
        text(ax, x0 + item_w / 2.0, 2.375, item, size=11.5)
    for centre in (chain_x[1], chain_x[2], chain_x[3], chain_x[4]):
        arrow(ax, (centre + box_w / 2.0, 3.10), (centre + box_w / 2.0, chain_y0), lw=1.6, scale=16)

    # The annotation box.
    rounded(ax, 0.26, 0.30, 15.48, 1.30, CALLOUT, lw=1.5, radius=0.07)
    text(ax, 0.60, 1.36, "Why four steps", size=12.0, bold=True, ha="left")
    body = (
        "Each step the transformer predicts a flow direction from the noisy latent plus the conditioning, and the "
        "scheduler moves the latent a quarter of the remaining distance. Four steps suffice because the base model "
        "is step-distilled. Eight steps looked the same."
    )
    text(ax, 0.60, 1.14, wrap(body, width=155), size=10.5, colour=INK, ha="left", va="top", linespacing=1.5)

    fig.savefig(LOOP_PATH, dpi=DPI, facecolor="white")
    plt.close(fig)


def describe(path: Path, want_aspect: float) -> dict:
    """Print and return the checks for one written figure."""
    with Image.open(path) as image:
        width, height = image.size
    pixels = np.asarray(Image.open(path).convert("RGB"))
    non_white = float(np.mean(np.any(pixels < 250, axis=2)))
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    aspect = width / height

    print(f"{path.name}")
    print(f"  bytes            {len(payload)} ({len(payload) / 1024.0:.1f} KiB)")
    print(f"  pixels           {width} x {height}")
    print(f"  aspect           {aspect:.4f} (target {want_aspect:.4f})")
    print(f"  non-white        {non_white:.4f}")
    print(f"  sha256           {digest}")
    return {"bytes": len(payload), "width": width, "height": height, "aspect": aspect, "non_white": non_white}


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    draw_architecture()
    draw_denoising_loop()

    checks = (
        (ARCH_PATH, ARCH_ASPECT, (0.02, 0.60)),
        (LOOP_PATH, LOOP_ASPECT, (0.02, 0.60)),
    )
    ok = True
    for path, want_aspect, limits in checks:
        if not path.is_file():
            print(f"{path.name}: FAIL, the file does not exist")
            ok = False
            continue
        result = describe(path, want_aspect)
        if result["bytes"] <= 100 * 1024:
            print("  FAIL             the file is not larger than 100 KiB")
            ok = False
        if abs(result["aspect"] - want_aspect) > 0.02:
            print(f"  FAIL             the aspect is not near {want_aspect:.4f}")
            ok = False
        if not (limits[0] < result["non_white"] < limits[1]):
            print(f"  FAIL             the non-white fraction is outside {limits}")
            ok = False
    print("checks:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
