"""Smooth the temporal flicker out of a directory of face-swapped frames.

Two methods are available.  Both run on the CPU.  Both use numpy and OpenCV
only.  No torch, no ONNX Runtime, no downloaded weights and no GPU.

`deflicker` is the cheap baseline.  For every pixel it blends the frame toward
the temporal mean of a small window of neighbouring frames:

    out_t = (1 - strength) * x_t + strength * mean(x over the window)

The window holds 3 frames, the frame itself and its two closest neighbours.
At the end of the range the window holds the frames that exist.  The method
removes high-frequency temporal noise.  It also blurs motion, because it
averages the same pixel position and not the same surface point.

`flow` is motion compensated, after the Ebsynth idea.  For frame t it
estimates the dense optical flow between frame t and frame t-1, and between
frame t and frame t+1.  It warps each neighbour into frame t with
`cv2.remap`.  Then it blends the three frames:

    out_t = (x_t + 0.5 * strength * warped_t-1
                 + 0.5 * strength * warped_t+1) / (1 + strength)

The current frame keeps the highest weight, 1.0.  Each warped neighbour gets
`strength * 0.5`.  The weights are renormalised by their sum, so the output
stays in range.  The neighbour frames are sampled at the same surface point,
so a moving head does not smear.  The first and the last frame of the range
have one neighbour only.  Those frames blend with the single neighbour.

The flow sign convention was fixed by a test, not by guesswork.  A synthetic
shift of +5 pixels along x gives a DIS flow of +5.  So for
`flow = dis.calc(grey_a, grey_b)`, the map `identity + flow` warps frame b
onto the grid of frame a.  The test gave a mean absolute grey difference of
0.09 after the warp, against 13.28 before it.

The script writes one output frame per input frame as
`faceswap_step3_<method>_<frame:04d>.png`.  It also writes a small run JSON
with the settings, the timing and three CPU self-checks.

Run from the project root:

    .venv/bin/python scripts/video/smooth_temporal.py \
        --frames out/video_report/01_raw_faceswap \
        --pattern "faceswap_frame_*.png" \
        --min-frame 175 --max-frame 270 \
        --out out/video_report/04_step3_smoothing \
        --method deflicker --strength 0.6
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FRAMES = ROOT / "out" / "video_report" / "01_raw_faceswap"
DEFAULT_OUT = ROOT / "out" / "video_report" / "04_step3_smoothing"
DEFAULT_PATTERN = "faceswap_frame_*.png"

MIN_FRAME = 175
MAX_FRAME = 270

# Output naming: faceswap_step3_<method>_<frame:04d>.png
OUT_PREFIX = "faceswap_step3"
PAD = 4

# The deflicker window.  A window of 3 covers the frame and its two closest
# neighbours.  A window of 5 is the other allowed size, but it costs more
# motion sharpness.  This script uses 3.
DEFLICKER_WINDOW = 3

DEFAULT_STRENGTH = 0.6

# The DIS preset named by the task.
DIS_PRESET = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
DIS_PRESET_NAME = "DISOpticalFlow PRESET_MEDIUM"


def frame_number(name: str) -> int | None:
    """Return the last whole number in a file name, or None."""
    groups = re.findall(r"\d+", name)
    return int(groups[-1]) if groups else None


def gather_frames(directory: Path, pattern: str, lo: int, hi: int):
    """Return (ordered, missing) for the frame numbers in [lo, hi].

    `ordered` holds (number, path) pairs in ascending frame order.
    `missing` holds the numbers in the range with no matching file.
    """
    number_to_path: dict[int, Path] = {}
    for path in directory.glob(pattern):
        number = frame_number(path.name)
        if number is not None:
            number_to_path[number] = path
    ordered: list[tuple[int, Path]] = []
    missing: list[int] = []
    for number in range(lo, hi + 1):
        if number in number_to_path:
            ordered.append((number, number_to_path[number]))
        else:
            missing.append(number)
    return ordered, missing


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Clamp a float image to 0-255, round it and return uint8."""
    return np.clip(image, 0.0, 255.0).round().astype(np.uint8)


def read_frames(ordered: list[tuple[int, Path]]) -> list[np.ndarray]:
    """Read every frame once, in order."""
    frames: list[np.ndarray] = []
    for number, path in ordered:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"cannot read frame {number}: {path}")
        if image.shape[:2] != (512, 512):
            print(f"WARNING: frame {number} is {image.shape[1]}x{image.shape[0]}, "
                  "expected 512x512.", file=sys.stderr)
        frames.append(image)
    return frames


# ---------------------------------------------------------------------------
# Method 1: deflicker
# ---------------------------------------------------------------------------


def smooth_deflicker(frames: list[np.ndarray], strength: float,
                     window: int) -> list[np.ndarray]:
    """Blend every frame toward the temporal mean of its window."""
    n = len(frames)
    radius = window // 2
    outputs: list[np.ndarray] = []
    for index in range(n):
        lo = max(0, index - radius)
        hi = min(n, index + radius + 1)
        window_mean = np.stack([f.astype(np.float32) for f in frames[lo:hi]]).mean(axis=0)
        blended = ((1.0 - strength) * frames[index].astype(np.float32)
                   + strength * window_mean)
        outputs.append(to_uint8(blended))
    return outputs


# ---------------------------------------------------------------------------
# Method 2: flow
# ---------------------------------------------------------------------------


def warp_neighbour(dis, current_grey: np.ndarray, neighbour_grey: np.ndarray,
                   neighbour_bgr: np.ndarray, grid_x: np.ndarray,
                   grid_y: np.ndarray) -> np.ndarray:
    """Warp the neighbour frame onto the grid of the current frame.

    `flow = dis.calc(current, neighbour)` follows the DIS convention that was
    verified with a synthetic shift: `map = identity + flow` samples the
    neighbour at the point that matches the current frame.
    """
    flow = dis.calc(current_grey, neighbour_grey, None)
    return cv2.remap(neighbour_bgr, grid_x + flow[..., 0], grid_y + flow[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def smooth_flow(frames: list[np.ndarray], strength: float) -> list[np.ndarray]:
    """Blend every frame with its two motion-compensated neighbours."""
    n = len(frames)
    height, width = frames[0].shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                 np.arange(height, dtype=np.float32))
    greys = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    dis = cv2.DISOpticalFlow_create(DIS_PRESET)

    neighbour_weight = strength * 0.5
    outputs: list[np.ndarray] = []
    for index in range(n):
        accumulated = frames[index].astype(np.float32)
        weight_sum = 1.0
        for neighbour in (index - 1, index + 1):
            if 0 <= neighbour < n:
                warped = warp_neighbour(dis, greys[index], greys[neighbour],
                                        frames[neighbour], grid_x, grid_y)
                accumulated += neighbour_weight * warped.astype(np.float32)
                weight_sum += neighbour_weight
        outputs.append(to_uint8(accumulated / weight_sum))
    return outputs


# ---------------------------------------------------------------------------
# CPU self-checks
# ---------------------------------------------------------------------------


def self_checks(raw: list[np.ndarray], out: list[np.ndarray]) -> dict[str, object]:
    """Measure three CPU-only facts about the output against the input.

    1. The mean absolute difference from the raw frame, over all frames.
    2. The whole-frame inter-frame change, raw and smoothed.
    3. The mean Laplacian variance, raw and smoothed.  A lower value means
       less high-frequency detail.  The measure mixes noise removal with
       softness, so read it next to the other two.
    """
    diffs = [float(np.abs(o.astype(np.float32) - r.astype(np.float32)).mean())
             for r, o in zip(raw, out)]

    raw_grey = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in raw]
    out_grey = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in out]

    def interframe_change(greys: list[np.ndarray]) -> float:
        return float(np.mean([np.abs(greys[i] - greys[i + 1]).mean()
                              for i in range(len(greys) - 1)]))

    def laplacian_variance(frames: list[np.ndarray]) -> float:
        return float(np.mean([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY),
                                            cv2.CV_64F).var() for f in frames]))

    raw_change = interframe_change(raw_grey)
    out_change = interframe_change(out_grey)
    raw_lap = laplacian_variance(raw)
    out_lap = laplacian_variance(out)
    return {
        "mean_abs_diff_from_raw_frame": {
            "mean": float(np.mean(diffs)),
            "median": float(np.median(diffs)),
            "max": float(np.max(diffs)),
        },
        "interframe_grey_change_full_frame": {
            "raw": raw_change,
            "out": out_change,
            "out_over_raw": out_change / raw_change if raw_change else None,
        },
        "laplacian_variance": {
            "raw": raw_lap,
            "out": out_lap,
            "out_over_raw": out_lap / raw_lap if raw_lap else None,
        },
    }


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run(args) -> int:
    """Smooth the frames and write the output frames plus the run JSON."""
    ordered, missing = gather_frames(args.frames, args.pattern,
                                     args.min_frame, args.max_frame)
    if missing:
        pretty = ", ".join(str(n) for n in missing)
        print(f"ERROR: missing {len(missing)} frame(s) in range "
              f"{args.min_frame}-{args.max_frame} matching {args.pattern!r}: {pretty}",
              file=sys.stderr)
        return 2
    if not ordered:
        print(f"ERROR: no frames in range {args.min_frame}-{args.max_frame} "
              f"matching {args.pattern!r} in {args.frames}", file=sys.stderr)
        return 2
    if not 0.0 <= args.strength <= 1.0:
        print(f"ERROR: --strength must be in [0, 1], got {args.strength}", file=sys.stderr)
        return 2

    print(f"method            : {args.method}")
    print(f"strength          : {args.strength}")
    if args.method == "deflicker":
        print(f"window            : {DEFLICKER_WINDOW} frames")
    else:
        print(f"flow              : {DIS_PRESET_NAME}  (identity + flow, warped with remap)")
    print(f"frames            : {args.frames}  pattern {args.pattern!r}")
    print(f"range             : {args.min_frame}-{args.max_frame}  "
          f"using {len(ordered)} frame(s), {ordered[0][0]} -> {ordered[-1][0]}")

    raw = read_frames(ordered)

    t0 = time.perf_counter()
    if args.method == "deflicker":
        out = smooth_deflicker(raw, args.strength, DEFLICKER_WINDOW)
    else:
        out = smooth_flow(raw, args.strength)
    smooth_seconds = time.perf_counter() - t0

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    failed: list[int] = []
    for index, (number, _path) in enumerate(ordered):
        target = args.out / f"{args.out_prefix}_{args.method}_{number:0{PAD}d}.png"
        if cv2.imwrite(str(target), out[index]):
            written += 1
        else:
            failed.append(number)

    checks = self_checks(raw, out)
    seconds_per_frame = smooth_seconds / len(ordered)
    run_record: dict[str, object] = {
        "method": args.method,
        "out_prefix": args.out_prefix,
        "strength": args.strength,
        "window": DEFLICKER_WINDOW if args.method == "deflicker" else None,
        "flow": DIS_PRESET_NAME if args.method == "flow" else None,
        "source_directory": str(args.frames.resolve()),
        "source_pattern": args.pattern,
        "range": [args.min_frame, args.max_frame],
        "frame_count": len(ordered),
        "frame_numbers": [n for n, _p in ordered],
        "out_directory": str(args.out.resolve()),
        "out_pattern": f"{args.out_prefix}_{args.method}_*.png",
        "frames_written": written,
        "frames_failed": failed,
        "smooth_seconds": round(smooth_seconds, 4),
        "seconds_per_frame": round(seconds_per_frame, 6),
        "self_checks": checks,
    }
    json_path = args.out / f"{args.out_prefix}_{args.method}_run.json"
    with open(json_path, "w") as handle:
        json.dump(run_record, handle, indent=2)

    def fmt(value: object) -> str:
        return "null" if value is None else f"{float(value):.4f}"

    print(f"\nframes written    : {written} of {len(ordered)}  (failed: {failed or 'none'})")
    print(f"smoothing seconds : {smooth_seconds:.3f}  "
          f"({seconds_per_frame:.4f} s per frame, compute only)")
    print("\nCPU self-checks")
    print(f"  abs diff from raw frame : mean {fmt(checks['mean_abs_diff_from_raw_frame']['mean'])}"
          f"  median {fmt(checks['mean_abs_diff_from_raw_frame']['median'])}"
          f"  max {fmt(checks['mean_abs_diff_from_raw_frame']['max'])}")
    change = checks["interframe_grey_change_full_frame"]
    print(f"  inter-frame grey change : raw {fmt(change['raw'])} -> out {fmt(change['out'])}"
          f"  (out/raw {fmt(change['out_over_raw'])})")
    lap = checks["laplacian_variance"]
    print(f"  Laplacian variance      : raw {fmt(lap['raw'])} -> out {fmt(lap['out'])}"
          f"  (out/raw {fmt(lap['out_over_raw'])})")
    print(f"\nrun JSON          : {json_path}")
    print(f"HEADLINE {args.method} @ strength {args.strength}: "
          f"{seconds_per_frame:.4f} s/frame | "
          f"inter-frame change x{fmt(change['out_over_raw'])} | "
          f"Laplacian variance x{fmt(lap['out_over_raw'])}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser."""
    parser = argparse.ArgumentParser(
        prog="smooth_temporal.py",
        description="Smooth temporal flicker in a directory of 512x512 PNG frames. "
                    "CPU only, numpy and OpenCV only.",
    )
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES,
                        help=f"directory of PNG frames (default {DEFAULT_FRAMES})")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN,
                        help=f"glob pattern for the frames (default {DEFAULT_PATTERN!r})")
    parser.add_argument("--min-frame", type=int, default=MIN_FRAME,
                        help=f"lowest frame number to keep (default {MIN_FRAME})")
    parser.add_argument("--max-frame", type=int, default=MAX_FRAME,
                        help=f"highest frame number to keep (default {MAX_FRAME})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"output directory (default {DEFAULT_OUT})")
    parser.add_argument("--out-prefix", default=OUT_PREFIX,
                        help=f"prefix of the output file names (default {OUT_PREFIX!r})")
    parser.add_argument("--method", required=True, choices=("deflicker", "flow"),
                        help="deflicker: temporal mean blend. flow: motion compensated blend.")
    parser.add_argument("--strength", type=float, default=DEFAULT_STRENGTH,
                        help=f"blend strength in [0, 1] (default {DEFAULT_STRENGTH})")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the smoothing from the command line."""
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
