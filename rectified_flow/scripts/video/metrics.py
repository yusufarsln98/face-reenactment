"""Measure how much a face-swapped frame directory flickers.

`metrics.py` runs three metric families over one directory of same-size PNG
frames, in frame-number order.  It writes one JSON, one CSV and one hair-mask
PNG, and it prints every number and a short headline.

The three families:

1. Face-region inter-frame change.  For each consecutive pair, the mean
   absolute grey difference (0-255) inside the largest detected face box.
2. Identity consonance.  The ArcFace cosine of each frame to the portrait, and
   the cosine of each frame to the next frame.
3. Hair temporal noise.  The per-pixel grey standard deviation across the whole
   temporal axis, inside ONE hair mask built by the BiSeNet parser.

Run from the project root:

    .venv/bin/python scripts/video/metrics.py --frames out/video_report/01_raw_faceswap \
        --pattern "faceswap_body_*.png" --label source_body_crop
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FRAMES = ROOT / "out" / "video_report" / "01_raw_faceswap"
DEFAULT_REFERENCE = ROOT / "IMG_20260817_111005.jpg"
DEFAULT_OUT = ROOT / "out" / "video_report" / "01_raw_faceswap" / "metrics"

# The frame numbers that hold.  Everything outside this range is dropped.  The
# directory holds two earlier probe frames numbered 0000 and 0001.
MIN_FRAME = 175
MAX_FRAME = 270

# ---------------------------------------------------------------------------
# GPU setup.  cuda_paths.apply() loads the NVIDIA cu12 shared libraries.  It
# runs before onnxruntime is imported anywhere below, per the project rule.
# ---------------------------------------------------------------------------

import swapface.cuda_paths

swapface.cuda_paths.apply()

# Import the GPU modules only after the apply call.  models.py imports
# insightface and onnxruntime and calls cuda_paths.apply() itself.
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from swapface.detect import detect_faces  # noqa: E402
from swapface.models import load_face_analysis  # noqa: E402
from swapface.pipeline import _read_reference_embedding  # noqa: E402
from swapface.region_mask import class_index, get_parser, parse_crop  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def frame_number(name: str) -> int | None:
    """Return the last whole number in a file name, or None."""
    groups = re.findall(r"\d+", name)
    return int(groups[-1]) if groups else None


def gather_frames(directory: Path, pattern: str, lo: int, hi: int) -> tuple[list[int], list[tuple[int, Path]], list[tuple[int, Path]]]:
    """Return (used, kept, excluded) frame tuples sorted by frame number.

    `used` is the list of frame numbers inside [lo, hi].  `kept` holds
    (number, path) for those.  `excluded` holds the matches outside the range.
    """
    matches: list[tuple[int, Path]] = []
    for path in directory.glob(pattern):
        number = frame_number(path.name)
        if number is None:
            continue
        matches.append((number, path))
    kept = sorted(((n, p) for n, p in matches if lo <= n <= hi), key=lambda t: t[0])
    excluded = sorted(((n, p) for n, p in matches if not (lo <= n <= hi)), key=lambda t: t[0])
    return [n for n, _ in kept], kept, excluded


def box_area(box: tuple[int, int, int, int]) -> int:
    """Return the pixel area of a face box."""
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def clip_box(box: tuple[int, int, int, int], h: int, w: int) -> tuple[int, int, int, int]:
    """Return a face box clipped to the frame."""
    x1, y1, x2, y2 = box
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def largest_face(frame_bgr: np.ndarray) -> tuple[tuple[int, int, int, int] | None, np.ndarray | None]:
    """Return (largest bbox, normed embedding) of the largest face, or Nones.

    The bbox comes from swapface.detect.detect_faces, as the task requires.
    The embedding comes from the same recognizer that detect_faces uses, run on
    the same frame.  Both lists are empty when the frame holds no face.
    """
    boxes = detect_faces(frame_bgr)
    region: tuple[int, int, int, int] | None = None
    if boxes:
        face = max(boxes, key=lambda b: box_area(b.bbox))
        region = face.bbox
    embedding: np.ndarray | None = None
    try:
        faces, _provider = load_face_analysis()
        found = faces.get(frame_bgr)
        if found:
            big = max(found, key=lambda f: box_area(tuple(int(v) for v in f.bbox)))
            emb = np.asarray(big.normed_embedding, dtype=np.float32)
            if emb.shape == (512,):
                embedding = emb
    except Exception as error:  # pragma: no cover - reported by the caller
        print(f"WARNING: recognizer failed on a frame: {error!r}", file=sys.stderr)
    return region, embedding


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity of two vectors."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def stats(values: list[float | None]) -> dict[str, object]:
    """Return mean, median, max and 95th percentile of the non-null values."""
    clean = [float(v) for v in values if v is not None]
    count = len(clean)
    if count == 0:
        return {"mean": None, "median": None, "max": None, "p95": None, "count": 0, "null_count": len([v for v in values if v is None])}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "p95": float(np.percentile(arr, 95)),
        "count": int(count),
        "null_count": int(len([v for v in values if v is None])),
    }


# ---------------------------------------------------------------------------
# The work
# ---------------------------------------------------------------------------


def run_metrics(frames_dir: Path, pattern: str, label: str, out_dir: Path,
                reference: Path, lo: int, hi: int) -> dict[str, object]:
    """Compute the three metric families and write JSON, CSV and the mask."""
    numbers, kept, excluded = gather_frames(frames_dir, pattern, lo, hi)
    if not kept:
        print(f"[{label}] no frames in range {lo}-{hi}. "
              f"matched {len(kept) + len(excluded)} file(s), used 0, excluded {len(excluded)}.")
        return {}
    print(f"[{label}] matched {len(kept) + len(excluded)} file(s), "
          f"excluded {len(excluded)} outside {lo}-{hi}, using {len(kept)} frame(s).")
    if excluded:
        bad = ", ".join(str(n) for n, _ in excluded)
        print(f"[{label}] excluded frame numbers: {bad}")

    # Read every frame once, in order, and keep a grey copy for the metrics.
    bgr_frames: list[np.ndarray] = []
    grey_frames: list[np.ndarray] = []
    for number, path in kept:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"cannot read frame {number}: {path}")
        if image.shape[:2] != (512, 512):
            print(f"WARNING: frame {number} is {image.shape[1]}x{image.shape[0]}, "
                  "expected 512x512.", file=sys.stderr)
        bgr_frames.append(image)
        grey_frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32))
    n = len(kept)

    # Load the detector and the recognizer once, and confirm a real GPU run.
    app, detector_provider = load_face_analysis()
    detector_session_providers = list(app.det_model.session.get_providers())
    parser_net, parser_device = get_parser()

    # For every frame: the face region (from detect_faces) and the embedding.
    regions: list[tuple[int, int, int, int] | None] = []
    embeddings: list[np.ndarray | None] = []
    t0 = time.perf_counter()
    for index in range(n):
        region, embedding = largest_face(bgr_frames[index])
        regions.append(region)
        embeddings.append(embedding)
    detect_seconds = time.perf_counter() - t0

    faces_without = [numbers[i] for i in range(n) if regions[i] is None or embeddings[i] is None]
    if faces_without:
        print(f"[{label}] frames with no detected face: {', '.join(map(str, faces_without))}")
    else:
        print(f"[{label}] every one of the {n} frame(s) yielded a detected face.")

    # Family 1: face-region inter-frame change.
    change_values: list[float | None] = []
    change_pairs: list[list[int]] = []
    change_region_pixels: list[int | None] = []
    for index in range(n - 1):
        a, b = index, index + 1
        change_pairs.append([numbers[a], numbers[b]])
        if regions[a] is None or regions[b] is None:
            change_values.append(None)
            change_region_pixels.append(None)
            continue
        height, width = grey_frames[a].shape
        x1, y1, x2, y2 = clip_box(regions[a], height, width)
        if x2 <= x1 or y2 <= y1:
            change_values.append(None)
            change_region_pixels.append(None)
            continue
        diff = np.abs(grey_frames[a][y1:y2, x1:x2] - grey_frames[b][y1:y2, x1:x2])
        change_values.append(float(diff.mean()))
        change_region_pixels.append(int((x2 - x1) * (y2 - y1)))
    change_stats = stats(change_values)
    region_counts = [p for p in change_region_pixels if p]
    region_median = float(np.median(region_counts)) if region_counts else None

    # Family 2: identity consonance.
    reference_embedding = _read_reference_embedding(reference)
    reference_resolved = reference.resolve()
    to_portrait_values: list[float | None] = []
    for index in range(n):
        if embeddings[index] is None:
            to_portrait_values.append(None)
        else:
            to_portrait_values.append(cosine(embeddings[index], reference_embedding))
    consecutive_values: list[float | None] = []
    consecutive_pairs: list[list[int]] = []
    for index in range(n - 1):
        consecutive_pairs.append([numbers[index], numbers[index + 1]])
        if embeddings[index] is None or embeddings[index + 1] is None:
            consecutive_values.append(None)
        else:
            consecutive_values.append(cosine(embeddings[index], embeddings[index + 1]))
    to_portrait_stats = stats(to_portrait_values)
    consecutive_stats = stats(consecutive_values)

    # Family 3: hair temporal noise.
    representative_index = next((i for i in range(n) if regions[i] is not None), 0)
    representative_number = numbers[representative_index]
    argmax = parse_crop(parser_net, parser_device, bgr_frames[representative_index])
    hair_class = class_index("hair")
    hair_mask = argmax == hair_class
    hair_pixels = int(hair_mask.sum())

    grey_stack = np.stack(grey_frames, axis=0)  # (N, 512, 512)
    std_map = grey_stack.std(axis=0)  # (512, 512)
    hair_std = std_map[hair_mask]
    if hair_std.size:
        hair_noise = {
            "mean": float(hair_std.mean()),
            "median": float(np.median(hair_std)),
            "p95": float(np.percentile(hair_std, 95)),
            "count": int(hair_std.size),
        }
    else:
        hair_noise = {"mean": None, "median": None, "p95": None, "count": 0}

    # Write outputs.
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / f"faceswap_hair_mask_{label}.png"
    cv2.imwrite(str(mask_path), (hair_mask.astype(np.uint8) * 255))

    gpu_confirmed = (
        detector_provider == "CUDAExecutionProvider"
        and "CUDAExecutionProvider" in detector_session_providers
        and any(r is not None for r in regions)
        and any(e is not None for e in embeddings)
        and str(parser_device) != "cpu"
    )

    result: dict[str, object] = {
        "label": label,
        "directory": str(frames_dir.resolve()),
        "pattern": pattern,
        "reference": str(reference_resolved),
        "frame_count": n,
        "frame_numbers": numbers,
        "frames_directory_resolved": str(frames_dir.resolve()),
        "excluded_outside_range": [num for num, _ in excluded],
        "range": [lo, hi],
        "min_frame": lo,
        "max_frame": hi,
        "used_frame_numbers": numbers,
        "detector_provider": detector_provider,
        "detector_session_providers": detector_session_providers,
        "parser_device": str(parser_device),
        "gpu_inference_confirmed": bool(gpu_confirmed),
        "frames_without_face": faces_without,
        "recognitions_seconds": round(detect_seconds, 3),
        "face_region_interframe_change": {
            "mean": change_stats["mean"],
            "median": change_stats["median"],
            "max": change_stats["max"],
            "p95": change_stats["p95"],
            "num_pairs": change_stats["count"],
            "null_pairs": [
                change_pairs[i] for i in range(len(change_values)) if change_values[i] is None
            ],
            "region_pixels_median": region_median,
            "raw": change_values,
            "pairs": change_pairs,
            "region_pixels": change_region_pixels,
        },
        "identity_consonance": {
            "to_portrait": {
                **to_portrait_stats,
                "raw": to_portrait_values,
                "frame_numbers": numbers,
            },
            "consecutive": {
                **consecutive_stats,
                "raw": consecutive_values,
                "pairs": consecutive_pairs,
            },
        },
        "hair_temporal_noise": {
            "mean": hair_noise["mean"],
            "median": hair_noise["median"],
            "p95": hair_noise["p95"],
            "hair_pixel_count": hair_pixels,
            "hair_class_index": hair_class,
            "representative_frame_number": representative_number,
            "mask_path": str(mask_path.resolve()),
            "raw_per_pixel_std": [round(float(x), 4) for x in hair_std.tolist()] if hair_std.size else [],
        },
    }

    json_path = out_dir / f"faceswap_metrics_{label}.json"
    csv_path = out_dir / f"faceswap_metrics_{label}.csv"
    with open(json_path, "w") as handle:
        json.dump(result, handle, indent=2)

    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "key", "value"])
        writer.writerow(["meta", "label", label])
        writer.writerow(["meta", "directory", result["directory"]])
        writer.writerow(["meta", "pattern", pattern])
        writer.writerow(["meta", "reference", reference_resolved])
        writer.writerow(["meta", "frame_count", n])
        writer.writerow(["meta", "detector_provider", detector_provider])
        writer.writerow(["meta", "parser_device", parser_device])
        writer.writerow(["meta", "gpu_inference_confirmed", bool(gpu_confirmed)])
        for key in ("mean", "median", "max", "p95"):
            writer.writerow(["face_region_interframe_change", key, change_stats[key]])
        writer.writerow(["face_region_interframe_change", "region_pixels_median", region_median])
        for key in ("mean", "median", "max", "p95"):
            writer.writerow(["identity_to_portrait", key, to_portrait_stats[key]])
        for key in ("mean", "median", "max", "p95"):
            writer.writerow(["identity_consecutive", key, consecutive_stats[key]])
        for key in ("mean", "median", "p95"):
            writer.writerow(["hair_temporal_noise", key, hair_noise[key]])
        writer.writerow(["hair_temporal_noise", "hair_pixel_count", hair_pixels])

    print_report(result, json_path, csv_path, mask_path)
    return result


def print_report(result: dict[str, object], json_path: Path, csv_path: Path, mask_path: Path) -> None:
    """Print every number, then the three headline numbers."""
    label = result["label"]
    fac = result["face_region_interframe_change"]
    to_portrait = result["identity_consonance"]["to_portrait"]
    consecutive = result["identity_consonance"]["consecutive"]
    hair = result["hair_temporal_noise"]

    def fmt(value: object) -> str:
        return "null" if value is None else f"{float(value):.4f}"

    print(f"\n===== {label} =====")
    print(f"directory         : {result['directory']}")
    print(f"pattern           : {result['pattern']}")
    print(f"reference portrait: {result['reference']}")
    print(f"frame_count       : {result['frame_count']}  "
          f"range {result['range'][0]}-{result['range'][1]}  "
          f"first/last {result['frame_numbers'][0]}/{result['frame_numbers'][-1]}")
    print(f"detector provider : {result['detector_provider']}  "
          f"session {result['detector_session_providers']}")
    print(f"parser device     : {result['parser_device']}")
    print(f"gpu confirmed     : {result['gpu_inference_confirmed']}")
    print(f"detection seconds : {result['recognitions_seconds']}")

    print("\n1. face-region inter-frame change (mean abs grey diff, 0-255)")
    print(f"   mean {fmt(fac['mean'])}  median {fmt(fac['median'])}  "
          f"max {fmt(fac['max'])}  p95 {fmt(fac['p95'])}  "
          f"pairs {fac['num_pairs']}  null {len(fac['null_pairs'])}")
    print(f"   face-region pixels (median over pairs): {fac['region_pixels_median']}")

    print("\n2. identity consonance (ArcFace cosine)")
    print(f"   to portrait  : mean {fmt(to_portrait['mean'])}  "
          f"median {fmt(to_portrait['median'])}  max {fmt(to_portrait['max'])}  "
          f"p95 {fmt(to_portrait['p95'])}")
    print(f"   consecutive  : mean {fmt(consecutive['mean'])}  "
          f"median {fmt(consecutive['median'])}  max {fmt(consecutive['max'])}  "
          f"p95 {fmt(consecutive['p95'])}")

    print("\n3. hair temporal noise (per-pixel grey std across time, 0-255)")
    print(f"   mean {fmt(hair['mean'])}  median {fmt(hair['median'])}  "
          f"p95 {fmt(hair['p95'])}")
    print(f"   hair pixels {hair['hair_pixel_count']}  "
          f"(class {hair['hair_class_index']} on frame {hair['representative_frame_number']})")
    print(f"   mask: {mask_path}")

    print(f"\nfiles: {json_path.name}  {csv_path.name}  {mask_path.name}")
    print(f"\nHEADLINE  {label}: "
          f"face change mean {fmt(fac['mean'])} | "
          f"consecutive identity mean {fmt(consecutive['mean'])} | "
          f"identity-to-portrait mean {fmt(to_portrait['mean'])} | "
          f"hair noise mean {fmt(hair['mean'])}")


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser."""
    parser = argparse.ArgumentParser(
        prog="metrics.py",
        description="Measure flicker and identity in a directory of 512x512 face frames.",
    )
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES,
                        help=f"directory of PNG frames (default {DEFAULT_FRAMES})")
    parser.add_argument("--pattern", required=True,
                        help='glob pattern for the frames, e.g. "faceswap_frame_*.png"')
    parser.add_argument("--label", required=True,
                        help="short label used in the output file names")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"output directory (default {DEFAULT_OUT})")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
                        help=f"portrait JPG that supplies the identity (default {DEFAULT_REFERENCE})")
    parser.add_argument("--min-frame", type=int, default=MIN_FRAME,
                        help=f"lowest frame number to keep (default {MIN_FRAME})")
    parser.add_argument("--max-frame", type=int, default=MAX_FRAME,
                        help=f"highest frame number to keep (default {MAX_FRAME})")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the metrics from the command line."""
    args = build_parser().parse_args(argv)
    result = run_metrics(
        frames_dir=args.frames,
        pattern=args.pattern,
        label=args.label,
        out_dir=args.out,
        reference=args.reference,
        lo=args.min_frame,
        hi=args.max_frame,
    )
    return 0 if result else 0


if __name__ == "__main__":
    raise SystemExit(main())
