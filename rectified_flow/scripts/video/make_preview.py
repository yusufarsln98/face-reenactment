"""Build an animated GIF preview and a labelled contact sheet from a frame range.

`make_preview.py` reads one numeric range of zero-padded PNG frames and makes two
report figures:

1. An animated GIF.  The wanted frames are staged in a throwaway temporary
   directory as a clean contiguous `%04d` sequence (source frame 175 becomes
   0001), then ffmpeg's two-pass palette method makes a 480-wide, 10 fps GIF.
   Feeding the staged sequence at 10 fps keeps every frame in the GIF, so the
   whole clip is shown at 96 frames.
2. A contact-sheet PNG.  A grid of 12 evenly spaced frames from the clip, each
   labelled with its source frame number.

Both artefacts let the report embed the motion without a video player.

Run from the project root:

    .venv/bin/python scripts/video/make_preview.py \
        --frames out/video_report/01_raw_faceswap \
        --start 175 --end 270 \
        --gif out/video_report/02_step1_reassembly/faceswap_step1_preview.gif \
        --sheet out/video_report/02_step1_reassembly/faceswap_step1_contact_sheet.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FRAMES = ROOT / "out" / "video_report" / "01_raw_faceswap"
DEFAULT_GIF = ROOT / "out" / "video_report" / "02_step1_reassembly" / "faceswap_step1_preview.gif"
DEFAULT_SHEET = ROOT / "out" / "video_report" / "02_step1_reassembly" / "faceswap_step1_contact_sheet.png"
DEFAULT_PATTERN = "faceswap_frame_*.png"

MIN_FRAME = 175
MAX_FRAME = 270

STAGED_PREFIX = "frame_"
PAD = 4

GIF_FPS = "10"
GIF_WIDTH = 480
SHEET_TILES = 12
SHEET_COLS = 4


def frame_number(name: str) -> int | None:
    """Return the last whole number in a file name, or None."""
    groups = re.findall(r"\d+", name)
    return int(groups[-1]) if groups else None


def gather_frames(directory: Path, pattern: str, lo: int, hi: int):
    """Return (ordered, missing): ordered [(number, path)] in [lo, hi], and the
    sorted numbers in [lo, hi] that had no matching file."""
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


def run(cmd: list[str], cwd: Path | None = None) -> int:
    """Print the exact command, then run it with its output shown in full."""
    print("$ " + shlex.join(cmd) + (f"    (cwd: {cwd})" if cwd else ""))
    sys.stdout.flush()
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        print(f"ERROR: command failed with exit code {result.returncode}", file=sys.stderr)
    return result.returncode


def make_gif(staging: Path, out: Path) -> int:
    """Run ffmpeg's two-pass palette method over the staged sequence."""
    filt = (f"fps={GIF_FPS},scale={GIF_WIDTH}:-1:flags=lanczos,"
            "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse")
    cmd = ["ffmpeg", "-y",
           "-framerate", GIF_FPS,
           "-start_number", "1",
           "-i", f"{STAGED_PREFIX}%0{PAD}d.png",
           "-vf", filt,
           str(out.resolve())]
    return run(cmd, cwd=staging)


def pick_tile_indices(clip_len: int, tiles: int) -> list[int]:
    """Return `tiles` evenly spaced clip indices over [0, clip_len-1]."""
    return sorted({int(round(x)) for x in np.linspace(0, clip_len - 1, tiles)})


def build_contact_sheet(ordered: list[tuple[int, Path]], out: Path,
                        tiles: int = SHEET_TILES, cols: int = SHEET_COLS) -> list[int]:
    """Write a grid of `tiles` evenly spaced frames, each labelled with its
    frame number.  Return the chosen frame numbers."""
    clip_len = len(ordered)
    indices = pick_tile_indices(clip_len, tiles)
    rows = math.ceil(tiles / cols)

    # Read the first chosen frame to learn the source cell size.
    probe = cv2.imread(str(ordered[indices[0]][1]))
    if probe is None:
        raise RuntimeError(f"cannot read frame image: {ordered[indices[0]][1]}")
    cell_h, cell_w = probe.shape[:2]
    label_h = max(40, cell_h // 8)

    sheet = np.full((rows * (cell_h + label_h), cols * cell_w, 3), 255, np.uint8)
    chosen_numbers: list[int] = []
    for tile, index in enumerate(indices):
        number, path = ordered[index]
        chosen_numbers.append(number)
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"cannot read frame {number}: {path}")
        if image.shape[:2] != (cell_h, cell_w):
            image = cv2.resize(image, (cell_w, cell_h))
        row, col = divmod(tile, cols)
        y, x = row * (cell_h + label_h), col * cell_w
        sheet[y:y + cell_h, x:x + cell_w] = image
        text = str(number)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        tx = x + (cell_w - tw) // 2
        ty = y + cell_h + (label_h + th) // 2
        cv2.putText(sheet, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    ok = cv2.imwrite(str(out), sheet)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {out}")
    return chosen_numbers


def gif_facts(path: Path) -> tuple[int, int, float]:
    """Return (byte size, decoded frame count, duration) of the GIF."""
    frames = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)])
    fmt = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_format", "-of", "json", str(path)]))
    return path.stat().st_size, int(frames.decode().strip()), float(fmt["format"]["duration"])


def main(argv: list[str] | None = None) -> int:
    """Run the preview builder from the command line."""
    parser = argparse.ArgumentParser(
        prog="make_preview.py",
        description="Make a palette-method GIF preview and a labelled contact sheet.",
    )
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES,
                        help=f"directory of PNG frames (default {DEFAULT_FRAMES})")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN,
                        help=f"glob pattern for the frames (default {DEFAULT_PATTERN})")
    parser.add_argument("--start", type=int, default=MIN_FRAME,
                        help=f"first frame number (default {MIN_FRAME})")
    parser.add_argument("--end", type=int, default=MAX_FRAME,
                        help=f"last frame number, inclusive (default {MAX_FRAME})")
    parser.add_argument("--gif", type=Path, default=DEFAULT_GIF,
                        help=f"output GIF path (default {DEFAULT_GIF})")
    parser.add_argument("--sheet", type=Path, default=DEFAULT_SHEET,
                        help=f"output contact-sheet PNG path (default {DEFAULT_SHEET})")
    args = parser.parse_args(argv)

    ordered, missing = gather_frames(args.frames, args.pattern, args.start, args.end)
    if missing:
        pretty = ", ".join(str(n) for n in missing)
        print(f"ERROR: missing {len(missing)} frame(s) in range "
              f"{args.start}-{args.end} matching {args.pattern!r}: {pretty}",
              file=sys.stderr)
        return 2
    print(f"using {len(ordered)} frames: {ordered[0][0]} -> {ordered[-1][0]} "
          f"(pattern {args.pattern!r} in {args.frames})")

    args.gif.parent.mkdir(parents=True, exist_ok=True)

    # GIF: stage the clean 0001..NNNN sequence, run ffmpeg there, then drop it.
    staging = Path(tempfile.mkdtemp(prefix="preview_tmp_", dir=str(args.gif.parent)))
    try:
        for index, (_number, source) in enumerate(ordered, start=1):
            shutil.copy2(source, staging / f"{STAGED_PREFIX}{index:0{PAD}d}.png")
        print(f"staged directory: {staging}")
        print(f"\n=== GIF ({GIF_WIDTH} wide, {GIF_FPS} fps) ===")
        rc = make_gif(staging, args.gif)
        if rc != 0:
            return rc
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        print(f"removed staging directory: {staging}")

    # Contact sheet: read the 12 chosen frames straight from the source.
    print(f"\n=== contact sheet ({SHEET_TILES} tiles) ===")
    args.sheet.parent.mkdir(parents=True, exist_ok=True)
    chosen = build_contact_sheet(ordered, args.sheet)
    for number in chosen:
        print(f"  tile frame {number}")

    # Read-backs.
    gif_bytes, gif_frames, gif_duration = gif_facts(args.gif)
    print("\n--- read-back ---")
    print(f"gif               : {args.gif}")
    print(f"gif size          : {gif_bytes} bytes")
    print(f"gif frame count   : {gif_frames}")
    print(f"gif duration (s)  : {gif_duration:.3f}")
    sheet_bytes = args.sheet.stat().st_size
    sheet_img = cv2.imread(str(args.sheet))
    sh, sw = (sheet_img.shape[:2] if sheet_img is not None else (None, None))
    print(f"contact sheet     : {args.sheet}")
    print(f"contact size      : {sheet_bytes} bytes")
    print(f"contact size (px) : {sw}x{sh}")
    print(f"contact tiles     : {len(chosen)} ({', '.join(map(str, chosen))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
