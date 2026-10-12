"""Reassemble one numeric range of PNG frames into an MP4 with ffmpeg.

`reassemble.py` reads a consecutive zero-padded frame range from a directory,
stages the wanted frames in ascending frame order inside a throwaway temporary
directory with sequential `%04d` names, and encodes them to an MP4.  When an
audio source is given, its audio is extracted to a WAV and muxed into the same
command.  The output is composed at the frame size: no scale, no pad.

Why the temporary directory.  The source directory holds two stray probe frames
numbered 0000 and 0001 that sit before the clip's first frame (175).  Feeding
the source `%04d` pattern directly would pull in the strays or need a fragile
offset.  Copying only the wanted frames to a fresh directory as 0001..0096
gives one clean contiguous sequence, so `-start_number 1` maps source frame 175
to sequence index 0001 and source frame 270 to 0096.  That directory is removed
when the run ends.

Run from the project root:

    .venv/bin/python scripts/video/reassemble.py \
        --frames out/video_report/01_raw_faceswap \
        --start 175 --end 270 --fps 29.969 \
        --audio "WhatsApp Video 2026-10-04 at 18.51.45.mp4" \
        --out out/video_report/02_step1_reassembly/faceswap_step1_reassembly.mp4
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FRAMES = ROOT / "out" / "video_report" / "01_raw_faceswap"
DEFAULT_OUT = ROOT / "out" / "video_report" / "02_step1_reassembly" / "faceswap_step1_reassembly.mp4"
DEFAULT_AUDIO = ROOT / "WhatsApp Video 2026-10-04 at 18.51.45.mp4"
DEFAULT_PATTERN = "faceswap_frame_*.png"

MIN_FRAME = 175
MAX_FRAME = 270
DEFAULT_FPS = "29.969"

# Staged temp file names: frame_0001.png .. frame_0096.png, a clean contiguous
# sequence that starts at 0001, so -start_number 1 is always correct.
STAGED_PREFIX = "frame_"
PAD = 4
WAV_NAME = "audio.wav"


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


def human_bytes(n: int) -> str:
    """Return a byte count as a short human string."""
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{n} B"


def probe(path: Path) -> dict:
    """Read the output back with ffprobe and return the facts we print."""
    fmt = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_format", "-of", "json", str(path)]))
    streams = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)]))["streams"]
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    nb = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)])
    return {
        "size_bytes": path.stat().st_size,
        "duration": float(fmt["format"]["duration"]),
        "width": video["width"] if video else None,
        "height": video["height"] if video else None,
        "video_codec": video["codec_name"] if video else None,
        "video_frames": nb.decode().strip(),
        "avg_frame_rate": video.get("avg_frame_rate") if video else None,
        "audio_codec": audio["codec_name"] if audio else None,
        "audio_sample_rate": audio.get("sample_rate") if audio else None,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the reassembly from the command line."""
    parser = argparse.ArgumentParser(
        prog="reassemble.py",
        description="Reassemble a numeric PNG frame range into an MP4 with ffmpeg.",
    )
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES,
                        help=f"directory of PNG frames (default {DEFAULT_FRAMES})")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN,
                        help=f"glob pattern for the frames (default {DEFAULT_PATTERN})")
    parser.add_argument("--start", type=int, default=MIN_FRAME,
                        help=f"first frame number (default {MIN_FRAME})")
    parser.add_argument("--end", type=int, default=MAX_FRAME,
                        help=f"last frame number, inclusive (default {MAX_FRAME})")
    parser.add_argument("--fps", default=DEFAULT_FPS,
                        help=f"output frame rate (default {DEFAULT_FPS})")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO,
                        help=f"source with the audio to mux (default {DEFAULT_AUDIO})")
    parser.add_argument("--no-audio", action="store_true",
                        help="encode video only, no audio track")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"output MP4 path (default {DEFAULT_OUT})")
    args = parser.parse_args(argv)

    audio_source = None if args.no_audio else args.audio.resolve()
    if audio_source is not None and not audio_source.exists():
        print(f"ERROR: audio source does not exist: {audio_source}", file=sys.stderr)
        return 2

    # Collect the wanted frames; fail before any work if any are missing.
    ordered, missing = gather_frames(args.frames, args.pattern, args.start, args.end)
    if missing:
        pretty = ", ".join(str(n) for n in missing)
        print(f"ERROR: missing {len(missing)} frame(s) in range "
              f"{args.start}-{args.end} matching {args.pattern!r}: {pretty}",
              file=sys.stderr)
        return 2
    count = len(ordered)
    print(f"staging {count} frames: {ordered[0][0]} -> {ordered[-1][0]} "
          f"(pattern {args.pattern!r} in {args.frames})")

    # The output directory holds the throwaway staging directory too.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="reassemble_tmp_", dir=str(args.out.parent)))
    try:
        # Copy only the wanted frames, renamed into a clean 0001..NNNN sequence.
        for index, (_number, source) in enumerate(ordered, start=1):
            target = staging / f"{STAGED_PREFIX}{index:0{PAD}d}.png"
            shutil.copy2(source, target)
        print(f"staged directory: {staging}")

        if audio_source is not None:
            rc = run(["ffmpeg", "-y", "-i", str(audio_source),
                      "-vn", "-acodec", "pcm_s16le", WAV_NAME], cwd=staging)
            if rc != 0:
                return rc

        # Compose at the frame size (512x512). No scale, no pad.
        cmd = ["ffmpeg", "-y",
               "-framerate", str(args.fps),
               "-start_number", "1",
               "-i", f"{STAGED_PREFIX}%0{PAD}d.png"]
        if audio_source is not None:
            cmd += ["-i", WAV_NAME]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]
        if audio_source is not None:
            cmd += ["-c:a", "aac", "-shortest"]
        cmd += [str(args.out.resolve())]
        rc = run(cmd, cwd=staging)
        if rc != 0:
            return rc
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        print(f"removed staging directory: {staging}")

    facts = probe(args.out)
    print("\n--- read-back with ffprobe ---")
    print(f"output            : {args.out}")
    print(f"file size         : {facts['size_bytes']} bytes ({human_bytes(facts['size_bytes'])})")
    print(f"duration (s)      : {facts['duration']:.3f}")
    print(f"resolution        : {facts['width']}x{facts['height']}")
    print(f"frame rate        : {facts['avg_frame_rate']}")
    print(f"video codec       : {facts['video_codec']}")
    print(f"video frame count : {facts['video_frames']}")
    print(f"audio codec       : {facts['audio_codec'] or 'none'}")
    if facts["audio_codec"]:
        print(f"audio sample rate : {facts['audio_sample_rate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
