"""Step 2: the face restoration pass over a directory of swapped frames.

The stage reads an already-swapped frame sequence, restores the face in every
frame with GFPGANv1.4, and writes one PNG per input frame at the same 512x512
size.  The file name is `faceswap_step2_gfpgan_w<weight>_<frame:04d>.png`, and
`faceswap_step2_gfpgan_<paste>_w<weight>_<frame:04d>.png` when `--paste` names a
mode other than the default `feathered`.

The inference contract is a port of the official one, TencentARC/GFPGAN tag
v1.3.4 `gfpgan/utils.py` lines 114-129:

1. The aligned 512x512 BGR crop goes through `img2tensor(crop / 255.,
   bgr2rgb=True, float32=True)` and then `normalize(..., (0.5, 0.5, 0.5),
   (0.5, 0.5, 0.5))`.
2. `output = self.gfpgan(cropped_face_t, return_rgb=False)[0]`.  The call uses
   the defaults of `GFPGANv1Clean.forward`, so `randomize_noise=True`: the
   network draws fresh style noise for every call.
3. The output goes through `tensor2img(output.squeeze(0), rgb2bgr=True,
   min_max=(-1, 1))`, which clamps to [-1, 1], maps to [0, 255], rounds, and
   returns uint8 BGR.

The `--weight` value does NOT reach the network, and no blend is added here.
The official source has no arithmetic that reads it:

- v1.3.4, the tag that released `GFPGANv1.4.pth`, has no weight argument at all.
- v1.3.6 and later add `weight=0.5` to `GFPGANer.enhance` and pass it to the
  model, and `GFPGANv1Clean.forward(..., **kwargs)` absorbs that keyword and
  never reads it.

The value is checked for range, printed, and encoded in the output file name,
so the two runs stay apart in the folder.  Both runs produce the same restored
pixels apart from the fresh noise draw of step 2 above.

The face comes from the project's own machinery: `swapface.detect.detect_faces`
for the box and the five keypoints, the project gate for the one-face rule, and
`insightface.utils.face_align.norm_crop2` with a 512 crop, which is the call
`swapface.pipeline._align_crop` makes with a 128 crop for the swapper.

The restored crop goes back by one of two project composites, chosen with
`--paste`:

- `feathered` (the default) calls `swapface.pipeline.paste_back`.  It keeps the
  historical file name `faceswap_step2_gfpgan_w<weight>_<frame:04d>.png`.
- `region` calls `swapface.pipeline.paste_swapped` in its `region` mode, the
  BiSeNet region composite that the inswapper frame loop uses.  The parse keeps
  only the skin and the features, so the crop the stage pastes shrinks to that
  region.  The file name becomes
  `faceswap_step2_gfpgan_region_w<weight>_<frame:04d>.png`.

A frame with no face, or with more than one face, is copied through unchanged,
so the sequence stays complete.  The gate still refuses to choose between
faces; this stage adds no bypass.

Run from the project root:

    .venv/bin/python scripts/video/gfpgan_restore.py \
        --frames out/video_report/01_raw_faceswap --start 175 --end 270 \
        --out out/video_report/03_step2_restoration --weight 0.7
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

import swapface.cuda_paths

# REQUIRED before onnxruntime is imported.  Loads the NVIDIA cu12 libraries.
swapface.cuda_paths.apply()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from insightface.utils import face_align  # noqa: E402

from gfpgan_arch import GFPGANv1Clean  # noqa: E402
from swapface.detect import detect_faces  # noqa: E402
from swapface.gate import Outcome, evaluate  # noqa: E402
from swapface.pipeline import PASTE_REGION, paste_back, paste_swapped  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = ROOT / "models" / "GFPGANv1.4.pth"
DEFAULT_FRAMES = ROOT / "out" / "video_report" / "01_raw_faceswap"
DEFAULT_OUT = ROOT / "out" / "video_report" / "03_step2_restoration"
DEFAULT_PATTERN = "faceswap_frame_*.png"

# The frame numbers of the report clip.  The source folder holds two earlier
# probe frames numbered 0000 and 0001.
MIN_FRAME = 175
MAX_FRAME = 270

# GFPGANv1.4 takes a 512x512 aligned crop; the swapper takes 128.
RESTORE_SIZE = 512

# How the restored crop goes back onto the frame.  `feathered` is the project's
# `paste_back`, the blurred inner square.  `region` is the project's BiSeNet
# region composite, the same call the inswapper frame loop makes for its own
# crop: `paste_swapped(..., PASTE_REGION)`.  The default keeps the historical
# behaviour and the historical file name.
PASTE_FEATHERED = "feathered"
PASTE_MODES = (PASTE_FEATHERED, PASTE_REGION)

# The frame number in the output file name.
FRAME_DIGITS = 4

# The exact hyper-parameters that the official `GFPGANer` passes for the v1.4
# model: `arch='clean'`, `channel_multiplier=2`.
MODEL_ARGS = dict(
    out_size=512,
    num_style_feat=512,
    channel_multiplier=2,
    decoder_load_path=None,
    fix_decoder=False,
    num_mlp=8,
    input_is_latent=True,
    different_w=True,
    narrow=1,
    sft_half=True,
)


def weight_tag(weight: float) -> str:
    """Return the weight as it appears in the file name, e.g. 0.7 -> `w070`."""
    return "w" + f"{weight:.2f}".replace(".", "")


def output_name(weight: float, number: int, paste: str = PASTE_FEATHERED) -> str:
    """Return the output file name of one frame.

    The feathered paste keeps the historical name.  Any other paste mode puts
    its own word before the weight tag, so the two runs stay apart in one
    folder.
    """
    middle = "" if paste == PASTE_FEATHERED else f"{paste}_"
    return (
        f"faceswap_step2_gfpgan_{middle}{weight_tag(weight)}"
        f"_{number:0{FRAME_DIGITS}d}.png"
    )


def output_tag(weight: float, paste: str = PASTE_FEATHERED) -> str:
    """Return the output file name part that carries no frame number."""
    middle = "" if paste == PASTE_FEATHERED else f"{paste}_"
    return f"faceswap_step2_gfpgan_{middle}{weight_tag(weight)}"


def frame_number(name: str) -> int | None:
    """Return the last whole number in a file name, or None."""
    groups = re.findall(r"\d+", name)
    return int(groups[-1]) if groups else None


def gather_frames(directory: Path, pattern: str, lo: int, hi: int):
    """Return (ordered, missing) over the inclusive range [lo, hi].

    `ordered` holds (number, path) in ascending frame order.  `missing` holds
    the frame numbers in the range that have no file.
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


def crop_to_tensor(crop_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Return the network input of one 512x512 BGR uint8 crop.

    Port of the official two lines:
    `img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)` followed by
    `normalize(x, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`, from basicsr v1.4.2
    `basicsr/utils/img_util.py` and torchvision.
    """
    scaled = crop_bgr / 255.0
    rgb = cv2.cvtColor(scaled.astype(np.float32), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()
    tensor = (tensor - 0.5) / 0.5  # the official normalize call, in place
    return tensor.unsqueeze(0).to(device)


def tensor_to_bgr(output: torch.Tensor) -> np.ndarray:
    """Return the 512x512 BGR uint8 crop of one network output.

    Port of `tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))`
    from basicsr v1.4.2 `basicsr/utils/img_util.py`.
    """
    tensor = output.squeeze(0).float().detach().cpu().clamp_(-1.0, 1.0)
    tensor = (tensor - (-1.0)) / (1.0 - (-1.0))
    image = tensor.numpy().transpose(1, 2, 0)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return (image * 255.0).round().astype(np.uint8)


def build_model() -> GFPGANv1Clean:
    """Return an empty GFPGANv1Clean with the official v1.4 parameters."""
    return GFPGANv1Clean(**MODEL_ARGS)


def load_model(device: torch.device) -> tuple[GFPGANv1Clean, int]:
    """Return the model with the official weights and the tensor count.

    The function raises when the checkpoint is absent, when it holds no
    `params_ema` entry, or when the state dict does not match the ported
    architecture.  A partial load is never accepted.
    """
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"checkpoint not found: {CHECKPOINT}")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    if "params_ema" not in checkpoint:
        raise RuntimeError(
            f"{CHECKPOINT} holds no 'params_ema' entry. Keys: {sorted(checkpoint)}"
        )
    state = checkpoint["params_ema"]

    model = build_model()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"the checkpoint does not match the ported architecture: "
            f"{len(missing)} missing key(s), {len(unexpected)} unexpected key(s). "
            f"missing: {sorted(missing)[:10]} unexpected: {sorted(unexpected)[:10]}"
        )
    # The load the official code does.  It cannot fail now, and it proves it.
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    return model, len(state)


@torch.no_grad()
def restore_crop(model: GFPGANv1Clean, crop_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    """Return the restored 512x512 BGR uint8 crop of one aligned crop."""
    tensor = crop_to_tensor(crop_bgr, device)
    output = model(tensor, return_rgb=False)[0]
    return tensor_to_bgr(output)


def paste_restored(
    frame: np.ndarray,
    restored_crop: np.ndarray,
    matrix: np.ndarray,
    paste: str,
) -> np.ndarray:
    """Return the frame with the restored crop put back by the chosen mode.

    `PASTE_FEATHERED` calls `swapface.pipeline.paste_back` on the 512 crop.
    `PASTE_REGION` calls `swapface.pipeline.paste_swapped` in its region mode,
    the project's BiSeNet composite.  No compositor is written here.
    """
    if paste == PASTE_FEATHERED:
        return paste_back(frame, restored_crop, matrix)
    return paste_swapped(frame, restored_crop, matrix, PASTE_REGION)


def main(argv: list[str] | None = None) -> int:
    """Run the restoration pass from the command line."""
    parser = argparse.ArgumentParser(
        prog="gfpgan_restore.py",
        description="Restore the face in every frame of a swapped frame sequence.",
    )
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES,
                        help=f"directory of input frames (default {DEFAULT_FRAMES})")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN,
                        help=f"glob pattern for the input frames (default {DEFAULT_PATTERN})")
    parser.add_argument("--start", type=int, default=MIN_FRAME,
                        help=f"first frame number (default {MIN_FRAME})")
    parser.add_argument("--end", type=int, default=MAX_FRAME,
                        help=f"last frame number, inclusive (default {MAX_FRAME})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"output directory (default {DEFAULT_OUT})")
    parser.add_argument("--weight", type=float, default=0.7,
                        help="fidelity weight recorded in the output file name. Default: 0.7")
    parser.add_argument("--paste", choices=PASTE_MODES, default=PASTE_FEATHERED,
                        help="how the restored crop goes back: 'feathered' is the "
                             "project's paste_back inner square, 'region' is the "
                             "project's BiSeNet region composite (the inswapper "
                             "frame loop paste). Default: feathered")
    args = parser.parse_args(argv)

    if not 0.0 <= args.weight <= 1.0:
        print(f"ERROR: --weight must be inside [0, 1]. Got {args.weight}.", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This stage runs on the GPU only.",
              file=sys.stderr)
        return 2

    ordered, missing = gather_frames(args.frames, args.pattern, args.start, args.end)
    if missing:
        pretty = ", ".join(str(n) for n in missing)
        print(f"ERROR: missing {len(missing)} frame(s) in range {args.start}-{args.end} "
              f"matching {args.pattern!r}: {pretty}", file=sys.stderr)
        return 2

    device = torch.device("cuda")
    print(f"device           : {device}  {torch.cuda.get_device_name(0)}")
    print(f"torch            : {torch.__version__}  cuda {torch.version.cuda}")
    model, tensor_count = load_model(device)
    print(f"checkpoint       : {CHECKPOINT}")
    print(f"torch.load       : map_location='cpu', key 'params_ema'")
    print(f"tensors loaded   : {tensor_count}")
    print(f"strict=True load : missing 0, unexpected 0 (the run stops on any other count)")
    print(f"model parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"weight           : {args.weight} -> {weight_tag(args.weight)}")
    print(f"paste mode       : {args.paste}")
    print(f"output tag       : {output_tag(args.weight, args.paste)}")
    print("weight note      : the value is recorded in the file name only. The official "
          "v1.3.4 call has no weight argument, and from v1.3.6 the clean architecture "
          "absorbs the keyword in **kwargs and never reads it. No blend is added here.")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"input            : {len(ordered)} frame(s) {args.pattern!r} in {args.frames}")
    print(f"output directory : {args.out}")

    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    restored_count = 0
    copied: list[str] = []
    first_evidence: list[str] = []

    for number, source in ordered:
        frame = cv2.imread(str(source))
        if frame is None:
            print(f"ERROR: cannot read frame {number}: {source}", file=sys.stderr)
            return 2

        decision = evaluate(detect_faces(frame))
        target = args.out / output_name(args.weight, number, args.paste)

        if decision.outcome is not Outcome.PASS or decision.face is None:
            reason = ("no face detected" if decision.outcome is Outcome.SKIP
                      else "more than one face detected")
            shutil.copy2(source, target)
            copied.append(f"{number} ({reason})")
            print(f"  frame {number:4d}  {reason}; the input frame is copied unchanged",
                  flush=True)
            continue

        crop, matrix = face_align.norm_crop2(frame, decision.face.kps, RESTORE_SIZE)
        restored_crop = restore_crop(model, crop, device)
        restored = paste_restored(frame, restored_crop, matrix, args.paste)

        if not first_evidence:
            difference = np.abs(restored_crop.astype(np.float32) - crop.astype(np.float32))
            first_evidence.append(
                f"  frame {number}: input crop mean {crop.mean():.4f}, "
                f"restored crop mean {restored_crop.mean():.4f}, "
                f"mean abs difference {difference.mean():.4f}, "
                f"max abs difference {difference.max():.0f} (0-255)"
            )

        if not cv2.imwrite(str(target), restored):
            print(f"ERROR: cannot write {target}", file=sys.stderr)
            return 2
        restored_count += 1

    elapsed = time.perf_counter() - start_time
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    print("\n--- evidence ---")
    for line in first_evidence:
        print(line)
    print(f"restored frames  : {restored_count}")
    print(f"copied frames    : {len(copied)}")
    for line in copied:
        print(f"  copied {line}")
    print(f"peak VRAM        : allocated {peak_allocated / 2**20:.1f} MiB, "
          f"reserved {peak_reserved / 2**20:.1f} MiB")
    print(f"seconds          : {elapsed:.1f} total, "
          f"{elapsed / max(len(ordered), 1):.3f} per frame "
          f"({len(ordered)} frames)")
    print(f"pattern for the next stage: {output_tag(args.weight, args.paste)}_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
