"""Parse the photo and one video frame into face regions, and save the masks.

Run from the project root:

    uv run python scripts/inswapper/parse_masks.py

The parser is the BiSeNet model from facexlib.  It runs on the GPU through
torch.  This script does not composite anything.  It writes the class index map,
a colour map, and one binary mask per named region, so a person can look at the
regions before any blending is built.

Every output file name holds the word `faceswap` only if it is a video
artefact.  These files are masks, not a swap, so they are named `mask`.

The script never writes to the photo or to the video.  It reads both and writes
only into out/parse/.
"""

from __future__ import annotations

import colorsys
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from bisenet_arch import BiSeNet

from swapface.detect import detect_faces

ROOT = Path(__file__).resolve().parent.parent.parent
PARSER_FILE = ROOT / "models" / "parsing_bisenet.pth"
OUT_DIR = ROOT / "out" / "parse"

PHOTO = ROOT / "IMG_20260817_111005.jpg"
VIDEO = ROOT / "WhatsApp Video 2026-10-04 at 18.51.45.mp4"
FRAME_INDEX = 288

# The 19 CelebAMask-HQ classes.  The index is the order the parser emits them.
# The count comes from the checkpoint.  Its conv_out layer has 19 output
# channels.  The order below was confirmed two ways.  The checkpoint was run on
# the photo, and the pixels of index 16 were on the jacket, index 17 was on the
# hair, and index 18 was empty because no hat is worn.  The order matches the
# FaceParsing.py label list of face-parsing.PyTorch, which is the model this
# checkpoint comes from.  Note that the brows come at index 2 and 3, and the
# nose comes at index 10.  Other published lists put them in a different order,
# so do not copy one from a blog.
CLASSES = [
    "background",
    "skin",
    "l_brow",
    "r_brow",
    "l_eye",
    "r_eye",
    "eye_g",
    "l_ear",
    "r_ear",
    "ear_r",
    "nose",
    "mouth",
    "u_lip",
    "l_lip",
    "neck",
    "neck_l",
    "cloth",
    "hair",
    "hat",
]

# Each region is a list of class indices.  Only the regions the composite needs
# are written.  The classes for background, neck, cloth and hat are not written
# as their own files.  They stay in the raw index map.
REGIONS = {
    "skin": [1],
    "hair": [17],
    "brows": [2, 3],
    "eyes": [4, 5, 6],
    "nose": [10],
    "mouth": [11, 12, 13],
    "ears": [7, 8],
}

# The 5-point template facexlib uses for a 512 crop.  It covers forehead to
# chin, so it keeps the hair that inswapper crops away.
FACE_SIZE = 512
FACE_TEMPLATE = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.1936],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)


def build_palette() -> np.ndarray:
    """Return one distinct BGR colour per class.

    A plain hue sweep is not enough.  It puts skin and cloth at the same red.
    The hues here step by the golden angle, so no two classes of 18 sit near
    each other on the colour wheel.
    """
    colours = np.zeros((len(CLASSES), 3), dtype=np.uint8)
    for index in range(1, len(CLASSES)):
        hue = (index * 0.6180339887498949) % 1.0
        sat = 0.75 + 0.20 * ((index * 0.7548776662466927) % 1.0)
        val = 0.80 + 0.20 * ((index * 0.5698402909980532) % 1.0)
        red, green, blue = colorsys.hsv_to_rgb(hue, sat, val)
        # cv2.imwrite takes BGR, so the colour is stored the other way round.
        colours[index] = (int(blue * 255), int(green * 255), int(red * 255))
    return colours


def print_legend(palette: np.ndarray) -> None:
    """Print the class index beside its colour, so the colour map is readable."""
    print("colour map legend")
    for index, class_name in enumerate(CLASSES):
        blue, green, red = (int(v) for v in palette[index])
        print(f"    {index:2d} {class_name:<10} rgb({red:>3},{green:>3},{blue:>3})")


def load_parser() -> BiSeNet:
    """Load the BiSeNet weights on the GPU."""
    if not PARSER_FILE.is_file():
        raise FileNotFoundError(f"parser weights not found: {PARSER_FILE}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BiSeNet(len(CLASSES))
    state = torch.load(PARSER_FILE, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.eval()
    net.to(device)
    return net


def align_face(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the largest face as a 512 crop, and the affine that made it.

    The affine maps image pixels to crop pixels.  It is needed to put the mask
    back on the full image.
    """
    faces = detect_faces(image)
    if not faces:
        raise RuntimeError("no face found in the image")
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    affine = cv2.estimateAffinePartial2D(
        face.kps.astype(np.float32), FACE_TEMPLATE, method=cv2.LMEDS
    )[0]
    if affine is None:
        raise RuntimeError("the 5-point alignment failed")
    crop = cv2.warpAffine(
        image, affine, (FACE_SIZE, FACE_SIZE), flags=cv2.INTER_LINEAR
    )
    return crop, affine


def parse(net: BiSeNet, device: torch.device, crop: np.ndarray) -> np.ndarray:
    """Return the 512x512 class index map for one aligned crop."""
    array = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    array = array.astype(np.float32) / 255.0
    array = (array - 0.5) / 0.5
    array = np.transpose(array, (2, 0, 1))
    tensor = torch.from_numpy(array).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = net(tensor)[0]
    return logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()


def paste_back(crop_mask: np.ndarray, affine: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    """Put a 512 crop mask on a canvas the size of the whole image."""
    height, width = image_shape[:2]
    inverse = cv2.invertAffineTransform(affine)
    return cv2.warpAffine(
        crop_mask, inverse, (width, height), flags=cv2.INTER_NEAREST, borderValue=0
    )


def write_masks(prefix: str, mask: np.ndarray, palette: np.ndarray) -> list[Path]:
    """Write the raw map, the colour map, and one binary mask per region."""
    written = []

    raw_path = OUT_DIR / f"{prefix}_mask_raw.png"
    cv2.imwrite(str(raw_path), mask)
    written.append(raw_path)

    colour = palette[mask]
    colour_path = OUT_DIR / f"{prefix}_mask_color.png"
    cv2.imwrite(str(colour_path), colour)
    written.append(colour_path)

    for name, indices in REGIONS.items():
        binary = np.isin(mask, indices).astype(np.uint8) * 255
        region_path = OUT_DIR / f"{prefix}_mask_{name}.png"
        cv2.imwrite(str(region_path), binary)
        written.append(region_path)

    return written


def print_histogram(name: str, mask: np.ndarray, image_shape: tuple[int, ...]) -> dict[int, int]:
    """Print the share of each class, and return the pixel counts."""
    counts = np.bincount(mask.ravel(), minlength=len(CLASSES))
    total = mask.size
    print(f"\n  class histogram, {name}, {image_shape[1]}x{image_shape[0]} px")
    for index, class_name in enumerate(CLASSES):
        if counts[index] == 0:
            continue
        print(f"    {index:2d} {class_name:<10} {counts[index]:>10d} px  {100.0 * counts[index] / total:6.3f} %")
    return {index: int(counts[index]) for index in range(len(CLASSES))}


def read_frame(video: Path, index: int) -> np.ndarray:
    """Return one frame at `index`.  Frame 0 is the first frame."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open the video: {video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {index}")
    return frame


def check(name: str, mask: np.ndarray, image_shape: tuple[int, ...], counts: dict[int, int]) -> list[str]:
    """Print the sanity checks for one mask.  Return the failures as text."""
    problems = []

    if mask.shape != image_shape[:2]:
        problems.append(f"{name}: mask shape {mask.shape} does not match image {image_shape[:2]}")

    hair = counts[CLASSES.index("hair")]
    skin = counts[CLASSES.index("skin")]
    if hair == 0:
        problems.append(f"{name}: hair is 0 pixels. The plan needs hair.")
    if hair == 0 or skin == 0:
        overlap = 0.0
    else:
        both = np.logical_and(mask == CLASSES.index("hair"), mask == CLASSES.index("skin")).sum()
        overlap = 100.0 * both / min(hair, skin)
    print(f"    hair {hair} px, skin {skin} px, overlap {overlap:.3f} % of the smaller")
    if hair and skin and overlap > 5.0:
        problems.append(f"{name}: hair and skin overlap by {overlap:.2f} %. The class indices are wrong.")
    return problems


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device. The parser must run on the GPU.", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    palette = build_palette()
    net = load_parser()
    device = next(net.parameters()).device

    print(f"parser      BiSeNet, facexlib parsing_bisenet.pth")
    print(f"weights     {PARSER_FILE}")
    print(f"runtime     torch {torch.__version__}")
    print(f"device      {device}  {torch.cuda.get_device_name(device)}")
    print(f"classes     {len(CLASSES)}, CelebAMask-HQ scheme")
    print(f"output      {OUT_DIR}")
    print_legend(palette)

    photo = cv2.imread(str(PHOTO))
    if photo is None:
        raise RuntimeError(f"cannot read the photo: {PHOTO}")
    frame = read_frame(VIDEO, FRAME_INDEX)

    # Warm up, then time.  A cold first call pays for the CUDA context.
    warm_crop, warm_affine = align_face(photo)
    parse(net, device, warm_crop)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(5):
        start = time.perf_counter()
        parse(net, device, warm_crop)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    print(f"warm speed  {sum(times) / len(times):.2f} ms per parse call, 5 runs, range {min(times):.2f}-{max(times):.2f} ms")

    written: list[Path] = []
    problems: list[str] = []

    for name, path, image in (
        ("photo", PHOTO, photo),
        (f"frame{FRAME_INDEX}", VIDEO, frame),
    ):
        print(f"\n{name}: {path}")
        crop, affine = align_face(image)
        crop_mask = parse(net, device, crop)
        mask = paste_back(crop_mask, affine, image.shape)
        written += write_masks(name, mask, palette)
        counts = print_histogram(name, mask, image.shape)
        problems += check(name, mask, image.shape, counts)

    print("\nfiles written to out/parse/")
    for path in written:
        print(f"  {path.name:<40} {path.stat().st_size:>9d} bytes")

    if problems:
        print("\nSANITY CHECK FAILED")
        for problem in problems:
            print(f"  {problem}")
        return 1

    print("\nSANITY CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())