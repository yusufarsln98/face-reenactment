"""Composite the swapper output through a face-parsing region mask.

`inswapper_128.onnx` receives the subject crop and a 512-value identity vector.
It does not receive, and it cannot produce, hair.  An elliptical paste therefore
leaves the subject's own hair and beard anchored while the face geometry under
them changes, so the hairline reads stretched.

This module follows the documented FaceFusion recipe
(`face_masker.create_region_mask`):

1. Parse the SWAPPED crop with BiSeNet, at the size the parser was trained at.
2. Build a boolean mask with `numpy.isin` over the class indices to keep.
3. Blur the mask with `cv2.GaussianBlur(mask, (0, 0), sigma=5)`.
4. Re-threshold the blur, so the alpha has a hard edge:
   `np.clip(blur * 2 - 0.5, 0, 1)`.
5. Warp the alpha and the swapped crop back to the frame with the transform
   that inswapper returned.
6. Paste with a linear alpha, not Poisson blending.

The classes to keep are the skin and the features.  Hair, cloth, hat, neck and
ears stay as they are in the original frame.  Ears are on the keep list on
purpose.  The class indices come from the vendored parser in `scripts/`; they
were confirmed on the photo and on a video frame by `scripts/inswapper/parse_masks.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = ROOT / "scripts" / "inswapper"
if str(_SCRIPTS) not in sys.path:
    # The parser model is vendored under scripts/, beside the script that uses it.
    # It is imported here, not copied, so there is one parser in the project.
    sys.path.insert(0, str(_SCRIPTS))

from parse_masks import CLASSES, load_parser, parse  # noqa: E402

# The classes that come FROM the swapper: skin, both brows, both eyes, nose,
# mouth, upper lip, lower lip.
SWAP_IDS = frozenset({1, 2, 3, 4, 5, 10, 11, 12, 13})

# The classes that stay from the ORIGINAL frame: background, glasses, both ears,
# ear ring, neck, neck lace, cloth, hair, hat.  Hair is 17 and cloth is 16.
KEEP_IDS = frozenset({0, 6, 7, 8, 9, 14, 15, 16, 17, 18})

# The FaceFusion recipe uses a blur of sigma 5 on the 512 mask.  The same value
# is used here, on the map the parser returns, so the edge lands in the same
# place.
BLUR_SIGMA = 5.0

# A morphological close of this many pixels fills the small parser holes at the
# hairline.  It is applied before the blur.
CLOSE_PX = 5

# The size the BiSeNet checkpoint was trained at.  The swapped crop is upscaled
# to this size before the parse.  A parse of the raw 128 crop is degenerate: it
# returned 96 % background, 0 % skin, so the mask came out empty.  The upscale
# changes no content, only the size the network sees.
MODEL_SIZE = 512

_parser = None


def get_parser():
    """Return the loaded parser and its device.  The weights load once."""
    global _parser
    if _parser is None:
        net = load_parser()
        _parser = (net, next(net.parameters()).device)
    return _parser


def parse_crop(net, device, swapped_crop: np.ndarray) -> np.ndarray:
    """Return the class map of one swapped crop, at `MODEL_SIZE` square.

    The crop is the swapper output, never the original frame.  The alpha must
    describe the pixels that the composite will take from the swapper.
    """
    upscaled = cv2.resize(
        swapped_crop, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_CUBIC
    )
    return parse(net, device, upscaled)


def region_mask_from_parse(
    parse_argmax: np.ndarray,
    region_ids,
    sigma: float = BLUR_SIGMA,
) -> np.ndarray:
    """Return the float32 alpha for the classes in `region_ids`.

    The alpha has the same height and width as `parse_argmax`.  It is 1 inside
    a region and 0 outside, with a narrow soft edge from the blur.
    """
    mask = np.isin(parse_argmax, sorted(region_ids)).astype(np.float32)

    # Fill the small holes the parser leaves at the hairline, so the skin of the
    # forehead is not speckled.
    kernel = np.ones((CLOSE_PX, CLOSE_PX), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    blur = cv2.GaussianBlur(mask, (0, 0), sigma)
    # Re-threshold, so the blur becomes a hard edge with a one pixel ramp.
    return np.clip(blur * 2.0 - 0.5, 0.0, 1.0).astype(np.float32)


def region_alpha(swapped_crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the class map and the alpha for one swapped crop.

    Both are the size of `swapped_crop`.  The parse runs at `MODEL_SIZE`, then
    the map and the alpha come back down to the crop size, so `paste_with_mask`
    can warp them with the transform that made the crop.
    """
    net, device = get_parser()
    crop_size = (swapped_crop.shape[1], swapped_crop.shape[0])

    argmax = parse_crop(net, device, swapped_crop)
    alpha = region_mask_from_parse(argmax, SWAP_IDS)

    return (
        cv2.resize(argmax, crop_size, interpolation=cv2.INTER_NEAREST),
        cv2.resize(alpha, crop_size, interpolation=cv2.INTER_LINEAR),
    )


def paste_with_mask(
    dst_frame: np.ndarray,
    src_swapped: np.ndarray,
    mask: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """Return the frame with the masked part of `src_swapped` put back.

    `matrix` is the 2x3 transform that `swapper.get` returned.  It made the
    aligned crop, so its inverse puts the swapped crop and its alpha back onto
    the frame.  A box cannot carry that geometry, so the transform is used.

    The blend is linear: `dst * (1 - a) + src * a`.  No Poisson blending.
    """
    height, width = dst_frame.shape[:2]
    size = (width, height)
    inverse = cv2.invertAffineTransform(matrix)

    warped = cv2.warpAffine(src_swapped, inverse, size, borderValue=0.0)
    alpha = cv2.warpAffine(mask, inverse, size, flags=cv2.INTER_LINEAR, borderValue=0.0)
    alpha = alpha[:, :, np.newaxis]

    merged = warped.astype(np.float32) * alpha + dst_frame.astype(np.float32) * (1.0 - alpha)
    return np.clip(merged, 0.0, 255.0).astype(np.uint8)


def class_index(name: str) -> int:
    """Return the parser class index of `name`, so a caller never hard-codes it."""
    return CLASSES.index(name)