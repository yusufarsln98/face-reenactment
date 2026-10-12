"""The frame loop: video + reference picture -> video.

The loop runs one pass over the video.  For each frame it does the next steps:

1. Detect the faces with SCRFD.
2. Ask the bystander gate for a decision.
3. Act on the decision.  PASS swaps the face.  SKIP and REFUSE pass the frame
   through unchanged.

Two rules shape the module:

- The frame rate and the frame size come from the source video.  They are
  never assumed, because a wrong frame rate changes the speed of the result.
- The identity embedding is computed once, before the loop.  The recogniser
  runs once for the reference picture, not once per frame.

No enhancement stage runs here.  There is no colour fix, no sharpening, and no
upscaling.  The output frame is the input frame with the face region replaced.
The enhancement stages come later and will insert two more steps.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .detect import detect_faces
from .gate import Outcome, RefusalLimitExceeded, RefusalTracker, evaluate
from .models import load_face_analysis, load_swapper
from .region_mask import paste_with_mask, region_alpha

logger = logging.getLogger(__name__)

# inswapper_128 takes a 128x128 aligned crop.
SWAP_SIZE = 128

# How the swapper output is put back onto the face.  `region` parses the swapped
# crop and keeps only the skin and the features.  `ellipse` is the old blend.  It
# stays so the two can be compared on the same frame.
PASTE_REGION = "region"
PASTE_ELLIPSE = "ellipse"
PASTE_MODES = (PASTE_REGION, PASTE_ELLIPSE)

# The frame number in an intermediate file name, zero padded to this width.
FRAME_DIGITS = 6


class RunStoppedByGate(RuntimeError):
    """The gate stopped the run because too many frames held two faces."""


class VideoFormatError(RuntimeError):
    """The source video did not report a usable frame rate or size."""


@dataclass(frozen=True)
class Identity:
    """The identity that every swapped face receives.

    The swapper reads `normed_embedding`, so this class exposes that name.
    """

    normed_embedding: np.ndarray


@dataclass
class SwapSummary:
    """What one run did.  The CLI prints this and nothing else."""

    total_frames: int = 0
    passed: int = 0
    skipped: int = 0
    refused: int = 0
    elapsed_s: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    swapped_frames: list[int] = field(default_factory=list)
    intermediates: list[Path] = field(default_factory=list)
    stopped_by_gate: bool = False

    def ms_per_frame(self) -> float:
        """Return the mean time for one frame, in milliseconds."""
        if self.total_frames == 0:
            return 0.0
        return self.elapsed_s * 1000.0 / self.total_frames


def _read_reference_embedding(path: Path) -> np.ndarray:
    """Return the 512-value identity vector of the reference picture.

    The vector is computed once.  The caller reuses it for every frame, so the
    recogniser never runs inside the frame loop.

    The function raises when the picture holds no face, or more than one face.
    It never chooses between faces, because a wrong choice would put the wrong
    identity into the video.
    """
    if not path.is_file():
        raise FileNotFoundError(f"reference picture not found: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise VideoFormatError(f"cannot read the reference picture: {path}")

    app, _provider = load_face_analysis()
    faces = app.get(image)
    if len(faces) == 0:
        raise VideoFormatError(
            f"no face found in the reference picture: {path}. "
            "The swap needs one clear face to copy the identity from."
        )
    if len(faces) > 1:
        raise VideoFormatError(
            f"{len(faces)} faces found in the reference picture: {path}. "
            "The swap needs one face, and it never chooses between faces. "
            "Crop the picture to a single face and run it again."
        )

    vector = np.asarray(faces[0].normed_embedding, dtype=np.float32)
    if vector.shape != (512,):
        raise VideoFormatError(
            f"the recogniser returned {vector.shape} values, expected 512."
        )
    return vector


def _open_video(path: Path) -> tuple[cv2.VideoCapture, float, int, int]:
    """Return the opened capture, the frame rate, and the frame size.

    The function raises when OpenCV reports a frame rate or a size that the
    writer cannot copy.  It never guesses a value.
    """
    if not path.is_file():
        raise FileNotFoundError(f"source video not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoFormatError(f"OpenCV cannot open the source video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0.0:
        capture.release()
        raise VideoFormatError(
            f"the source video reports a frame rate of {fps}. "
            "The frame rate is never assumed, so the run stops here."
        )

    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if width <= 0 or height <= 0:
        capture.release()
        raise VideoFormatError(
            f"the source video reports a frame size of {width}x{height}."
        )

    return capture, fps, width, height


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Return a writer that uses the frame rate and the size of the source."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise VideoFormatError(
            f"OpenCV cannot open the output video for writing: {path}"
        )
    return writer


def _align_crop(frame: np.ndarray, kps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the aligned 128x128 crop and the 2x3 matrix that made it."""
    from insightface.utils import face_align

    return face_align.norm_crop2(frame, kps, SWAP_SIZE)


def paste_back(frame: np.ndarray, swapped: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return the frame with `swapped` put back over the original face.

    The swapper works in a 128x128 aligned crop.  This function inverts the
    matrix that made the crop, so the swapper output lands on the face in the
    original frame.

    The mask is eroded before it is blurred.  Without the erosion the mask
    covers the whole warped crop square, and the join shows as a rectangle
    around the face.  The erosion keeps the join on the face itself.
    """
    height, width = frame.shape[:2]
    size = (width, height)
    inverse = cv2.invertAffineTransform(matrix)

    warped = cv2.warpAffine(swapped, inverse, size, borderValue=0.0)
    mask = np.full((swapped.shape[0], swapped.shape[1]), 255.0, dtype=np.float32)
    mask = cv2.warpAffine(mask, inverse, size, borderValue=0.0)
    mask[mask > 20.0] = 255.0

    # Keep only the inner part of the crop, so the join sits on the face.
    ys, xs = np.nonzero(mask == 255.0)
    if ys.size and xs.size:
        box_h = int(ys.max() - ys.min())
        box_w = int(xs.max() - xs.min())
        erode = max(int(math.sqrt(max(box_h * box_w, 1)) // 10), 4)
        kernel = np.ones((erode, erode), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

    mask = cv2.GaussianBlur(mask, (0, 0), 5.0)

    alpha = (mask / 255.0)[:, :, np.newaxis]
    merged = warped.astype(np.float32) * alpha + frame.astype(np.float32) * (1.0 - alpha)
    return np.clip(merged, 0.0, 255.0).astype(np.uint8)


def paste_swapped(
    frame: np.ndarray, swapped: np.ndarray, matrix: np.ndarray, paste_mode: str = PASTE_REGION
) -> np.ndarray:
    """Return the frame with `swapped` put back, by the requested paste mode.

    `PASTE_REGION` parses the swapped 128 crop and pastes only the skin and the
    features.  The hair, the cloth and the ears then stay as they are in the
    original frame.  `PASTE_ELLIPSE` calls `paste_back` for comparison.

    The function raises when `paste_mode` names neither mode.
    """
    if paste_mode == PASTE_ELLIPSE:
        return paste_back(frame, swapped, matrix)
    if paste_mode == PASTE_REGION:
        _argmax, alpha = region_alpha(swapped)
        return paste_with_mask(frame, swapped, alpha, matrix)
    raise ValueError(
        f"paste_mode must be one of {PASTE_MODES}. Got {paste_mode!r}. "
        f"The mode is never guessed."
    )


def _save_intermediates(
    directory: Path, index: int, crop: np.ndarray, swapped: np.ndarray, pasted: np.ndarray
) -> list[Path]:
    """Write the three PNG stages of one frame and return the paths.

    Only the stages that exist now are written.  There is no enhancement stage
    in this task, so there is no restored face and no final face.  A later
    task adds two more files here, one for each enhancement stage.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{index:0{FRAME_DIGITS}d}_stage"
    written = []
    for suffix, image in (("crop", crop), ("swapped", swapped), ("pasted", pasted)):
        path = directory / f"{stem}_{suffix}.png"
        cv2.imwrite(str(path), image)
        written.append(path)
    return written


def _write_samples(
    video_path: Path,
    reference: Path,
    output_path: Path,
    indices: list[int],
    directory: Path,
    paste_mode: str = PASTE_REGION,
) -> list[Path]:
    """Re-read `indices` from the source video and write their PNG stages.

    The frame loop cannot know the middle and the last PASS frame until the
    run ends.  This function runs after the loop.  It re-opens the source and
    swaps only the few frames that were sampled, so the memory use stays flat
    on a long video.
    """
    if not indices:
        return []

    identity = Identity(normed_embedding=_read_reference_embedding(reference))
    swapper, _provider = load_swapper()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoFormatError(f"OpenCV cannot reopen the source video: {video_path}")

    written: list[Path] = []
    try:
        for index in sorted(set(indices)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                logger.warning("cannot seek back to frame %d, no stages written", index)
                continue
            target = _single_face(frame)
            crop, _matrix = _align_crop(frame, target.kps)
            swapped, matrix = swapper.get(frame, target, identity, paste_back=False)
            pasted = paste_swapped(frame, swapped, matrix, paste_mode)
            written.extend(_save_intermediates(directory, index, crop, swapped, pasted))
    finally:
        capture.release()
    return written


def _single_face(frame: np.ndarray):
    """Return the one detected face of a frame that the gate passed."""
    boxes = detect_faces(frame)
    decision = evaluate(boxes)
    if decision.face is None:
        raise VideoFormatError(
            "the sampled frame no longer holds exactly one face. "
            "The stage images for that frame are not written."
        )
    return decision.face


def _sample_indices(pass_indices: list[int]) -> list[int]:
    """Return the first, the middle, and the last PASS frame."""
    if not pass_indices:
        return []
    return [pass_indices[0], pass_indices[len(pass_indices) // 2], pass_indices[-1]]


def run_swap(
    video_path: str | Path,
    reference_image_path: str | Path,
    output_path: str | Path,
    *,
    debug_range: tuple[int, int] | None = None,
    max_frames: int | None = None,
    progress_every: int = 30,
    paste_mode: str = PASTE_REGION,
) -> SwapSummary:
    """Swap the face in `video_path` and write the result to `output_path`.

    The identity comes from `reference_image_path`.  The frame rate and the
    frame size come from `video_path`.  No enhancement stage runs.

    `paste_mode` chooses how the swapper output is put back.  The default is
    `PASTE_REGION`, which parses the swapped crop and keeps only the skin and
    the features, so the subject's own hair and ears stay.  `PASTE_ELLIPSE` uses
    the old elliptical blend.  The function raises for any other value.

    `debug_range` holds an inclusive start and end frame number.  When it is
    set, only the frames in that range write their stage PNGs.  When it is
    None, the first, the middle, and the last PASS frame write them.

    `max_frames` stops the run after that many frames.  It exists to test a
    short prefix of a long video.

    The function raises `RunStoppedByGate` when more than the allowed share of
    frames held more than one face.  It never adds a way past the gate.
    """
    video_path = Path(video_path)
    reference = Path(reference_image_path)
    output = Path(output_path)

    # The identity is computed once, before the loop, and reused below.
    identity = Identity(normed_embedding=_read_reference_embedding(reference))
    swapper, _provider = load_swapper()

    capture, fps, width, height = _open_video(video_path)
    writer = _open_writer(output, fps, (width, height))

    tracker = RefusalTracker()
    summary = SwapSummary(fps=fps, width=width, height=height)
    start_time = time.perf_counter()

    index = 0
    try:
        while True:
            if max_frames is not None and index >= max_frames:
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            in_debug_range = debug_range is not None and debug_range[0] <= index <= debug_range[1]

            decision = evaluate(detect_faces(frame))
            try:
                tracker.record(decision)
            except RefusalLimitExceeded as error:
                summary.stopped_by_gate = True
                summary.refused = tracker.refused
                summary.total_frames = tracker.processed
                raise RunStoppedByGate(
                    f"{tracker.refused} of {tracker.processed} frames held more than "
                    f"one face. This is above the limit of {RefusalTracker.LIMIT_PERCENT} "
                    f"percent, so the run stopped. The footage does not match the "
                    f"expected source. Cause: {error}"
                ) from error

            summary.total_frames = index + 1

            if decision.outcome is Outcome.PASS and decision.face is not None:
                summary.passed += 1
                summary.swapped_frames.append(index)
                crop, _matrix = _align_crop(frame, decision.face.kps)
                swapped, matrix = swapper.get(
                    frame, decision.face, identity, paste_back=False
                )
                frame = paste_swapped(frame, swapped, matrix, paste_mode)
                if in_debug_range:
                    summary.intermediates.extend(
                        _save_intermediates(
                            output.parent / "intermediates", index, crop, swapped, frame
                        )
                    )
            elif decision.outcome is Outcome.SKIP:
                summary.skipped += 1
            else:
                summary.refused += 1

            writer.write(frame)

            if progress_every > 0 and summary.total_frames % progress_every == 0:
                rate = summary.total_frames / max(time.perf_counter() - start_time, 1e-9)
                print(
                    f"  frame {summary.total_frames:6d}  "
                    f"pass {summary.passed}  skip {summary.skipped}  "
                    f"refuse {summary.refused}  {rate:.1f} fps",
                    flush=True,
                )
            index += 1
    except BaseException:
        writer.release()
        capture.release()
        raise

    writer.release()
    capture.release()
    summary.elapsed_s = time.perf_counter() - start_time

    if debug_range is None:
        sampled = _sample_indices(summary.swapped_frames)
        directory = output.parent / "intermediates"
        summary.intermediates.extend(
            _write_samples(video_path, reference, output, sampled, directory, paste_mode)
        )
    else:
        summary.intermediates = [p for p in summary.intermediates if p.is_file()]

    return summary
