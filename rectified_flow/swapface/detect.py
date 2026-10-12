"""Detect the faces in one video frame.

`detect_faces` takes a BGR frame as a numpy array, in the OpenCV channel order,
at any size.  It returns one `FaceBox` per face that SCRFD finds.

An empty list is a normal answer.  It means the frame holds no face, or the
detector failed.  A failed frame must not stop a long run, so an error in the
detector is logged and reported as an empty list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .models import load_face_analysis

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceBox:
    """One detected face.

    `bbox` holds x1, y1, x2, y2 as whole pixels.  `kps` holds the 5 SCRFD
    keypoints as a (5, 2) array of x and y.  `det_score` is the detector
    confidence.
    """

    bbox: tuple[int, int, int, int]
    kps: np.ndarray
    det_score: float


def detect_faces(frame: np.ndarray) -> list[FaceBox]:
    """Return every face that SCRFD finds in `frame`.

    The list is empty when the frame holds no face, or when the detector
    raises.  The caller gets no way to tell those two cases apart, because the
    frame loop must skip the frame either way.
    """
    try:
        app, _provider = load_face_analysis()
        faces = app.get(frame)
        boxes = []
        for face in faces:
            x1, y1, x2, y2 = (int(round(v)) for v in face.bbox)
            boxes.append(
                FaceBox(
                    bbox=(x1, y1, x2, y2),
                    kps=np.asarray(face.kps, dtype=np.float32),
                    det_score=float(face.det_score),
                )
            )
        return boxes
    except Exception:
        logger.warning("face detection failed on a frame", exc_info=True)
        return []
