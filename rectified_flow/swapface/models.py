"""Load the face analysis pack and the inswapper from the project weights.

Two rules drive this module:

- The weights live in the project `models/` directory, not in `~/.insightface`.
  insightface downloads a model pack when it does not find one.  Every loader
  here checks the exact weight file first, so a missing file raises before
  insightface can start a download.
- A requested provider is not a realised provider.  Each loader reports the
  provider that the ONNX Runtime session actually got.

Both loaders cache their result.  The second call returns the same object and
does not read the weights again.
"""

from __future__ import annotations

from pathlib import Path

import swapface.cuda_paths

# REQUIRED before onnxruntime is imported.  Loads the NVIDIA cu12 libraries.
swapface.cuda_paths.apply()

from insightface.app import FaceAnalysis  # noqa: E402
from insightface.model_zoo import get_model  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BUFFALO_L_DIR = MODELS_DIR / "buffalo_l"
DETECTOR_FILE = BUFFALO_L_DIR / "det_10g.onnx"
SWAPPER_FILE = MODELS_DIR / "inswapper_128.onnx"

PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# The swapper takes a 128x128 crop, so SCRFD detects at the same size.
DET_SIZE = (128, 128)

_face_analysis: FaceAnalysis | None = None
_face_analysis_provider: str = ""
_swapper = None
_swapper_provider: str = ""


def _require(path: Path) -> Path:
    """Return `path`, or raise an error that names the missing file."""
    if not path.is_file():
        raise FileNotFoundError(f"model weight not found: {path}")
    return path


def _realised_provider(model: object) -> str:
    """Return the provider the ONNX Runtime session of `model` actually uses."""
    session = getattr(model, "session", None)
    if session is None:
        return "unknown"
    providers = session.get_providers()
    return providers[0] if providers else "unknown"


def load_face_analysis() -> tuple[FaceAnalysis, str]:
    """Return the buffalo_l `FaceAnalysis` app and its active provider.

    The app is prepared at DET_SIZE.  It is cached after the first call.
    """
    global _face_analysis, _face_analysis_provider
    if _face_analysis is None:
        # Check first.  insightface would download the whole pack otherwise.
        _require(DETECTOR_FILE)
        # insightface looks for `root/models/<name>`, so root is the directory
        # that holds `models/`.  That resolves to the pack in this project.
        app = FaceAnalysis(
            name="buffalo_l",
            root=str(MODELS_DIR.parent),
            providers=PROVIDERS,
        )
        app.prepare(ctx_id=0, det_size=DET_SIZE)
        _face_analysis = app
        _face_analysis_provider = _realised_provider(app.det_model)
    return _face_analysis, _face_analysis_provider


def load_swapper() -> tuple[object, str]:
    """Return the inswapper model and its active provider.

    The model is cached after the first call.
    """
    global _swapper, _swapper_provider
    if _swapper is None:
        _require(SWAPPER_FILE)
        _swapper = get_model(str(SWAPPER_FILE), providers=PROVIDERS)
        _swapper_provider = _realised_provider(_swapper)
    return _swapper, _swapper_provider
