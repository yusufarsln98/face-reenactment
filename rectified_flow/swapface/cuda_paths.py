"""Put the NVIDIA cu12 wheel libraries on the loader search path.

REQUIRED: call `import swapface.cuda_paths; swapface.cuda_paths.apply()`
as the first two lines of any script that imports onnxruntime and needs
the GPU.  Nothing else loads these libraries for you.

The nvidia-*-cu12 wheels unpack into `nvidia/<pkg>/lib/`, a directory the
dynamic linker does not search.  onnxruntime-gpu needs libcudart.so.12,
libcublas.so.12, libcublasLt.so.12, libcublas.so.12 and the cuDNN 9 sonames.
`apply()` loads each one with RTLD_GLOBAL so that `onnxruntime` can use the
CUDA execution provider.

This module lives in the project source, not in `.venv`, so it survives a
`uv sync` that rebuilds the virtual environment from scratch.
"""

import ctypes
import os
import sysconfig

# soname -> nvidia wheel subdirectory that ships it
_SONAMES = {
    "libcudart.so.12": "cuda_runtime",
    "libcublas.so.12": "cublas",
    "libcublasLt.so.12": "cublas",
    "libcurand.so.10": "curand",
    "libcufft.so.11": "cufft",
    "libcudnn.so.9": "cudnn",
}


def _purelib():
    """Return the site-packages directory of the active environment."""
    return sysconfig.get_paths()["purelib"]


def apply():
    """Load the NVIDIA cu12 shared libraries with RTLD_GLOBAL.

    Call this before `import onnxruntime`.  A missing optional library is
    ignored so that a CPU-only machine can still import the module.
    """
    if os.name != "posix":
        return
    purelib = _purelib()
    for soname, package in _SONAMES.items():
        path = os.path.join(purelib, "nvidia", package, "lib", soname)
        if os.path.exists(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass