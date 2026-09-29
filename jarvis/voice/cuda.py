"""CUDA runtime discovery for the ML stacks on Windows.

CTranslate2 (Whisper) and ONNX Runtime both load cuBLAS/cuDNN by bare DLL
name. A machine with the NVIDIA *driver* installed almost never has the CUDA
*runtime* DLLs, so Whisper fails with "Library cublas64_12.dll is not found".
The `nvidia-cublas-cu12` and friends wheels ship those DLLs inside the venv,
which is enough -- but the loader has to be told where they are.

This must run *before* ctranslate2 or onnxruntime is imported, and the returned
handles must stay referenced for the process lifetime. Losing them lets Windows
unload the directories and later loads fail.
"""

from __future__ import annotations

import contextlib
import os
import sysconfig
from pathlib import Path

# Keep module-level so the OS does not unload the directories mid-process.
_HANDLES: list[object] = []
_DONE = False


def _nvidia_bin_dirs() -> list[Path]:
    """bin/ directories inside the nvidia-* wheels, if any are installed."""
    roots = []
    for key in ("purelib", "platlib"):
        try:
            roots.append(Path(sysconfig.get_paths()[key]))
        except KeyError:
            continue
    seen: list[Path] = []
    for root in roots:
        nvidia = root / "nvidia"
        if not nvidia.is_dir():
            continue
        for pkg in sorted(nvidia.iterdir()):
            bindir = pkg / "bin"
            if bindir.is_dir() and bindir not in seen:
                seen.append(bindir)
    return seen


def enable_cuda_dlls() -> list[str]:
    """Put the venv's CUDA runtime DLLs on the loader path.

    Idempotent. Returns the directories that were registered, for logging.
    """
    global _DONE
    if _DONE:
        return []
    _DONE = True

    dirs = _nvidia_bin_dirs()
    if not dirs:
        return []

    for d in dirs:
        # Windows 3.8+ no longer searches PATH for LoadLibrary targets, so the
        # directory handle is the part that actually matters. PATH is set too
        # for child processes and for libraries loaded by name from C++.
        with contextlib.suppress(OSError):
            _HANDLES.append(os.add_dll_directory(str(d)))
    os.environ["PATH"] = os.pathsep.join([str(d) for d in dirs] + [os.environ.get("PATH", "")])
    return [str(d) for d in dirs]


def cuda_available() -> bool:
    """Whether CTranslate2 can actually see a usable GPU."""
    enable_cuda_dlls()
    try:
        import ctranslate2
    except ImportError:
        return False
    try:
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False
