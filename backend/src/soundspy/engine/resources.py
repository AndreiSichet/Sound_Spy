"""What the backend process uses: RAM (working set) and graphics-card memory (step 08).

RAM comes from the operating system (``GetProcessMemoryInfo`` on Windows, ``getrusage``
elsewhere: no extra package); graphics-card memory from PyTorch, only if PyTorch is already
imported and CUDA initialised (asking would otherwise start CUDA just to say "0").
"""

from __future__ import annotations

import sys
from typing import Any


def process_memory_mb() -> float | None:
    """The process's resident memory in MB (``None`` if the OS does not tell)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        try:
            kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
            # 64-bit handles: without these, ctypes passes the pseudo-handle as a 32-bit int
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters),
                                                   wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                            ctypes.byref(counters), counters.cb)
        except (AttributeError, OSError):
            return None
        return round(counters.WorkingSetSize / 2**20, 1) if ok else None
    try:
        import resource

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(kb / 1024.0, 1)
    except (ImportError, OSError):
        return None


def gpu_memory_mb() -> dict[str, float] | None:
    """``{allocated, reserved}`` in MB on the current CUDA device, if CUDA is in use."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return None
        return {"allocated": round(torch.cuda.memory_allocated() / 2**20, 1),
                "reserved": round(torch.cuda.memory_reserved() / 2**20, 1)}
    except Exception:  # noqa: BLE001 - a broken driver must not break a status request
        return None


def snapshot() -> dict[str, Any]:
    gpu = gpu_memory_mb()
    return {"ram_mb": process_memory_mb(),
            "vram_allocated_mb": gpu["allocated"] if gpu else None,
            "vram_reserved_mb": gpu["reserved"] if gpu else None}
