"""Keeping a thread on the CPU's fast cores (step 07c).

Intel's 12th-generation and later laptop CPUs have two kinds of cores: performance cores and
slower efficiency cores. Windows moves a thread that works in short bursts (the audio thread:
a few milliseconds every 20 ms) to efficiency cores. Measured on the laptop (i5-12450H, 4 + 4
cores, Balanced plan), one DeepFilterNet3 step paced like a live session: mean 7.0 ms, p95
11.5 ms unpinned; 4.1 / 5.5 ms pinned to the performance cores.

:func:`prefer_performance_cores` asks Windows which logical processors have the highest
efficiency class (``GetSystemCpuSetInformation``) and restricts the **calling thread** to them.
On a CPU with one kind of core, or outside Windows, it does nothing.
"""

from __future__ import annotations

import sys


def performance_core_mask() -> int | None:
    """Affinity mask of the fastest logical processors (processor group 0), or ``None`` if
    the CPU has one kind of core or the information is not available."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetSystemCpuSetInformation.argtypes = (ctypes.c_void_p, wintypes.ULONG,
                                                   ctypes.POINTER(wintypes.ULONG),
                                                   wintypes.HANDLE, wintypes.ULONG)
        k32.GetSystemCpuSetInformation.restype = wintypes.BOOL
        length = wintypes.ULONG(0)
        k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(length), None, 0)
        if not length.value:
            return None
        buf = (ctypes.c_ubyte * length.value)()
        if not k32.GetSystemCpuSetInformation(buf, length, ctypes.byref(length), None, 0):
            return None
        data = bytes(buf)
    except (OSError, AttributeError):
        return None
    entries: list[tuple[int, int, int]] = []  # (group, logical processor, efficiency class)
    offset = 0
    while offset + 20 <= len(data):
        size = int.from_bytes(data[offset:offset + 4], "little")
        kind = int.from_bytes(data[offset + 4:offset + 8], "little")
        if size <= 0:
            break
        if kind == 0:  # CpuSetInformation
            group = int.from_bytes(data[offset + 12:offset + 14], "little")
            entries.append((group, data[offset + 14], data[offset + 18]))
        offset += size
    group0 = [(lp, eff) for group, lp, eff in entries if group == 0]
    if not group0:
        return None
    classes = {eff for _, eff in group0}
    if len(classes) < 2:
        return None
    best = max(classes)
    mask = 0
    for lp, eff in group0:
        if eff == best and lp < 64:
            mask |= 1 << lp
    return mask or None


def prefer_performance_cores() -> int | None:
    """Pin the calling thread to the performance cores; returns the mask used, or ``None``
    when nothing was changed."""
    mask = performance_core_mask()
    if mask is None:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentThread.restype = wintypes.HANDLE
        k32.SetThreadAffinityMask.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
        k32.SetThreadAffinityMask.restype = ctypes.c_size_t
        previous = k32.SetThreadAffinityMask(k32.GetCurrentThread(), mask)
    except (OSError, AttributeError):
        return None
    return mask if previous else None
