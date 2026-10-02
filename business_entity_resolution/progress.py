"""Progress bars + timestamped logging that stay readable both on a TTY and in `nohup ... > run.log`."""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager

from tqdm import tqdm

_IS_TTY = sys.stderr.isatty()


def pbar(iterable=None, total: int | None = None, desc: str = "", unit: str = "it", leave: bool = True, **kw):
    """tqdm with sane defaults: stderr, dynamic width, slower refresh when logging to a file."""
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.5 if _IS_TTY else 5.0,
        smoothing=0.1,
        **kw,
    )


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    tqdm.write(f"[{ts}] {msg}", file=sys.stderr)


@contextmanager
def stage(name: str):
    """Print a banner before/after a pipeline stage with the elapsed time."""
    bar = "=" * 78
    log(f"{bar}\n[{time.strftime('%H:%M:%S')}] >>> STAGE {name}\n{bar}")
    t0 = time.time()
    try:
        yield
    finally:
        log(f"<<< STAGE {name} done in {fmt_secs(time.time() - t0)}")


def rss_gb(peak: bool = True) -> float | None:
    """Process resident memory in GB: the peak so far (default) or the current value. No psutil dependency."""
    try:
        import os
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):   # PROCESS_MEMORY_COUNTERS (psapi.h), all ten fields: cb must be its full size
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            k32 = ctypes.WinDLL("kernel32")
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi")
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return None
            return (pmc.PeakWorkingSetSize if peak else pmc.WorkingSetSize) / 2**30
        if not peak:
            with open("/proc/self/statm") as f:
                return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20        # KiB on Linux
    except Exception:
        return None


def available_ram_gb() -> float | None:
    """Physical memory currently available to new allocations (GB)."""
    try:
        import os
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):   # MEMORYSTATUSEX
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            return ms.ullAvailPhys / 2**30 if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)) else None
        with open("/proc/meminfo") as f:
            info = {l.split(":")[0]: int(l.split()[1]) for l in f}
        return info["MemAvailable"] / 2**20
    except Exception:
        return None


def fmt_secs(s: float) -> str:
    s = int(s)
    h, r = divmod(s, 3600)
    m, r = divmod(r, 60)
    return f"{h}h{m:02d}m{r:02d}s" if h else (f"{m}m{r:02d}s" if m else f"{r}s")
