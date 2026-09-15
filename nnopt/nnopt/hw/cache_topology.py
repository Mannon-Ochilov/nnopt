"""Physical cache hierarchy discovery for the target compute platform.

Implements the hardware-fact side of README.md Sec 2.1 / Sec 2.2:
for a matrix operator executed on a set of logical processors, we need
the *nearest common cache* those processors share (L2 if they share an
L2 instance, otherwise L3). L1 is deliberately excluded from this
selection (README Sec 2.2: "L1 kesh operatorning umumiy ishchi
to'plamini joylashtirish mezoni sifatida olinmaydi").

On Windows this is read via GetLogicalProcessorInformationEx(RelationCache),
which returns, per physical cache instance, its level/size and the affinity
mask of logical processors that share it. That mask *is* the answer to
"which cores use this cache together" -- no separate core-to-cache join is
needed.

If the WinAPI call is unavailable (non-Windows, or call failure), we fall
back to a conservative heuristic built from psutil + py-cpuinfo: L3 is
assumed shared by all logical processors, L2 is assumed private per
physical core. This fallback is clearly flagged in CacheTopology.source.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CacheInstance:
    """One physical cache instance (e.g. "the L2 cache of core 3")."""

    level: int  # 1, 2, or 3
    size_bytes: int
    line_size: int
    associativity: int
    cache_type: str  # "unified" | "instruction" | "data" | "trace" | "unknown"
    group: int  # Windows processor group id (almost always 0)
    core_ids: frozenset[int]  # logical processor indices sharing this instance


@dataclass
class CacheTopology:
    instances: list[CacheInstance]
    logical_processor_count: int
    source: str  # "winapi" | "fallback-heuristic"

    def by_level(self, level: int) -> list[CacheInstance]:
        return [c for c in self.instances if c.level == level]

    def nearest_shared_cache(
        self, core_ids: frozenset[int], group: int = 0, min_level: int = 2
    ) -> CacheInstance | None:
        """Smallest-level cache (>= min_level) that fully covers core_ids.

        This is the direct implementation of README Sec 2.1's target-cache
        rule: prefer L2 if the operator's executing cores share one L2
        instance, otherwise fall back to the shared L3 instance.
        Returns None if no cache instance covers the full core set (should
        not normally happen for level 3 on a single-socket machine).
        """
        candidates = sorted(
            (c for c in self.instances if c.level >= min_level and c.group == group),
            key=lambda c: (c.level, c.size_bytes),
        )
        for cache in candidates:
            if core_ids <= cache.core_ids:
                return cache
        return None

    def global_shared_cache(self, group: int = 0) -> CacheInstance:
        """The cache level GUARANTEED shared by every logical processor on
        the machine -- the conservative, correct-by-default target when an
        operator's actual executing cores are not pinned/known in advance
        (no explicit thread-affinity control). On this machine's topology
        (8x L2 shared only in pairs, 1x L3 shared by all 16 cores), that is
        L3: an operator whose threads the OS scheduler could place on ANY
        core pair cannot rely on landing in one specific L2 instance, so
        assuming L2 as the target without pinning threads is optimistic and
        unsound. Use `nearest_shared_cache(core_ids)` directly instead of
        this when threads ARE pinned to a known core set (README Sec 2.1).
        """
        all_cores = frozenset(range(self.logical_processor_count))
        cache = self.nearest_shared_cache(all_cores, group=group, min_level=1)
        if cache is None:
            raise RuntimeError(
                "no single cache instance covers every logical processor -- "
                "unexpected on a single-socket machine; check CacheTopology.instances"
            )
        return cache

    def summary(self) -> str:
        lines = [f"CacheTopology(source={self.source}, logical_processors={self.logical_processor_count})"]
        for level in (1, 2, 3):
            insts = self.by_level(level)
            if not insts:
                continue
            sizes = sorted({c.size_bytes for c in insts})
            lines.append(
                f"  L{level}: {len(insts)} instance(s), "
                f"size(s)={[f'{s / 1024:.0f} KiB' for s in sizes]}, "
                f"cores/instance~={sorted({len(c.core_ids) for c in insts})}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Windows implementation (ctypes / GetLogicalProcessorInformationEx)
# --------------------------------------------------------------------------

def _query_windows() -> CacheTopology | None:
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:  # pragma: no cover - ctypes is stdlib, defensive only
        return None

    RELATION_CACHE = 2
    ERROR_INSUFFICIENT_BUFFER = 122

    class GROUP_AFFINITY(ctypes.Structure):
        _fields_ = [
            ("Mask", ctypes.c_uint64),
            ("Group", ctypes.c_uint16),
            ("Reserved", ctypes.c_uint16 * 3),
        ]

    class CACHE_RELATIONSHIP(ctypes.Structure):
        _fields_ = [
            ("Level", ctypes.c_ubyte),
            ("Associativity", ctypes.c_ubyte),
            ("LineSize", ctypes.c_uint16),
            ("CacheSize", ctypes.c_uint32),
            ("Type", ctypes.c_int),
            ("Reserved", ctypes.c_ubyte * 18),
            ("GroupCount", ctypes.c_uint16),
            ("GroupMask", GROUP_AFFINITY),
            # NOTE: on machines with >64 logical processors (GroupCount > 1)
            # the OS appends additional GROUP_AFFINITY entries after this
            # struct. We do not parse those extra entries (rare case for
            # the desktop/edge platforms this dissertation targets); we
            # still advance by the OS-reported entry size so parsing of
            # subsequent entries stays correct.
        ]

    class SLPI_EX(ctypes.Structure):
        _fields_ = [
            ("Relationship", ctypes.c_int),
            ("Size", ctypes.c_uint32),
            ("Cache", CACHE_RELATIONSHIP),
        ]

    CACHE_TYPE_NAMES = {0: "unified", 1: "instruction", 2: "data", 3: "trace"}

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetLogicalProcessorInformationEx.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    kernel32.GetLogicalProcessorInformationEx.restype = ctypes.c_int

    length = ctypes.c_uint32(0)
    ok = kernel32.GetLogicalProcessorInformationEx(RELATION_CACHE, None, ctypes.byref(length))
    if ok:
        # Zero relationships of this type on the system -- nothing to report.
        return CacheTopology(instances=[], logical_processor_count=0, source="winapi")
    err = ctypes.get_last_error()
    if err != ERROR_INSUFFICIENT_BUFFER or length.value == 0:
        return None

    buf = ctypes.create_string_buffer(length.value)
    ok = kernel32.GetLogicalProcessorInformationEx(RELATION_CACHE, buf, ctypes.byref(length))
    if not ok:
        return None

    instances: list[CacheInstance] = []
    max_core_bit = -1
    offset = 0
    total = length.value
    while offset < total:
        remaining = total - offset
        if remaining < ctypes.sizeof(ctypes.c_int) + ctypes.sizeof(ctypes.c_uint32):
            break
        entry = SLPI_EX.from_buffer_copy(buf, offset)
        entry_size = entry.Size
        if entry_size <= 0 or offset + entry_size > total:
            break
        cache = entry.Cache
        mask = cache.GroupMask.Mask
        core_ids = frozenset(i for i in range(64) if (mask >> i) & 1)
        if core_ids:
            max_core_bit = max(max_core_bit, max(core_ids))
        instances.append(
            CacheInstance(
                level=int(cache.Level),
                size_bytes=int(cache.CacheSize),
                line_size=int(cache.LineSize),
                associativity=int(cache.Associativity),
                cache_type=CACHE_TYPE_NAMES.get(int(cache.Type), "unknown"),
                group=int(cache.GroupMask.Group),
                core_ids=core_ids,
            )
        )
        offset += entry_size

    if not instances:
        return None

    return CacheTopology(
        instances=instances,
        logical_processor_count=max_core_bit + 1,
        source="winapi",
    )


# --------------------------------------------------------------------------
# Portable fallback (psutil + py-cpuinfo based heuristic)
# --------------------------------------------------------------------------

def _query_fallback() -> CacheTopology:
    import psutil

    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
    except Exception:
        info = {}

    logical = psutil.cpu_count(logical=True) or 1
    physical = psutil.cpu_count(logical=False) or logical

    def _size(key: str) -> int:
        val = info.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
        return 0

    # py-cpuinfo reports l2/l3 as *total* size on most backends; treat as such.
    l1d = _size("l1_data_cache_size") or 32 * 1024
    l2_total = _size("l2_cache_size") or (256 * 1024 * physical)
    l3_total = _size("l3_cache_size") or (8 * 1024 * 1024)

    instances: list[CacheInstance] = []

    # L1: private per logical processor (heuristic).
    for lp in range(logical):
        instances.append(
            CacheInstance(
                level=1,
                size_bytes=l1d,
                line_size=64,
                associativity=0,
                cache_type="data",
                group=0,
                core_ids=frozenset({lp}),
            )
        )

    # L2: assume private per *physical* core, shared by its hyperthread siblings.
    threads_per_core = max(1, logical // max(1, physical))
    l2_per_core = l2_total // max(1, physical)
    for core in range(physical):
        siblings = frozenset(
            range(core * threads_per_core, min(logical, (core + 1) * threads_per_core))
        )
        instances.append(
            CacheInstance(
                level=2,
                size_bytes=l2_per_core,
                line_size=64,
                associativity=0,
                cache_type="unified",
                group=0,
                core_ids=siblings,
            )
        )

    # L3: assume one shared instance across all logical processors
    # (reasonable default for single-socket desktop/edge CPUs; AMD CCX-split
    # L3 is NOT modeled here -- flagged via `source`).
    instances.append(
        CacheInstance(
            level=3,
            size_bytes=l3_total,
            line_size=64,
            associativity=0,
            cache_type="unified",
            group=0,
            core_ids=frozenset(range(logical)),
        )
    )

    return CacheTopology(
        instances=instances,
        logical_processor_count=logical,
        source="fallback-heuristic",
    )


def detect_cache_topology() -> CacheTopology:
    """Best-effort cache topology detection for the current machine."""
    topo = _query_windows()
    if topo is not None and topo.instances:
        return topo
    return _query_fallback()


if __name__ == "__main__":
    topo = detect_cache_topology()
    print(topo.summary())
    print()
    print(f"platform: {platform.platform()}")
    for c in topo.instances:
        print(
            f"  L{c.level} size={c.size_bytes / 1024:.0f}KiB type={c.cache_type} "
            f"group={c.group} cores={sorted(c.core_ids)}"
        )
