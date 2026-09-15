"""Real-execution latency measurement -- README.md Sec 8.2 (`bench/`) and
Sec 9 rule 4: "har o'zgarishdan keyin latency o'lchanadi (warmup >=10,
median >=50 run)". Nazariy FLOPs/xotira yutug'i hech qachon o'z-o'zidan
"tezlashuv" deb taqdim etilmaydi -- bu modul kaskadning har bir qaroriga
kerak bo'ladigan haqiqiy o'lchov manbai.

Also implements a minimal thread-affinity control (README Sec 2.1/2.2:
"maqsadli kesh operatorni bajaruvchi yadrolar guruhiga bog'liq") so that
a given benchmark run can be pinned to a known set of logical processors,
letting cache-fit predictions (K_cache) be matched against the cache
instance those specific cores actually share.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort


@dataclass
class LatencyResult:
    op_or_model_name: str
    warmup_runs: int
    measured_runs: int
    median_ms: float
    mean_ms: float
    stddev_ms: float
    min_ms: float
    max_ms: float
    p90_ms: float
    raw_ms: list[float] = field(repr=False, default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.op_or_model_name}: median={self.median_ms:.3f}ms "
            f"mean={self.mean_ms:.3f}ms (+/-{self.stddev_ms:.3f}) "
            f"p90={self.p90_ms:.3f}ms min={self.min_ms:.3f}ms max={self.max_ms:.3f}ms "
            f"[{self.measured_runs} runs after {self.warmup_runs} warmup]"
        )


def make_session(
    model_path: str,
    intra_op_threads: int | None = None,
    inter_op_threads: int = 1,
    providers: list[str] | None = None,
) -> ort.InferenceSession:
    """Create an ONNX Runtime session with explicit, reproducible threading.

    intra_op_threads controls how many logical processors a single op's
    computation is allowed to use -- this is the runtime knob that
    determines which cores (and therefore which shared cache, see
    nnopt.hw.cache_topology) actually execute a given operator.
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_op_threads is not None:
        so.intra_op_num_threads = intra_op_threads
    so.inter_op_num_threads = inter_op_threads
    providers = providers or ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, sess_options=so, providers=providers)


def _random_feed(session: ort.InferenceSession, seed: int = 0) -> dict[str, np.ndarray]:
    """Generate a random feed dict matching the session's declared input
    shapes/dtypes. Dynamic dims (== None / symbolic) default to a small
    representative size (documented in README Sec 8.5.3) -- override via
    `fixed_feed` in measure_latency() for anything model-specific (e.g.
    real audio features for Whisper).
    """
    rng = np.random.default_rng(seed)
    ORT_TO_NP = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    feed = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        np_dtype = ORT_TO_NP.get(inp.type, np.float32)
        if np_dtype == np.bool_:
            arr = rng.integers(0, 2, size=shape).astype(np.bool_)
        elif np.issubdtype(np_dtype, np.integer):
            arr = rng.integers(0, 100, size=shape).astype(np_dtype)
        else:
            arr = rng.standard_normal(size=shape).astype(np_dtype)
        feed[inp.name] = arr
    return feed


def measure_latency(
    session: ort.InferenceSession,
    name: str = "model",
    fixed_feed: dict[str, np.ndarray] | None = None,
    warmup_runs: int = 10,
    measured_runs: int = 50,
    seed: int = 0,
) -> LatencyResult:
    """Warmup + median-of-N wall-clock latency for one InferenceSession.run()."""
    feed = fixed_feed if fixed_feed is not None else _random_feed(session, seed=seed)
    output_names = [o.name for o in session.get_outputs()]

    for _ in range(warmup_runs):
        session.run(output_names, feed)

    timings_ms: list[float] = []
    for _ in range(measured_runs):
        t0 = time.perf_counter()
        session.run(output_names, feed)
        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1000.0)

    timings_sorted = sorted(timings_ms)
    p90_idx = min(len(timings_sorted) - 1, int(round(0.9 * (len(timings_sorted) - 1))))
    return LatencyResult(
        op_or_model_name=name,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        median_ms=statistics.median(timings_ms),
        mean_ms=statistics.fmean(timings_ms),
        stddev_ms=statistics.pstdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        min_ms=min(timings_ms),
        max_ms=max(timings_ms),
        p90_ms=timings_sorted[p90_idx],
        raw_ms=timings_ms,
    )


def compare(baseline: LatencyResult, candidate: LatencyResult) -> dict:
    """README Sec 9 rule 4 helper: is `candidate` actually faster than
    `baseline`, measured (never assumed from theoretical FLOPs/memory)?
    """
    speedup = baseline.median_ms / candidate.median_ms if candidate.median_ms > 0 else float("inf")
    return {
        "baseline_median_ms": baseline.median_ms,
        "candidate_median_ms": candidate.median_ms,
        "speedup": speedup,
        "improved": candidate.median_ms < baseline.median_ms,
    }


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    if model_path is None:
        print("usage: python -m nnopt.bench.latency <model.onnx>")
        raise SystemExit(1)
    sess = make_session(model_path)
    result = measure_latency(sess, name=model_path)
    print(result.summary())
