from nnopt.profiler.graph_profiler import profile_onnx_model, evaluate_against_cache
from nnopt.hw.cache_topology import detect_cache_topology

topo = detect_cache_topology()
l2 = topo.by_level(2)[0]
l3 = topo.by_level(3)[0]

print("=== ENCODER (batch=1) ===")
enc_profiles = profile_onnx_model(
    "models/uzbek_stt_v1_onnx/encoder_model.onnx", free_dims={"batch_size": 1}, on_error="warn"
)
print("total Gemm/MatMul ops profiled:", len(enc_profiles))
enc_sorted = sorted(enc_profiles, key=lambda p: -p.M_total)
print("--- top 8 by M_total ---")
for p in enc_sorted[:8]:
    fit_l2 = evaluate_against_cache(p, l2, alpha=0.7)
    fit_l3 = evaluate_against_cache(p, l3, alpha=0.7)
    print(
        f"{p.name[:45]:45s} m={p.m:5d} k={p.k:5d} n={p.n:5d} "
        f"M_total={p.M_total/1024:8.1f}KiB AI={p.arithmetic_intensity:6.2f} "
        f"K_L2={fit_l2.k_cache:6.2f} K_L3={fit_l3.k_cache:6.3f}"
    )
total_mtotal = sum(p.M_total for p in enc_profiles)
print("sum M_total all matmul-like ops (MiB):", total_mtotal / 1024 / 1024)
n_critical_l2 = sum(1 for p in enc_profiles if evaluate_against_cache(p, l2, 0.7).is_critical)
n_critical_l3 = sum(1 for p in enc_profiles if evaluate_against_cache(p, l3, 0.7).is_critical)
print(f"operators critical vs L2: {n_critical_l2}/{len(enc_profiles)}, vs L3: {n_critical_l3}/{len(enc_profiles)}")
