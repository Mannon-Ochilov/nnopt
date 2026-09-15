from nnopt.profiler.graph_profiler import profile_onnx_model, evaluate_against_cache
from nnopt.hw.cache_topology import detect_cache_topology

topo = detect_cache_topology()
l2 = topo.by_level(2)[0]
l3 = topo.by_level(3)[0]

for dec_len in (1, 32):
    print(f"=== DECODER (batch=1, decoder_seq_len={dec_len}, encoder_seq_len=1500) ===")
    profiles = profile_onnx_model(
        "models/uzbek_stt_v1_onnx/decoder_model.onnx",
        free_dims={"batch_size": 1, "decoder_sequence_length": dec_len, "encoder_sequence_length": 1500},
        on_error="warn",
    )
    print("total Gemm/MatMul ops profiled:", len(profiles))
    top = sorted(profiles, key=lambda p: -p.M_total)[:6]
    for p in top:
        fit_l2 = evaluate_against_cache(p, l2, alpha=0.7)
        fit_l3 = evaluate_against_cache(p, l3, alpha=0.7)
        print(
            f"{p.name[:45]:45s} m={p.m:5d} k={p.k:5d} n={p.n:5d} "
            f"M_total={p.M_total/1024:9.1f}KiB AI={p.arithmetic_intensity:6.2f} "
            f"K_L2={fit_l2.k_cache:7.2f} K_L3={fit_l3.k_cache:6.3f}"
        )
    total_mtotal = sum(p.M_total for p in profiles)
    print("sum M_total (MiB):", total_mtotal / 1024 / 1024)
    n_l2 = sum(1 for p in profiles if evaluate_against_cache(p, l2, 0.7).is_critical)
    n_l3 = sum(1 for p in profiles if evaluate_against_cache(p, l3, 0.7).is_critical)
    print(f"critical vs L2: {n_l2}/{len(profiles)}, vs L3: {n_l3}/{len(profiles)}")
    print()
