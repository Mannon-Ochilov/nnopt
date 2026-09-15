"""Tests for nnopt.hw.cache_topology (README Sec 2.1 cache-selection logic)."""

from __future__ import annotations

import pytest

from nnopt.hw.cache_topology import CacheInstance, CacheTopology, detect_cache_topology


def _synthetic_topology():
    """4 logical processors: 2x L2 (each shared by a pair), 1x L3 (all 4)."""
    l1_insts = [
        CacheInstance(1, 32 * 1024, 64, 8, "data", 0, frozenset({i})) for i in range(4)
    ]
    l2_insts = [
        CacheInstance(2, 1 * 1024 * 1024, 64, 8, "unified", 0, frozenset({0, 1})),
        CacheInstance(2, 1 * 1024 * 1024, 64, 8, "unified", 0, frozenset({2, 3})),
    ]
    l3_insts = [
        CacheInstance(3, 8 * 1024 * 1024, 64, 16, "unified", 0, frozenset({0, 1, 2, 3})),
    ]
    return CacheTopology(instances=l1_insts + l2_insts + l3_insts, logical_processor_count=4, source="synthetic-test")


def test_nearest_shared_cache_picks_l2_when_cores_share_one_instance():
    topo = _synthetic_topology()
    cache = topo.nearest_shared_cache(frozenset({0, 1}))
    assert cache.level == 2


def test_nearest_shared_cache_falls_back_to_l3_across_l2_domains():
    topo = _synthetic_topology()
    cache = topo.nearest_shared_cache(frozenset({0, 2}))
    assert cache.level == 3


def test_global_shared_cache_is_l3_when_l2_only_covers_pairs():
    """The core scenario motivating this module: without explicit thread
    pinning, an operator could run on any core pair, so the only cache
    level SAFE to target is the one covering every logical processor."""
    topo = _synthetic_topology()
    cache = topo.global_shared_cache()
    assert cache.level == 3
    assert cache.core_ids == frozenset({0, 1, 2, 3})


def test_global_shared_cache_picks_l2_when_l2_itself_spans_all_cores():
    """If a (hypothetical) machine's L2 already covers every core, that IS
    the correct global answer -- global_shared_cache must not hard-code L3,
    it must find the smallest level that actually qualifies."""
    l1_insts = [CacheInstance(1, 32 * 1024, 64, 8, "data", 0, frozenset({i})) for i in range(4)]
    l2_all = CacheInstance(2, 4 * 1024 * 1024, 64, 8, "unified", 0, frozenset({0, 1, 2, 3}))
    l3_all = CacheInstance(3, 16 * 1024 * 1024, 64, 16, "unified", 0, frozenset({0, 1, 2, 3}))
    topo = CacheTopology(instances=l1_insts + [l2_all, l3_all], logical_processor_count=4, source="synthetic-test")

    cache = topo.global_shared_cache()
    assert cache.level == 2  # smallest (and thus most useful/precise) qualifying level


def test_real_machine_global_shared_cache_matches_l3():
    """Sanity check against this machine's actual detected topology
    (README session finding: 8x L2 shared only in pairs, 1x L3 shared by
    all 16 logical processors) -- global_shared_cache() must resolve to L3,
    not silently fall back to an arbitrary L2 instance."""
    topo = detect_cache_topology()
    cache = topo.global_shared_cache()
    assert cache.core_ids == frozenset(range(topo.logical_processor_count))
    # On machines with a single shared L3 (the common case), this must be L3;
    # skip the strict level assertion only if some future test machine's L2
    # itself spans every core (see the synthetic test above for that case).
    l2_instances = topo.by_level(2)
    l2_spans_all = any(c.core_ids == frozenset(range(topo.logical_processor_count)) for c in l2_instances)
    if not l2_spans_all:
        assert cache.level == 3
