import numpy as np

from india_s2s_bench.bootstrap import circular_block_indices, paired_interval


def test_circular_blocks_are_deterministic_and_in_range():
    first = circular_block_indices(7, 4, np.random.default_rng(12))
    second = circular_block_indices(7, 4, np.random.default_rng(12))
    assert np.array_equal(first, second)
    assert len(first) == 7
    assert np.all((first >= 0) & (first < 7))


def test_paired_interval_keeps_leads_with_initialization():
    dates = np.repeat(np.array(["a", "b", "c", "d"]), 6)
    a = np.arange(24.0) + 2.0
    b = np.arange(24.0)
    result = paired_interval(a, b, dates, draws=100, block_length=2, seed=3)
    assert np.isclose(result["effect"], 2.0)
    assert np.isclose(result["ci_low"], 2.0)
    assert np.isclose(result["ci_high"], 2.0)
