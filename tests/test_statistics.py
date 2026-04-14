import math
from pathlib import Path

import pytest

from benchr._results import (
    _compute_metric_stats,
    _compute_metric_ratio,
    _geomean_with_sigma,
    _group_execution_result,
    _extract_unique_names,
    _scale_unit,
    build_summary_data,
    ExecutionResult,
    Measurement,
    MetricRatio,
    MetricStats,
    GroupedResult,
)
from conftest import make_execution


# ---------------------------------------------------------------------------
# _compute_metric_stats
# ---------------------------------------------------------------------------

def test_compute_metric_stats_single_value():
    stats = _compute_metric_stats([7.0], "runtime", "s", True)
    assert stats.n == 1
    assert stats.mean == 7.0
    assert stats.median == 7.0
    assert stats.stdev == 0.0
    assert stats.min == 7.0
    assert stats.max == 7.0


def test_compute_metric_stats_known_list():
    stats = _compute_metric_stats([2.0, 4.0, 6.0], "runtime", "s", True)
    assert stats.n == 3
    assert stats.mean == 4.0
    assert stats.median == 4.0
    assert stats.stdev == 2.0
    assert stats.min == 2.0
    assert stats.max == 6.0


# ---------------------------------------------------------------------------
# _compute_metric_ratio
# ---------------------------------------------------------------------------

def test_compute_metric_ratio_lower_is_better():
    baseline = _compute_metric_stats([100.0], "runtime", "s", True)
    current = _compute_metric_stats([50.0], "runtime", "s", True)
    ratio = _compute_metric_ratio(baseline, current)
    assert ratio is not None
    # lower_is_better=True, baseline=100, current=50 => display_ratio = 100/50 = 2.0
    assert ratio.display_ratio == 2.0


def test_compute_metric_ratio_higher_is_better():
    baseline = _compute_metric_stats([100.0], "throughput", "ops/s", False)
    current = _compute_metric_stats([200.0], "throughput", "ops/s", False)
    ratio = _compute_metric_ratio(baseline, current)
    assert ratio is not None
    # lower_is_better=False, display_ratio = raw = current/baseline = 200/100 = 2.0
    assert ratio.display_ratio == 2.0


def test_compute_metric_ratio_zero_median_returns_none():
    baseline = _compute_metric_stats([0.0], "runtime", "s", True)
    current = _compute_metric_stats([50.0], "runtime", "s", True)
    assert _compute_metric_ratio(baseline, current) is None


def test_compute_metric_ratio_none_lower_is_better_returns_none():
    baseline = _compute_metric_stats([100.0], "score", "", None)
    current = _compute_metric_stats([50.0], "score", "", None)
    assert _compute_metric_ratio(baseline, current) is None


# ---------------------------------------------------------------------------
# _geomean_with_sigma
# ---------------------------------------------------------------------------

def test_geomean_with_sigma_two_ratios():
    mr1 = MetricRatio(
        metric="runtime", unit="s", lower_is_better=True,
        raw_ratio=0.5, display_ratio=2.0, sigma=0.0,
        baseline_center=100.0, baseline_stdev=0.0,
        current_center=50.0, current_stdev=0.0,
    )
    mr2 = MetricRatio(
        metric="runtime", unit="s", lower_is_better=True,
        raw_ratio=0.125, display_ratio=8.0, sigma=0.0,
        baseline_center=100.0, baseline_stdev=0.0,
        current_center=12.5, current_stdev=0.0,
    )
    geo, sigma = _geomean_with_sigma([mr1, mr2])
    assert math.isclose(geo, 4.0, rel_tol=1e-9)
    # With zero stdevs sigma is 0
    assert sigma == 0.0


def test_geomean_with_sigma_nonzero_sigma():
    mr1 = MetricRatio(
        metric="runtime", unit="s", lower_is_better=True,
        raw_ratio=0.5, display_ratio=2.0, sigma=0.1,
        baseline_center=100.0, baseline_stdev=5.0,
        current_center=50.0, current_stdev=2.5,
    )
    mr2 = MetricRatio(
        metric="runtime", unit="s", lower_is_better=True,
        raw_ratio=0.125, display_ratio=8.0, sigma=0.4,
        baseline_center=100.0, baseline_stdev=10.0,
        current_center=12.5, current_stdev=1.0,
    )
    geo, sigma = _geomean_with_sigma([mr1, mr2])
    assert math.isclose(geo, 4.0, rel_tol=1e-9)
    assert sigma > 0


# ---------------------------------------------------------------------------
# _group_execution_result
# ---------------------------------------------------------------------------

def test_group_execution_result_basic():
    exe1_r1 = make_execution(benchmark_name="b1", suite="s1", info={"v": "1"}, run=1)
    exe1_r2 = make_execution(benchmark_name="b1", suite="s1", info={"v": "1"}, run=2)
    er = ExecutionResult(measurements=[
        Measurement(execution=exe1_r1, metric="runtime", value=1.0, unit="s", lower_is_better=True),
        Measurement(execution=exe1_r2, metric="runtime", value=1.5, unit="s", lower_is_better=True),
    ])
    grouped = _group_execution_result(er, "test")
    assert grouped.name == "test"
    assert len(grouped.benchmarks) == 1
    bg = grouped.benchmarks[0]
    assert bg.suite == "s1"
    assert bg.benchmark == "b1"
    assert bg.metrics[("runtime", "s")] == [1.0, 1.5]
    assert bg.run_counts.successes == 2
    assert bg.run_counts.failures == 0


def test_group_execution_result_failure_counted():
    exe = make_execution(benchmark_name="b1", suite="s1", info={}, run=1)
    er = ExecutionResult(measurements=[
        Measurement(execution=exe, metric="failed", value=1),
        Measurement(execution=exe, metric="runtime", value=0.0, unit="s", lower_is_better=True),
    ])
    grouped = _group_execution_result(er, "test")
    assert grouped.benchmarks[0].run_counts.failures == 1


def test_group_execution_result_different_info_separate_groups():
    exe_a = make_execution(benchmark_name="b1", suite="s1", info={"opt": "O0"}, run=1)
    exe_b = make_execution(benchmark_name="b1", suite="s1", info={"opt": "O2"}, run=1)
    er = ExecutionResult(measurements=[
        Measurement(execution=exe_a, metric="runtime", value=2.0, unit="s", lower_is_better=True),
        Measurement(execution=exe_b, metric="runtime", value=1.0, unit="s", lower_is_better=True),
    ])
    grouped = _group_execution_result(er, "test")
    assert len(grouped.benchmarks) == 2


# ---------------------------------------------------------------------------
# _extract_unique_names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "paths, expected",
    [
        ([], []),
        ([Path("results/data.csv")], ["data"]),
        (
            [Path("vm1/results.csv"), Path("vm2/results.csv")],
            ["vm1", "vm2"],
        ),
        (
            [Path("prefix/a/suffix.json"), Path("prefix/b/suffix.json")],
            ["a", "b"],
        ),
    ],
    ids=["empty", "single", "common_prefix_stripped", "common_suffix_stripped"],
)
def test_extract_unique_names(paths, expected):
    assert _extract_unique_names(paths) == expected


# ---------------------------------------------------------------------------
# _scale_unit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, unit, expected_scale, expected_unit",
    [
        (0.5, "s", 1e3, "ms"),
        (0.0005, "s", 1e6, "\u00b5s"),
        (1024, "kB", 1 / 1024, "MB"),
        (1024 * 1024, "kB", 1 / (1024 * 1024), "GB"),
        (42, "flops", 1, "flops"),
    ],
    ids=["s_to_ms", "s_to_us", "kB_to_MB", "kB_to_GB", "unknown_unit"],
)
def test_scale_unit(value, unit, expected_scale, expected_unit):
    scale, scaled_unit = _scale_unit(value, unit)
    assert math.isclose(scale, expected_scale, rel_tol=1e-9)
    assert scaled_unit == expected_unit


# ---------------------------------------------------------------------------
# build_summary_data (no baselines)
# ---------------------------------------------------------------------------

def test_build_summary_data_no_baselines():
    exe = make_execution(benchmark_name="b1", suite="s1", info={}, run=1)
    er = ExecutionResult(measurements=[
        Measurement(execution=exe, metric="runtime", value=1.0, unit="s", lower_is_better=True),
    ])
    summary = build_summary_data(er, baseline_paths=[])
    assert len(summary.groups) == 1
    assert summary.baseline is None
    assert summary.comparees == []
    assert summary.comparee_names == []
    assert summary.ratios == {}
    assert summary.geomeans == {}
