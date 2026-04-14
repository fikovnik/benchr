import json
from pathlib import Path

from benchr._results import (
    ExecutionResult,
    Measurement,
    execution_result_to_json,
    execution_result_from_json,
)
from conftest import make_execution, sample_execution_result


# ---------------------------------------------------------------------------
# ExecutionResult.info_columns
# ---------------------------------------------------------------------------

def test_info_columns_deduplicates_and_preserves_order():
    exe_a = make_execution(info={"variant": "a", "compiler": "gcc"})
    exe_b = make_execution(info={"compiler": "clang", "variant": "b"})
    result = ExecutionResult(measurements=[
        Measurement(execution=exe_a, metric="runtime", value=1.0),
        Measurement(execution=exe_b, metric="runtime", value=2.0),
    ])
    cols = result.info_columns()
    # First execution introduces "variant" then "compiler"; second has them
    # in reversed order but they should NOT be duplicated.
    assert cols == ["variant", "compiler"]


# ---------------------------------------------------------------------------
# ExecutionResult.metrics
# ---------------------------------------------------------------------------

def test_metrics_deduplicates_and_preserves_order():
    exe = make_execution()
    result = ExecutionResult(measurements=[
        Measurement(execution=exe, metric="runtime", value=1.0),
        Measurement(execution=exe, metric="max_rss", value=512),
        Measurement(execution=exe, metric="runtime", value=2.0),
    ])
    assert result.metrics() == ["runtime", "max_rss"]


# ---------------------------------------------------------------------------
# ExecutionResult.to_data_frame
# ---------------------------------------------------------------------------

def test_to_data_frame_basic():
    pd = __import__("pytest").importorskip("pandas")
    er = sample_execution_result()
    df = er.to_data_frame()
    # 4 measurements -> 4 rows
    assert len(df) == 4
    assert "metric" in df.reset_index().columns
    assert "value" in df.reset_index().columns


def test_to_data_frame_pivoted():
    pd = __import__("pytest").importorskip("pandas")
    er = sample_execution_result()
    df = er.to_data_frame(pivoted=True)
    cols = df.reset_index().columns.tolist()
    assert "runtime" in cols
    assert "max_rss" in cols
    # pivoted collapses rows: 2 benchmarks * 1 run each = 2 rows
    assert len(df) == 2


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

def test_json_round_trip():
    exe1 = make_execution(
        benchmark_name="bench_a",
        suite="suite_x",
        command=["./run", "--fast"],
        working_directory=Path("/work"),
        env={"CC": "gcc"},
        timeout=10.0,
        info={"variant": "opt"},
        run=1,
    )
    exe2 = make_execution(
        benchmark_name="bench_b",
        suite="suite_y",
        command=["./run2"],
        working_directory=Path("/work2"),
        env={},
        timeout=None,
        info={"variant": "debug", "arch": "x86"},
        run=2,
    )

    original = ExecutionResult(measurements=[
        Measurement(execution=exe1, metric="runtime", value=1.5, unit="s", lower_is_better=True),
        Measurement(execution=exe1, metric="max_rss", value=1024, unit="kB", lower_is_better=True),
        # unit="" and lower_is_better=None -- both should be omitted from JSON
        Measurement(execution=exe2, metric="score", value=42.0, unit="", lower_is_better=None),
        Measurement(execution=exe2, metric="throughput", value=100.0, unit="ops/s", lower_is_better=False),
    ])

    json_text = execution_result_to_json(original)
    restored = execution_result_from_json(json_text)

    assert len(restored.measurements) == len(original.measurements)

    for orig, rest in zip(original.measurements, restored.measurements):
        assert rest.execution.benchmark_name == orig.execution.benchmark_name
        assert rest.execution.suite == orig.execution.suite
        assert rest.execution.command == orig.execution.command
        assert rest.execution.working_directory == orig.execution.working_directory
        assert rest.execution.env == orig.execution.env
        assert rest.execution.timeout == orig.execution.timeout
        assert rest.execution.info == orig.execution.info
        assert rest.execution.run == orig.execution.run
        assert rest.metric == orig.metric
        assert rest.value == orig.value
        assert rest.unit == orig.unit
        assert rest.lower_is_better == orig.lower_is_better
        # parser is always None after deserialization
        assert rest.execution.parser is None

    # Verify that empty unit and None lower_is_better are omitted from JSON
    parsed = json.loads(json_text)
    score_md = parsed["executions"][1]["measurements"][0]
    assert score_md["metric"] == "score"
    assert "unit" not in score_md
    assert "lower_is_better" not in score_md


# ---------------------------------------------------------------------------
# Golden file test
# ---------------------------------------------------------------------------

def test_golden_file_clox():
    golden_path = Path(__file__).resolve().parent.parent / "examples" / "clox.json"
    text = golden_path.read_text()
    result = execution_result_from_json(text)
    assert len(result.measurements) > 0

    # Re-serialize and re-deserialize for round-trip check
    json_text2 = execution_result_to_json(result)
    result2 = execution_result_from_json(json_text2)
    assert len(result2.measurements) == len(result.measurements)

    for m1, m2 in zip(result.measurements, result2.measurements):
        assert m1.metric == m2.metric
        assert m1.value == m2.value
        assert m1.unit == m2.unit
        assert m1.lower_is_better == m2.lower_is_better
        assert m1.execution.benchmark_name == m2.execution.benchmark_name
