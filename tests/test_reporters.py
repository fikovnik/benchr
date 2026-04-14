import csv
import json
from pathlib import Path

from benchr._output import (
    CsvReporter,
    JsonReporter,
    TableReporter,
    SummaryReporter,
    DirReporter,
    MixedReporter,
    Reporter,
)
from benchr._types import Execution, ProcessResult, SuccessfulProcessResult
from benchr._results import Measurement, ExecutionResult

from conftest import make_execution, make_success, sample_execution_result


def _make_process_and_parsed():
    """Helper: build a SuccessfulProcessResult and its parsed ExecutionResult."""
    exe = make_execution(benchmark_name="bench1", suite="suite1", info={"variant": "a"})
    pr = SuccessfulProcessResult(
        execution=exe, runtime=1.5, stdout="out", stderr="err", rusage=None
    )
    parsed = ExecutionResult(
        measurements=[
            Measurement(execution=exe, metric="runtime", value=1.5, unit="s"),
            Measurement(execution=exe, metric="max_rss", value=1024, unit="kB"),
        ]
    )
    return exe, pr, parsed


def test_csv_reporter(tmp_path):
    """CsvReporter writes a header row and data rows to a CSV file."""
    filepath = tmp_path / "results.csv"
    reporter = CsvReporter(filepath)
    exe, pr, parsed = _make_process_and_parsed()

    reporter.start([exe])
    reporter.report(pr, parsed)
    reporter.finalize()

    with open(filepath, "r", newline="") as f:
        reader = list(csv.reader(f))

    assert len(reader) == 3  # header + 2 measurements
    header = reader[0]
    assert "benchmark" in header
    assert "metric" in header
    assert "value" in header
    assert reader[1][header.index("benchmark")] == "bench1"
    assert reader[1][header.index("metric")] == "runtime"
    assert reader[1][header.index("value")] == "1.5"
    assert reader[2][header.index("metric")] == "max_rss"


def test_json_reporter(tmp_path):
    """JsonReporter writes valid JSON that round-trips."""
    filepath = tmp_path / "results.json"
    reporter = JsonReporter(filepath)
    exe, pr, parsed = _make_process_and_parsed()

    reporter.report(pr, parsed)
    reporter.finalize()

    text = filepath.read_text()
    data = json.loads(text)
    assert "executions" in data
    assert len(data["executions"]) == 1
    assert len(data["executions"][0]["measurements"]) == 2
    # Verify round-trip: re-serialize and compare
    text2 = json.dumps(data, indent=2)
    assert json.loads(text2) == data


def test_table_reporter(capsys):
    """TableReporter output contains benchmark names and metric values."""
    reporter = TableReporter()
    exe, pr, parsed = _make_process_and_parsed()

    reporter.report(pr, parsed)
    reporter.finalize()

    captured = capsys.readouterr()
    assert "bench1" in captured.out
    assert "runtime" in captured.out
    assert "1.5" in captured.out


def test_summary_reporter(capsys):
    """SummaryReporter output is non-empty after finalize."""
    reporter = SummaryReporter()
    exe, pr, parsed = _make_process_and_parsed()

    reporter.start([exe])
    reporter.report(pr, parsed)
    reporter.finalize()

    captured = capsys.readouterr()
    assert len(captured.out.strip()) > 0


def test_dir_reporter(tmp_path):
    """DirReporter creates the expected directory tree."""
    output = tmp_path / "output"
    reporter = DirReporter(output)
    exe, pr, parsed = _make_process_and_parsed()

    reporter.start([exe])
    reporter.report(pr, parsed)

    run_dir = output / "suite1" / "bench1" / "1"
    assert run_dir.exists()
    assert (run_dir / "stdout").exists()
    assert (run_dir / "stderr").exists()
    assert (run_dir / "exitcode").exists()
    assert (run_dir / "seq").exists()
    assert (run_dir / "result.csv").exists()
    assert (run_dir / "exitcode").read_text().strip() == "0"


def test_mixed_reporter_delegates():
    """MixedReporter delegates start/report/finalize to all children."""

    class TrackingReporter(Reporter):
        def __init__(self):
            self.started = False
            self.reported = False
            self.finalized = False

        def start(self, executions):
            self.started = True

        def report(self, process_result, parsed):
            self.reported = True

        def finalize(self):
            self.finalized = True

    r1 = TrackingReporter()
    r2 = TrackingReporter()
    mixed = MixedReporter(r1, r2)

    exe, pr, parsed = _make_process_and_parsed()

    mixed.start([exe])
    assert r1.started and r2.started

    mixed.report(pr, parsed)
    assert r1.reported and r2.reported

    mixed.finalize()
    assert r1.finalized and r2.finalized
