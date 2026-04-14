from pathlib import Path

from benchr._output import DryExecutor, DefaultExecutor, Reporter
from benchr._parsers import PlainFloatParser, ResultParser
from benchr._types import Execution, ProcessResult
from benchr._results import ExecutionResult, Measurement


class CollectingReporter(Reporter):
    """A simple reporter that collects results into a list for assertion."""

    def __init__(self):
        self.results: list[ExecutionResult] = []
        self.process_results: list[ProcessResult] = []
        self.started = False
        self.finalized = False

    def start(self, executions):
        self.started = True

    def report(self, process_result, parsed):
        self.process_results.append(process_result)
        self.results.append(parsed)

    def finalize(self):
        self.finalized = True


def _make_exec(command, parser=None, working_directory=None, timeout=None):
    """Helper to build an Execution for testing."""
    if parser is None:
        parser = PlainFloatParser(unit="s")
    if working_directory is None:
        working_directory = Path("/tmp")
    return Execution(
        benchmark_name="test_bench",
        suite="test_suite",
        parser=parser,
        command=command,
        working_directory=working_directory,
        env={},
        timeout=timeout,
        info={},
        run=1,
    )


def test_dry_executor_prints_command(capsys):
    """DryExecutor prints command and working directory without running anything."""
    exe = _make_exec(["echo", "42"], working_directory=Path("/tmp"))
    executor = DryExecutor()
    executor.execute(exe)

    captured = capsys.readouterr()
    assert "echo 42" in captured.out
    assert "/tmp" in captured.out


def test_default_executor_echo():
    """DefaultExecutor runs a real echo command and parses the output."""
    reporter = CollectingReporter()
    exe = _make_exec(["echo", "42"])

    with DefaultExecutor(reporter) as executor:
        executor.execute_all([exe])

    assert reporter.started
    assert len(reporter.results) == 1
    measurements = reporter.results[0].measurements
    assert len(measurements) == 1
    assert measurements[0].value == 42.0
    assert measurements[0].metric == "runtime"


def test_default_executor_command_not_found():
    """DefaultExecutor handles a nonexistent command gracefully."""
    reporter = CollectingReporter()
    exe = _make_exec(["__nonexistent_command_xyz__", "arg1"])

    with DefaultExecutor(reporter) as executor:
        executor.execute_all([exe])

    assert executor.failed_executions == 1


def test_default_executor_non_zero_exit():
    """DefaultExecutor reports failure on non-zero exit code."""
    reporter = CollectingReporter()
    exe = _make_exec(["python3", "-c", "import sys; sys.exit(1)"])

    with DefaultExecutor(reporter) as executor:
        executor.execute_all([exe])

    assert executor.failed_executions == 1
