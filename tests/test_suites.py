import pytest
from pathlib import Path

from benchr._types import Parameters, Execution, Benchmark
from benchr._parsers import ResultParser, PlainFloatParser
from benchr._suites import (
    suite,
    BaseSuite,
    MatrixSuite,
    Matrix,
    RunsSuite,
    TimeoutSuite,
    Config,
)
from benchr._results import ExecutionResult


class _DummyParser(ResultParser):
    def parse(self, process_result):
        return ExecutionResult()


def test_suite_factory_with_string_benchmarks():
    """Strings passed to suite() are converted to Benchmark objects."""
    s = suite("s1", ["alpha", "beta"])
    exes = list(s.get_executions(Parameters()))
    names = [e.benchmark_name for e in exes]
    assert names == ["alpha", "beta"]
    for e in exes:
        assert isinstance(e, Execution.Incomplete)


def test_suite_factory_with_benchmark_objects():
    """Benchmark objects passed directly are used as-is."""
    b1 = Benchmark("b1")
    b2 = Benchmark("b2")
    s = suite("s2", [b1, b2])
    exes = list(s.get_executions(Parameters()))
    assert [e.benchmark_name for e in exes] == ["b1", "b2"]


def test_base_suite_get_executions():
    """BaseSuite.get_executions produces Execution.Incomplete per benchmark."""
    s = suite(
        "my_suite",
        ["x", "y"],
        command=lambda ps, b: ["run", b.name],
        working_directory=Path("/tmp"),
        parser=_DummyParser(),
    )
    exes = list(s.get_executions(Parameters()))
    assert len(exes) == 2
    for e in exes:
        assert isinstance(e, Execution.Incomplete)
        assert e.suite == "my_suite"
    assert exes[0].command == ["run", "x"]
    assert exes[1].command == ["run", "y"]


def test_empty_benchmarks_raises_value_error():
    """An empty benchmarks list raises ValueError."""
    s = suite("empty", [])
    with pytest.raises(ValueError, match="No benchmarks"):
        list(s.get_executions(Parameters()))


def test_matrix_suite():
    """2 benchmarks x 2 matrix params = 4 executions with info populated."""
    base = suite("mat", ["a", "b"])
    # Matrix default info uses {name: str(p)} when no custom info callback is set
    m = Matrix("opt", ["O0", "O2"])
    decorated = base.matrix(m)
    exes = list(decorated.get_executions(Parameters()))
    assert len(exes) == 4
    infos = [e.info for e in exes]
    assert {"opt": "O0"} in infos
    assert {"opt": "O2"} in infos
    # Each benchmark appears twice
    names = [e.benchmark_name for e in exes]
    assert names.count("a") == 2
    assert names.count("b") == 2


def test_runs_suite():
    """.runs(3) produces 3 copies with run=1,2,3."""
    base = suite("r", ["x"])
    decorated = base.runs(3)
    exes = list(decorated.get_executions(Parameters()))
    assert len(exes) == 3
    runs = [e.run for e in exes]
    assert runs == [1, 2, 3]


def test_timeout_suite_does_not_mutate_original():
    """.timeout(10.0) sets timeout on executions but does NOT mutate original."""
    base = suite("t", ["x"])
    original_exes = list(base.get_executions(Parameters()))
    assert original_exes[0].timeout is None

    decorated = base.timeout(10.0)
    timeout_exes = list(decorated.get_executions(Parameters()))
    assert timeout_exes[0].timeout == 10.0

    # Original is unaffected
    original_exes2 = list(base.get_executions(Parameters()))
    assert original_exes2[0].timeout is None


def test_config_get_executions_applies_defaults():
    """Config.get_executions applies default parser, command, working_directory, env."""
    parser = _DummyParser()
    s = suite("cfg", ["a"])
    cfg = (
        Config([s])
        .parser(parser)
        .command(["echo", "hello"])
        .working_directory(Path("/tmp"))
        .env({"MY_VAR": "1"})
    )
    exes = cfg.get_executions(Parameters())
    assert len(exes) == 1
    e = exes[0]
    assert isinstance(e, Execution)
    assert e.parser is parser
    assert e.command == ["echo", "hello"]
    assert e.working_directory == Path("/tmp")
    assert "MY_VAR" in e.env
    assert e.env["MY_VAR"] == "1"


def test_config_missing_parser_raises():
    """Config.get_executions without a parser raises ValueError."""
    s = suite("no_parser", ["a"])
    cfg = Config([s]).command(["echo"]).working_directory(Path("/tmp"))
    with pytest.raises(ValueError, match="parser"):
        cfg.get_executions(Parameters())
