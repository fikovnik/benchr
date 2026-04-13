import abc
import argparse
import csv
import dataclasses
import json
import math
import os
import re
import resource
import shutil
import statistics
import subprocess
import tempfile
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from threading import Lock
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Iterator, Literal, Optional, Sequence

# --------------------------------------
#           HELPERS
# --------------------------------------


def const[T](x: T) -> Callable[..., T]:
    return lambda *args, **kwargs: x


# --------------------------------------
#           DEFINITIONS
# --------------------------------------

Env = dict[str, str]
Command = list[str]


class Parameters(SimpleNamespace):
    def __or__(self, other: "Parameters") -> "Parameters":
        return Parameters(**vars(self), **vars(other))

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    @staticmethod
    def from_namespace(ns: argparse.Namespace) -> "Parameters":
        """
        Convert the argparse.Namespace to Parameters
        """
        return Parameters(**vars(ns))


@dataclass
class SuccesfulProcessResult:
    execution: "Execution"
    runtime: float
    stdout: str
    stderr: str
    rusage: Optional[resource.struct_rusage]


@dataclass
class FailedProcessResult:
    execution: "Execution"
    runtime: Optional[float]
    stdout: Optional[str]
    stderr: Optional[str]
    rusage: Optional[resource.struct_rusage]

    returncode: int
    # Only set for pre-execution failures (command not found, spawn OSError).
    # For process exits `returncode` is the single source of truth: 124 means
    # timed out (by convention of coreutils' `timeout(1)`), any other non-zero
    # value is a crash, and 0 is unreachable here.
    reason: Optional[str] = None

    @staticmethod
    def empty(execution: "Execution", reason: str) -> "FailedProcessResult":
        return FailedProcessResult(
            execution=execution,
            runtime=None,
            stdout=None,
            stderr=None,
            rusage=None,
            returncode=0,
            reason=reason,
        )


ProcessResult = SuccesfulProcessResult | FailedProcessResult

# --------------------------------------
#          INPUT DEFINITIONS
# --------------------------------------


@dataclass
class Execution:
    """
    A definition of ready-to-run command
    """

    benchmark_name: str
    suite: str
    parser: Optional["ResultParser"]

    command: Command
    working_directory: Path
    env: Env

    timeout: Optional[float]

    info: dict[str, str]
    run: int = 1

    def as_identifier(self) -> str:
        id = f"{self.suite},{self.benchmark_name}"

        parts = [f"{k}={v}" for k, v in self.info.items()]
        parts.append(f"run={self.run}")
        if parts:
            id += " (" + ", ".join(parts) + ")"

        return id

    @dataclass
    class Incomplete:
        """
        A intermediate state of Execution
        """

        benchmark_name: str
        data: tuple[Any, ...] | Any
        keys: SimpleNamespace

        suite: str
        parser: Optional["ResultParser"]

        command: Optional[Command]
        working_directory: Optional[Path]
        env: Env

        timeout: Optional[float]

        info: dict[str, str]
        run: int = 1

        def finalize(self) -> "Execution":
            if self.parser is None:
                raise ValueError(
                    f"Benchmark {self.benchmark_name} in suite {self.suite} is missing a parser"
                )

            if self.working_directory is None:
                raise ValueError(
                    f"Benchmark {self.benchmark_name} in suite {self.suite} is missing working directory"
                )

            if self.command is None:
                raise ValueError(
                    f"Benchmark {self.benchmark_name} in suite {self.suite} is missing a command"
                )

            return Execution(
                benchmark_name=self.benchmark_name,
                suite=self.suite,
                parser=self.parser,
                command=self.command,
                working_directory=self.working_directory,
                env=self.env,
                timeout=self.timeout,
                info=self.info,
                run=self.run,
            )


@dataclass
class Benchmark:
    """
    A definition of one benchmark. data and keys can be any benchmark-specific
    data that are needed for its execution.

    `data` is specified by positional arguments. If there is a single argument,
    `data` will not be a tuple but just that argument.

    `keys` are specified by keyword arguments
    """

    name: str
    data: tuple[Any, ...] | Any
    keys: SimpleNamespace

    def __init__(self, name: str, *data: Any, **keys: Any) -> None:
        self.name = name

        if len(data) == 1:
            self.data = data[0]
        else:
            self.data = data

        self.keys = SimpleNamespace(keys)

    @staticmethod
    def from_files(*files: Path) -> list["Benchmark"]:
        """
        Create benchmarks from Path, where the name will be the filename
        without extension, and `keys.path` is the full given path
        """
        return [
            Benchmark(
                file.stem,
                path=file,
            )
            for file in files
        ]

    @staticmethod
    def from_folder(folder: Path, extension: Optional[str] = None) -> list["Benchmark"]:
        """
        Recursively walk the given folder, collecting all files with the given
        extension (or all if no extension is given) into Benchmarks
        """
        res = []
        for path, _, files in folder.walk():
            for file in files:
                p = path / file
                if extension is None or p.suffix.lower() == ("." + extension.lower()):
                    res.append(p)

        return Benchmark.from_files(*res)


B = Benchmark

# --------------------------------------
#          SUITES OF BENCHMARKS
# --------------------------------------


class BenchmarkCollection[This](abc.ABC):
    """
    Abstract superclass of Suite and Benchmark - implementation detail
    """

    @abc.abstractmethod
    def apply_suite_decorator(self, decorator: Callable[["Suite"], "Suite"]) -> This:
        """
        Apply a decorator to all Suites inside this collection
        """
        ...

    def matrix[T](
        self,
        matrix: "Matrix[T]",
    ) -> This:
        """
        Add a matrix parameter. Each benchmark is going to be duplicated with
        each instance having one value from `parameters`.

        If `working_directory` is not None, it will also change the working
        directory of each benchmark.

        If `env` is not None, it will add a new environment variables to the
        benchmarks environment.
        """
        return self.apply_suite_decorator(matrix.build)

    def runs(self, value: int) -> This:
        """
        Run each benchmark `value` times without any other modification
        """
        return self.apply_suite_decorator(lambda suite: RunsSuite(suite, value))

    def timeout(self, timeout: float) -> This:
        """
        Set the timeout of benchmarks (in seconds)
        """
        return self.apply_suite_decorator(
            lambda suite: TimeoutSuite(
                suite,
                timeout,
            )
        )


class Suite(BenchmarkCollection["Suite"]):
    """
    A collection of benchmarks
    """

    def apply_suite_decorator(self, decorator: Callable[["Suite"], "Suite"]) -> "Suite":
        return decorator(self)

    @abc.abstractmethod
    def get_executions(
        self, parameters: Parameters
    ) -> Iterator[Execution.Incomplete]: ...

    def to_config(self) -> "Config":
        """
        Create a simplified config with only one Suite
        """
        return Config([self])


class BaseSuite(Suite):
    name: str
    benchmarks: Callable[[Parameters], list[Benchmark]]

    command: Optional[Callable[[Parameters, Benchmark], Command]]
    working_directory: Optional[Callable[[Parameters, Benchmark], Path]]
    env: Callable[[Parameters, Benchmark], Env]

    parser: Optional["ResultParser"]

    def __init__(
        self,
        name: str,
        benchmarks: Callable[[Parameters], list[Benchmark]],
        command: Optional[Callable[[Parameters, Benchmark], Command]],
        working_directory: Optional[Callable[[Parameters, Benchmark], Path]],
        env: Callable[[Parameters, Benchmark], Env],
        parser: Optional["ResultParser"],
    ) -> None:
        self.name = name
        self.benchmarks = benchmarks

        self.command = command
        self.working_directory = working_directory
        self.env = env

        self.parser = parser

    def get_executions(self, parameters: Parameters) -> Iterator[Execution.Incomplete]:
        benchs = self.benchmarks(parameters)

        if len(benchs) == 0:
            raise ValueError(f"No benchmarks defined in {self.name}!")

        for b in benchs:
            if self.command is not None:
                command = self.command(parameters, b)
            else:
                command = None

            if self.working_directory is not None:
                working_directory = self.working_directory(parameters, b)
            else:
                working_directory = None

            env = self.env(parameters, b)

            yield Execution.Incomplete(
                benchmark_name=b.name,
                data=b.data,
                keys=b.keys,
                suite=self.name,
                command=command,
                parser=self.parser,
                working_directory=working_directory,
                env=env,
                timeout=None,
                info={},
            )


def suite(
    name: str,
    benchmarks: Sequence[Benchmark | str] | Callable[[Parameters], list[Benchmark]],
    *,
    command: Optional[Callable[[Parameters, Benchmark], Command]] = None,
    working_directory: Optional[Callable[[Parameters, Benchmark], Path] | Path] = None,
    env: Callable[[Parameters, Benchmark], Env] | Env = {},
    parser: Optional["ResultParser"] = None,
) -> Suite:
    """
    Flexible way of constructing a Suite
    """
    if not callable(benchmarks):
        benchmarks = const(
            [Benchmark(b) if isinstance(b, str) else b for b in benchmarks]
        )

    if working_directory is not None and not callable(working_directory):
        working_directory = const(working_directory)

    if not callable(env):
        env = const(env)

    return BaseSuite(
        name=name,
        benchmarks=benchmarks,
        command=command,
        working_directory=working_directory,
        env=env,
        parser=parser,
    )


# --------------------------------------
#          SUITE DECORATORS
# --------------------------------------


class SuiteDecorator(Suite):
    """
    A Suite that extends another Suite
    """

    parent: Suite

    def __init__(self, parent: Suite) -> None:
        self.parent = parent

    def get_executions(self, parameters: Parameters) -> Iterator[Execution.Incomplete]:
        for pexe in self.parent.get_executions(parameters):
            for exe in self.extend_execution(parameters, pexe):
                yield exe

    @abc.abstractmethod
    def extend_execution(
        self, parameters: Parameters, execution: Execution.Incomplete
    ) -> Iterator[Execution.Incomplete]:
        """
        This method needs to be implemented, as it is the one that extends the
        parent suite
        """
        ...


type MatrixCallable[T, R] = Callable[[Parameters, Execution.Incomplete, T], R]


class MatrixSuite[T](SuiteDecorator):
    name: str
    parameters: Sequence[T]

    matrix_command: Optional[MatrixCallable[T, Command]]
    matrix_working_directory: Optional[MatrixCallable[T, Path]]
    matrix_env: MatrixCallable[T, Env]
    matrix_info: Optional[Callable[[T], dict[str, str]]]

    def __init__(
        self,
        name: str,
        parent: Suite,
        parameters: Sequence[T],
        matrix_command: Optional[MatrixCallable[T, Command]],
        matrix_working_directory: Optional[MatrixCallable[T, Path]],
        matrix_env: MatrixCallable[T, Env],
        matrix_info: Optional[Callable[[T], dict[str, str]]],
    ) -> None:
        super().__init__(parent)

        self.name = name
        self.parameters = parameters

        self.matrix_command = matrix_command
        self.matrix_working_directory = matrix_working_directory
        self.matrix_env = matrix_env
        self.matrix_info = matrix_info

    def extend_execution(
        self, parameters: Parameters, execution: Execution.Incomplete
    ) -> Iterator[Execution.Incomplete]:
        for p in self.parameters:
            if self.matrix_command is not None:
                c = self.matrix_command(parameters, execution, p)
            else:
                c = execution.command

            e = execution.env | self.matrix_env(parameters, execution, p)

            if self.matrix_working_directory is not None:
                wd = self.matrix_working_directory(parameters, execution, p)
            else:
                wd = execution.working_directory

            if self.matrix_info is not None:
                i = execution.info | self.matrix_info(p)
            else:
                i = execution.info | {self.name: str(p)}

            yield dataclasses.replace(
                execution,
                command=c,
                env=e,
                working_directory=wd,
                info=i,
            )


class RunsSuite(SuiteDecorator):
    count: int

    def __init__(self, parent: Suite, count: int) -> None:
        super().__init__(parent)
        self.count = count

    def extend_execution(
        self, parameters: Parameters, execution: Execution.Incomplete
    ) -> Iterator[Execution.Incomplete]:
        for i in range(1, self.count + 1):
            yield dataclasses.replace(execution, run=i)


@dataclass
class Matrix[T]:
    """
    The MatrixSuite builder
    """

    name: str
    parameters: Sequence[T]

    matrix_command: Optional[MatrixCallable[T, Command]] = None
    matrix_working_directory: Optional[MatrixCallable[T, Path]] = None
    matrix_env: MatrixCallable[T, Env] = const({})
    matrix_info: Optional[Callable[[T], dict[str, str]]] = None

    def __init__(
        self,
        name: str,
        parameters: Sequence[T],
    ) -> None:
        self.name = name
        self.parameters = parameters

    def command(self, callback: Callable[[T], Command]):
        return self.command_full(lambda ps, ex, p: callback(p))

    def command_full(self, callback: MatrixCallable[T, Command]):
        if self.matrix_command is not None:
            raise ValueError("Multiple definitions of command")

        return dataclasses.replace(self, matrix_command=callback)

    def working_directory(self, callback: Callable[[T], Path]):
        return self.working_directory_full(lambda ps, ex, p: callback(p))

    def working_directory_full(self, callback: MatrixCallable[T, Path]):
        if self.matrix_working_directory is not None:
            raise ValueError("Multiple definitions of working directory")

        return dataclasses.replace(self, matrix_working_directory=callback)

    def env(self, name: Optional[str]):
        if name is None:
            name = self.name

        return self.env_callback_full(lambda ps, ex, p: {name: str(p)})

    def env_callback(self, callback: Callable[[T], Env]):
        return self.env_callback_full(lambda ps, ex, p: callback(p))

    def env_callback_full(self, callback: MatrixCallable[T, Env]):
        prev_mk_env = self.matrix_env
        mk_env = lambda ps, ex, p: prev_mk_env(ps, ex, p) | callback(ps, ex, p)

        return dataclasses.replace(self, matrix_env=mk_env)

    def info(self, callback: Callable[[T], dict[str, str]]):
        if self.matrix_info is not None:
            raise ValueError("Multiple definitions of info")

        return dataclasses.replace(self, matrix_info=callback)

    def build(self, suite: Suite) -> MatrixSuite:
        return MatrixSuite(
            name=self.name,
            parent=suite,
            parameters=self.parameters,
            matrix_command=self.matrix_command,
            matrix_working_directory=self.matrix_working_directory,
            matrix_env=self.matrix_env,
            matrix_info=self.matrix_info,
        )


class TimeoutSuite(SuiteDecorator):
    timeout_value: float

    def __init__(
        self,
        parent: Suite,
        timeout_value: float,
    ) -> None:
        super().__init__(parent)
        self.timeout_value = timeout_value

    def extend_execution(
        self, parameters: Parameters, execution: Execution.Incomplete
    ) -> Iterator[Execution.Incomplete]:
        execution.timeout = self.timeout_value
        yield execution


# --------------------------------------
#          CONFIGURATION
# --------------------------------------


@dataclass
class Config(BenchmarkCollection["Config"]):
    """
    The full configuration of all suites.
    """

    suites: list[Suite]

    default_parser: Optional["ResultParser"] = None
    default_command: Optional[Callable[[Parameters, Execution.Incomplete], Command]] = (
        None
    )
    default_working_directory: Optional[
        Callable[[Parameters, Execution.Incomplete], Path]
    ] = None
    default_env: Callable[[Parameters, Execution.Incomplete], Env] = const({})

    def parser(self, default_parser: "ResultParser") -> "Config":
        if self.default_parser is not None:
            raise ValueError("Multiple definitions of default parser")

        return dataclasses.replace(self, default_parser=default_parser)

    def command(
        self,
        default_command: Callable[[Parameters, Execution.Incomplete], Command]
        | Command,
    ) -> "Config":
        """
        Define a default command for all benchmarks
        """
        if self.default_command is not None:
            raise ValueError("Multiple definitions of default command")

        if not callable(default_command):
            default_command = const(default_command)

        return dataclasses.replace(self, default_command=default_command)

    def working_directory(
        self,
        default_working_directory: Callable[[Parameters, Execution.Incomplete], Path]
        | Path,
    ) -> "Config":
        """
        Define a default working directory for all benchmarks
        """
        if self.default_working_directory is not None:
            raise ValueError("Multiple definitions of default working directory")

        if not callable(default_working_directory):
            default_working_directory = const(default_working_directory)

        return dataclasses.replace(
            self, default_working_directory=default_working_directory
        )

    def env(
        self,
        default_env: Callable[[Parameters, Execution.Incomplete], Env] | Env,
    ) -> "Config":
        """
        Define a default environment for all benchmarks
        """
        if not callable(default_env):
            default_env = const(default_env)

        if self.default_env is not None:
            prev_default_env = self.default_env
            callback = lambda ps, e: prev_default_env(ps, e) | default_env(ps, e)
        else:
            callback = default_env

        return dataclasses.replace(self, default_env=callback)

    def get_executions(self, parameters: Parameters) -> list[Execution]:
        """
        Return all executions for this configuration
        """
        res = []
        for suite in self.suites:
            for exe in suite.get_executions(parameters):
                if exe.parser is None:
                    if self.default_parser is None:
                        raise ValueError(
                            f"No result parser for benchmark {exe.benchmark_name} in suite {exe.suite}"
                        )

                    exe.parser = self.default_parser

                if exe.command is None:
                    if self.default_command is None:
                        raise ValueError(
                            f"No command for benchmark {exe.benchmark_name} in suite {exe.suite}"
                        )

                    exe.command = self.default_command(parameters, exe)

                if exe.working_directory is None:
                    if self.default_working_directory is None:
                        raise ValueError(
                            f"No working directory for benchmark {exe.benchmark_name} in suite {exe.suite}"
                        )

                    exe.working_directory = self.default_working_directory(
                        parameters, exe
                    )

                exe.env = self.default_env(parameters, exe) | exe.env

                res.append(exe.finalize())
        return res

    def apply_suite_decorator(
        self, decorator: Callable[["Suite"], "Suite"]
    ) -> "Config":
        return dataclasses.replace(self, suites=list(map(decorator, self.suites)))


# --------------------------------------
#           RESULT DEFINITIONS
# --------------------------------------


@dataclass
class Measurement:
    execution: Execution
    metric: str
    value: float
    unit: str = ""
    # True = lower is better, False = higher is better, None = not comparable
    lower_is_better: Optional[bool] = None

    @staticmethod
    def runtime(
        execution: Execution,
        value: float,
        unit: str,
    ) -> "Measurement":
        return Measurement(
            execution=execution,
            metric="runtime",
            value=value,
            unit=unit,
        )


@dataclass
class ExecutionResult:
    measurements: list[Measurement] = dataclasses.field(default_factory=list)

    def info_columns(self) -> list[str]:
        """
        Get all info categories on all Executions
        """
        out = []

        for measure in self.measurements:
            for col in measure.execution.info.keys():
                if col not in out:
                    out.append(col)

        return out

    def metrics(self) -> list[str]:
        """
        Get all metrics in the result
        """
        out = []

        for measure in self.measurements:
            if measure.metric not in out:
                out.append(measure.metric)

        return out

    def to_data_frame(self, pivoted: bool = False, units: Optional[bool] = None):
        import pandas as pd

        if units is None:
            units = not pivoted

        info_cols = self.info_columns()

        rows = []
        for m in self.measurements:
            row: dict[str, Any] = {
                "benchmark": m.execution.benchmark_name,
                "suite": m.execution.suite,
                "run": m.execution.run,
            }

            for col in info_cols:
                row[col] = m.execution.info.get(col, "")

            row["lower_is_better"] = m.lower_is_better

            if pivoted:
                row[m.metric] = m.value
                if units:
                    row[m.metric + "_unit"] = m.unit
            else:
                row["metric"] = m.metric
                row["value"] = m.value
                if units:
                    row["unit"] = m.unit

            rows.append(row)

        index_cols = ["benchmark", "suite", "run"] + info_cols + ["lower_is_better"]

        df = pd.DataFrame(rows)
        if pivoted:
            df = df.groupby(index_cols).agg(
                lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else float("nan")
            )
        else:
            df.set_index(index_cols, inplace=True)
        return df


# --------------------------------------
#           SERIALIZATION
# --------------------------------------


def execution_result_to_json(result: ExecutionResult) -> str:
    """Serialize an ExecutionResult to a JSON string (indented for diffability)."""
    return json.dumps(_execution_result_to_dict(result), indent=2)


def execution_result_from_json(text: str) -> ExecutionResult:
    """Load an ExecutionResult from a JSON string produced by `execution_result_to_json`."""
    return _dict_to_execution_result(json.loads(text))


def _execution_result_to_dict(result: ExecutionResult) -> dict:
    # Group measurements by execution object to avoid repeating execution data
    grouped: dict[int, tuple[Execution, list[Measurement]]] = {}
    order: list[int] = []
    for m in result.measurements:
        key = id(m.execution)
        if key not in grouped:
            grouped[key] = (m.execution, [])
            order.append(key)
        grouped[key][1].append(m)

    return {
        "executions": [
            {
                "benchmark_name": exc.benchmark_name,
                "suite": exc.suite,
                "command": exc.command,
                "working_directory": str(exc.working_directory),
                "env": exc.env,
                "timeout": exc.timeout,
                "info": exc.info,
                "run": exc.run,
                "measurements": [
                    {
                        "metric": m.metric,
                        "value": m.value,
                        **({"unit": m.unit} if m.unit else {}),
                        **(
                            {"lower_is_better": m.lower_is_better}
                            if m.lower_is_better is not None
                            else {}
                        ),
                    }
                    for m in measurements
                ],
            }
            for exc, measurements in (grouped[k] for k in order)
        ]
    }


def _dict_to_execution_result(d: dict) -> ExecutionResult:
    measurements: list[Measurement] = []
    for ed in d["executions"]:
        execution = Execution(
            benchmark_name=ed["benchmark_name"],
            suite=ed["suite"],
            parser=None,  # loaded results are never re-executed
            command=ed["command"],
            working_directory=Path(ed["working_directory"]),
            env=ed["env"],
            timeout=ed["timeout"],
            info=ed["info"],
            run=ed.get("run", 1),
        )
        for md in ed["measurements"]:
            measurements.append(
                Measurement(
                    execution=execution,
                    metric=md["metric"],
                    value=md["value"],
                    unit=md.get("unit", ""),
                    lower_is_better=md.get("lower_is_better"),
                )
            )
    return ExecutionResult(measurements=measurements)


_META_METRICS = {"failed"}

COMPARE_STAT = statistics.median


def _extract_unique_names(paths: list[Path]) -> list[str]:
    """
    Extract short, unique display names from a list of file paths.
    Strips extension and any common path prefix/suffix components.
    Example: ['vm1/results.csv', 'vm2/results.csv'] -> ['vm1', 'vm2']
    """
    if not paths:
        return []
    if len(paths) == 1:
        return [paths[0].stem]

    parts_list = [list(p.with_suffix("").parts) for p in paths]

    while all(len(p) > 1 for p in parts_list) and len({p[0] for p in parts_list}) == 1:
        for p in parts_list:
            p.pop(0)

    while all(len(p) > 1 for p in parts_list) and len({p[-1] for p in parts_list}) == 1:
        for p in parts_list:
            p.pop()

    return ["/".join(p) for p in parts_list]


MetricKey = tuple[str, str]  # (metric_name, unit)

# A hashable, canonically ordered snapshot of the `execution.info` entries
# that identify one benchmark variant, *excluding* the per-run counter.
#
# Stored as a tuple of ``(key, value)`` pairs sorted by key, so two variants
# with the same logical info compare equal regardless of the original
# insertion order in `Execution.info`. The tuple form is required because
# this value is used as a dict key when grouping/indexing variants for
# summary and comparison (see `_group_execution_result`, `compare_and_print`).
VariantInfo = tuple[tuple[str, str], ...]


def _variant_info_of(execution: "Execution") -> VariantInfo:
    """Build the `VariantInfo` identity from an execution's info dict."""
    return tuple(sorted(execution.info.items()))


@dataclass
class BenchmarkRunCounts:
    """Outcome counts for all runs of a single benchmark variant."""

    failures: int  # runs that did not produce a successful result (crashes and timeouts alike)
    successes: int  # runs that produced parsed measurements


@dataclass
class BenchmarkGroup:
    """
    All runs of one benchmark variant, aggregated for summary/comparison.

    A "variant" is a unique (suite, benchmark, info-minus-run) triple: for a
    benchmark with `.runs(2)` there is one BenchmarkGroup containing values
    for both runs; for a MatrixSuite there is one BenchmarkGroup per matrix
    cell.
    """

    suite: str
    benchmark: str
    # Info items that disambiguate this variant (excludes "run"). See
    # `VariantInfo` for the exact shape and rationale.
    info: VariantInfo
    # Measured values across runs, grouped by (metric, unit).
    metrics: dict[MetricKey, list[float]]
    run_counts: BenchmarkRunCounts


@dataclass
class GroupedResult:
    """
    An ExecutionResult reshaped for summary and comparison.
    """

    name: str  # display label (e.g. JSON filename stem, or "current")
    benchmarks: list[BenchmarkGroup]  # one entry per unique benchmark variant
    # Per-metric direction: True = lower is better, False = higher is better.
    # Metrics absent from this map are not comparable.
    lower_is_better: dict[MetricKey, bool]


def _group_execution_result(result: ExecutionResult, name: str) -> GroupedResult:
    """
    Reshape an ExecutionResult into a GroupedResult for comparison/summary.

    Groups measurements by (suite, benchmark, info-minus-run), folds the
    meta-metric "failed" into run counts, and collects per-metric
    `lower_is_better` annotations.
    """
    # Scratch buckets keyed by (suite, benchmark, non_run_info)
    variant_order: list[tuple] = []
    variant_metrics: dict[tuple, dict[MetricKey, list[float]]] = {}
    variant_runs: dict[tuple, set[int]] = {}  # observed "run" info values
    variant_fails: dict[tuple, int] = {}
    lower_is_better: dict[MetricKey, bool] = {}

    def remember(identity: tuple) -> None:
        if identity not in variant_metrics:
            variant_metrics[identity] = {}
            variant_order.append(identity)

    for m in result.measurements:
        variant_info = _variant_info_of(m.execution)
        identity = (m.execution.suite, m.execution.benchmark_name, variant_info)
        remember(identity)

        variant_runs.setdefault(identity, set()).add(m.execution.run)

        if m.metric == "failed" and m.value == 1:
            variant_fails[identity] = variant_fails.get(identity, 0) + 1

        if m.metric in _META_METRICS:
            continue

        metric_key = (m.metric, m.unit)
        if m.lower_is_better is not None:
            lower_is_better[metric_key] = m.lower_is_better

        per_variant = variant_metrics[identity]
        per_variant.setdefault(metric_key, []).append(m.value)

    benchmarks: list[BenchmarkGroup] = []
    for identity in variant_order:
        suite, bench, info = identity
        total = len(variant_runs.get(identity, set())) or 1
        failed = variant_fails.get(identity, 0)
        benchmarks.append(
            BenchmarkGroup(
                suite=suite,
                benchmark=bench,
                info=info,
                metrics=variant_metrics[identity],
                run_counts=BenchmarkRunCounts(
                    failures=failed,
                    successes=total - failed,
                ),
            )
        )

    return GroupedResult(
        name=name, benchmarks=benchmarks, lower_is_better=lower_is_better
    )


# --------------------------------------
#           PARSERS
# --------------------------------------


class ResultParser(abc.ABC):
    """
    Parse stdout and stderr into results
    """

    @abc.abstractmethod
    def parse(self, process_result: ProcessResult) -> ExecutionResult: ...

    def ignore_fail(self) -> "ResultParser":
        """
        Ignore failed executions, parsing them as succesful
        """
        return IgnoreFailParserDecorator(self)

    def lower_is_better(self) -> "ResultParser":
        """Tag every parsed measurement as a lower-is-better metric."""
        return DirectionParserDecorator(self, lower_is_better=True)

    def higher_is_better(self) -> "ResultParser":
        """Tag every parsed measurement as a higher-is-better metric."""
        return DirectionParserDecorator(self, lower_is_better=False)

    def __and__(self, other) -> "ResultParser":
        return MixedResultParser(self, other)


class MixedResultParser(ResultParser):
    """
    Multiple parsers posing as one
    """

    parsers: list[ResultParser]

    @staticmethod
    def canonize(parsers: Iterable[ResultParser]) -> Iterator[ResultParser]:
        """
        Flatten the representation of MixedResultParser (one in another)
        """
        for parser in parsers:
            if isinstance(parser, MixedResultParser):
                for subparser in parser.parsers:
                    yield subparser
            else:
                yield parser

    def __init__(self, *parsers: ResultParser) -> None:
        self.parsers = list(canon_p for canon_p in MixedResultParser.canonize(parsers))

    def parse(self, process_result: ProcessResult) -> ExecutionResult:
        result = ExecutionResult()

        for parser in self.parsers:
            result.measurements += parser.parse(process_result).measurements

        return result


class PlainFloatParser(ResultParser):
    """
    Try to parse simple floats on each line as seconds. Only on succesful
    runs.
    """

    unit: str
    metric: str

    def __init__(self, unit: str, metric: str = "runtime") -> None:
        self.unit = unit
        self.metric = metric

    def parse(self, process_result: ProcessResult) -> ExecutionResult:
        if isinstance(process_result, FailedProcessResult):
            return ExecutionResult()

        result = ExecutionResult()

        for line in process_result.stdout.split("\n"):
            try:
                value = float(line)
                result.measurements.append(
                    Measurement(
                        execution=process_result.execution,
                        metric=self.metric,
                        value=value,
                        unit=self.unit,
                    )
                )
            except ValueError:
                pass

        return result


class LineParser(ResultParser):
    """
    Extract a single non-empty line from stdout/stderr and pass it to a subparser.

    The line parameter selects which non-empty line to extract:
    - Positive values are 1-based from the top (1 = first, 2 = second, ...)
    - Negative values index from the bottom (-1 = last, -2 = second to last, ...)
    - 0 is forbidden
    """

    subparser: ResultParser
    line: int

    def __init__(self, subparser: ResultParser, line: int = -1) -> None:
        if line == 0:
            raise ValueError(
                "line must be non-zero (positive from top, negative from bottom)"
            )
        self.subparser = subparser
        self.line = line

    @staticmethod
    def _select_line(text: str, line: int) -> str:
        lines = [l for l in text.split("\n") if l.strip() != ""]
        try:
            # Convert 1-based positive index to 0-based
            idx = line - 1 if line > 0 else line
            return lines[idx]
        except IndexError:
            return ""

    def parse(self, process_result: ProcessResult) -> ExecutionResult:
        if isinstance(process_result, FailedProcessResult):
            return ExecutionResult()

        return self.subparser.parse(
            dataclasses.replace(
                process_result,
                stdout=self._select_line(process_result.stdout, self.line),
                stderr=self._select_line(process_result.stderr, self.line),
            )
        )


class RegexParser(ResultParser):
    """
    Parse the output of a succesful run based on a regex
    """

    type MatchGroup = str | int
    type OutputType = Literal["stdout", "stderr", "both"]

    metric: str
    regex: re.Pattern[str]
    output: OutputType

    match_group: MatchGroup
    process: Callable[[str], float]

    unit: Optional[str]
    unit_match_group: Optional[MatchGroup]

    iterations: bool

    def __init__(
        self,
        metric: str,
        regex: re.Pattern[str],
        output: OutputType,
        match_group: MatchGroup,
        process: Callable[[str], float] = float,
        unit: Optional[str] = None,
        unit_match_group: Optional[MatchGroup] = None,
        iterations: bool = False,
    ) -> None:
        self.metric = metric
        self.regex = regex
        self.output = output

        self.match_group = match_group
        self.process = process

        if unit is None and unit_match_group is None:
            raise ValueError("Missing unit specification")
        self.unit = unit
        self.unit_match_group = unit_match_group

        self.iterations = iterations

    def parse(self, process_result: ProcessResult) -> ExecutionResult:
        if isinstance(process_result, FailedProcessResult):
            return ExecutionResult()

        result = ExecutionResult()
        if self.output == "stdout":
            outputs = [process_result.stdout]
        elif self.output == "stderr":
            outputs = [process_result.stderr]
        elif self.output == "both":
            outputs = [process_result.stdout, process_result.stderr]
        else:
            raise ValueError(f"Unknown output type {self.output}")

        for output in outputs:
            pos = 0
            while (match := self.regex.search(output, pos)) is not None:
                pos = match.end()
                value = self.process(match.group(self.match_group))

                if self.unit_match_group is not None:
                    unit = match.group(self.unit_match_group)
                elif self.unit is not None:
                    unit = self.unit
                else:
                    unit = ""

                result.measurements.append(
                    Measurement(
                        process_result.execution,
                        self.metric,
                        value,
                        unit,
                    )
                )

        return result


class RebenchParser(ResultParser):
    """
    Format used by the ReBench (https://github.com/smarr/ReBench) benchmarker,
    mostly copied from the RebenchLogAdapter. The supported format is:
    ```
    optional_prefix: benchmark_name optional_criterion: iterations=123 runtime: 1000[ms|us]
    ```
    or for non-runtime
    ```
    optional_prefix: benchmark_name: criterion: number_with_unit
    ```

    Unlike ReBench, benchr only reports runtime in ms. Runtime report with other
    criterion other than "total" (or none) are ignored.

    When a runtime with no criterion (or criterion "total") or non-runtime
    criterion "total" is parsed, a new iteration is assumed. This should be
    equivalent to ReBench.
    """

    re_log_line = re.compile(
        r"^(?:.*: )?([^\s]+)( [\w\.]+)?: iterations=([0-9]+) "
        + r"runtime: (?P<runtime>(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)"
        + r"(?P<unit>[mu])s"
    )

    re_extra_criterion_log_line = re.compile(
        r"^(?:.*: )?([^\s]+): (?P<criterion>[^:]{1,30}):\s*"
        + r"(?P<value>(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)"
        + r"(?P<unit>[a-zA-Z]+)"
    )

    def parse(self, process_result: ProcessResult) -> ExecutionResult:
        if process_result.stdout is None:
            return ExecutionResult()

        result = ExecutionResult()

        for line in process_result.stdout.split("\n"):
            match = self.re_log_line.match(line)
            if match is not None:
                # Match runtime
                time = float(match.group("runtime"))
                if match.group("unit") == "u":
                    time /= 1000

                # Match criterion, maybe skip
                criterion = match.group(2)
                if criterion is not None and criterion.strip() != "total":
                    continue

                result.measurements.append(
                    Measurement(
                        process_result.execution,
                        "runtime",
                        time,
                        "ms",
                    )
                )
                continue

            match = self.re_extra_criterion_log_line.match(line)
            if match is not None:
                # Match groups
                value = float(match.group("value"))
                unit = match.group("unit")
                criterion = match.group("criterion")

                # Add measurement
                result.measurements.append(
                    Measurement(
                        process_result.execution,
                        criterion,
                        value,
                        unit,
                    )
                )
                continue

        return result


class SingleResourceUsageParser(ResultParser):
    RUsageField = Literal[
        "ru_utime",
        "ru_stime",
        "ru_maxrss",
        "ru_ixrss",
        "ru_idrss",
        "ru_isrss",
        "ru_minflt",
        "ru_majflt",
        "ru_nswap",
        "ru_inblock",
        "ru_oublock",
        "ru_msgsnd",
        "ru_msgrcv",
        "ru_nsignals",
        "ru_nvcsw",
        "ru_nivcsw",
    ]

    field: RUsageField
    metric: str
    unit: str

    def __init__(self, field: RUsageField, metric: str, unit: str) -> None:
        self.field = field
        self.metric = metric
        self.unit = unit

    def parse(self, process_result: ProcessResult) -> ExecutionResult:
        if process_result.rusage is None:
            return ExecutionResult()

        value = getattr(process_result.rusage, self.field)

        # MacOS reports in B, not kB
        if sys.platform == "darwin" and self.field == "ru_maxrss":
            value /= 1024

        return ExecutionResult(
            [
                Measurement(
                    execution=process_result.execution,
                    metric=self.metric,
                    value=value,
                    unit=self.unit,
                )
            ]
        )


def MaxRssParser() -> ResultParser:
    return SingleResourceUsageParser("ru_maxrss", "max_rss", "kB")


class TimeParser(ResultParser):
    """
    Emit up to three time measurements (in seconds): "elapsed" (wall clock,
    from process_result.runtime), "user" (rusage.ru_utime), and
    "system" (rusage.ru_stime). At least one flag must be true.
    """

    elapsed: bool
    system: bool
    user: bool

    def __init__(
        self, elapsed: bool = True, system: bool = False, user: bool = False
    ) -> None:
        if not (elapsed or system or user):
            raise ValueError(
                "TimeParser requires at least one of elapsed, system, user to be True"
            )
        self.elapsed = elapsed
        self.system = system
        self.user = user

    def parse(self, process_result: ProcessResult) -> ExecutionResult:
        measurements: list[Measurement] = []

        if self.elapsed and process_result.runtime is not None:
            measurements.append(
                Measurement(
                    execution=process_result.execution,
                    metric="elapsed",
                    value=process_result.runtime,
                    unit="s",
                )
            )

        if process_result.rusage is not None:
            if self.user:
                measurements.append(
                    Measurement(
                        execution=process_result.execution,
                        metric="user",
                        value=process_result.rusage.ru_utime,
                        unit="s",
                    )
                )
            if self.system:
                measurements.append(
                    Measurement(
                        execution=process_result.execution,
                        metric="system",
                        value=process_result.rusage.ru_stime,
                        unit="s",
                    )
                )

        return ExecutionResult(measurements)


class FailedParser(ResultParser):
    def parse(self, process_result: ProcessResult) -> ExecutionResult:
        return ExecutionResult(
            [
                Measurement(
                    execution=process_result.execution,
                    metric="failed",
                    value=1 if isinstance(process_result, FailedProcessResult) else 0,
                )
            ]
        )


# --------------------------------------
#           PARSER DECORATORS
# --------------------------------------


class IgnoreFailParserDecorator(ResultParser):
    subparser: ResultParser

    def __init__(self, subparser: ResultParser) -> None:
        self.subparser = subparser

    def parse(self, process_result: ProcessResult) -> ExecutionResult:
        if isinstance(process_result, FailedProcessResult):
            process_result = SuccesfulProcessResult(
                execution=process_result.execution,
                runtime=process_result.runtime or -1,
                stdout=process_result.stdout or "",
                stderr=process_result.stderr or "",
                rusage=process_result.rusage,
            )

        return self.subparser.parse(process_result)


class DirectionParserDecorator(ResultParser):
    subparser: ResultParser
    _lib: bool  # True = lower is better, False = higher is better

    def __init__(self, subparser: ResultParser, lower_is_better: bool) -> None:
        self.subparser = subparser
        self._lib = lower_is_better

    def parse(self, process_result: ProcessResult) -> ExecutionResult:
        result = self.subparser.parse(process_result)

        for m in result.measurements:
            m.lower_is_better = self._lib

        return result


# --------------------------------------
#           TUI
# --------------------------------------


class TUI:
    if sys.stdout.isatty():
        RESET = "\033[0m"
        BOLD = "\033[1m"
        BLACK = "\033[30m"
        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        MAGENTA = "\033[35m"
        CYAN = "\033[36m"
        WHITE = "\033[37m"
    else:
        RESET = ""
        BOLD = ""
        BLACK = ""
        RED = ""
        GREEN = ""
        YELLOW = ""
        BLUE = ""
        MAGENTA = ""
        CYAN = ""
        WHITE = ""


# --------------------------------------
#           REPORTERS
# --------------------------------------


class Reporter(abc.ABC):
    """
    Reports the results. Lifecycle:
      1. start(executions) - once before any execution runs.
      2. report(process_result, parsed) - once per finished execution.
      3. finalize() - once after all executions have completed.
    """

    def start(self, executions: list[Execution]) -> None:
        """Called once before any execution runs. Default no-op."""
        pass

    @abc.abstractmethod
    def report(self, process_result: ProcessResult, parsed: ExecutionResult) -> None:
        """Called once per finished execution (success or failure)."""
        ...

    def finalize(self) -> None:
        """Called once after all executions have completed. Default no-op."""
        pass


class MixedReporter(Reporter):
    """
    Multiple reporters posing as one
    """

    reporters: list[Reporter]

    def __init__(self, *reporters: Reporter) -> None:
        self.reporters = list(reporters)

    def start(self, executions: list[Execution]) -> None:
        for r in self.reporters:
            r.start(executions)

    def report(self, process_result: "ProcessResult", parsed: ExecutionResult):
        for r in self.reporters:
            r.report(process_result, parsed)

    def finalize(self) -> None:
        for r in self.reporters:
            r.finalize()


class CsvReporter(Reporter):
    """
    Report into CSV file. Streams rows to disk as executions finish.

    The header schema (info columns) is fixed from the first reported result;
    subsequent measurements are written using that same schema.
    """

    filepath: Path
    separator: str

    def __init__(self, filepath: Path, separator: str = ",") -> None:
        self.filepath = filepath
        self.separator = separator
        self._file = None
        self._info_cols: Optional[list[str]] = None

    @staticmethod
    def escape_text(text: str) -> str:
        if "," in text:
            return '"' + text.replace('"', r"\"") + '"'

        return text

    def format_line(self, line: list[str]) -> str:
        return self.separator.join(map(self.escape_text, line)) + "\n"

    def start(self, executions: list[Execution]) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.filepath, "wt")
        self._info_cols = None

    def report(self, process_result: ProcessResult, parsed: ExecutionResult):
        if self._file is None:
            # Reporter used without start(); open lazily.
            self.start([])
        assert self._file is not None

        if self._info_cols is None:
            self._info_cols = parsed.info_columns()
            columns = (
                ["benchmark", "suite", "run"]
                + self._info_cols
                + ["lower_is_better", "metric", "value", "unit"]
            )
            self._file.write(self.format_line(columns))

        for measure in parsed.measurements:
            line: list[str] = [
                measure.execution.benchmark_name,
                measure.execution.suite,
                str(measure.execution.run),
            ]

            for col in self._info_cols:
                line.append(measure.execution.info.get(col, ""))

            lib_str = "" if measure.lower_is_better is None else str(measure.lower_is_better)
            line += [lib_str, measure.metric, str(measure.value), measure.unit]

            self._file.write(self.format_line(line))

        self._file.flush()

    def finalize(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class JsonReporter(Reporter):
    """
    Report as a single JSON file (ExecutionResult serialized).
    Buffers measurements in memory and writes the file on finalize().
    """

    filepath: Path

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self._result = ExecutionResult()

    def report(self, process_result: ProcessResult, parsed: ExecutionResult):
        self._result.measurements.extend(parsed.measurements)

    def finalize(self) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "wt") as file:
            file.write(execution_result_to_json(self._result))


class TableReporter(Reporter):
    """
    Report into CLI. Buffers measurements and prints the full table on
    finalize().
    """

    def __init__(self) -> None:
        self._result = ExecutionResult()

    def report(self, process_result: "ProcessResult", parsed: ExecutionResult):
        self._result.measurements.extend(parsed.measurements)

    def finalize(self) -> None:
        self.print_result(self._result)

    def print_result(self, result: ExecutionResult):
        info_cols = result.info_columns()

        def lib_str(m: Measurement) -> str:
            return "" if m.lower_is_better is None else str(m.lower_is_better)

        # Measure widths
        benchmark_col_w = len("benchmark")
        suite_col_w = len("suite")
        run_w = len("run")
        info_cols_w = {info: len(info) for info in info_cols}
        lib_w = len("lower_is_better")
        metric_w = len("metric")
        value_w = len("value")
        unit_w = len("unit")

        for measure in result.measurements:
            benchmark_col_w = max(
                len(measure.execution.benchmark_name), benchmark_col_w
            )
            suite_col_w = max(len(measure.execution.suite), suite_col_w)
            run_w = max(len(str(measure.execution.run)), run_w)

            for i in info_cols:
                info_cols_w[i] = max(
                    len(measure.execution.info.get(i, "")), info_cols_w[i]
                )

            lib_w = max(len(lib_str(measure)), lib_w)
            metric_w = max(len(measure.metric), metric_w)
            value_w = max(len(str(measure.value)), value_w)
            unit_w = max(len(measure.unit), unit_w)

        # Print header
        sep_size = sum(
            [
                benchmark_col_w + 2,
                suite_col_w + 2,
                run_w + 2,
                sum(info_cols_w.values()),
                len(info_cols_w) * 2,
                lib_w + 2,
                metric_w + 2,
                value_w + 2,
                unit_w,
            ]
        )

        print("\n" + "-" * sep_size)
        print("benchmark".ljust(benchmark_col_w + 2), end="")
        print("suite".ljust(suite_col_w + 2), end="")
        print("run".ljust(run_w + 2), end="")

        for i in info_cols:
            print(i.ljust(info_cols_w[i] + 2), end="")

        print("lower_is_better".ljust(lib_w + 2), end="")
        print("metric".ljust(metric_w + 2), end="")
        print("value".ljust(value_w + 2), end="")
        print("unit".ljust(unit_w), end="")
        print("\n" + "-" * sep_size)

        # Print
        for measure in result.measurements:
            print(measure.execution.benchmark_name.ljust(benchmark_col_w + 2), end="")
            print(measure.execution.suite.ljust(suite_col_w + 2), end="")
            print(str(measure.execution.run).ljust(run_w + 2), end="")

            for i in info_cols:
                print(
                    measure.execution.info.get(i, "").ljust(
                        info_cols_w[i] + 2,
                    ),
                    end="",
                )

            print(lib_str(measure).ljust(lib_w + 2), end="")
            print(measure.metric.ljust(metric_w + 2), end="")
            print(str(measure.value).ljust(value_w + 2), end="")
            print((measure.unit).ljust(unit_w + 2), end="")
            print()

        print("-" * sep_size)


class SummaryReporter(Reporter):
    """
    Buffers measurements and prints a compact statistical summary on
    finalize(). Reuses `_group_execution_result` so the grouping stays in
    lockstep with `compare_and_print`.
    """

    def __init__(self) -> None:
        self._result = ExecutionResult()

    def report(self, process_result: ProcessResult, parsed: ExecutionResult):
        self._result.measurements.extend(parsed.measurements)

    def finalize(self) -> None:
        self.print_result(self._result)

    def print_result(self, result: ExecutionResult):
        grouped = _group_execution_result(result, name="current")

        print()
        for i, group in enumerate(grouped.benchmarks):
            if i > 0:
                print()
            self._print_benchmark_group(group)

    @staticmethod
    def _scale_unit(mean_val: float, unit: str) -> tuple[float, str]:
        abs_val = abs(mean_val)
        if unit == "s":
            if 0 < abs_val < 0.001:
                return 1e6, "\u00b5s"
            if 0 < abs_val < 1:
                return 1e3, "ms"
        elif unit == "kB":
            if abs_val >= 1024 * 1024:
                return 1 / (1024 * 1024), "GB"
            if abs_val >= 1024:
                return 1 / 1024, "MB"
        return 1, unit

    def _print_benchmark_group(self, group: BenchmarkGroup):
        name = f"{group.suite}/{group.benchmark}"
        if group.info:
            info_str = ", ".join(f"{k}={v}" for k, v in group.info)
            name += f" ({info_str})"

        rc = group.run_counts
        total_runs = (rc.failures + rc.successes) or 1
        run_word = "run" if total_runs == 1 else "runs"

        f_s = f"{TUI.RED}{rc.failures}{TUI.RESET}" if rc.failures else str(rc.failures)
        s_s = f"{TUI.GREEN}{rc.successes}{TUI.RESET}"

        print(f"{TUI.BOLD}{name}:{TUI.RESET} {f_s}/{s_s} {run_word}")

        if not group.metrics:
            return

        # Pre-compute scaled units and labels for alignment
        scaled_info: dict[MetricKey, tuple[float, str]] = {}
        for metric_key, values in group.metrics.items():
            _, unit = metric_key
            mean_val = statistics.mean(values)
            scaled_info[metric_key] = self._scale_unit(mean_val, unit)

        multi_run = total_runs > 1
        suffix = " (mean \u00b1 \u03c3):" if multi_run else ":"
        labels = {k: f"{k[0]} [{scaled_info[k][1]}]{suffix}" for k in group.metrics}
        max_label_w = max(len(l) for l in labels.values())

        for metric_key, values in group.metrics.items():
            label = labels[metric_key].ljust(max_label_w)
            scale, _ = scaled_info[metric_key]
            n = len(values)
            mean_val = statistics.mean(values) * scale

            if n >= 2:
                stddev_val = statistics.stdev(values) * scale
                min_val = min(values) * scale
                max_val = max(values) * scale

                print(
                    f"  {TUI.BOLD}{label}{TUI.RESET}"
                    f"  {TUI.GREEN}{TUI.BOLD}{mean_val:.2f}{TUI.RESET}"
                    f" \u00b1 {TUI.GREEN}{stddev_val:.2f}{TUI.RESET}"
                    f"    ({TUI.CYAN}{min_val:.2f}{TUI.RESET}"
                    f" \u2026 {TUI.MAGENTA}{max_val:.2f}{TUI.RESET})"
                )
            else:
                print(f"  {label}  {TUI.GREEN}{TUI.BOLD}{mean_val:.2f}{TUI.RESET}")


class DirReporter(Reporter):
    """
    Stream per-execution raw artifacts into a directory tree:

        <output_dir>/<suite>/<benchmark>/<run_id>/
            seq, stdout, stderr, exitcode, rusage, result.csv

    Stable per-benchmark run ids are pre-computed in submission order from
    the list passed to `start`, so numbering is deterministic under parallel
    execution.
    """

    output_dir: Path

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._run_ids: dict[int, int] = {}

    def start(self, executions: list[Execution]) -> None:
        self._run_ids = {}
        counts: dict[tuple[str, str], int] = {}
        for exe in executions:
            key = (exe.suite, exe.benchmark_name)
            counts[key] = counts.get(key, 0) + 1
            self._run_ids[id(exe)] = counts[key]

    def report(self, process_result: ProcessResult, parsed: ExecutionResult):
        pr = process_result
        exe = pr.execution
        run_id = self._run_ids.get(id(exe), exe.run)
        run_dir = self.output_dir / exe.suite / exe.benchmark_name / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        # seq: cwd, command, then info k=v lines
        seq_lines = [str(exe.working_directory), " ".join(exe.command)]
        seq_lines.append(f"run={exe.run}")
        seq_lines.extend(f"{k}={v}" for k, v in exe.info.items())
        (run_dir / "seq").write_text("\n".join(seq_lines) + "\n")

        if pr.stdout is not None:
            (run_dir / "stdout").write_text(pr.stdout)
        if pr.stderr is not None:
            (run_dir / "stderr").write_text(pr.stderr)

        if isinstance(pr, SuccesfulProcessResult):
            (run_dir / "exitcode").write_text("0\n")
        else:
            (run_dir / "exitcode").write_text(f"{pr.returncode}\n")

        if pr.rusage is not None:
            ru_lines = [
                f"{field}={getattr(pr.rusage, field)}"
                for field in dir(pr.rusage)
                if field.startswith("ru_")
            ]
            (run_dir / "rusage").write_text("\n".join(ru_lines) + "\n")

        # Per-execution result.csv: drive a fresh CsvReporter so the schema
        # matches the top-level CSV exactly.
        csv = CsvReporter(run_dir / "result.csv")
        csv.start([])
        csv.report(pr, parsed)
        csv.finalize()


def compare_and_print(datasets: list[GroupedResult]):
    """
    Compare N grouped result sets. The first dataset is the baseline; all
    subsequent datasets are compared against it.
    """
    if len(datasets) < 2:
        return

    baseline = datasets[0]
    others = datasets[1:]

    def center(values: list[float]) -> float:
        if not values:
            return float("nan")
        return COMPARE_STAT(values)

    def stdev(values: list[float]) -> float:
        if len(values) >= 2:
            return statistics.stdev(values)
        return 0.0

    # Index each dataset by (suite, benchmark, info) so we can look up a group
    # by identity regardless of its position in `.benchmarks`.
    def _index(ds: GroupedResult) -> dict[tuple[str, str, VariantInfo], BenchmarkGroup]:
        return {(g.suite, g.benchmark, g.info): g for g in ds.benchmarks}

    other_indices = [(ds, _index(ds)) for ds in others]

    all_lower_is_better: dict[MetricKey, bool] = {}
    for ds in datasets:
        all_lower_is_better.update(ds.lower_is_better)

    def compare_pair(
        bl_center: float,
        bl_sd: float,
        other_center: float,
        other_sd: float,
        lower_is_better: bool,
    ) -> tuple[float, float, str]:
        """Return (display_ratio, sigma, word) where word is 'better' or 'worse'."""
        if lower_is_better:
            is_better = other_center < bl_center
        else:
            is_better = other_center > bl_center

        if is_better:
            ratio = (
                (bl_center / other_center)
                if lower_is_better
                else (other_center / bl_center)
            )
            word = "better"
        else:
            ratio = (
                (other_center / bl_center)
                if lower_is_better
                else (bl_center / other_center)
            )
            word = "worse"

        rel_err_sq = 0.0
        if bl_center != 0 and bl_sd > 0:
            rel_err_sq += (bl_sd / bl_center) ** 2
        if other_center != 0 and other_sd > 0:
            rel_err_sq += (other_sd / other_center) ** 2
        sigma = ratio * math.sqrt(rel_err_sq)
        return ratio, sigma, word

    def format_ratio_line(
        indent: str,
        name: str,
        ratio: float,
        sigma: float,
        word: str,
        baseline_name: str,
    ) -> str:
        err_str = f" \u00b1 {sigma:.2f}" if sigma > 0 else ""
        word_color = TUI.GREEN if word == "better" else TUI.RED
        word_str = f"{word_color}{TUI.BOLD}{word}{TUI.RESET}"
        return (
            f"{indent}{TUI.MAGENTA}{name}{TUI.RESET} was"
            f" {TUI.GREEN}{TUI.BOLD}{ratio:.2f}{TUI.RESET}{err_str}"
            f" times {word_str} than {TUI.GREEN}{TUI.BOLD}{baseline_name}{TUI.RESET}"
        )

    def format_runs(name: str, rc: BenchmarkRunCounts) -> str:
        f_s = f"{TUI.RED}{rc.failures}{TUI.RESET}" if rc.failures else str(rc.failures)
        s_s = f"{TUI.GREEN}{rc.successes}{TUI.RESET}"
        return f"{name}: {f_s} failed / {s_s} succeeded"

    # Per-benchmark comparison (baseline identities only)
    print()
    first = True
    for bl_group in baseline.benchmarks:
        identity = (bl_group.suite, bl_group.benchmark, bl_group.info)

        present_others = [
            (ds, idx[identity]) for ds, idx in other_indices if identity in idx
        ]
        if not present_others:
            continue

        if not first:
            print()
        first = False

        name = f"{bl_group.suite}/{bl_group.benchmark}"
        if bl_group.info:
            info_str = ", ".join(f"{k}={v}" for k, v in bl_group.info)
            name += f" ({info_str})"
        print(f"{TUI.BOLD}{name}:{TUI.RESET}")

        print(f"  runs:")
        print(f"    {format_runs(baseline.name, bl_group.run_counts)}")
        for ds, other_group in present_others:
            print(f"    {format_runs(ds.name, other_group.run_counts)}")

        for metric_key, bl_vals in bl_group.metrics.items():
            if metric_key not in all_lower_is_better:
                continue
            metric, unit = metric_key
            lower_is_better = all_lower_is_better[metric_key]

            bl_c = center(bl_vals)
            bl_sd = stdev(bl_vals)
            if bl_c == 0 or math.isnan(bl_c):
                continue

            metric_printed = False
            for ds, other_group in present_others:
                other_vals = other_group.metrics.get(metric_key)
                if not other_vals:
                    continue
                other_c = center(other_vals)
                other_sd = stdev(other_vals)
                if other_c == 0 or math.isnan(other_c):
                    continue

                if not metric_printed:
                    print(f"  {TUI.CYAN}{metric}{TUI.RESET}:")
                    metric_printed = True

                ratio, sigma, word = compare_pair(
                    bl_c, bl_sd, other_c, other_sd, lower_is_better
                )
                print(
                    format_ratio_line(
                        "    ", ds.name, ratio, sigma, word, baseline.name
                    )
                )

    # Summary: per-suite geometric mean of raw ratios
    suites_in_order: list[str] = []
    for g in baseline.benchmarks:
        if g.suite not in suites_in_order:
            suites_in_order.append(g.suite)

    if not suites_in_order:
        return

    print(f"\n{TUI.BOLD}Summary (geometric mean of ratios):{TUI.RESET}")

    for suite in suites_in_order:
        suite_groups = [g for g in baseline.benchmarks if g.suite == suite]
        print(f"  {TUI.BOLD}{suite}:{TUI.RESET}")

        def sum_counts(groups: list[BenchmarkGroup]) -> BenchmarkRunCounts:
            f = s = 0
            for g in groups:
                f += g.run_counts.failures
                s += g.run_counts.successes
            return BenchmarkRunCounts(f, s)

        def sum_counts_for(
            ds_index: dict[tuple[str, str, VariantInfo], BenchmarkGroup],
        ) -> BenchmarkRunCounts:
            matched = [
                ds_index[(g.suite, g.benchmark, g.info)]
                for g in suite_groups
                if (g.suite, g.benchmark, g.info) in ds_index
            ]
            return sum_counts(matched)

        # Aggregated run counts per dataset across benchmarks in this suite
        print(f"    runs:")
        print(f"      {format_runs(baseline.name, sum_counts(suite_groups))}")
        for ds, idx in other_indices:
            print(f"      {format_runs(ds.name, sum_counts_for(idx))}")

        # Collect metrics present in the baseline for this suite
        suite_metric_keys: list[MetricKey] = []
        for g in suite_groups:
            for mk in g.metrics:
                if mk not in suite_metric_keys:
                    suite_metric_keys.append(mk)

        for metric_key in suite_metric_keys:
            if metric_key not in all_lower_is_better:
                continue
            metric, unit = metric_key
            lower_is_better = all_lower_is_better[metric_key]

            metric_printed = False
            for ds, idx in other_indices:
                # Collect raw ratios (other/baseline) and their relative errors
                ratios: list[float] = []
                rel_errs_sq: list[float] = []
                for bl_g in suite_groups:
                    key = (bl_g.suite, bl_g.benchmark, bl_g.info)
                    other_g = idx.get(key)
                    if other_g is None:
                        continue
                    bl_vals = bl_g.metrics.get(metric_key)
                    other_vals = other_g.metrics.get(metric_key)
                    if not bl_vals or not other_vals:
                        continue
                    bl_c = center(bl_vals)
                    other_c = center(other_vals)
                    if (
                        bl_c == 0
                        or other_c == 0
                        or math.isnan(bl_c)
                        or math.isnan(other_c)
                    ):
                        continue
                    raw = other_c / bl_c
                    rsq = 0.0
                    bl_sd = stdev(bl_vals)
                    other_sd = stdev(other_vals)
                    if bl_sd > 0:
                        rsq += (bl_sd / bl_c) ** 2
                    if other_sd > 0:
                        rsq += (other_sd / other_c) ** 2
                    ratios.append(raw)
                    rel_errs_sq.append(rsq)

                if not ratios:
                    continue

                geo_raw = math.exp(statistics.mean(math.log(r) for r in ratios))

                if lower_is_better:
                    is_better = geo_raw < 1
                    display = (1 / geo_raw) if is_better else geo_raw
                else:
                    is_better = geo_raw > 1
                    display = geo_raw if is_better else (1 / geo_raw)
                word = "better" if is_better else "worse"

                # Uncertainty on geomean via error propagation on log-space average
                N = len(ratios)
                sigma_log = math.sqrt(sum(rel_errs_sq)) / N if rel_errs_sq else 0.0
                sigma_display = display * sigma_log

                if not metric_printed:
                    print(f"    {TUI.CYAN}{metric}{TUI.RESET}:")
                    metric_printed = True

                print(
                    format_ratio_line(
                        "      ", ds.name, display, sigma_display, word, baseline.name
                    )
                )


# --------------------------------------
#           EXECUTORS
# --------------------------------------


class Executor(abc.ABC):
    """
    Execute the executions
    """

    @abc.abstractmethod
    def execute(self, execution: Execution):
        """
        Run single execution - implementation detail, use `execute_all`
        """
        ...

    def execute_all(self, executions: list[Execution]) -> Optional[ExecutionResult]:
        """
        Run all executions - this is the prefered way of running executions
        """
        for execution in executions:
            self.execute(execution)
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class DefaultExecutor(Executor):
    """
    The main Executor
    """

    all_executions: Optional[int]
    finished_executions: int
    failed_executions: int

    reporter: Reporter

    result: ExecutionResult

    def __init__(self, reporter: Reporter) -> None:
        self.all_executions = None
        self.finished_executions = 0
        self.failed_executions = 0

        self.reporter = reporter

        self.result = ExecutionResult()

    def execute_all(self, executions: list[Execution]) -> ExecutionResult:
        self.all_executions = len(executions)
        self.reporter.start(executions)
        super().execute_all(executions)
        return self.result

    def execute(self, execution: Execution):
        cmd = shutil.which(execution.command[0])
        if cmd is None:
            self.error_execution(
                FailedProcessResult.empty(
                    execution=execution,
                    reason=f"Command not found ({execution.command[0]})",
                )
            )
            return

        execution.command[0] = cmd

        self.start_execution(execution)

        stdout_file = tempfile.TemporaryFile()
        stderr_file = tempfile.TemporaryFile()
        try:
            proc = subprocess.Popen(
                execution.command,
                cwd=execution.working_directory,
                env=execution.env,
                stdin=None,
                stdout=stdout_file,
                stderr=stderr_file,
                shell=False,
            )
            starttime = time.monotonic()

            rusage = None
            waitstatus = None
            timed_out = False
            if execution.timeout is not None:
                stoptime = time.monotonic() + execution.timeout

                # Poll for the child with WNOHANG until it exits or the
                # timeout is exceeded. The child has not been reaped yet, so
                # os.wait4 cannot raise ChildProcessError here.
                while True:
                    pid, waitstatus, rusage = os.wait4(proc.pid, os.WNOHANG)
                    assert pid == proc.pid or pid == 0

                    if pid == proc.pid:
                        break  # child exited, waitstatus/rusage populated

                    if stoptime - time.monotonic() <= 0:
                        timed_out = True
                        proc.kill()
                        break

                    time.sleep(0.01)

            # Reap the child if the polling loop did not already get it
            # (either because there was no timeout, or because we just killed
            # it after the timeout expired).
            if waitstatus is None or timed_out:
                _, waitstatus, rusage = os.wait4(proc.pid, 0)

            endtime = time.monotonic()
            runtime = endtime - starttime
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read().decode(errors="replace")
            stderr = stderr_file.read().decode(errors="replace")
            returncode = 124 if timed_out else os.waitstatus_to_exitcode(waitstatus)

            if returncode != 0:
                result = FailedProcessResult(
                    execution=execution,
                    runtime=runtime,
                    stdout=stdout,
                    stderr=stderr,
                    rusage=rusage,
                    returncode=returncode,
                )
                self.error_execution(result)
            else:
                result = SuccesfulProcessResult(
                    execution=execution,
                    runtime=runtime,
                    stdout=stdout,
                    stderr=stderr,
                    rusage=rusage,
                )

            self.finalize(result)

        except OSError as e:
            self.error_execution(
                FailedProcessResult.empty(
                    execution=execution,
                    reason=str(e),
                )
            )
        finally:
            stdout_file.close()
            stderr_file.close()

    def start_execution(self, execution: Execution) -> None:
        print(
            "["
            + f"{TUI.RED}{TUI.BOLD}{self.failed_executions}{TUI.RESET}"
            + f"/{TUI.GREEN}{TUI.BOLD}{self.finished_executions}{TUI.RESET}"
            + (
                f"/{TUI.BLUE}{TUI.BOLD}{self.all_executions}{TUI.RESET}"
                if self.all_executions is not None
                else ""
            )
            + "] "
            + execution.as_identifier()
            + "\n",
            end="",
        )

    def error_execution(self, process_result: FailedProcessResult):
        self.failed_executions += 1
        lines = [
            f"{TUI.RED}{TUI.BOLD}Error in {process_result.execution.as_identifier()}{TUI.RESET}"
        ]
        if process_result.returncode == 124:
            lines.append(
                f"Program timed out after {process_result.execution.timeout} seconds"
            )
        elif process_result.returncode != 0:
            lines.append(
                f"Program ended with non-zero return code ({process_result.returncode})"
            )
        else:
            # Pre-execution failure (command not found, spawn OSError).
            lines.append(process_result.reason or "Unknown error")
        print("\n".join(lines), file=sys.stderr)

    def finalize(self, process_result: ProcessResult) -> None:
        self.finished_executions += 1
        parser = process_result.execution.parser
        assert parser is not None, "finalize only called for fully-resolved executions"
        parsed = parser.parse(process_result)
        self.result.measurements.extend(parsed.measurements)
        self.reporter.report(process_result, parsed)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.reporter.finalize()
        return False


class ParallelExecutor(DefaultExecutor):
    """
    An executor that runs multiple tasks in parallel - mostly usable for
    collecting metrics other that runtime
    """

    pool: ThreadPoolExecutor
    futures: list[Future]
    lock: Lock
    in_process_runs: int
    last_info: Optional[str]

    def __init__(
        self,
        ncores: int,
        reporter: Reporter,
    ) -> None:
        super().__init__(reporter)

        self.pool = ThreadPoolExecutor(max_workers=ncores)
        self.futures = []
        self.lock = Lock()
        self.in_process_runs = 0
        self.last_info = None

    def execute(self, execution: Execution):
        future = self.pool.submit(super().execute, execution)
        self.futures.append(future)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.pool.shutdown(wait=True)
        exceptions = [
            e for e in map(lambda f: f.exception(None), self.futures) if e is not None
        ]

        if len(exceptions) != 0:
            raise ExceptionGroup("In ParallelExecutor:", exceptions)  # type: ignore
        super().__exit__(*args)
        return False

    def print_execution(self):
        assert self.last_info is not None

        print(
            "["
            + f"{TUI.MAGENTA}{TUI.BOLD}{self.in_process_runs}{TUI.RESET}"
            + f"/{TUI.RED}{TUI.BOLD}{self.failed_executions}{TUI.RESET}"
            + f"/{TUI.GREEN}{TUI.BOLD}{self.finished_executions}{TUI.RESET}"
            + (
                f"/{TUI.BLUE}{TUI.BOLD}{self.all_executions}{TUI.RESET}"
                if self.all_executions is not None
                else ""
            )
            + "] "
            + self.last_info
            + "\n",
            end="",
        )

    def start_execution(self, execution: Execution) -> None:
        with self.lock:
            self.in_process_runs += 1
            self.last_info = execution.as_identifier()
            self.print_execution()

    def error_execution(self, process_result: FailedProcessResult):
        with self.lock:
            super().error_execution(process_result)

    def finalize(self, process_result: ProcessResult) -> None:
        with self.lock:
            self.in_process_runs -= 1
            super().finalize(process_result)
            self.print_execution()


class DryExecutor(Executor):
    """
    Pseudo-executor which only prints the execution plan
    """

    def execute(self, execution: Execution):
        print("Execution:", " ".join(execution.command))
        print("Working working_directory:", str(execution.working_directory))
        print("Environment: ", end="")
        pprint(execution.env)
        print("Info: ", end="")
        pprint(execution.info)
        print("-" * 10)


# --------------------------------------
#          ARGUMENT PARSING
# --------------------------------------


def make_argparser(*params: str, **kwarg_params: Any) -> argparse.ArgumentParser:
    """
    Create a default argument parser from the given parameters
    """
    parser = argparse.ArgumentParser()
    user = parser.add_argument_group("User specified parameters")

    for p in params:
        user.add_argument(
            f"--{p}",
            metavar="str",
            type=str,
            required=True,
            dest=p,
        )

    for k, v in kwarg_params.items():
        t = type(v) if v is not None else str
        user.add_argument(
            f"--{k}",
            help=f"(Default: {v})",
            metavar=t.__name__,
            type=t,
            default=v,
            dest=k,
        )

    return parser


def parse_params(*params: str, **kwarg_params: Any) -> Parameters:
    """
    Create a default argument parser and run it on argv
    """
    parser = make_argparser(*params, **kwarg_params)
    args = parser.parse_args()
    return Parameters.from_namespace(args)


# --------------------------------------
#          DEFAULT MAIN
# --------------------------------------


def main(
    config: Config,
    params: list[str],
    kwarg_params: dict[str, Any],
    reporter: Optional[Reporter] = None,
    executor: Optional[Executor] = None,
) -> Optional[ExecutionResult]:
    """
    Sane default main. config is the benchmarks configuration that will be
    executed, params is a list of required parameters from the user,
    kwarg_params are optional parameters with their default value.

    Three independent output flags are available: `--output-csv <file>`,
    `--output-json <file>` and `--output <dir>` (full per-execution tree).
    They may be combined; if none is given, results are only summarized to
    the terminal via SummaryReporter.

    If no executor is specified, it defaults to DefaultExecutor, which can be
    changed with CLI arguments --dry and --jobs/-j.
    """
    parser = make_argparser(*params, **kwarg_params)

    defp = parser.add_argument_group("Default benchr parameters")
    defp.add_argument(
        "--output",
        help="Directory to export a full per-execution tree "
        "(<suite>/<benchmark>/<run>/{stdout,stderr,seq,exitcode,rusage,result.csv})",
        metavar="dir",
        type=str,
        default=None,
        dest="__output",
    )
    defp.add_argument(
        "--output-csv",
        help="Export results as a single CSV file",
        metavar="file",
        type=str,
        default=None,
        dest="__output_csv",
    )
    defp.add_argument(
        "--output-json",
        help="Export results as a single JSON file (ExecutionResult)",
        metavar="file",
        type=str,
        default=None,
        dest="__output_json",
    )

    if executor is None:
        defp.add_argument(
            "--jobs",
            "-j",
            help="Allow this many runs in parallel (Default: 1)",
            metavar="jobs",
            type=int,
            default=1,
            dest="__jobs",
        )
        defp.add_argument(
            "--dry",
            help="Do not run, only print what would be runned",
            action="store_true",
            dest="__dry",
        )

    defp.add_argument(
        "--compare",
        help="Compare results against baseline JSON file(s); first is baseline",
        metavar="json",
        nargs="+",
        type=str,
        default=None,
        dest="__compare",
    )

    ps = Parameters.from_namespace(parser.parse_args())

    executions = list(config.get_executions(ps))

    if executor is None:
        if ps.__dry:
            executor = DryExecutor()
        else:
            if reporter is None:
                reporters: list[Reporter] = [SummaryReporter()]
                if ps.__output_csv:
                    reporters.append(CsvReporter(Path(ps.__output_csv)))
                if ps.__output_json:
                    reporters.append(JsonReporter(Path(ps.__output_json)))
                if ps.__output:
                    reporters.append(DirReporter(Path(ps.__output).resolve()))
                reporter = (
                    reporters[0] if len(reporters) == 1 else MixedReporter(*reporters)
                )

            if ps.__jobs > 1:
                executor = ParallelExecutor(ps.__jobs, reporter)
            else:
                executor = DefaultExecutor(reporter)

    result = None
    try:
        with executor:
            result = executor.execute_all(executions)
    except KeyboardInterrupt:
        print("Interrupted")
        sys.exit(0)

    if ps.__compare is not None and result is not None:
        baseline_paths = [Path(f) for f in ps.__compare]
        baseline_names = _extract_unique_names(baseline_paths)
        baselines = [execution_result_from_json(p.read_text()) for p in baseline_paths]
        datasets = [
            _group_execution_result(r, n) for r, n in zip(baselines, baseline_names)
        ]
        datasets.append(_group_execution_result(result, "current"))
        compare_and_print(datasets)

    return result


# TODO: Run info - date, reflog

# --------------------------------------
#           EXPORTS
# --------------------------------------

__all__ = [
    # Definitions
    "Env",
    "Command",
    "Parameters",
    "ProcessResult",
    "SuccesfulProcessResult",
    "FailedProcessResult",
    # Input definitions
    "Execution",
    "Benchmark",
    "B",
    # Suites of benchmarks
    "Suite",
    "suite",
    # Suite decorators
    "SuiteDecorator",
    "MatrixSuite",
    "Matrix",
    "TimeoutSuite",
    # Configuration
    "Config",
    # Result definitions
    "Measurement",
    "ExecutionResult",
    # Serialization
    "execution_result_to_json",
    "execution_result_from_json",
    # Grouped (comparison) types
    "MetricKey",
    "VariantInfo",
    "BenchmarkRunCounts",
    "BenchmarkGroup",
    "GroupedResult",
    # Parsers
    "ResultParser",
    "MixedResultParser",
    "PlainFloatParser",
    "LineParser",
    "RegexParser",
    "RebenchParser",
    "SingleResourceUsageParser",
    "MaxRssParser",
    "TimeParser",
    "FailedParser",
    # Parser decorators
    "IgnoreFailParserDecorator",
    "DirectionParserDecorator",
    # Reporters
    "Reporter",
    "MixedReporter",
    "CsvReporter",
    "JsonReporter",
    "TableReporter",
    "SummaryReporter",
    "DirReporter",
    # Executors
    "Executor",
    "DefaultExecutor",
    "ParallelExecutor",
    "DryExecutor",
    # ArgumentParsing
    "make_argparser",
    "parse_params",
    # Comparison
    "compare_and_print",
    # Default main
    "main",
    # Reexports
    "Path",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchr - benchmark comparison tool")
    sub = parser.add_subparsers(dest="command")

    compare_parser = sub.add_parser(
        "compare",
        help="Compare benchmark JSON result files; first file is the baseline",
    )
    compare_parser.add_argument(
        "files", nargs="+", type=str, help="JSON result files to compare"
    )

    args = parser.parse_args()

    if args.command == "compare":
        files = [Path(f) for f in args.files]
        for f in files:
            if not f.exists():
                print(f"Error: file not found: {f}", file=sys.stderr)
                sys.exit(1)

        names = _extract_unique_names(files)
        results = [execution_result_from_json(f.read_text()) for f in files]

        if len(results) == 1:
            SummaryReporter().print_result(results[0])
        else:
            grouped = [_group_execution_result(r, n) for r, n in zip(results, names)]
            compare_and_print(grouped)
    else:
        parser.print_help()
        sys.exit(1)
