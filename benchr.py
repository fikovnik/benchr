import abc
import argparse
import dataclasses
import functools
import inspect
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from pprint import pprint
from threading import Lock
from types import ClassMethodDescriptorType, SimpleNamespace
from typing import Any, Callable, Iterator, Literal, Optional, Self


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


# --------------------------------------
#           HELPER
# --------------------------------------


# TODO: maybe adjust api, maybe remove
def run_cmd(*args, **kwargs):
    """
    Run a very simple command, wrapper around subprocess.run
    """
    return subprocess.run(*args, check=True, **kwargs)


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

    command: Command
    parser: "ResultParser"
    working_directory: Path
    env: Env
    info: dict[str, str]

    def as_identifier(self) -> str:
        id = f"{self.suite},{self.benchmark_name}"

        if len(self.info) != 0:
            id += " ("
            id += ", ".join((f"{k}={v}" for k, v in self.info.items()))
            id += ")"

        return id


@dataclass
class Benchmark:
    """
    A definition of one benchmark. data can be any benchmark-specific data
    that are needed for its execution.
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
        return [
            Benchmark(
                file.stem,
                path=file,
            )
            for file in files
        ]

    @staticmethod
    def from_folder(folder: Path, extension: Optional[str] = None) -> list["Benchmark"]:
        res = []
        for path, _, files in folder.walk():
            for file in files:
                p = path / file
                if extension is None or p.suffix.lower() == "." + extension.lower():
                    res.append(path / file)

        return Benchmark.from_files(*res)


B = Benchmark

# TODO: Benchmarks from files


@dataclass
class Suite:
    """
    A collection of benchmarks. They should all be connected with similar
    structure.
    """

    type SuiteFactory[T] = Callable[[Parameters, Benchmark], T]
    type ConstructBenchList = list[Benchmark] | list[Benchmark] | list[Benchmark | str]

    name: str
    benchmarks: list[Benchmark] | Callable[[Parameters], list[Benchmark]]

    command: SuiteFactory[Command]
    parser: "ResultParser"

    working_directory: Optional[SuiteFactory[Path] | Path]
    env: SuiteFactory[Env]

    def __init__(
        self,
        name: str,
        benchmarks: ConstructBenchList | Callable[[Parameters], list[Benchmark]],
        command: SuiteFactory[Command],
        parser: "ResultParser",
        working_directory: Optional[SuiteFactory[Path] | Path] = None,
        env: SuiteFactory[Env] | Env = {},
    ) -> None:
        self.name = name
        if callable(benchmarks):
            self.benchmarks = benchmarks
        else:
            self.benchmarks = [
                Benchmark(b) if isinstance(b, str) else b for b in benchmarks
            ]

        self.command = command
        self.parser = parser

        self.working_directory = working_directory

        if callable(env):
            self.env = env
        else:
            self.env = const(env)

    def get_benchmarks(self, parameters: Parameters) -> list[Benchmark]:
        if callable(self.benchmarks):
            bs = self.benchmarks(parameters)
        else:
            bs = self.benchmarks

        if len(bs) == 0:
            raise ValueError(f"No benchmarks defined in {self.name}!")

        return bs

    def mk_command(self, parameters: Parameters, benchmark: Benchmark) -> Command:
        return self.command(parameters, benchmark)


# --------------------------------------
#          CONFIGURATION
# --------------------------------------


type ConfigFactory[T] = Callable[[Parameters, Suite, Benchmark], T]
type MatrixFactory[T, U] = Callable[[T, U], U]


default_info = const({})


class Config(abc.ABC):
    @abc.abstractmethod
    def to_executions(self, parameters: Parameters) -> Iterator[Execution]: ...

    def matrix[T](
        self,
        name: str,
        *params: T,
        working_directory: Optional[MatrixFactory[T, Path]] = None,
        env: Optional[MatrixFactory[T, Env] | str | Literal[True]] = None,
    ) -> "MatrixConfig[T]":
        return MatrixConfig(
            self,
            name,
            *params,
            working_directory=working_directory,
            env=env,
        )

    def runs(self, value: int) -> "Config":
        return self.matrix("run", *[i for i in range(1, value + 1)])

    def time(self, *args: str) -> "Config":
        time = shutil.which("time")

        if time is None:
            raise ValueError("time utility is not available")
        # TODO:
        return self


def config(
    *suites: Suite,
    working_directory: Optional[ConfigFactory[Path] | Path] = None,
    env: ConfigFactory[Env] | Env = {},
    info: ConfigFactory[dict[str, str]] = default_info,
) -> Config:
    return BaseConfig(*suites, working_directory=working_directory, env=env, info=info)


class BaseConfig(Config):
    """
    The full configuration of all suites.
    """

    suites: list[Suite]

    working_directory: Optional[ConfigFactory[Path] | Path]
    env: ConfigFactory[Env]

    info: ConfigFactory[dict[str, str]]

    def __init__(
        self,
        *suites: Suite,
        working_directory: Optional[ConfigFactory[Path] | Path] = None,
        env: ConfigFactory[Env] | Env = {},
        info: ConfigFactory[dict[str, str]] = default_info,
    ) -> None:
        if len(suites) == 0:
            raise ValueError("No suites defined!")

        self.suites = list(suites)
        self.working_directory = working_directory

        if callable(env):
            self.env = env
        else:
            self.env = const(env)

        self.info = info

    def to_executions(self, parameters: Parameters) -> Iterator[Execution]:
        for suite in self.suites:
            for bench in suite.get_benchmarks(parameters):
                command = suite.mk_command(parameters, bench)

                # WD
                if suite.working_directory is not None:
                    if callable(suite.working_directory):
                        wd = suite.working_directory(parameters, bench)
                    else:
                        wd = suite.working_directory

                elif self.working_directory is not None:
                    if callable(self.working_directory):
                        wd = self.working_directory(parameters, suite, bench)
                    else:
                        wd = self.working_directory
                else:
                    raise ValueError(
                        f"No working directory defined for suite {suite.name}"
                    )

                env = suite.env(parameters, bench) | self.env(parameters, suite, bench)

                info = self.info(parameters, suite, bench)

                yield Execution(
                    benchmark_name=bench.name,
                    suite=suite.name,
                    command=command,
                    parser=suite.parser,
                    working_directory=wd,
                    env=env,
                    info=info,
                )


class MatrixConfig[T](Config):
    parent: Config
    name: str
    parameters: list[T]

    working_directory: MatrixFactory[T, Path]
    env: MatrixFactory[T, Env]

    def __init__(
        self,
        parent: Config,
        name: str,
        *params: T,
        working_directory: Optional[MatrixFactory[T, Path]] = None,
        env: Optional[MatrixFactory[T, Env] | str | Literal[True]] = None,
    ) -> None:
        self.parent = parent
        self.name = name
        self.parameters = list(params)

        if working_directory is None:
            self.working_directory = lambda _, path: path
        else:
            self.working_directory = working_directory

        if env is None:
            self.env = lambda _, prev_env: prev_env
        elif env is True:
            self.env = lambda param, prev_env: prev_env | {name: str(param)}
        elif isinstance(env, str):
            self.env = lambda param, prev_env: prev_env | {str(env): str(param)}
        else:
            self.env = env

    def to_executions(self, parameters: Parameters) -> Iterator[Execution]:
        for exe in self.parent.to_executions(parameters):
            for param in self.parameters:
                yield dataclasses.replace(
                    exe,
                    working_directory=self.working_directory(
                        param, exe.working_directory
                    ),
                    env=self.env(param, exe.env),
                    info=exe.info | {self.name: str(param)},
                )


C = BaseConfig

# --------------------------------------
#           RESULT DEFINITIONS
# --------------------------------------


@dataclass
class Measurement:
    execution: Execution
    measurement_info: dict[str, str]
    metric: str
    value: float
    unit: str

    @staticmethod
    def runtime(
        execution: Execution,
        value: float,
        unit: str,
        measurement_info: dict[str, str] = {},
    ) -> "Measurement":
        return Measurement(
            execution=execution,
            measurement_info=measurement_info,
            metric="runtime",
            value=value,
            unit=unit,
        )


@dataclass
class ExecutionResult:
    measurements: list[Measurement] = dataclasses.field(default_factory=list)


# --------------------------------------
#           PARSER
# --------------------------------------


class ResultParser(abc.ABC):
    @abc.abstractmethod
    def parse(
        self, execution: Execution, stdout: str, stderr: str
    ) -> ExecutionResult: ...


class PlainSecondsParser(ResultParser):
    def parse(self, execution: Execution, stdout: str, stderr: str) -> ExecutionResult:
        result = ExecutionResult()

        for line in stdout.split("\n"):
            try:
                time = float(line) * 1000
                result.measurements.append(Measurement.runtime(execution, time, "ms"))
            except ValueError:
                pass

        return result


class LastLineParser(ResultParser):
    subparser: ResultParser

    def __init__(self, subparser: ResultParser) -> None:
        self.subparser = subparser

    def parse(self, execution: Execution, stdout: str, stderr: str) -> ExecutionResult:
        stdout_line = ""
        for stdout_line in reversed(stdout.split("\n")):
            if stdout_line.strip() != "":
                break

        stderr_line = ""
        for stderr_line in reversed(stderr.split("\n")):
            if stderr_line.strip() != "":
                break

        return self.subparser.parse(execution, stdout_line, stderr)


class RegexParser(ResultParser):
    metric: str
    regex: re.Pattern[str]
    match_group: str | int
    unit: str
    iterations: bool

    def __init__(
        self,
        regex: str,
        match_group: str | int,
        metric: str,
        unit: str,
        iterations: bool = True,
    ) -> None:
        self.regex = re.compile(regex)
        self.match_group = match_group
        self.metric = metric
        self.unit = unit
        self.iterations = iterations

    def parse(self, execution: Execution, stdout: str, stderr: str) -> ExecutionResult:
        result = ExecutionResult()
        iteration = 1

        for line in stdout.split("\n"):
            match = self.regex.match(line)
            if match is not None:
                value = float(match.group(self.match_group))

                if self.iterations:
                    info = {"iteration": str(iteration)}
                    iteration += 1
                else:
                    info = {}

                result.measurements.append(
                    Measurement(execution, info, self.metric, value, self.unit)
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

    def parse(self, execution: Execution, stdout: str, stderr: str) -> ExecutionResult:
        result = ExecutionResult()
        iteration = 0

        for line in stdout.split("\n"):
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
                        execution, {"iteration": str(iteration)}, "runtime", time, "ms"
                    )
                )
                iteration += 1
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
                        execution, {"iteration": str(iteration)}, criterion, value, unit
                    )
                )

                # Force new iteration
                if criterion == "total":
                    iteration += 1
                continue

        return result


class MixedResultParser(ResultParser):
    parsers: list[ResultParser]

    def __init__(self, *parsers: ResultParser) -> None:
        self.parsers = list(*parsers)

    def parse(self, execution: Execution, stdout: str, stderr: str) -> ExecutionResult:
        result = ExecutionResult()

        for parser in self.parsers:
            result.measurements += parser.parse(execution, stdout, stderr).measurements

        return result


class TimeParser(ResultParser):
    type Column = Literal["maximum_resident_size", "average_resident_size"]

    time: str
    columns: list[Column]

    def __init__(self, *columns: Column) -> None:
        time = shutil.which("time")

        if time is None:
            raise ValueError("time utility is not available")

        self.time = time
        self.columns = list(columns)

    def parse(self, execution: Execution, stdout: str, stderr: str) -> ExecutionResult:
        return super().parse(execution, stdout, stderr)


# --------------------------------------
#           TUI
# --------------------------------------


class TUI:
    if sys.stdout.isatty():
        # TODO:
        # RESET_LINE = "\r"
        RESET_LINE = ""
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
        RESET_LINE = "\n"
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
#           FORMATTERS
# --------------------------------------


class Formatter(abc.ABC):
    @abc.abstractmethod
    def format(self, results: list[ExecutionResult]): ...

    @staticmethod
    def metrics(results: list[ExecutionResult]) -> list[str]:
        out = []

        for result in results:
            for measure in result.measurements:
                if measure.metric not in out:
                    out.append(measure.metric)

        return out

    @staticmethod
    def measurement_info_columns(results: list[ExecutionResult]) -> list[str]:
        out = []

        for result in results:
            for measure in result.measurements:
                for col in measure.measurement_info.keys():
                    if col not in out:
                        out.append(col)

        return out

    @staticmethod
    def info_columns(results: list[ExecutionResult]) -> list[str]:
        out = []

        for result in results:
            for measure in result.measurements:
                for col in measure.execution.info.keys():
                    if col not in out:
                        out.append(col)

        return out


class CsvFormatter(Formatter):
    filepath: Path
    separator: str

    def __init__(self, filepath: Path, separator: str = ",") -> None:
        self.filepath = filepath
        self.separator = separator

    @staticmethod
    def escape_text(text: str) -> str:
        if "," in text:
            return '"' + text.replace('"', r"\"") + '"'

        return text

    def format_line(self, line: list[str]) -> str:
        return self.separator.join(map(self.escape_text, line)) + "\n"

    def format(self, results: list[ExecutionResult]):
        info_cols = Formatter.info_columns(results)
        measurement_info_cols = Formatter.measurement_info_columns(results)
        metrics = Formatter.metrics(results)

        columns = (
            ["benchmark", "suite"]
            + info_cols
            + measurement_info_cols
            + metrics
            + ["unit"]
        )

        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(self.filepath, "wt") as file:
            file.write(self.format_line(columns))

            for result in results:
                for measure in result.measurements:
                    line: list[str] = [
                        measure.execution.benchmark_name,
                        measure.execution.suite,
                    ]

                    for col in info_cols:
                        line.append(measure.execution.info.get(col, ""))

                    for col in measurement_info_cols:
                        line.append(measure.measurement_info.get(col, ""))

                    for metric in metrics:
                        if measure.metric == metric:
                            line.append(str(measure.value))
                        else:
                            line.append("")

                    line.append(measure.unit)

                    file.write(self.format_line(line))


class FlatCsvFormatter(Formatter):
    filepath: Path
    separator: str

    def __init__(self, filepath: Path, separator: str = ",") -> None:
        self.filepath = filepath
        self.separator = separator

    @staticmethod
    def escape_text(text: str) -> str:
        if "," in text:
            return '"' + text.replace('"', r"\"") + '"'

        return text

    def format_line(self, line: list[str]) -> str:
        pprint(line)
        return self.separator.join(map(self.escape_text, line)) + "\n"

    def format(self, results: list[ExecutionResult]):
        info_cols = Formatter.info_columns(results)
        measurement_info_cols = Formatter.measurement_info_columns(results)

        pprint(info_cols)
        pprint(measurement_info_cols)

        columns = (
            ["benchmark", "suite"]
            + info_cols
            + measurement_info_cols
            + ["metric", "value", "unit"]
        )

        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(self.filepath, "wt") as file:
            file.write(self.format_line(columns))

            for result in results:
                for measure in result.measurements:
                    line: list[str] = [
                        measure.execution.benchmark_name,
                        measure.execution.suite,
                    ]

                    for col in info_cols:
                        line.append(measure.execution.info.get(col, ""))

                    for col in measurement_info_cols:
                        line.append(measure.measurement_info.get(col, ""))

                    line += [measure.metric, str(measure.value), measure.unit]

                    file.write(self.format_line(line))


# --------------------------------------
#           EXECUTORS
# --------------------------------------


class Executor(abc.ABC):
    @abc.abstractmethod
    def execute(self, execution: Execution): ...

    def execute_all(self, executions: list[Execution]):
        for execution in executions:
            self.execute(execution)


class DefaultExecutor(Executor):
    """
    The main Executor, which executes given commands, reporting success or
    failures to reporter
    """

    all_executions: Optional[int]
    finished_executions: int
    failed_executions: int

    formatter: Formatter
    crash_folder: Path

    results: list[ExecutionResult]

    def __init__(self, crash_folder: Path, formatter: Formatter) -> None:
        self.all_executions = None
        self.finished_executions = 0
        self.failed_executions = 0

        self.formatter = formatter
        self.crash_folder = crash_folder

        self.results = []

    def execute_all(self, executions: list[Execution]):
        self.all_executions = len(executions)
        return super().execute_all(executions)

    def execute(self, execution: Execution):
        cmd = shutil.which(execution.command[0])
        if cmd is None:
            self.error_execution(
                execution, f"Command not found ({execution.command[0]})"
            )
            return

        execution.command[0] = cmd

        self.start_execution(execution)

        proc = None
        try:
            # TODO: group, process group?
            proc = subprocess.run(
                execution.command,
                # run spec
                capture_output=True,
                check=False,
                # Popen spec
                stdin=None,
                shell=False,
                cwd=execution.working_directory,
                env=execution.env,
                text=True,
            )

            if proc.returncode != 0:
                self.error_execution(
                    execution,
                    f"Program ended with non-zero return code ({proc.returncode})",
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                )
                return

            self.finalize(
                execution,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )

        except OSError as e:
            self.error_execution(execution, str(e))

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

    def error_execution(
        self,
        execution: Execution,
        msg: str,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
    ):
        self.failed_executions += 1
        print(
            f"{TUI.RED}{TUI.BOLD}Error in {execution.as_identifier()}{TUI.RESET}\n{msg}\n"
        )

        # TODO: save to folder

        if stdout is not None or stderr is not None:
            run_path = self.crash_folder / execution.as_identifier().replace(" ", "_")
            run_path.mkdir(parents=True, exist_ok=True)

            if stdout is not None:
                with open(run_path / "stdout", "wt") as file:
                    file.write(stdout)

            if stderr is not None:
                with open(run_path / "stderr", "wt") as file:
                    file.write(stderr)

            print(
                f"{TUI.RED}stdout and stderr are saved in {str(run_path)}{TUI.RESET}\n"
            )

    def finalize(self, execution: Execution, stdout: str, stderr: str) -> None:
        self.finished_executions += 1
        self.results.append(execution.parser.parse(execution, stdout, stderr))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.formatter.format(self.results)
        return False


# class ParallelExecutor(DefaultExecutor):
#     pool: ThreadPoolExecutor
#     lock: Lock
#
#     in_process_runs: int
#
#     def __init__(self, ncores: int) -> None:
#         super().__init__()
#         self.pool = ThreadPoolExecutor(max_workers=ncores)
#         self.lock = Lock()
#
#     def execute(self, run: Run):
#         self.pool.submit(super().execute, run)
#
#     def __enter__(self):
#         return self
#
#     def __exit__(self, *args):
#         self.pool.shutdown(wait=True)
#         super().__exit__(*args)
#         return False


class DryExecutor(Executor):
    """
    Simple executor which only prints what would be executed
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


def namespace_to_parameters(ns: argparse.Namespace) -> Parameters:
    """
    Convert the argparse.Namespace to Parameters
    """
    return Parameters(**vars(ns))


def parse_params(*params: str, **kwarg_params: Any) -> Parameters:
    """
    Create a default argument parser and run it on argv
    """
    parser = make_argparser(*params, **kwarg_params)
    args = parser.parse_args()
    return namespace_to_parameters(args)


# --------------------------------------
#          DEFAULT RUN
# --------------------------------------


def main(
    config: Config,
    *params: str,
    formatter: Optional[Formatter] = None,
    derived: Optional[Callable[[Parameters], Parameters]] = None,
    **kwarg_params: Any,
) -> None:
    """
    Sane default main. config is the benchmarks configuration that will be
    executed, params is a list of required parameters from the user,
    kwarg_params are optional parameters with their default value.
    """
    parser = make_argparser(*params, **kwarg_params)

    defp = parser.add_argument_group("Default benchr parameters")
    defp.add_argument(
        "--output",
        help="Where to store the results (Default: ./output)",
        metavar="file",
        type=str,
        default="./output",
        dest="__output",
    )
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

    ps = namespace_to_parameters(parser.parse_args())
    if derived is not None:
        ps |= derived(ps)

    executions = list(config.to_executions(ps))

    output: Path = Path(ps.__output).resolve()
    if formatter is None:
        formatter = FlatCsvFormatter(output / "results.csv")

    if ps.__dry:
        DryExecutor().execute_all(executions)
    elif ps.__jobs > 1:
        # TODO: parallel executor
        pass
    else:
        with DefaultExecutor(output / "crash", formatter) as executor:
            executor.execute_all(executions)


# TODO: Catch keyboard_interrupts
# TODO: Run info - date, reflog
