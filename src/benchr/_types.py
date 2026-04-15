import argparse
import dataclasses
import resource
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from benchr._parsers import ResultParser


def const[T](x: T) -> Callable[..., T]:
    return lambda *args, **kwargs: x


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
class SuccessfulProcessResult:
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


ProcessResult = SuccessfulProcessResult | FailedProcessResult


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
        ident = f"{self.suite},{self.benchmark_name}"

        parts = [f"{k}={v}" for k, v in self.info.items()]
        parts.append(f"run={self.run}")
        if parts:
            ident += " (" + ", ".join(parts) + ")"

        return ident

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
        extension (or all if no extension is given) into Benchmarks.
        The benchmark name is the path relative to `folder` without extension,
        e.g. for folder="tests" and file="tests/assignment/global.lox" the
        name will be "assignment/global".
        """
        res = []
        for path, _, files in folder.walk():
            for file in files:
                p = path / file
                if extension is None or p.suffix.lower() == ("." + extension.lower()):
                    name = str(p.relative_to(folder).with_suffix(""))
                    res.append(Benchmark(name, path=p))

        return res
