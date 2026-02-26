import abc
import argparse
import pprint
import subprocess
import types
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TextIO, Callable

# --------------------------------------
#           DEFINITIONS
# --------------------------------------

Env = dict[str, str]

Parameters = types.SimpleNamespace

# --------------------------------------
#           HELPER
# --------------------------------------


def run_cmd(*args, **kwargs):
    """
    Run a very simple command, wrapper around subprocess.run
    """
    return subprocess.run(*args, check=True, **kwargs)


# --------------------------------------
#           UI
# --------------------------------------


class Reporter:
    def new_run(self): ...

    def failed_run(
        self,
        command: list[str],
        info: dict[str, str],
        msg: str,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
    ): ...

    def finished_run(
        self,
        command: list[str],
        info: dict[str, str],
        stdout: str,
        stderr: str,
    ): ...


# --------------------------------------
#           EXECUTORS
# --------------------------------------


class Executor:
    """
    The main Executor, which executes given commands, reporting success or
    failures to reporter
    """

    reporter: Reporter

    def execute(
        self,
        command: list[str],
        working_directory: Path,
        env: Env,
        info: dict[str, str],
    ):
        self.reporter.new_run()

        cmd = shutil.which(command[0])
        if cmd is None:
            self.reporter.failed_run(command, info, "Command does not exist")
            return

        command[0] = cmd

        proc = None
        try:
            # TODO: group, process group?
            proc = subprocess.run(
                command,
                # run spec
                capture_output=True,
                check=False,
                # Popen spec
                stdin=None,
                shell=False,
                cwd=working_directory,
                env=env,
                text=True,
            )

            if proc.returncode != 0:
                self.reporter.failed_run(
                    command,
                    info,
                    f"Non-zero return code ({proc.returncode})",
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                )

            self.reporter.finished_run(command, info, proc.stdout, proc.stderr)

        except OSError as e:
            self.reporter.failed_run(
                command,
                info,
                f"OSError: {e.strerror or ''}",
                stdout=proc.stdout if proc is not None else None,
                stderr=proc.stderr if proc is not None else None,
            )


class DryExecutor(Executor):
    def execute(
        self,
        command: list[str],
        working_directory: Path,
        env: Env,
        info: dict[str, str],
    ):
        print("Run:", " ".join(command))
        print("Working working_directory:", str(working_directory))
        print("Environment: ", end="")
        pprint.pprint(env)
        print("Info: ", end="")
        pprint.pprint(info)
        print("-" * 10)


class ParallelExecutor(Executor):
    pool: ThreadPoolExecutor

    def __init__(self, ncores: int) -> None:
        super().__init__()
        self.pool = ThreadPoolExecutor(max_workers=ncores)

    def execute(self, *args, **kwargs):
        self.pool.submit(super().execute, *args, **kwargs)

    def __enter__(self):
        self.pool.__enter__()
        return self

    def __exit__(self, *args):
        return self.pool.__exit__(*args)


# TODO: parsing results
class BenchmarkExecutor(Executor):
    result_path: Path
    result: TextIO
    csv_headers: list[str]

    def __init__(self, result_path: Path, csv_headers: list[str]) -> None:
        super().__init__()
        self.result_path = result_path
        self.csv_headers = csv_headers

    def execute(
        self,
        command: list[str],
        working_directory: Path,
        env: Env,
        info: dict[str, str],
    ): ...

    def __enter__(self):
        # TODO: Check rebench-denoise
        self.result = open(self.result_path, "w")
        self.result.__enter__()
        self.result.write(",".join(self.csv_headers))
        self.result.write("\n")
        return self

    def __exit__(self, *args):
        return self.result.__exit__(*args)


# --------------------------------------
#          DATA DEFINITIONS
# --------------------------------------


@dataclass
class Benchmark:
    """
    A definition of one benchmark. data can be any benchmark-specific data
    that are needed for its execution.
    """

    name: str
    data: tuple[Any, ...]

    def __init__(self, name: str, *data: Any) -> None:
        self.name = name
        self.data = data


B = Benchmark


class Suite(abc.ABC):
    """
    A collection of benchmarks. They should all be connected with similar
    structure.

    The mk_command needs to implemented, providing a command that should be
    executed, unique for each benchmark.

    Environment can be defined statically (with env) or per benchmark (by
    overriding mk_env), or defaults to empty.

    Similarly, working directory can be either specified statically with
    working_directory, or by overriding mk_working_directory. If none of these
    is defined, configuration needs to provide a working directory.
    """

    name: str
    benchmarks: list[Benchmark]

    working_directory: Optional[Path] = None
    env: Env = {}

    def mk_working_directory(
        self, parameters: Parameters, benchmark: Benchmark
    ) -> Optional[Path]:
        return self.working_directory

    def mk_env(self, parameters: Parameters, benchmark: Benchmark) -> Env:
        return self.env

    @abc.abstractmethod
    def mk_command(self, parameters: Parameters, benchmark: Benchmark) -> list[str]: ...


class Config:
    """
    The full configuration of all suites.

    If a suite does not provide a working directory, it can be specified in
    here, either statically (with working_directory), or dynamically with
    mk_working_directory.

    Specifying an environment through env merges it with the environment
    provided by the suite. Alternatively, mk_env can be overriden to modify
    the behavior per benchmark.
    """

    suites: list[Suite]

    working_directory: Optional[Path]
    env: Env

    def __init__(
        self,
        *suites: Suite,
        working_directory: Optional[Path] = None,
        env: Env = {},
    ):
        if len(suites) == 0:
            raise ValueError("No suites defined!")

        for suite in suites:
            if len(suite.benchmarks) == 0:
                raise ValueError(f"Suite {suite.name} has no benchmarks!")

        self.suites = list(suites)
        self.working_directory = working_directory
        self.env = env

    def mk_working_directory(
        self, parameters: Parameters, suite: Suite, benchmark: Benchmark
    ) -> Path:
        working_directory = suite.mk_working_directory(parameters, benchmark)
        if working_directory is not None:
            return working_directory

        if self.working_directory is not None:
            return self.working_directory

        raise ValueError("Working directory is not defined!")

    def mk_env(self, parameters: Parameters, suite: Suite, benchmark: Benchmark) -> Env:
        return self.env | suite.mk_env(parameters, benchmark)


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
    Convert the argparse.Namespace to Parameters (aka SimpleNamespace)
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


DEFAULT_CSV_HEADERS = ["benchmark", "suite"]


def default_mk_info(benchmark: Benchmark, suite: Suite) -> dict[str, str]:
    return {"benchmark": benchmark.name, "suite": suite.name}


def run_config(
    executor: Executor,
    conf: Config,
    params: Parameters,
    mk_info: Callable[[Benchmark, Suite], dict[str, str]],
) -> None:
    """
    Run a given config on the executor and with the given parameters and info
    """
    for suite in conf.suites:
        try:
            for bench in suite.benchmarks:
                try:
                    cmd = suite.mk_command(params, bench)
                    env = conf.mk_env(params, suite, bench)
                    working_directory = conf.mk_working_directory(params, suite, bench)

                    executor.execute(
                        cmd,
                        working_directory,
                        env,
                        mk_info(bench, suite),
                    )
                except ValueError as e:
                    e.add_note(f"In benchmark {bench.name}")
                    raise

        except ValueError as e:
            e.add_note(f"In suite {suite.name}")
            raise


def main(config: Config, *params: str, **kwarg_params: Any) -> None:
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
        help="Do not run; only print what would be runned",
        action="store_true",
        dest="__dry",
    )
    defp.add_argument(
        "--verbose",
        help="Be more verbose with output",
        action="store_true",
        dest="__verbose",
    )

    ps = namespace_to_parameters(parser.parse_args())

    # TODO: Threading, verbose

    if ps.__dry:
        run_config(DryExecutor(), config, ps, default_mk_info)
    else:
        with BenchmarkExecutor(Path(ps.__output), DEFAULT_CSV_HEADERS) as executor:
            run_config(executor, config, ps, default_mk_info)
