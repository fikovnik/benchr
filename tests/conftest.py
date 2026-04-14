from pathlib import Path
from types import SimpleNamespace

from benchr._types import (
    Execution,
    SuccessfulProcessResult,
    FailedProcessResult,
)
from benchr._results import Measurement, ExecutionResult


def make_execution(**overrides) -> Execution:
    defaults = dict(
        benchmark_name="bench1",
        suite="suite1",
        parser=None,
        command=["echo", "hello"],
        working_directory=Path("/tmp"),
        env={},
        timeout=None,
        info={},
        run=1,
    )
    defaults.update(overrides)
    return Execution(**defaults)


def make_success(stdout="", stderr="", runtime=1.0, rusage=None, **exe_kw):
    return SuccessfulProcessResult(
        execution=make_execution(**exe_kw),
        runtime=runtime,
        stdout=stdout,
        stderr=stderr,
        rusage=rusage,
    )


def make_failure(returncode=1, stdout=None, stderr=None, runtime=None, rusage=None, reason=None, **exe_kw):
    return FailedProcessResult(
        execution=make_execution(**exe_kw),
        runtime=runtime,
        stdout=stdout,
        stderr=stderr,
        rusage=rusage,
        returncode=returncode,
        reason=reason,
    )


def make_rusage(**fields):
    defaults = {
        "ru_utime": 0.5,
        "ru_stime": 0.1,
        "ru_maxrss": 10240,
        "ru_ixrss": 0,
        "ru_idrss": 0,
        "ru_isrss": 0,
        "ru_minflt": 100,
        "ru_majflt": 0,
        "ru_nswap": 0,
        "ru_inblock": 0,
        "ru_oublock": 0,
        "ru_msgsnd": 0,
        "ru_msgrcv": 0,
        "ru_nsignals": 0,
        "ru_nvcsw": 10,
        "ru_nivcsw": 5,
    }
    defaults.update(fields)
    return SimpleNamespace(**defaults)


def sample_execution_result() -> ExecutionResult:
    exe1 = make_execution(benchmark_name="bench1", suite="suite1", info={"variant": "a"})
    exe2 = make_execution(benchmark_name="bench2", suite="suite1", info={"variant": "a"})
    return ExecutionResult(measurements=[
        Measurement(execution=exe1, metric="runtime", value=1.5, unit="s", lower_is_better=True),
        Measurement(execution=exe1, metric="max_rss", value=1024, unit="kB", lower_is_better=True),
        Measurement(execution=exe2, metric="runtime", value=2.0, unit="s", lower_is_better=True),
        Measurement(execution=exe2, metric="max_rss", value=2048, unit="kB", lower_is_better=True),
    ])
