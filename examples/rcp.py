from pathlib import Path
import os
from types import SimpleNamespace
import tempfile
from contextlib import ExitStack
import pprint

import benchr
from benchr import (
    Benchmark,
    BenchmarkExecutor,
    Config,
    DryExecutor,
    Executor,
    ParallelExecutor,
    Suite,
    default_mk_info,
)

# Assuming this is in `rcp` subfolder of PRL_PRG/rcp
CWD = Path("/home/rihafilip/code/r/rcp/rcp")
# TODO:
# CWD = Path(__file__).parent


class RCPSuite(Suite):
    name = "RCPSuite"
    working_directory = CWD
    env = {}

    def __init__(self, benchmarks: list[Benchmark]) -> None:
        super().__init__()
        self.benchmarks = benchmarks

    def mk_command(
        self, parameters: SimpleNamespace, benchmark: Benchmark
    ) -> list[str]:
        path: Path = benchmark.data[0]
        bench_opts: str = parameters.bench_opts

        return (
            ["/usr/bin/time", "-v"]
            + [str(parameters.R), "--slave", "--no-restore"]
            + ["-f", str(parameters.harness_bin), "--args"]
            + [
                "--output-dir",
                parameters.output,
                "--runs",
                str(parameters.runs),
            ]
            + bench_opts.split()
            + [str(path.with_suffix(""))]
        )


def check_microbenchmark(Rscript: Path):
    benchr.run_cmd(
        [
            Rscript,
            "-e"
            """if (!requireNamespace("microbenchmark", quietly=TRUE)) quit(status=1)""",
        ]
    )


def main():
    with ExitStack() as estack:
        params = benchr.parse_params(
            RSH_HOME=CWD / ".." / "external" / "rsh" / "client" / "rsh",
            R_HOME=CWD / ".." / "external" / "rsh" / "external" / "R",
            bench_opts="--rcp",
            filter="",
            parallel=str(os.cpu_count()),
            runs=1,
            output=None,
        )

        params.RSH_HOME = Path(params.RSH_HOME).resolve()
        params.R_HOME = Path(params.R_HOME).resolve()

        params.bench_dir = params.RSH_HOME / "inst" / "benchmarks"
        params.harness_bin = params.RSH_HOME / "inst" / "benchmarks" / "harness.R"

        params.R = params.R_HOME / "bin" / "R"
        params.Rscript = params.R_HOME / "bin" / "Rscript"

        if params.output is None:
            params.output = estack.enter_context(tempfile.TemporaryDirectory())

        pprint.pprint(params)

        all_benchmarks = [
            Benchmark(path.stem, path)
            for path in params.bench_dir.rglob(f"*{params.filter}*.R")
            # Top level has main program and harness -> we want benchmarks
            if path.parent != params.bench_dir
        ]

        check_microbenchmark(params.Rscript)

        conf = Config(RCPSuite(all_benchmarks))

        # TODO: Executor
        executor = DryExecutor()
        benchr.run_config(executor, conf, params, default_mk_info)


if __name__ == "__main__":
    main()
