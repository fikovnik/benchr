from pathlib import Path
import os
import shutil
import tempfile
from contextlib import ExitStack

import benchr
from benchr import (
    Benchmark,
    Config,
    DefaultExecutor,
    DryExecutor,
    Suite,
)

CWD = Path("/home/rihafilip/code/r/rcp/rcp")
# TODO:
# Assuming this is in `rcp` subfolder of PRL_PRG/rcp
# CWD = Path(__file__).parent


def check_namespace(Rscript: Path, namespace: str):
    benchr.run_cmd(
        [
            Rscript,
            "-e",
            f"""if (!requireNamespace("{namespace}", quietly=TRUE)) quit(status=1)""",
        ]
    )


def main():
    with ExitStack() as estack:
        params = benchr.parse_params(
            RSH_HOME=CWD / ".." / "external" / "rsh" / "client" / "rsh",
            R_HOME=CWD / ".." / "external" / "rsh" / "external" / "R",
            bench_opts="--rcp",
            path_filter="",
            parallel=os.cpu_count(),
            runs=1,
            output=None,
        )

        RSH_HOME: Path = params.RSH_HOME.resolve()
        R_HOME: Path = params.R_HOME.resolve()
        bench_opts = params.bench_opts.split()
        path_filter = params.path_filter
        parallel = params.parallel
        executions = params.runs
        output = (
            Path(params.output)
            if params.output is not None
            else Path(estack.enter_context(tempfile.TemporaryDirectory()))
        )

        bench_dir = RSH_HOME / "inst" / "benchmarks"
        harness_bin = RSH_HOME / "inst" / "benchmarks" / "harness.R"

        R = R_HOME / "bin" / "R"
        Rscript = R_HOME / "bin" / "Rscript"

        benchmarks = list(
            filter(
                lambda b: (
                    b.keys.path.parent != bench_dir
                    and (path_filter == "" or path_filter not in str(b.keys.path))
                ),
                Benchmark.from_folder(bench_dir, extension="R"),
            )
        )

        RCPSuite = (
            benchr.config(
                Suite(
                    name="RCPSuite",
                    benchmarks=benchmarks,
                    parser=benchr.RebenchParser(),
                    working_directory=CWD,
                    command=lambda _, benchmark: (
                        [
                            str(R),
                            "--slave",
                            "--no-restore",
                            "-f",
                            str(harness_bin),
                            "--args",
                        ]
                        + ["--output-dir", str(output)]
                        + ["--runs", str(executions)]
                        + bench_opts
                        + [str(benchmark.keys.path.with_suffix(""))]
                    ),
                )
            )
            .matrix("RCP_ON", 0, 1, env=True)
            .time("maximum_resident_size", "average_resident_size")
        )

        executions = list(RCPSuite.to_executions(params))

        dry_executor = DryExecutor()
        dry_executor.execute_all(executions)

        # check_namespace(Rscript, "microbenchmark")
        # check_namespace(Rscript, "rcp")
        #
        # with DefaultExecutor(
        #     output / "crash", benchr.CsvFormatter(output / "result.csv")
        # ) as executor:
        # executor.execute_all(list(executions))


if __name__ == "__main__":
    main()
