from subprocess import CompletedProcess
from benchr import *


conf = (
    Config(
        [
            suite(
                name="LoxSuite",
                benchmarks=lambda ps: list(
                    filter(
                        lambda b: b.name != "zoo_batch",
                        Benchmark.from_files(ps.cwd / "benchmarks", pattern=r"\.lox$"),
                    )
                ),
                parser=(
                    LineParser(PlainFloatParser("s")) & MaxRssParser() & TimeParser()
                ).lower_is_better()  # All of them are less is better
                & FailedParser(),
            ).timeout(20),
            suite(
                name="ZooBatch",
                benchmarks=lambda ps: [
                    B("zoo_batch", path=ps.cwd / "benchmarks" / "zoo_batch.lox")
                ],
                parser=LineParser(
                    PlainFloatParser("iter", metric="throughput"), line=2
                ).higher_is_better()
                & FailedParser(),
            ).timeout(12),
        ]
    )
    .runs(2)
    .working_directory(lambda ps, _: ps.cwd / "benchmarks")
    .command(
        lambda ps, bench: [
            ps.lox,
            str(bench.keys.path),
        ]
    )
)

if __name__ == "__main__":
    main(
        conf,
        ["lox"],
        {"cwd": Path(__file__).parent},
        reporter=SummaryReporter(formatter=CompactFormatter("runtime")),
    )
