from pathlib import Path

import benchr
from benchr import config, Suite

conf = (
    benchr.config(
        benchr.Suite(
            name="LoxSuite",
            working_directory=lambda ps, _: ps.cwd / "benchmarks",
            parser=benchr.LastLineParser(benchr.PlainSecondsParser()),
            command=lambda ps, bench: [
                ps.lox,
                str(bench.keys.path),
            ],
            benchmarks=lambda ps: benchr.Benchmark.from_folder(
                ps.cwd / "benchmarks", extension="lox"
            ),
        )
    )
    .runs(5)
    .time("maximum_resident_size", "average_resident_size")
)


if __name__ == "__main__":
    benchr.main(conf, "lox", cwd=Path(__file__).parent)
