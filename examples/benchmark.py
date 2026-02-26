from pathlib import Path

import benchr
from benchr import B, Config, Suite

INPUTS = Path(__file__).resolve() / "inputs"
BENCHMARKS = INPUTS / "Benchmarks"


LOCALE = {
    "LC_CTYPE": "en_US.UTF-8",
    "LC_TIME": "en_US.UTF-8",
    "LC_MONETARY": "en_US.UTF-8",
    "LC_PAPER": "en_US.UTF-8",
    "LC_ADDRESS": "C",
    "LC_MEASUREMENT": "en_US.UTF-8",
    "LC_NUMERIC": "C",
    "LC_COLLATE": "en_US.UTF-8",
    "LC_MESSAGES": "en_US.UTF-8",
    "LC_NAME": "C",
    "LC_TELEPHONE": "C",
    "LC_IDENTIFICATION": "C",
}

DEFAULT_ENV = LOCALE | {
    "PIR_OSR": "0",
    "PIR_WARMUP": "10",
    "PIR_GLOBAL_SPECIALIZATION_LEVEL": "0",
    "PIR_DEFAULT_SPECULATION": "0",
    "STATS_USE_RIR_NAMES": "1",
}


class AreWeFast(Suite):
    name = "areWeFast"
    working_directory = BENCHMARKS / "areWeFast"
    benchmarks = [
        B("Mandelbrot", 500),
        B("Bounce", 35),
        B("Bounce_nonames", 35),
        B("Bounce_nonames_simple", 35),
        B("Storage", 100),
    ]

    def mk_command(self, parameters, benchmark):
        return [
            str(Path(parameters.Rpath) / "bin" / "Rscript"),
            "harness.r",
            benchmark.name,
            str(parameters.iterations),
            str(benchmark.data[0]),
        ]


class Shootout(Suite):
    name = "shootout"
    benchmarks = [
        B("binarytrees", "binarytrees", 9),
        B("fannkuchredux", "fannkuch", 9),
        B("fasta", "fasta", 60000),
        B("fastaredux", "fastaredux", 80000),
        B("knucleotide", "knucleotide", 2000),
        B("mandelbrot_ascii", "mandelbrot", 300),
        B("mandelbrot_naive_ascii", "mandelbrot", 200),
        B("nbody", "nbody", 25000),
        B("nbody_naive", "nbody", 20000),
        B("pidigits", "pidigits", 30),
        B("regexdna", "regexdna", 500000),
        B("reversecomplement", "reversecomplement", 150000),
        B("spectralnorm", "spectralnorm", 1200),
        B("spectralnorm_math", "spectralnorm", 1200),
    ]

    def mk_working_directory(self, parameters, benchmark):
        return BENCHMARKS / "shootout" / benchmark.data[0]

    def mk_command(self, parameters, benchmark):
        return [
            str(Path(parameters.Rpath) / "bin" / "Rscript"),
            "harness.r",
            benchmark.name,
            str(parameters.iterations),
            str(benchmark.data[1]),
        ]


class RealThing(Suite):
    name = "RealThing"
    working_directory = BENCHMARKS / "RealThing"
    benchmarks = [
        B("convolution", 500),
        B("convolution_slow", 1500),
        B("volcano", 1),
        B("flexclust", 5),
    ]

    def mk_command(self, parameters, benchmark):
        return [
            str(Path(parameters.Rpath) / "bin" / "Rscript"),
            "harness.r",
            benchmark.name,
            str(parameters.iterations),
            str(benchmark.data[0]),
        ]


class Kaggles(Suite):
    name = "kaggle"
    benchmarks = [
        B("basic-analysis"),
        B("bolt-driver"),
        B("london-airbnb"),
        B("placement"),
        B("titanic"),
    ]

    def mk_working_directory(self, parameters, benchmark):
        return INPUTS / "kaggle" / benchmark.name

    def mk_command(self, parameters, benchmark):
        return [
            str(Path(parameters.Rpath) / "bin" / "Rscript"),
            "../../harness.r",
            benchmark.name,
            str(parameters.iterations),
        ]


class Recommenderlab(Suite):
    name = "recommenderlab"
    benchmarks = [B("recommenderlab")]
    working_directory = INPUTS / "recommenderlab"

    def mk_command(self, parameters, benchmark):
        return [
            str(Path(parameters.Rpath) / "bin" / "Rscript"),
            "runner.r",
        ]


conf = Config(
    AreWeFast(),
    Shootout(),
    RealThing(),
    Kaggles(),
    Recommenderlab(),
    env=DEFAULT_ENV,
)

if __name__ == "__main__":
    benchr.main(conf, "Rpath", iterations=15)
