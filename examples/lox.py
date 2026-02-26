from pathlib import Path

import benchr
from benchr import B, Config, Suite


class LoxSuite(Suite):
    name = "LoxSuite"
    working_directory = Path(__file__) / "benchmarks"
    benchmarks = [
        B("binary_trees"),
        B("equality"),
        B("fib"),
        B("instantiation"),
        B("invocation"),
        B("method_call"),
        B("properties"),
        B("string_equality"),
        B("zoo_batch"),
        B("zoo"),
    ]

    def mk_command(self, parameters, benchmark) -> list[str]:
        return [
            parameters.lox,
            f"{benchmark.name}.lox",
        ]


conf = Config(LoxSuite())

if __name__ == "__main__":
    benchr.main(conf, "lox")
