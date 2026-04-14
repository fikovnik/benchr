import argparse
from pathlib import Path

from benchr._types import const, Parameters, Benchmark


def test_const_returns_constant():
    """const returns the same value regardless of args/kwargs."""
    f = const(42)
    assert f() == 42
    assert f(1, 2, 3) == 42
    assert f(a="x", b="y") == 42
    assert f(1, key="val") == 42


def test_parameters_or_merges():
    """Parameters.__or__ merges two Parameters."""
    p1 = Parameters(a=1, b=2)
    p2 = Parameters(c=3, d=4)
    merged = p1 | p2
    assert merged.a == 1
    assert merged.b == 2
    assert merged.c == 3
    assert merged.d == 4


def test_parameters_getitem():
    """Parameters.__getitem__ provides attribute access via []."""
    p = Parameters(x=10, y=20)
    assert p["x"] == 10
    assert p["y"] == 20


def test_parameters_from_namespace():
    """Parameters.from_namespace converts argparse.Namespace."""
    ns = argparse.Namespace(foo="bar", count=5)
    p = Parameters.from_namespace(ns)
    assert p.foo == "bar"
    assert p.count == 5


def test_benchmark_single_data_unwrapped():
    """Single positional data is unwrapped (not a tuple)."""
    b = Benchmark("test", 42)
    assert b.data == 42
    assert b.name == "test"


def test_benchmark_multiple_data_tuple():
    """Multiple positional data stay as a tuple."""
    b = Benchmark("test", 1, 2, 3)
    assert b.data == (1, 2, 3)


def test_benchmark_keys_accessible():
    """Keyword arguments are accessible via keys."""
    b = Benchmark("test", path="/foo")
    assert b.keys.path == "/foo"


def test_benchmark_from_files():
    """Benchmark.from_files creates benchmarks with stem name and keys.path."""
    p1 = Path("/data/foo.txt")
    p2 = Path("/data/bar.csv")
    benchmarks = Benchmark.from_files(p1, p2)
    assert len(benchmarks) == 2
    assert benchmarks[0].name == "foo"
    assert benchmarks[0].keys.path == p1
    assert benchmarks[1].name == "bar"
    assert benchmarks[1].keys.path == p2


def test_benchmark_from_folder(tmp_path):
    """Benchmark.from_folder creates benchmarks filtered by extension."""
    (tmp_path / "a.py").write_text("pass")
    (tmp_path / "b.py").write_text("pass")
    (tmp_path / "c.txt").write_text("hello")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.py").write_text("pass")

    benchmarks = Benchmark.from_folder(tmp_path, extension="py")
    names = sorted(b.name for b in benchmarks)
    assert len(names) == 3
    assert "a" in names
    assert "b" in names
    assert "d" in names
    # .txt file should be excluded
    assert "c" not in names
