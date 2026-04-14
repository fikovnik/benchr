import re

import pytest

from benchr._parsers import (
    PlainFloatParser,
    LineParser,
    RegexParser,
    RebenchParser,
    SingleResourceUsageParser,
    MaxRssParser,
    TimeParser,
    FailedParser,
    MixedResultParser,
    IgnoreFailParserDecorator,
    DirectionParserDecorator,
)
from benchr._results import ExecutionResult

from conftest import make_success, make_failure, make_rusage


# ---------------------------------------------------------------------------
# PlainFloatParser
# ---------------------------------------------------------------------------

class TestPlainFloatParser:
    def test_happy_path_multiple_floats(self):
        pr = make_success(stdout="1.5\n2.5\n3.5\n")
        result = PlainFloatParser(unit="s").parse(pr)
        assert len(result.measurements) == 3
        assert [m.value for m in result.measurements] == [1.5, 2.5, 3.5]
        assert all(m.unit == "s" for m in result.measurements)
        assert all(m.metric == "runtime" for m in result.measurements)

    def test_mixed_lines_non_floats_skipped(self):
        pr = make_success(stdout="hello\n1.0\nworld\n2.0\n")
        result = PlainFloatParser(unit="s").parse(pr)
        assert len(result.measurements) == 2
        assert [m.value for m in result.measurements] == [1.0, 2.0]

    def test_failed_process_returns_empty(self):
        pr = make_failure(stdout="1.0\n2.0\n")
        result = PlainFloatParser(unit="s").parse(pr)
        assert result.measurements == []

    def test_custom_metric_and_unit(self):
        pr = make_success(stdout="42.0\n")
        result = PlainFloatParser(unit="MB", metric="memory").parse(pr)
        assert len(result.measurements) == 1
        assert result.measurements[0].metric == "memory"
        assert result.measurements[0].unit == "MB"
        assert result.measurements[0].value == 42.0


# ---------------------------------------------------------------------------
# LineParser
# ---------------------------------------------------------------------------

class TestLineParser:
    def test_line_1_selects_first(self):
        inner = PlainFloatParser(unit="s")
        pr = make_success(stdout="1.0\n2.0\n3.0\n")
        result = LineParser(inner, line=1).parse(pr)
        assert len(result.measurements) == 1
        assert result.measurements[0].value == 1.0

    def test_line_minus1_selects_last(self):
        inner = PlainFloatParser(unit="s")
        pr = make_success(stdout="1.0\n2.0\n3.0\n")
        result = LineParser(inner, line=-1).parse(pr)
        assert len(result.measurements) == 1
        assert result.measurements[0].value == 3.0

    def test_line_0_raises_value_error(self):
        inner = PlainFloatParser(unit="s")
        with pytest.raises(ValueError, match="non-zero"):
            LineParser(inner, line=0)

    def test_blank_lines_skipped(self):
        inner = PlainFloatParser(unit="s")
        pr = make_success(stdout="\n\n1.0\n\n2.0\n\n")
        result = LineParser(inner, line=1).parse(pr)
        assert len(result.measurements) == 1
        assert result.measurements[0].value == 1.0

    def test_failed_process_returns_empty(self):
        inner = PlainFloatParser(unit="s")
        pr = make_failure(stdout="1.0\n")
        result = LineParser(inner, line=1).parse(pr)
        assert result.measurements == []

    def test_default_line_is_last(self):
        inner = PlainFloatParser(unit="s")
        pr = make_success(stdout="1.0\n2.0\n")
        result = LineParser(inner).parse(pr)
        assert result.measurements[0].value == 2.0


# ---------------------------------------------------------------------------
# RegexParser
# ---------------------------------------------------------------------------

class TestRegexParser:
    def test_single_match(self):
        parser = RegexParser(
            metric="runtime",
            regex=re.compile(r"time:\s*([\d.]+)"),
            output="stdout",
            match_group=1,
            unit="s",
        )
        pr = make_success(stdout="time: 1.234\n")
        result = parser.parse(pr)
        assert len(result.measurements) == 1
        assert result.measurements[0].value == pytest.approx(1.234)
        assert result.measurements[0].unit == "s"

    def test_multiple_matches_iteration(self):
        parser = RegexParser(
            metric="runtime",
            regex=re.compile(r"time:\s*([\d.]+)"),
            output="stdout",
            match_group=1,
            unit="ms",
        )
        pr = make_success(stdout="time: 1.0\ntime: 2.0\ntime: 3.0\n")
        result = parser.parse(pr)
        assert len(result.measurements) == 3
        assert [m.value for m in result.measurements] == [1.0, 2.0, 3.0]

    def test_unit_from_match_group(self):
        parser = RegexParser(
            metric="runtime",
            regex=re.compile(r"time:\s*([\d.]+)\s*(\w+)"),
            output="stdout",
            match_group=1,
            unit_match_group=2,
        )
        pr = make_success(stdout="time: 1.5 ms\n")
        result = parser.parse(pr)
        assert len(result.measurements) == 1
        assert result.measurements[0].unit == "ms"
        assert result.measurements[0].value == pytest.approx(1.5)

    def test_output_stderr(self):
        parser = RegexParser(
            metric="runtime",
            regex=re.compile(r"time:\s*([\d.]+)"),
            output="stderr",
            match_group=1,
            unit="s",
        )
        pr = make_success(stdout="nothing here", stderr="time: 5.0\n")
        result = parser.parse(pr)
        assert len(result.measurements) == 1
        assert result.measurements[0].value == pytest.approx(5.0)

    def test_output_both(self):
        parser = RegexParser(
            metric="runtime",
            regex=re.compile(r"time:\s*([\d.]+)"),
            output="both",
            match_group=1,
            unit="s",
        )
        pr = make_success(stdout="time: 1.0\n", stderr="time: 2.0\n")
        result = parser.parse(pr)
        assert len(result.measurements) == 2
        values = [m.value for m in result.measurements]
        assert 1.0 in values
        assert 2.0 in values

    def test_failed_returns_empty(self):
        parser = RegexParser(
            metric="runtime",
            regex=re.compile(r"time:\s*([\d.]+)"),
            output="stdout",
            match_group=1,
            unit="s",
        )
        pr = make_failure(stdout="time: 1.0\n")
        result = parser.parse(pr)
        assert result.measurements == []

    def test_missing_unit_spec_raises_value_error(self):
        with pytest.raises(ValueError, match="Missing unit"):
            RegexParser(
                metric="runtime",
                regex=re.compile(r"time:\s*([\d.]+)"),
                output="stdout",
                match_group=1,
            )


# ---------------------------------------------------------------------------
# RebenchParser
# ---------------------------------------------------------------------------

class TestRebenchParser:
    def test_runtime_ms(self):
        parser = RebenchParser()
        pr = make_success(stdout="bench1: iterations=1 runtime: 500ms\n")
        result = parser.parse(pr)
        assert len(result.measurements) == 1
        assert result.measurements[0].value == pytest.approx(500.0)
        assert result.measurements[0].unit == "ms"
        assert result.measurements[0].metric == "runtime"

    def test_runtime_us_divided_by_1000(self):
        parser = RebenchParser()
        pr = make_success(stdout="bench1: iterations=1 runtime: 2000us\n")
        result = parser.parse(pr)
        assert len(result.measurements) == 1
        assert result.measurements[0].value == pytest.approx(2.0)
        assert result.measurements[0].unit == "ms"

    def test_extra_criterion(self):
        parser = RebenchParser()
        pr = make_success(stdout="bench1: MemoryUsage: 1024kB\n")
        result = parser.parse(pr)
        assert len(result.measurements) == 1
        assert result.measurements[0].metric == "MemoryUsage"
        assert result.measurements[0].value == pytest.approx(1024.0)
        assert result.measurements[0].unit == "kB"

    def test_non_total_criterion_skipped(self):
        parser = RebenchParser()
        pr = make_success(stdout="bench1 warmup: iterations=10 runtime: 100ms\n")
        result = parser.parse(pr)
        assert result.measurements == []

    def test_total_criterion_accepted(self):
        parser = RebenchParser()
        pr = make_success(stdout="bench1 total: iterations=10 runtime: 100ms\n")
        result = parser.parse(pr)
        assert len(result.measurements) == 1
        assert result.measurements[0].value == pytest.approx(100.0)

    def test_stdout_none_returns_empty(self):
        parser = RebenchParser()
        pr = make_failure(stdout=None)
        result = parser.parse(pr)
        assert result.measurements == []


# ---------------------------------------------------------------------------
# SingleResourceUsageParser
# ---------------------------------------------------------------------------

class TestSingleResourceUsageParser:
    def test_extracts_field_from_rusage(self):
        rusage = make_rusage(ru_maxrss=20480)
        pr = make_success(rusage=rusage)
        parser = SingleResourceUsageParser("ru_maxrss", "max_rss", "kB")
        result = parser.parse(pr)
        assert len(result.measurements) == 1
        # On darwin ru_maxrss is divided by 1024
        # The actual value depends on platform; just check it was extracted

    def test_darwin_maxrss_correction(self, monkeypatch):
        monkeypatch.setattr("benchr._parsers.sys.platform", "darwin")
        rusage = make_rusage(ru_maxrss=10240)
        pr = make_success(rusage=rusage)
        parser = SingleResourceUsageParser("ru_maxrss", "max_rss", "kB")
        result = parser.parse(pr)
        assert len(result.measurements) == 1
        assert result.measurements[0].value == pytest.approx(10.0)

    def test_linux_maxrss_no_correction(self, monkeypatch):
        monkeypatch.setattr("benchr._parsers.sys.platform", "linux")
        rusage = make_rusage(ru_maxrss=10240)
        pr = make_success(rusage=rusage)
        parser = SingleResourceUsageParser("ru_maxrss", "max_rss", "kB")
        result = parser.parse(pr)
        assert len(result.measurements) == 1
        assert result.measurements[0].value == pytest.approx(10240)

    def test_no_rusage_returns_empty(self):
        pr = make_success(rusage=None)
        parser = SingleResourceUsageParser("ru_maxrss", "max_rss", "kB")
        result = parser.parse(pr)
        assert result.measurements == []

    def test_extracts_non_maxrss_field(self):
        rusage = make_rusage(ru_utime=1.25)
        pr = make_success(rusage=rusage)
        parser = SingleResourceUsageParser("ru_utime", "user_time", "s")
        result = parser.parse(pr)
        assert len(result.measurements) == 1
        assert result.measurements[0].value == pytest.approx(1.25)
        assert result.measurements[0].metric == "user_time"


# ---------------------------------------------------------------------------
# MaxRssParser
# ---------------------------------------------------------------------------

class TestMaxRssParser:
    def test_factory_returns_correct_parser(self):
        parser = MaxRssParser()
        assert isinstance(parser, SingleResourceUsageParser)
        assert parser.field == "ru_maxrss"
        assert parser.metric == "max_rss"
        assert parser.unit == "kB"


# ---------------------------------------------------------------------------
# TimeParser
# ---------------------------------------------------------------------------

class TestTimeParser:
    def test_elapsed_only(self):
        parser = TimeParser(elapsed=True, user=False, system=False)
        pr = make_success(runtime=2.5)
        result = parser.parse(pr)
        assert len(result.measurements) == 1
        assert result.measurements[0].metric == "elapsed"
        assert result.measurements[0].value == pytest.approx(2.5)
        assert result.measurements[0].unit == "s"

    def test_user_and_system_only(self):
        rusage = make_rusage(ru_utime=0.5, ru_stime=0.1)
        parser = TimeParser(elapsed=False, user=True, system=True)
        pr = make_success(runtime=1.0, rusage=rusage)
        result = parser.parse(pr)
        metrics = {m.metric: m.value for m in result.measurements}
        assert "user" in metrics
        assert "system" in metrics
        assert metrics["user"] == pytest.approx(0.5)
        assert metrics["system"] == pytest.approx(0.1)
        assert "elapsed" not in metrics

    def test_all_three(self):
        rusage = make_rusage(ru_utime=0.5, ru_stime=0.1)
        parser = TimeParser(elapsed=True, user=True, system=True)
        pr = make_success(runtime=1.0, rusage=rusage)
        result = parser.parse(pr)
        metrics = {m.metric: m.value for m in result.measurements}
        assert len(metrics) == 3
        assert metrics["elapsed"] == pytest.approx(1.0)
        assert metrics["user"] == pytest.approx(0.5)
        assert metrics["system"] == pytest.approx(0.1)

    def test_runtime_none_skips_elapsed(self):
        parser = TimeParser(elapsed=True, user=False, system=False)
        pr = make_failure(runtime=None)
        result = parser.parse(pr)
        assert result.measurements == []

    def test_rusage_none_skips_user_system(self):
        parser = TimeParser(elapsed=False, user=True, system=True)
        pr = make_success(runtime=1.0, rusage=None)
        result = parser.parse(pr)
        assert result.measurements == []

    def test_all_false_raises_value_error(self):
        with pytest.raises(ValueError, match="at least one"):
            TimeParser(elapsed=False, user=False, system=False)

    def test_default_is_elapsed_only(self):
        parser = TimeParser()
        pr = make_success(runtime=3.14)
        result = parser.parse(pr)
        assert len(result.measurements) == 1
        assert result.measurements[0].metric == "elapsed"


# ---------------------------------------------------------------------------
# FailedParser
# ---------------------------------------------------------------------------

class TestFailedParser:
    def test_success_returns_0(self):
        parser = FailedParser()
        pr = make_success()
        result = parser.parse(pr)
        assert len(result.measurements) == 1
        assert result.measurements[0].metric == "failed"
        assert result.measurements[0].value == 0

    def test_failure_returns_1(self):
        parser = FailedParser()
        pr = make_failure()
        result = parser.parse(pr)
        assert len(result.measurements) == 1
        assert result.measurements[0].metric == "failed"
        assert result.measurements[0].value == 1


# ---------------------------------------------------------------------------
# MixedResultParser
# ---------------------------------------------------------------------------

class TestMixedResultParser:
    def test_combines_two_parsers(self):
        p1 = PlainFloatParser(unit="s")
        p2 = TimeParser(elapsed=True)
        mixed = MixedResultParser(p1, p2)
        pr = make_success(stdout="1.5\n", runtime=2.0)
        result = mixed.parse(pr)
        metrics = {m.metric: m.value for m in result.measurements}
        assert "runtime" in metrics
        assert "elapsed" in metrics
        assert metrics["runtime"] == pytest.approx(1.5)
        assert metrics["elapsed"] == pytest.approx(2.0)

    def test_and_operator_works(self):
        p1 = PlainFloatParser(unit="s")
        p2 = TimeParser(elapsed=True)
        mixed = p1 & p2
        assert isinstance(mixed, MixedResultParser)
        pr = make_success(stdout="1.0\n", runtime=3.0)
        result = mixed.parse(pr)
        assert len(result.measurements) == 2

    def test_nested_flattening(self):
        p1 = PlainFloatParser(unit="s")
        p2 = TimeParser(elapsed=True)
        p3 = FailedParser()
        inner = MixedResultParser(p1, p2)
        outer = MixedResultParser(inner, p3)
        # After flattening, outer.parsers should be [p1, p2, p3]
        assert len(outer.parsers) == 3
        assert outer.parsers[0] is p1
        assert outer.parsers[1] is p2
        assert outer.parsers[2] is p3


# ---------------------------------------------------------------------------
# IgnoreFailParserDecorator
# ---------------------------------------------------------------------------

class TestIgnoreFailParserDecorator:
    def test_converts_failure_to_success_before_delegating(self):
        inner = PlainFloatParser(unit="s")
        parser = inner.ignore_fail()
        assert isinstance(parser, IgnoreFailParserDecorator)
        pr = make_failure(stdout="1.5\n", returncode=1)
        result = parser.parse(pr)
        assert len(result.measurements) == 1
        assert result.measurements[0].value == pytest.approx(1.5)

    def test_runtime_defaults_to_minus1(self):
        inner = TimeParser(elapsed=True)
        parser = inner.ignore_fail()
        pr = make_failure(runtime=None)
        result = parser.parse(pr)
        assert len(result.measurements) == 1
        assert result.measurements[0].value == pytest.approx(-1)

    def test_success_passed_through_unchanged(self):
        inner = PlainFloatParser(unit="s")
        parser = inner.ignore_fail()
        pr = make_success(stdout="2.0\n")
        result = parser.parse(pr)
        assert len(result.measurements) == 1
        assert result.measurements[0].value == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# DirectionParserDecorator
# ---------------------------------------------------------------------------

class TestDirectionParserDecorator:
    @pytest.mark.parametrize("lower_is_better", [True, False])
    def test_tags_all_measurements(self, lower_is_better):
        inner = PlainFloatParser(unit="s")
        if lower_is_better:
            parser = inner.lower_is_better()
        else:
            parser = inner.higher_is_better()
        pr = make_success(stdout="1.0\n2.0\n")
        result = parser.parse(pr)
        assert len(result.measurements) == 2
        for m in result.measurements:
            assert m.lower_is_better is lower_is_better

    def test_lower_is_better_via_method(self):
        inner = PlainFloatParser(unit="s")
        parser = inner.lower_is_better()
        assert isinstance(parser, DirectionParserDecorator)
        pr = make_success(stdout="5.0\n")
        result = parser.parse(pr)
        assert result.measurements[0].lower_is_better is True

    def test_higher_is_better_via_method(self):
        inner = PlainFloatParser(unit="ops")
        parser = inner.higher_is_better()
        assert isinstance(parser, DirectionParserDecorator)
        pr = make_success(stdout="100.0\n")
        result = parser.parse(pr)
        assert result.measurements[0].lower_is_better is False
