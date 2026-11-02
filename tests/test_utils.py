import numpy as np
import pandas as pd
import pytest

from metocean_stats import utils


def test_bin_range_labels():
    utils.bin_range_labels([0, 1, 2], unit=" m")


@pytest.mark.parametrize("by", ["month", "season", "year", "week", "day", "hour", "direction", "HS"])
def test_groupby_with_metadata(nora, by):
    utils.groupby_with_metadata(nora, by=by, var="HS", var_dir="DIRP")


def test_get_n_axes():
    utils.get_n_axes(5)


def test_resolve_axes_grid():
    utils.resolve_axes_grid(3)


def test_infer_step(nora):
    utils.infer_step(nora["HS"])


def test_bin_edges(nora):
    utils.bin_edges(nora["HS"], 0.5)


def test_aggregate_statistics(nora):
    utils.aggregate_statistics(nora["HS"], ["mean", "p90", "max"])


def test_aggregate_statistics_keeps_input_labels_and_sorts_output():
    s = pd.Series(np.arange(1, 101, dtype=float))
    stats = ["Count", "Min", "P5", "P25", "P50", "Mean", "P75", "P90", "P99", "Maximum"]
    result = utils.aggregate_statistics(s, stats)
    assert list(result.keys()) == stats


def test_aggregate_statistics_aliases_and_equivalent_percentiles():
    s = pd.Series(np.arange(1, 101, dtype=float))
    result = utils.aggregate_statistics(
        s, ["minimum", "MIN", "MAXIMUM", "max", "P90", "90%", "p090"]
    )
    assert result["minimum"] == result["MIN"]
    assert result["MAXIMUM"] == result["max"]
    assert result["P90"] == result["90%"] == result["p090"]


def test_aggregate_statistics_percent_of_total_requires_total():
    s = pd.Series(np.arange(10, dtype=float))
    with pytest.raises(ValueError, match="total"):
        utils.aggregate_statistics(s, ["%"])
    result = utils.aggregate_statistics(s, ["%", "density", "count"], total=100)
    assert result["%"] == result["density"] == 10.0


def test_aggregate_statistics_rejects_bare_number_and_unknown_token():
    s = pd.Series(np.arange(10, dtype=float))
    with pytest.raises(ValueError):
        utils.aggregate_statistics(s, ["90"])
    with pytest.raises(ValueError):
        utils.aggregate_statistics(s, ["bogus"])


def test_table_to_latex(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0]})
    utils.table_to_latex(df, str(tmp_path / "table.tex"), float_format=".2f")


def test_dirmag_to_uv():
    utils.dirmag_to_uv(np.array([90.0]), np.array([5.0]))


def test_uv_to_dirmag():
    utils.uv_to_dirmag(np.array([1.0]), np.array([0.0]))
