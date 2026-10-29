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


def test_table_to_latex(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0]})
    utils.table_to_latex(df, str(tmp_path / "table.tex"), float_format=".2f")


def test_dirmag_to_uv():
    utils.dirmag_to_uv(np.array([90.0]), np.array([5.0]))


def test_uv_to_dirmag():
    utils.uv_to_dirmag(np.array([1.0]), np.array([0.0]))
