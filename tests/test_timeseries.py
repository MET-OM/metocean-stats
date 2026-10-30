import matplotlib.pyplot as plt
import pytest

from metocean_stats.timeseries import CMAConfig, TimeSeries, Variable

HS = Variable(col="HS", name="Significant wave height", symbol="H_s", unit="m")
DIRP = Variable(col="DIRP", name="Peak direction", symbol="theta_p", unit="deg")


@pytest.fixture
def ts(nora):
    return TimeSeries(nora, primary=HS, direction=DIRP)


def test_variable():
    assert HS.axis_label
    assert HS.short_label


def test_cma_config():
    CMAConfig(vars=("HS", "TP"), model=lambda *args: None)


def test_init(nora):
    TimeSeries(nora, primary=HS, direction=DIRP)


def test_properties(ts):
    ts.duration
    ts.timestep
    ts.timestep_hours
    ts.n_years
    ts.columns


def test_smooth_discrete_geometric(ts):
    ts.smooth_discrete_geometric("TP")


def test_describe(ts):
    ts.describe()


def test_statistics(ts):
    ts.statistics()


def test_statistics_accepts_unified_stats_list_and_keeps_labels(ts):
    """The func/percentiles split used to force two parameters for one
    concept; a single `stats` list (as accepted by
    `utils.aggregate_statistics`) now covers both, and output columns use
    exactly the strings given."""
    stats = ["Count", "Min", "P5", "Mean", "P95", "Maximum"]
    df = ts.statistics(by="month", stats=stats)
    assert list(df.columns) == stats


def test_nonexceedance(ts):
    ts.nonexceedance()


def test_frequency_table(ts):
    ts.frequency_table(row="month", col="direction")


def test_block_maxima(ts):
    ts.block_maxima(by="direction", min_coverage=0.0)


def test_block_maxima_default_works_without_direction(nora):
    """Regression test: block_maxima's `by` parameter used to default to
    "direction", so the most basic call (no arguments) failed outright for
    any TimeSeries without a direction variable configured. The sensible,
    always-available default is "year"."""
    ts_no_direction = TimeSeries(nora, primary=HS)
    df = ts_no_direction.block_maxima(min_coverage=0.0)
    assert "All" in df.index
    assert all(label.isdigit() for label in df.index if label != "All")


def test_block_maxima_accepts_unified_stats_list(ts):
    stats = ["min", "P50", "mean", "max"]
    df = ts.block_maxima(by="year", stats=stats, min_coverage=0.0)
    assert list(df.columns) == stats


def test_persistence(ts):
    ts.persistence(threshold=2.0)


def test_persistence_accepts_unified_stats_list(ts):
    stats = ["count", "mean", "P90"]
    df = ts.persistence(threshold=2.0, stats=stats)
    assert list(df.columns) == stats + ["Yearly"]


def test_trends(ts):
    ts.trends(by="month")


def test_trends_percentile_equivalent_forms_agree(ts):
    df_p = ts.trends(by="month", stat="P90", min_entries_per_year=50)
    df_pct = ts.trends(by="month", stat="90%", min_entries_per_year=50)
    assert df_p.equals(df_pct)


def test_plot_rose(ts):
    ts.plot_rose()


def test_statistics_2d(ts):
    ts.statistics_2d()


def test_statistics_2d_density_and_percent_tokens_agree(ts):
    df_density = ts.statistics_2d(row="month", col="direction", stat="density")
    df_percent = ts.statistics_2d(row="month", col="direction", stat="%")
    assert df_density.equals(df_percent)


def test_plot_statistics(ts):
    ts.plot_statistics()


def test_plot_statistics_2d(ts):
    ts.plot_statistics_2d()


def test_save_figure(ts, tmp_path):
    fig = plt.figure()
    ts.save_figure(fig, str(tmp_path / "figure.png"))
