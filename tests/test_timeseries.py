import matplotlib.pyplot as plt
import pytest

from metocean_stats.timeseries import CMAConfig, TimeSeries, Variable

HS = Variable(col="HS", name="Significant wave height", symbol="H_s", unit="m")
DIRP = Variable(col="DIRP", name="Peak direction", symbol="theta_p", unit="deg")

NOT_IMPLEMENTED = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="not implemented yet")


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


@NOT_IMPLEMENTED
def test_resample(ts):
    ts.resample("1D")


@NOT_IMPLEMENTED
def test_fill_gaps(ts):
    ts.fill_gaps()


@NOT_IMPLEMENTED
def test_trim(ts):
    ts.trim()


def test_smooth_discrete_geometric(ts):
    ts.smooth_discrete_geometric("TP")


@NOT_IMPLEMENTED
def test_availability(ts):
    ts.availability()


def test_describe(ts):
    ts.describe()


def test_statistics(ts):
    ts.statistics()


def test_nonexceedance(ts):
    ts.nonexceedance()


def test_frequency_table(ts):
    ts.frequency_table(row="month", col="direction")


def test_block_maxima(ts):
    ts.block_maxima(by="direction", min_coverage=0.0)


def test_persistence(ts):
    ts.persistence(threshold=2.0)


def test_trends(ts):
    ts.trends(by="month")


def test_plot_rose(ts):
    ts.plot_rose()


def test_statistics_2d(ts):
    ts.statistics_2d()


def test_plot_statistics(ts):
    ts.plot_statistics()


def test_plot_statistics_2d(ts):
    ts.plot_statistics_2d()


@NOT_IMPLEMENTED
def test_plot_timeseries(ts):
    ts.plot_timeseries()


@NOT_IMPLEMENTED
def test_plot_monthly_boxplot(ts):
    ts.plot_monthly_boxplot()


@NOT_IMPLEMENTED
def test_plot_exceedance(ts):
    ts.plot_exceedance()


@NOT_IMPLEMENTED
def test_plot_histogram(ts):
    ts.plot_histogram()


@NOT_IMPLEMENTED
def test_climatology(ts):
    ts.climatology()


@NOT_IMPLEMENTED
def test_to_csv(ts, tmp_path):
    ts.to_csv(str(tmp_path / "ts.csv"))


def test_save_figure(ts, tmp_path):
    fig = plt.figure()
    ts.save_figure(fig, str(tmp_path / "figure.png"))
