import pytest

from metocean_stats.WW.weather_windows import WeatherWindow


@pytest.fixture
def ww(nora):
    return WeatherWindow(nora, timestep=3, direction="DIRP").fit(
        stages=[{"duration": 4, "criteria": {"HS": 3.0}}]
    )


def test_fit(nora):
    WeatherWindow(nora, timestep=3, direction="DIRP").fit(
        stages=[{"duration": 4, "criteria": {"HS": 3.0}}]
    )


def test_statistics(ww):
    ww.statistics(by="month")


def test_statistics_accepts_unified_stats_list(ww):
    stats = ["min", "mean", "P90", "count", "%"]
    df = ww.statistics(by="month", stats=stats)
    assert set(df.columns) == set(stats)


def test_plot_operability(ww):
    ww.plot_operability()


def test_plot_statistics(ww):
    ww.plot_statistics(by="month")


def test_availability_table(ww):
    ww.availability_table(row="direction", col="month")


def test_availability_table_accepts_percentile_stat(ww):
    ww.availability_table(row="direction", col="month", metric="waiting_time", stat="P90")


def test_plot_temporal_raster(ww):
    ww.plot_temporal_raster()


def test_plot_timeseries(ww):
    ww.plot_timeseries()
