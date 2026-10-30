import numpy as np
import pandas as pd
import pytest
import xarray as xr

from metocean_stats.profile import ProfileTimeSeries, tqdm_levels


@pytest.fixture
def pts():
    rng = np.random.default_rng(0)
    time = pd.date_range("2000-01-01", periods=10 * 365 * 4, freq="6h")
    height = np.array([10.0, 25.0, 50.0])
    ws = np.abs(rng.normal(8.0, 2.0, (len(time), len(height)))) + height / 10
    wd = rng.uniform(0.0, 360.0, (len(time), len(height)))
    ds = xr.Dataset(
        {"ws": (("time", "height"), ws), "wd": (("time", "height"), wd)},
        coords={"time": time, "height": height},
    )
    return ProfileTimeSeries(ds, level_convention="height", primary="ws", direction="wd")


@pytest.fixture
def pts_eva(pts):
    pts.EVA.get_extremes()
    pts.EVA.fit(AM_dist=["genextreme"], POT_dist=["genpareto"])
    return pts


def test_properties(pts):
    pts.time
    pts.levels()
    pts.duration
    pts.timestep


def test_statistics(pts):
    pts.statistics()


def test_plot_profile(pts):
    pts.plot_profile(time=pts.time[100])


def test_plot_statistics(pts):
    pts.plot_statistics()


def test_plot_rose(pts):
    pts.plot_rose()


def test_to_timeseries(pts):
    pts.to_timeseries(level=25.0)


def test_eva_get_extremes(pts):
    pts.EVA.get_extremes()


def test_eva_fit(pts):
    pts.EVA.get_extremes()
    pts.EVA.fit(AM_dist=["genextreme"], POT_dist=["genpareto"])


def test_eva_table_return_values(pts_eva):
    pts_eva.EVA.table_return_values(return_period=10, grouping="omni", method="AM", dist="genextreme")


def test_eva_plot_return_values(pts_eva):
    pts_eva.EVA.plot_return_values(return_periods=[10, 100], grouping="omni", method="AM", dist="genextreme")


def test_eva_table_turkstra(pts_eva):
    pts_eva.EVA.table_turkstra(basis="percentile", percentile=99)


def test_eva_plot_turkstra(pts_eva):
    pts_eva.EVA.plot_turkstra(basis="percentile", percentile=99)


def test_eva_get_extremes_sectors_param(pts):
    """Regression test: ProfileEVA.get_extremes used to read `sectors` via
    getattr(self._pts, "sectors", 12), but ProfileTimeSeries never sets a
    `sectors` attribute, so the requested sector count was always ignored
    and silently replaced with the hardcoded default of 12."""
    pts.EVA.get_extremes(sectors=4)
    model = next(iter(pts.EVA.models.values()))
    assert len(model.data_sectors) == 4


def test_to_timeseries_unknown_primary_raises_value_error(pts):
    """Regression test: an unknown `primary` used to raise a bare KeyError
    before the intended, friendlier ValueError check ran."""
    with pytest.raises(ValueError):
        pts.to_timeseries(level=25.0, primary="unknown_variable")


def test_eva_getitem_missing_level_error_message(pts):
    """Regression test: the error message used to suggest
    get_extremes(levels=[...]), a parameter that doesn't exist."""
    with pytest.raises(KeyError, match="get_extremes"):
        pts.EVA[10.0]


def test_tqdm_levels():
    list(tqdm_levels([1.0, 2.0], desc="levels"))
