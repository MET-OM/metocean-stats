"""
Generate the figures/tables embedded via ``.. image::``/``.. csv-table::``
directives in the v2 module docstrings (TimeSeries, ProfileTimeSeries, ...).

Output goes to docs/files_v2/, kept separate from the legacy v1 assets in
docs/files/ (which will go away once v1 is removed).

Re-run this whenever a v2 module's plotting/statistics output changes, so the
docstring examples stay in sync with the code:

    .conda/bin/python docs/generate_examples.py

Argument choices below are taken from real usage in .conda/metocean_examples/
notebooks (adapted to the small test dataset used throughout the test suite),
not invented call patterns.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metocean_stats.stats.aux_funcs import readNora10File
from metocean_stats.timeseries import TimeSeries, Variable
from metocean_stats.profile import ProfileTimeSeries
from metocean_stats.EVA.univariate_eva import UnivariateEVA
from metocean_stats.CMA.bivariate_eva import BivariateEVA
from metocean_stats.WW.weather_windows import WeatherWindow

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "tests" / "data" / "NORA_test.txt"
OUT_DIR = Path(__file__).resolve().parent / "files_v2"


def _load_data():
    return readNora10File(DATA_FILE)


def generate_timeseries_examples() -> None:
    """Examples for metocean_stats/timeseries.py (TimeSeries)."""
    df = _load_data()
    hs = Variable(col="HS", name="Significant wave height", symbol="H_s", unit="m")
    tp = Variable(col="TP", name="Peak period", symbol="T_p", unit="s")
    w10 = Variable(col="W10", name="Wind speed", symbol="U_10", unit="m/s")
    t2m = Variable(col="T2m", name="Air temperature", symbol="T_2m", unit="degC")
    dirp = Variable(col="DIRP", name="Peak direction", symbol="theta_p", unit="deg")
    ts = TimeSeries(df, primary=hs, secondary=[tp, w10, t2m], direction=dirp)

    # ts.smooth_discrete_geometric(...): Tp sits on NORA10's discrete
    # spectral grid, which looks like stripes in a Tp histogram/table.
    tp_before = ts.data["TP"].copy()
    ts.smooth_discrete_geometric("TP")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axes[0].hist(tp_before.dropna(), bins=60)
    axes[0].set_title("Tp -- raw (discrete grid)")
    axes[1].hist(ts.data["TP"].dropna(), bins=60)
    axes[1].set_title("Tp -- smoothed")
    for ax in axes:
        ax.set_xlabel("Tp (s)")
    axes[0].set_ylabel("Count")
    fig.savefig(OUT_DIR / "timeseries_smooth_discrete_geometric.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ts.statistics(...)
    table = ts.statistics(by="direction", stats=["P5", "P25", "P50", "P75", "P90", "P99", "Max"])
    table.round(2).to_csv(OUT_DIR / "timeseries_statistics_direction.csv")

    # ts.describe()
    ts.describe().round(2).to_csv(OUT_DIR / "timeseries_describe.csv")

    # ts.nonexceedance(...)
    table = ts.nonexceedance(variable="HS", by="month", step=1, rounding=1)
    table.to_csv(OUT_DIR / "timeseries_nonexceedance_month.csv")

    # ts.frequency_table(...)
    ts.frequency_table().to_csv(OUT_DIR / "timeseries_frequency_table.csv")

    # ts.block_maxima(...)
    table = ts.block_maxima(year_start=7, by="month")
    table.round(2).to_csv(OUT_DIR / "timeseries_block_maxima_month.csv")

    # ts.persistence(...)
    table = ts.persistence(threshold=4.0, by="month")
    table.round(2).to_csv(OUT_DIR / "timeseries_persistence_month.csv")

    # ts.trends(...)
    table = ts.trends(by="month")
    table.round(4).to_csv(OUT_DIR / "timeseries_trends_month.csv")

    # ts.statistics_2d(...)
    table = ts.statistics_2d(row="month", col="direction", stat="mean")
    table.round(2).to_csv(OUT_DIR / "timeseries_statistics_2d_month_direction.csv")

    # ts.plot_rose(...)
    fig = ts.plot_rose(
        magnitude="W10", direction="DIRM", by="month", figsize=(14, 20),
        mag_step=5, mag_max=25, ncols=3, cmap="viridis",
    )
    fig.savefig(OUT_DIR / "timeseries_plot_rose_month.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ts.plot_statistics(...)
    ax = ts.plot_statistics(variable="T2m", by="year")
    ax.figure.savefig(OUT_DIR / "timeseries_plot_statistics_year.png", dpi=120, bbox_inches="tight")
    plt.close(ax.figure)

    # ts.plot_statistics_2d(...)
    ax = ts.plot_statistics_2d(
        row="HS", col="TP", step_row=0.5, step_col=1, figsize=(15, 8),
        norm="log", cmap="Blues", stat_cells="density", fmt_cells=".2f", margins=True,
    )
    ax.figure.savefig(OUT_DIR / "timeseries_plot_statistics_2d.png", dpi=120, bbox_inches="tight")
    plt.close(ax.figure)


def generate_univariate_eva_examples() -> None:
    """Examples for metocean_stats/EVA/univariate_eva.py (UnivariateEVA)."""
    df = _load_data()
    uni = UnivariateEVA(df, "HS", "DIRP", "Significant wave height", "H_s", "m", sectors=4)
    uni.get_extremes(errors="ignore")
    uni.fit(AM_dist=["GEV"], errors="ignore")
    return_periods = [1, 10, 100]

    # uni.plot_return_value_comparison(...)
    ax = uni.plot_return_value_comparison(by="all", return_periods=return_periods)
    ax.figure.savefig(OUT_DIR / "univariate_eva_plot_return_value_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(ax.figure)

    # uni.plot_return_value_confidence(...)
    ax = uni.plot_return_value_confidence(by="all", method="AM", dist="GEV", return_periods=return_periods)
    ax.figure.savefig(OUT_DIR / "univariate_eva_plot_return_value_confidence.png", dpi=120, bbox_inches="tight")
    plt.close(ax.figure)

    # uni.table_model_parameters(...)
    table = uni.table_model_parameters(by="direction", method="AM", dist="GEV")
    table.round(3).to_csv(OUT_DIR / "univariate_eva_table_model_parameters_direction.csv")

    # uni.table_return_values_final(...)
    table = uni.table_return_values_final(by="direction", method="AM", dist="GEV", return_periods=return_periods)
    table.round(2).to_csv(OUT_DIR / "univariate_eva_table_return_values_final_direction.csv")

    # uni.plot_return_values_final(...)
    ax = uni.plot_return_values_final(by="direction", method="AM", dist="GEV", return_periods=return_periods)
    ax.figure.savefig(OUT_DIR / "univariate_eva_plot_return_values_final_direction.png", dpi=120, bbox_inches="tight")
    plt.close(ax.figure)


def generate_bivariate_eva_examples() -> None:
    """Examples for metocean_stats/CMA/bivariate_eva.py (BivariateEVA).

    Fit on a 10x-downsampled series: BivariateEVA.plot_marginal_quantiles
    has no speed knob of its own (unlike the underlying virocon model,
    which accepts `sample=`), so the cheapest way to keep every example
    here fast is to fit on less data in the first place.
    """
    df = _load_data().iloc[::10]
    joint = BivariateEVA(df, "HS", "TP", "DIRP", sectors=4)
    joint.fit()

    # joint.table_return_values(...)
    rve = joint.table_return_values(by="direction", return_periods=[1, 10, 100])
    rve.round(2).to_csv(OUT_DIR / "bivariate_eva_table_return_values_direction.csv")

    # joint.plot_isodensity_contours(...)
    ax = joint.plot_isodensity_contours(by="direction", RVE=rve)
    ax[0].figure.savefig(OUT_DIR / "bivariate_eva_plot_isodensity_contours_direction.png", dpi=120, bbox_inches="tight")
    plt.close(ax[0].figure)

    # joint.table_isodensity_contours(...)
    tables = joint.table_isodensity_contours(by="direction", RVE=rve)
    first_subset = next(iter(tables))
    tables[first_subset].round(2).to_csv(OUT_DIR / "bivariate_eva_table_isodensity_contours_direction.csv")

    # joint.table_conditional_return_values(...)
    table = joint.table_conditional_return_values(by="direction", RVE=rve)
    table.round(2).to_csv(OUT_DIR / "bivariate_eva_table_conditional_return_values_direction.csv")

    # joint.table_model_parameters(...)
    table = joint.table_model_parameters(by="direction")
    table.round(3).to_csv(OUT_DIR / "bivariate_eva_table_model_parameters_direction.csv")

    # joint.plot_marginal_quantiles(...)
    axes = joint.plot_marginal_quantiles(by="direction")
    axes[0].figure.savefig(OUT_DIR / "bivariate_eva_plot_marginal_quantiles_direction.png", dpi=120, bbox_inches="tight")
    plt.close(axes[0].figure)

    # joint.plot_dependence_functions(...)
    axes = joint.plot_dependence_functions(by="direction")
    axes[0].figure.savefig(OUT_DIR / "bivariate_eva_plot_dependence_functions_direction.png", dpi=120, bbox_inches="tight")
    plt.close(axes[0].figure)


def generate_profile_examples() -> None:
    """Examples for metocean_stats/profile.py (ProfileTimeSeries, ProfileEVA)."""
    rng = np.random.default_rng(0)
    time = pd.date_range("2000-01-01", periods=10 * 365 * 4, freq="6h")
    height = np.array([10.0, 25.0, 50.0])
    ws = np.abs(rng.normal(8.0, 2.0, (len(time), len(height)))) + height / 10
    wd = rng.uniform(0.0, 360.0, (len(time), len(height)))
    ds = xr.Dataset(
        {"ws": (("time", "height"), ws), "wd": (("time", "height"), wd)},
        coords={"time": time, "height": height},
    )
    pts = ProfileTimeSeries(ds, level_convention="height", primary="ws", direction="wd")

    # pts.statistics(...)
    pts.statistics(by="all").round(2).to_csv(OUT_DIR / "profile_statistics_all.csv")

    # pts.plot_profile(...)
    ax = pts.plot_profile(time=pts.time[100])
    ax.figure.savefig(OUT_DIR / "profile_plot_profile.png", dpi=120, bbox_inches="tight")
    plt.close(ax.figure)

    # pts.plot_statistics(...)
    ax = pts.plot_statistics(by="all", stat=["mean", "P90", "P99", "max"])
    ax.figure.savefig(OUT_DIR / "profile_plot_statistics.png", dpi=120, bbox_inches="tight")
    plt.close(ax.figure)

    # pts.plot_rose(...)
    axes = pts.plot_rose()
    axes[0].figure.savefig(OUT_DIR / "profile_plot_rose.png", dpi=120, bbox_inches="tight")
    plt.close(axes[0].figure)

    # pts.EVA.table_return_values(...) / pts.EVA.plot_return_values(...)
    pts.EVA.get_extremes(errors="ignore")
    pts.EVA.fit(AM_dist=["GEV"], errors="ignore")

    table = pts.EVA.table_return_values(10, "monthly", "AM", "GEV")
    table.round(2).to_csv(OUT_DIR / "profile_eva_table_return_values_monthly.csv")

    axes = pts.EVA.plot_return_values([1, 10, 100], "omni", "AM", "GEV")
    axes[0].figure.savefig(OUT_DIR / "profile_eva_plot_return_values_omni.png", dpi=120, bbox_inches="tight")
    plt.close(axes[0].figure)


def generate_weather_window_examples() -> None:
    """Examples for metocean_stats/WW/weather_windows.py (WeatherWindow)."""
    df = _load_data()
    ww = WeatherWindow(df, timestep=3, direction="DIRP").fit(
        stages=[{"duration": 4, "criteria": {"HS": 3.0}}]
    )

    # ww.statistics(...)
    ww.statistics(by="month").round(2).to_csv(OUT_DIR / "weather_window_statistics_month.csv")

    # ww.plot_operability(...)
    ax = ww.plot_operability(freqs=["M", "W", "D", "h"])
    ax.figure.savefig(OUT_DIR / "weather_window_plot_operability.png", dpi=120, bbox_inches="tight")
    plt.close(ax.figure)

    # ww.plot_statistics(...)
    ax = ww.plot_statistics(by="month")
    ax.figure.savefig(OUT_DIR / "weather_window_plot_statistics_month.png", dpi=120, bbox_inches="tight")
    plt.close(ax.figure)

    # ww.availability_table(...)
    ww.availability_table(row="direction", col="month").round(2).to_csv(
        OUT_DIR / "weather_window_availability_table.csv"
    )

    # ww.plot_temporal_raster(...)
    ax = ww.plot_temporal_raster()
    ax.figure.savefig(OUT_DIR / "weather_window_plot_temporal_raster.png", dpi=120, bbox_inches="tight")
    plt.close(ax.figure)

    # ww.plot_timeseries(...)
    axes = ww.plot_timeseries()
    fig = np.atleast_1d(axes).ravel()[0].figure
    fig.savefig(OUT_DIR / "weather_window_plot_timeseries.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_timeseries_examples()
    generate_univariate_eva_examples()
    generate_bivariate_eva_examples()
    generate_profile_examples()
    generate_weather_window_examples()


if __name__ == "__main__":
    main()
