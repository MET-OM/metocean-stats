import numpy as np
import pandas as pd

from metocean_stats.plotting import plot_rose, plot_scatter, plot_scatter_diagram, plot_spectra


def test_plot_rose():
    rng = np.random.default_rng(0)
    plot_rose(magnitude=rng.uniform(0.0, 10.0, 500), direction=rng.uniform(0.0, 360.0, 500))


def test_plot_scatter_diagram(nora):
    plot_scatter_diagram(nora, var1="TP", var2="HS")


def test_plot_scatter():
    rng = np.random.default_rng(0)
    x = rng.normal(5, 1.5, 500)
    y = x + rng.normal(0, 0.5, 500)
    df = pd.DataFrame({"obs": x, "model": y})
    ax = plot_scatter(df, "obs", "model")
    assert "CORR" in ax.texts[0].get_text()


def test_plot_spectra(spectra_ds):
    plot_spectra(spectra_ds["SPEC"].isel(time=0))
