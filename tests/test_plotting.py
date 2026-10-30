import numpy as np

from metocean_stats.plotting import plot_rose, plot_scatter_diagram, plot_spectra


def test_plot_rose():
    rng = np.random.default_rng(0)
    plot_rose(magnitude=rng.uniform(0.0, 10.0, 500), direction=rng.uniform(0.0, 360.0, 500))


def test_plot_scatter_diagram(nora):
    plot_scatter_diagram(nora, var1="TP", var2="HS")


def test_plot_spectra(spectra_ds):
    plot_spectra(spectra_ds["SPEC"].isel(time=0))
