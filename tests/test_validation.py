import pytest

from metocean_stats.validation import TimeSeriesValidation


@pytest.fixture
def val(nora):
    model = nora.copy()
    model["HS"] = nora["HS"] * 1.05
    return TimeSeriesValidation({"ref": nora, "model": model})


@pytest.fixture
def val_aligned(val):
    val.align(reference="ref")
    return val


def test_init(nora):
    model = nora.copy()
    TimeSeriesValidation({"ref": nora, "model": model})


def test_summary(val):
    val.summary()


def test_align(val):
    val.align(reference="ref")


def test_get_variable(val):
    val.get_variable("ref", "HS")


def test_plot_scatter(val_aligned):
    val_aligned.plot_scatter(x=("ref", "HS"), y=("model", "HS"))


def test_plot_rose(val_aligned):
    val_aligned.plot_rose(magnitude=("model", "HS"), direction=("model", "DIRP"))


def test_plot_timeseries(val_aligned):
    val_aligned.plot_timeseries(vars=[("ref", "HS"), ("model", "HS")])


def test_error_metrics(val_aligned):
    val_aligned.error_metrics(x=("ref", "HS"), y=("model", "HS"))
