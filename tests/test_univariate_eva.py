import pytest

from metocean_stats.timeseries import TimeSeries, Variable

NOT_IMPLEMENTED = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="not implemented yet")

HS = Variable(col="HS", name="Significant wave height", symbol="H_s", unit="m")
DIRP = Variable(col="DIRP", name="Peak direction", symbol="theta_p", unit="deg")


@pytest.fixture
def ts(nora):
    return TimeSeries(nora, primary=HS, direction=DIRP)


@pytest.fixture
def eva(ts):
    ts.EVA.get_extremes(errors="ignore")
    ts.EVA.fit(AM_dist=["genextreme"], POT_dist=["genpareto"], errors="ignore")
    return ts.EVA


def test_get_extremes(ts):
    ts.EVA.get_extremes(errors="ignore")


def test_fit(ts):
    ts.EVA.get_extremes(errors="ignore")
    ts.EVA.fit(AM_dist=["genextreme"], POT_dist=["genpareto"], errors="ignore")


@NOT_IMPLEMENTED
def test_plot_threshold_diagnostics(eva):
    eva.plot_threshold_diagnostics()


def test_plot_return_value_comparison(eva):
    eva.plot_return_value_comparison(grouping="omni", return_periods=[10, 100])


def test_plot_return_value_confidence(eva):
    eva.plot_return_value_confidence(grouping="omni", method="AM", dist="genextreme",
                                     return_periods=[10, 100], samples=20)


def test_table_model_parameters(eva):
    eva.table_model_parameters(grouping="monthly", method="AM", dist="genextreme")


def test_table_return_values_final(eva):
    eva.table_return_values_final(grouping="omni", method="AM", dist="genextreme",
                                  return_periods=[10, 100])


def test_plot_return_values_final(eva):
    eva.plot_return_values_final(grouping="omni", method="AM", dist="genextreme",
                                 return_periods=[10, 100])
