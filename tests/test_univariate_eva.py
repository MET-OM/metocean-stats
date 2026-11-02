import pytest

from metocean_stats.EVA.univariate_eva import UnivariateEVA
from metocean_stats.timeseries import TimeSeries, Variable

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


def test_plot_return_values_final_returns_axes(eva):
    """Regression test: used to return None because the function had no
    `return ax` statement."""
    ax = eva.plot_return_values_final(grouping="omni", method="AM", dist="genextreme",
                                      return_periods=[10, 100])
    assert ax is not None


def test_fit_errors_warn_does_not_crash_on_omni_failure(ts, monkeypatch):
    """Regression test: the omni fit used to ignore `errors` entirely and
    always raise, even with errors='warn', because it wasn't wrapped in the
    same try/except as the monthly and sector fits."""
    import pyextremes as pyex

    ts.EVA.get_extremes(errors="ignore")

    orig_fit_model = pyex.EVA.fit_model
    calls = {"n": 0}

    def patched(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated omni fit failure")
        return orig_fit_model(self, *args, **kwargs)

    monkeypatch.setattr(pyex.EVA, "fit_model", patched)

    with pytest.warns(UserWarning, match="omni"):
        ts.EVA.fit(AM_dist=["genextreme"], errors="warn")
    assert ("AM", "genextreme") not in ts.EVA.models_omni


def test_dist_name_map_honoured_by_plot_return_value_confidence(ts):
    """Regression test: used to look up models with the module-level
    _dist_name_map instead of self.dist_name_map, so a custom mapper was
    ignored and raised a KeyError."""
    def custom_map(dist, abbreviate=True):
        mapping = {"genextreme": "MYGEV"}
        if abbreviate:
            return mapping.get(dist, dist)
        reverse = {v: k for k, v in mapping.items()}
        return reverse.get(dist, dist)

    eva = UnivariateEVA(data=ts.data, var="HS", dist_name_map=custom_map)
    eva.get_extremes(errors="ignore")
    eva.fit(AM_dist=["genextreme"], errors="ignore")
    eva.plot_return_value_confidence(grouping="omni", method="AM", dist="MYGEV",
                                     return_periods=[10, 100], samples=20)


def test_get_extremes_with_missing_calendar_month(ts):
    """Regression test: get_extremes used to require exactly 12 monthly
    thresholds, so a record missing a whole calendar month failed even
    with the default, auto-derived thresholds."""
    data_missing_jan = ts.data[ts.data.index.month != 1]
    eva = UnivariateEVA(data=data_missing_jan, var="HS")
    assert len(eva.keys_monthly) == 11
    eva.get_extremes(errors="ignore")
    assert len(eva.th_monthly) == 11
