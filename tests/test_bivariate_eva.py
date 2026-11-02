import numpy as np
import pytest

from metocean_stats.CMA.bivariate_eva import BivariateEVA


@pytest.fixture
def bvm(nora):
    return BivariateEVA(data=nora, var1="HS", var2="TP", var_dir="DIRP", sectors=4)


@pytest.fixture
def bvm_fit(bvm):
    bvm.fit()
    return bvm


@pytest.fixture
def rve(bvm_fit):
    return bvm_fit.table_return_values(grouping="monthly", return_periods=[10, 100])


def test_init(nora):
    BivariateEVA(data=nora, var1="HS", var2="TP", var_dir="DIRP", sectors=4)


def test_fit(bvm):
    bvm.fit()


def test_plot_isodensity_contours(bvm_fit, rve):
    bvm_fit.plot_isodensity_contours(grouping="omni", RVE=rve)


def test_plot_isodensity_contours_with_explicit_ax(bvm_fit, rve):
    """Regression test: passing ax for grouping='omni' used to raise
    TypeError due to isinstance(ax, plt.axes) checking a function."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    bvm_fit.plot_isodensity_contours(grouping="omni", RVE=rve, ax=ax)


def test_table_isodensity_contours(bvm_fit, rve):
    bvm_fit.table_isodensity_contours(grouping="monthly", RVE=rve)


def test_table_isodensity_contours_column_count(bvm_fit, rve):
    """Regression test: columns used to be doubled (8 instead of 4) because
    the loop iterated over every RVE column (both var1 and var2) instead of
    just var1's return-value columns."""
    tables = bvm_fit.table_isodensity_contours(grouping="monthly", RVE=rve)
    any_table = next(iter(tables.values()))
    # 2 return periods -> 2 (low, high) columns each = 4, same count as RVE's
    # own columns (2 return periods x 2 variables).
    assert any_table.shape[1] == len(rve.columns) == 4


def test_table_conditional_return_values(bvm_fit, rve):
    bvm_fit.table_conditional_return_values(grouping="monthly", RVE=rve)


def test_table_conditional_return_values_matches_rve_hs(bvm_fit, rve):
    """Regression test: the HS column used to be computed from a mix of HS
    and TP return values zipped together incorrectly."""
    cond = bvm_fit.table_conditional_return_values(grouping="monthly", RVE=rve)
    hs_cols_rve = [c for c in rve.columns if c.endswith(" H_s")]
    hs_cols_cond = [c for c in cond.columns if "H_s" in c]
    assert len(hs_cols_cond) == len(hs_cols_rve)
    for rve_col, cond_col in zip(hs_cols_rve, hs_cols_cond):
        assert np.allclose(cond[cond_col].values, rve[rve_col].values)


def test_table_return_values(bvm_fit):
    bvm_fit.table_return_values(grouping="monthly", return_periods=[10, 100])


def test_table_model_parameters(bvm_fit):
    bvm_fit.table_model_parameters(grouping="monthly")


def test_plot_marginal_quantiles(bvm_fit):
    bvm_fit.plot_marginal_quantiles(grouping="monthly")


def test_plot_marginal_quantiles_with_explicit_axes(bvm_fit):
    """Regression test: passing axes explicitly for grouping='monthly' used
    to raise UnboundLocalError('fig') since fig was only set when axes was
    created internally."""
    import matplotlib.pyplot as plt
    n = len(bvm_fit.data_monthly)
    fig, axes = plt.subplots(2 * n, 1)
    bvm_fit.plot_marginal_quantiles(grouping="monthly", axes=axes.ravel())


def test_plot_dependence_functions(bvm_fit):
    bvm_fit.plot_dependence_functions(grouping="monthly")


def test_plot_dependence_functions_with_explicit_axes(bvm_fit):
    """Regression test: same fig-undefined bug as plot_marginal_quantiles."""
    import matplotlib.pyplot as plt
    n = len(bvm_fit.data_monthly)
    fig, axes = plt.subplots(2 * n, 1)
    bvm_fit.plot_dependence_functions(grouping="monthly", axes=axes.ravel())
