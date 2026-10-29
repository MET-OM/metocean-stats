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


def test_table_isodensity_contours(bvm_fit, rve):
    bvm_fit.table_isodensity_contours(grouping="monthly", RVE=rve)


def test_table_conditional_return_values(bvm_fit, rve):
    bvm_fit.table_conditional_return_values(grouping="monthly", RVE=rve)


def test_table_return_values(bvm_fit):
    bvm_fit.table_return_values(grouping="monthly", return_periods=[10, 100])


def test_table_model_parameters(bvm_fit):
    bvm_fit.table_model_parameters(grouping="monthly")


def test_plot_marginal_quantiles(bvm_fit):
    bvm_fit.plot_marginal_quantiles(grouping="monthly")


def test_plot_dependence_functions(bvm_fit):
    bvm_fit.plot_dependence_functions(grouping="monthly")
