import pytest
from metocean_stats.CMA import JointProbabilityModel
from metocean_stats.CMA.predefined import get_DNVGL_Hs_Tz, get_LiGaoMoan_U_hs_tp
from metocean_stats.stats.aux_funcs import readNora10File
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

@pytest.fixture
def get_data():
    data_path = Path(__file__).parent / "data" / "NORA_test.txt"
    return readNora10File(data_path)

@pytest.fixture
def two_dim_model(get_data):
    data = get_data
    model = JointProbabilityModel(get_DNVGL_Hs_Tz)
    model.fit(data,"HS","TP")
    return model

@pytest.fixture
def three_dim_model(get_data):
    data = get_data
    model = JointProbabilityModel(get_LiGaoMoan_U_hs_tp)
    model.fit(data,"W10","HS","TP")
    return model


def test_plot_marginal_quantiles(two_dim_model):
    result =  two_dim_model.plot_marginal_quantiles()
    assert result is not None

def test_plot_dependence_functions(two_dim_model):
    result = two_dim_model.plot_dependence_functions()
    assert result is not None

def test_plot_histograms(two_dim_model):
    result = two_dim_model.plot_histograms_of_interval_distributions()
    assert result is not None

def test_plot_isodensity(two_dim_model):
    result = two_dim_model.plot_isodensity_contours(points=[1,2,3,4,5])
    assert result is not None

def test_plot_iform(two_dim_model):
    result = two_dim_model.plot_contours(method="IFORM")
    assert result is not None

def test_plot_isorm(two_dim_model):
    result = two_dim_model.plot_contours(method="isorm")
    assert result is not None

def test_plot_pdf(two_dim_model):
    result = two_dim_model.plot_pdf_heatmap()
    assert result is not None

def test_get_dependent_given_marginal(two_dim_model):
    result = two_dim_model.get_dependent_given_marginal([1,2,3,4,5,6,7,8,9,10])
    assert len(result) == 10

def test_plot_dependent_percentiles(two_dim_model):
    result = two_dim_model.plot_dependent_percentiles()
    assert result is not None

def test_plot_DNVGL_steepness(two_dim_model):
    result = two_dim_model.plot_DNVGL_steepness_criterion()
    assert result is not None

def test_plot_data_scatter(two_dim_model):
    result = two_dim_model.plot_data_scatter()
    assert result is not None

def test_plot_data_density(two_dim_model):
    result = two_dim_model.plot_data_density()
    assert result is not None

def test_plot_semantics(two_dim_model):
    result = two_dim_model.plot_semantics()
    assert result is not None

def test_parameters(two_dim_model):
    result = two_dim_model.parameters(complete=False)
    assert type(result) is dict

def test_parameters_complete(two_dim_model):
    result = two_dim_model.parameters(complete=True)
    assert type(result) is pd.DataFrame

def test_reset_labels(two_dim_model):
    two_dim_model.reset_labels()

def test_plot_legend(two_dim_model):
    _,ax = plt.subplots()
    result = two_dim_model.plot_legend(ax)
    assert result is not None

def test_plot_3D_contour(three_dim_model):
    result = three_dim_model.plot_3D_contour()
    assert result is not None

def test_plot_3D_slices(three_dim_model):
    result = three_dim_model.plot_3D_contour_slices()
    assert result is not None

def test_plot_3D_isodensity(three_dim_model):
    result = three_dim_model.plot_3D_isodensity_contour(marginal_value=10.0,slice_values=[2,4,6,8])
    assert result is not None

def test_plot_3D_isodensity_slices(three_dim_model):
    result = three_dim_model.plot_3D_isodensity_contour_slice()
    assert result is not None

