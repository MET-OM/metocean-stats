import matplotlib.pyplot as plt
import numpy as np
import pytest

from metocean_stats.CMA.jointmodel import JointProbabilityModel


@pytest.fixture
def model2d(nora):
    model = JointProbabilityModel("hs_tp")
    model.fit(nora[["HS", "TP"]])
    return model


@pytest.fixture
def model3d(nora):
    model = JointProbabilityModel("u_hs_tp")
    model.fit(nora[["W10", "HS", "TP"]])
    return model


def test_str_returns_a_string(model2d):
    """Regression test: __str__ used to return the parameters DataFrame
    directly, so str(model) raised TypeError."""
    assert isinstance(str(model2d), str)


def test_plot_contours_with_single_string_label(model2d):
    """Regression test: a single string `labels` was reset to [] instead of
    [labels], so it was always rejected as not matching the number of
    periods."""
    model2d.plot_contours(periods=[10], labels="my-label")
    assert model2d.legend_labels == ["my-label"]


def test_plot_dnvgl_steepness_criterion_draws_one_line(model2d):
    """Regression test: the steepness line used to be drawn twice (once by
    the method itself, once by the shared plotting helper it also called)."""
    fig, ax = plt.subplots()
    model2d.plot_DNVGL_steepness_criterion(ax=ax)
    assert len(ax.lines) == 1


def test_plot_dnvgl_steepness_criterion_respects_xlim(model2d):
    """Regression test: ylim was always set to a data-derived default before
    the xlim-only code path could ever run, so an explicit xlim was ignored."""
    fig, ax = plt.subplots()
    model2d.plot_DNVGL_steepness_criterion(ax=ax, xlim=10)
    assert ax.lines[0].get_xdata().max() == pytest.approx(10.0)


def test_plot_data_scatter_rejects_bad_shape_clearly(model2d):
    """Regression test: a wrong-shape array raised a misleading
    NotImplementedError ("Only 2-dimensional data is supported") instead of
    a clear, shape-specific message."""
    with pytest.raises(ValueError, match="shape"):
        model2d.plot_data_scatter(data=np.arange(10))


def test_plot_data_density_rejects_bad_shape_clearly(model2d):
    with pytest.raises(ValueError, match="shape"):
        model2d.plot_data_density(data=np.arange(10))


def test_plot_data_scatter_single_point_still_works(model2d):
    model2d.plot_data_scatter(data=[1.5, 10.0])


def test_plot_3d_contour_slices_invalid_slice_dim_raises(model3d):
    """Regression test: an out-of-range slice_dim (e.g. a valid negative
    Python index that never matches 0/1/2) left axis_labels undefined,
    raising UnboundLocalError instead of a clear ValueError."""
    with pytest.raises(ValueError, match="slice_dim"):
        model3d.plot_3D_contour_slices(return_period=10, slice_dim=-1, slice_values=[1, 2])


def test_plot_3d_isodensity_contour_slice_invalid_slice_dim_raises(model3d):
    with pytest.raises(ValueError, match="slice_dim"):
        model3d.plot_3D_isodensity_contour_slice(marginal_values=[10], slice_dim=-1)


def test_plot_3d_isodensity_contour_slice_labels_with_default_levels(model3d):
    """Regression test: level_sort was only defined when explicit
    density_levels were given, so passing `labels` with the default levels
    raised UnboundLocalError."""
    model3d.plot_3D_isodensity_contour_slice(labels=[f"L{i}" for i in range(6)])


def test_plot_3d_isodensity_contour_handles_slices_with_no_contour(model3d, nora):
    """Regression test: a slice with no contour points used to assign None
    into a float array (X/Y/Z), raising a TypeError. These slices should
    now be left as NaN instead."""
    marginal_value = float(nora["W10"].quantile(0.99))
    ax = model3d.plot_3D_isodensity_contour(
        marginal_value=marginal_value, slice_dim=0, slice_values=30)
    assert ax is not None


def test_table_contour_and_table_isodensity_contour_are_finite(model2d):
    """table_contour and table_isodensity_contour derive their default
    range_values from the generated contour's own coordinate range (ceil
    for the lower bound, floor for the upper bound), so every requested
    value stays inside the contour and no row comes out all-NaN."""
    import numpy as np

    t1 = model2d.table_contour(return_period=10)
    assert len(t1) > 0
    assert np.isfinite(t1.select_dtypes("number").to_numpy()).all()

    t2 = model2d.table_isodensity_contour(points=[2.0, 4.0])
    assert len(t2) > 0
    # Different points' contours can have different valid ranges, so some
    # cells are legitimately NaN; only the first (range) column must be
    # fully finite.
    assert np.isfinite(t2.iloc[:, 0].to_numpy()).all()
    assert np.isfinite(t2.select_dtypes("number").to_numpy()).any()
