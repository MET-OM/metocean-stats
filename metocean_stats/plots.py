from __future__ import annotations

import itertools
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from typing import Literal

from .spectral import _ROLE_TIME, _ROLE_FREQ, _ROLE_DIR, _ALIASES_FREQ, _ALIASES_DIR

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cmocean
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import infer_step

_AUTO_FACET_WARN_THRESHOLD = 25
_AUTO_FACET_RAISE_THRESHOLD = 100

def _grid_shape(n_panels: int, max_cols: int = 4) -> tuple[int, int]:
    """
    Choose a (nrows, ncols) grid that comfortably fits `n_panels` panels,
    preferring a roughly-square layout up to `max_cols` columns, then
    widening to max_cols and growing rows beyond that. Never returns a
    grid with fewer cells than n_panels.
    """
    if n_panels <= 0:
        raise ValueError(f"n_panels should be a positive integer, got {n_panels}.")
 
    if n_panels < max_cols ** 2:
        table = np.array([
            (rows, cols)
            for cols in range(1, max_cols + 1)
            for rows in range(1, cols + 1)
        ])
        table = table[np.prod(table, axis=1) >= n_panels]
        nrows, ncols = table[np.argmin(np.prod(table, axis=1) - n_panels)]
        return int(nrows), int(ncols)
 
    nrows = int(np.ceil(n_panels / max_cols))
    return nrows, max_cols
 
 
def _check_panel_count(n_panels: int, dims: list, errors: str) -> None:
    """
    errors="ignore" -- never warns or raises, any n_panels.
    errors="raise"  -- warns above _AUTO_FACET_WARN_THRESHOLD, raises
                        above _AUTO_FACET_RAISE_THRESHOLD.
    """
    if errors not in ("ignore", "raise"):
        raise ValueError(f"errors must be 'ignore' or 'raise', got {errors!r}.")
    if errors == "ignore":
        return
 
    if n_panels > _AUTO_FACET_RAISE_THRESHOLD:
        raise ValueError(
            f"Auto-faceting/flattening over {dims} would produce "
            f"{n_panels} panels (> {_AUTO_FACET_RAISE_THRESHOLD}), which "
            "is almost certainly unintentional and too slow to render. "
            "Pass row=/col= to select a subset, reduce the data first "
            "(e.g. via .sel()/.isel()), or pass errors='ignore' to force "
            "it anyway."
        )
    if n_panels > _AUTO_FACET_WARN_THRESHOLD:
        warnings.warn(
            f"Auto-faceting/flattening over {dims} produces {n_panels} "
            f"panels (> {_AUTO_FACET_WARN_THRESHOLD}), which may be slow "
            "to render. Pass row=/col= to select a subset, reduce the "
            "data first, or pass errors='ignore' to silence this.",
            stacklevel=3,
        )
 


def _leftover_dims(da: xr.DataArray, *spectral_dims: str) -> tuple[str, ...]:
    """Every dimension of `da` except the named spectral dimension(s)."""
    return tuple(d for d in da.dims if d not in spectral_dims)


def _format_panel_value(v) -> str:
    """
    Format one facet coordinate value for a panel title. Datetime-like
    values are truncated to minute precision (YYYY-MM-DD HH:MM), never
    finer. Every other value type (str labels like "Jan"/"DJF", numeric
    bin edges, etc.) is passed through via plain str(), untouched.
    """
    if isinstance(v, np.datetime64):
        return pd.Timestamp(v).strftime("%Y-%m-%d %H:%M")
    return str(v)
 
 
def _colorbar_label(da) -> str:
    """
    'name [units]' from da.attrs, falling back gracefully if either is
    missing rather than raising -- a plot should still render for data
    that hasn't been through SpectralTimeSeries's attrs machinery.
    """
    # name = da.attrs.get("name") or da.name or "Spectral density"
    name = "Spectral density"
    units = da.attrs.get("units")
    return f"{name} [{units}]" if units else name


def _global_radial_zoom(
    da: xr.DataArray,
    freq_dim: str | None,
    dir_dim: str | None,
    vmin: float,
    radius: str = "frequency",
) -> tuple[float | None, float | None]:
    """
    Compute a single, GLOBAL radial limit for 2D polar spectrum panels,
    so every panel in a facet grid can share the same "zoomed in" view
    and stay comparable -- exactly like vmin/vmax already do for color.

    Finds the LOWEST and HIGHEST frequency at which ANY value anywhere
    in `da` (across every batch/facet dimension and the direction
    dimension) still exceeds `vmin`. Outside that frequency range, no
    panel has anything to show, so it's safe to clip there. Both edges
    are computed (not just the high-frequency one) because the "empty"
    side of the spectrum could be either end -- e.g. swell-only data
    wastes the high-frequency tail, wind-sea-only data wastes the
    low-frequency end.

    Frequency and period are RECIPROCAL axes, so a cutoff on one is a
    cutoff on the other end of the other:
      - radius="frequency": rmin <- lowest valid freq, rmax <- highest
        valid freq (direct, un-flipped mapping).
      - radius="period": rmin <- 1/(highest valid freq) [small period,
        near centre], rmax <- 1/(lowest valid freq) [large period,
        outer edge] -- flipped, because period grows as frequency
        shrinks.

    Values are returned in the DATA's native units (Hz or seconds) --
    any logradius transform is applied later by the caller, exactly
    like the un-logged vmin/vmax are.

    Returns (rmin, rmax), either or both None if there's no freq
    dimension to zoom along (i.e. this is a 1D spectrum), or if nothing
    in the data exceeds vmin (falls back to the full range, no clip).
    """
    if freq_dim is None or dir_dim is None:
        # Zoom only applies to the 2D polar case.
        return None, None

    other_dims = tuple(d for d in da.dims if d != freq_dim)
    # Max over every dimension except freq (dir + any leftover batch
    # dims) -- "is there ANYWHERE, in ANY panel, data above vmin at
    # this frequency?"
    max_per_freq = da.max(dim=other_dims, skipna=True)
    valid = max_per_freq.values > vmin
    if not np.any(valid):
        # Nothing exceeds vmin anywhere -- don't zoom into an empty plot.
        return None, None

    freqs = da.coords[freq_dim].values.astype(float)
    lowest_valid_freq = float(freqs[valid].min())
    highest_valid_freq = float(freqs[valid].max())

    if radius == "period":
        rmin = 1.0 / highest_valid_freq if highest_valid_freq > 0 else None
        rmax = 1.0 / lowest_valid_freq if lowest_valid_freq > 0 else None
        return rmin, rmax

    return lowest_valid_freq, highest_valid_freq

def _resolve_spectral_target(
    ds: xr.Dataset,
    var: str | None = None,
) -> tuple[str, str | None, str | None]:
    """
    Identify which data variable in `ds` is a spectrum, and which of its
    dims play the freq/dir roles -- via the same alias detection as
    SpectralTimeSeries.__init__, but WITHOUT requiring a time role. This
    is what lets plotting work identically on the dimension-agnostic
    output of aggregate()/groupby().aggregate() (whose non-spectral dims
    might be "month", "hs", a stacked combo, or none at all).

    Parameters
    ----------
    ds : xr.Dataset
    var : str, optional
        Name of the spectral variable to plot. If None, exactly one data
        variable in `ds` must carry a freq and/or dir dim (by alias) --
        if more than one qualifies, raises asking for var= to disambiguate.

    Returns
    -------
    (var_name, freq_dim, dir_dim)
        freq_dim / dir_dim are None if that role isn't present on the
        chosen variable.
    """
    dim_map: dict[str, str | None] = {}
    for role, aliases in ((_ROLE_FREQ, _ALIASES_FREQ), (_ROLE_DIR, _ALIASES_DIR)):
        dim_map[role] = next((a for a in aliases if a in ds.dims), None)

    freq_dim, dir_dim = dim_map[_ROLE_FREQ], dim_map[_ROLE_DIR]
    if freq_dim is None and dir_dim is None:
        raise ValueError(
            "No frequency or direction dimension found "
            f"(looked for aliases {_ALIASES_FREQ} / {_ALIASES_DIR} among "
            f"dims {list(ds.dims)}). Nothing spectral to plot."
        )

    def _dims_on(v):
        return (freq_dim if freq_dim in ds[v].dims else None,
                dir_dim if dir_dim in ds[v].dims else None)

    if var is not None:
        if var not in ds.data_vars:
            raise ValueError(f"var={var!r} not found. Available: {list(ds.data_vars)}.")
        f, d = _dims_on(var)
        return var, f, d

    candidates = [
        v for v in ds.data_vars
        if (freq_dim is not None and freq_dim in ds[v].dims)
        or (dir_dim  is not None and dir_dim  in ds[v].dims)
    ]
    if len(candidates) == 0:
        raise ValueError(
            f"No data variable has a freq or dir dimension. "
            f"Available: {list(ds.data_vars)}."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple candidate spectral variables found: {candidates}. "
            "Pass var=<name> to disambiguate."
        )
    f, d = _dims_on(candidates[0])
    return candidates[0], f, d

def _to_log_radius(r: np.ndarray) -> np.ndarray:
    """
    Map native radial values (Hz or seconds, always > 0 for a spectrum)
    onto a log10 coordinate for plotting. Kept as a tiny named helper
    (rather than inlining np.log10 everywhere) so every place that needs
    to move a radial value onto/off the log axis -- the mesh/contour
    coordinates themselves, r_zoom, and user rmin/rmax -- goes through
    exactly one definition and can't drift out of sync with each other.
    """
    r = np.asarray(r, dtype=float)
    return np.log10(r)


def _radial_ticks(rmin: float, rmax: float, logradius: bool, n: int = 6) -> np.ndarray:
    """
    Pick tick locations spanning [rmin, rmax] (already in PLOT
    coordinates -- i.e. already log10'd if logradius=True), returned in
    that same PLOT coordinate system so callers can hand them straight
    to ax.set_rticks().

    For the log case this delegates to matplotlib's own LogLocator
    (1-2-5-per-decade "nice" ticks) rather than a hand-rolled stepping
    loop -- a hand-rolled version risks landing on ugly values or, worse,
    an infinite loop for degenerate/near-equal rmin==rmax ranges, and
    LogLocator already solves both correctly. When the visible range
    spans less than one decade (e.g. a tightly-zoomed panel), log
    spacing has nothing to offer, so this falls back to plain "nice"
    linear ticks instead -- still returned in log-plot coordinates.
    """
    if not logradius:
        return np.linspace(rmin, rmax, n)

    native_min, native_max = 10 ** rmin, 10 ** rmax
    if not np.isfinite(native_min) or native_min <= 0:
        native_min = native_max / 100
    if native_min >= native_max:
        return _to_log_radius(np.array([native_min]))

    decades = np.log10(native_max / native_min)
    if decades < 1.0:
        ticks = mticker.MaxNLocator(nbins=n, steps=[1, 2, 2.5, 5, 10]).tick_values(native_min, native_max)
    else:
        loc = mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0))
        ticks = loc.tick_values(native_min, native_max)
        if len(ticks[(ticks >= native_min) & (ticks <= native_max)]) > n:
            loc = mticker.LogLocator(base=10, subs=(1.0,))
            ticks = loc.tick_values(native_min, native_max)

    ticks = ticks[(ticks >= native_min) & (ticks <= native_max)]
    if len(ticks) < 2:
        ticks = np.array([native_min, native_max])
    return _to_log_radius(ticks)

def _draw_polar_spectrum(
    ax, da, freq_dim: str, dir_dim: str,
    radius: str = "frequency", plot_type: str = "pcolormesh",
    cmap=cmocean.cm.thermal, vmin: float | None = None, vmax: float | None = None,
    dir_letters: bool = False, log_scale: bool = True,
    r_zoom: tuple[float | None, float | None] = (None, None),
    logradius: bool = True,
    r_limits: tuple[float | None, float | None] = (None, None),
    mask_below_vmin: bool = False,
):
    """
    Draw one 2D (freq, dir) spectrum slice onto a polar Axes. `da` must
    already be reduced to exactly (freq_dim, dir_dim).
 
    The directional axis is periodically closed in ANGLE space (after
    sorting theta), not in the original degree coordinate -- closing in
    degree space (e.g. appending dir=365) doesn't survive
    _direction_to_math_radians' internal `% 360`, which collapses such a
    point back onto an existing angle instead of extending past it. The
    fix here appends theta_sorted[0] + 2*pi (and the matching data
    column) as the literal final point AFTER theta has been computed and
    sorted, which is the only place the "one full lap later" value can
    be expressed without being wrapped away.
 
    log_scale=True (default) colors on a log scale via LogNorm. `vmin`/
    `vmax` should normally be supplied by the caller (plot_spectra) as
    GLOBAL values computed once across every panel in a grid.
 
    plot_type : {"pcolormesh", "contour", "contourf"}
        "contour" draws unfilled contour lines only; "contourf" is the
        filled equivalent. Colorbar ticks for log_scale=True are
        decade-aligned via LogLocator for every plot_type, stashed on
        the returned mesh as `mesh._log_cbar_ticks` -- see plot_spectra,
        which passes this to fig.colorbar(..., ticks=...).
 
    logradius : bool, default True
        Plot the radial (frequency/period) axis on a log10 scale rather
        than linear, by transforming the radial coordinate itself.
 
    r_zoom : (float | None, float | None), default (None, None)
        GLOBAL radial limits (native units), computed once across every
        panel by the caller. Overridden per-side by r_limits.
 
    r_limits : (float | None, float | None), default (None, None)
        Explicit, user-supplied radial limits (native units), taking
        precedence over r_zoom on whichever side(s) are not None.
 
    Returns the drawn mappable, carrying `_log_cbar_ticks` (array or
    None) for colorbar tick placement.
    """
    freqs = da.coords[freq_dim].values.astype(float)
    dirs  = da.coords[dir_dim].values.astype(float)
    rad_native = 1.0 / freqs if radius == "period" else freqs
    rad   = _to_log_radius(rad_native) if logradius else rad_native
    theta = np.deg2rad(dirs % 360)

    order = np.argsort(theta)
    theta = theta[order]
    vals  = da.transpose(freq_dim, dir_dim).values[:, order]

    theta = np.concatenate([theta, [theta[0] + 2.0 * np.pi]])
    vals = np.concatenate([vals, vals[:, :1]], axis=1)

    vmax = vmax if vmax is not None else float(np.nanmax(vals))

    # Resolve the colormap object (so we can set its "under" color
    # without mutating any shared/global colormap instance) and, if
    # requested, mark values below vmin to render white -- display-only,
    # the underlying `vals` are never modified.
    cmap_obj = plt.get_cmap(cmap).copy()
    if mask_below_vmin:
        cmap_obj.set_under("white")

    if log_scale:
        vmin = vmin if vmin is not None else max(vmax * 1e-2, 1e-12)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        color_kwargs = dict(norm=norm)
        cbar_ticks = mticker.LogLocator(base=10).tick_values(vmin, vmax)
        cbar_ticks = cbar_ticks[(cbar_ticks >= vmin) & (cbar_ticks <= vmax)]
    else:
        if vmin is None:
            vmin = float(np.nanmin(vals))
        color_kwargs = dict(vmin=vmin, vmax=vmax)
        cbar_ticks = None

    if plot_type == "pcolormesh":
        mesh = ax.pcolormesh(theta, rad, vals, cmap=cmap_obj, shading="auto", **color_kwargs)
    elif plot_type in ("contour", "contourf"):
        contour_fn = ax.contourf if plot_type == "contourf" else ax.contour
        extend = "min" if plot_type == "contourf" else "neither"
        if log_scale:
            levels = np.logspace(np.log10(vmin), np.log10(vmax), 11)
            mesh = contour_fn(theta, rad, vals, levels=levels, cmap=cmap_obj, norm=norm, extend=extend)
        else:
            step = max(np.round(vmax / 10, 2), 1e-3)
            levels = np.round(np.arange(0, vmax + step, step), 3)
            mesh = contour_fn(theta, rad, vals, levels=levels, cmap=cmap_obj, extend=extend)
    else:
        raise ValueError(
            f"plot_type must be 'pcolormesh', 'contour', or 'contourf', got {plot_type!r}."
        )

    mesh._log_cbar_ticks = cbar_ticks
 
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(315)
    ax.grid(True)
    if dir_letters:
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
 
    zoom_min, zoom_max = r_zoom
    user_min, user_max = r_limits
    final_min = user_min if user_min is not None else zoom_min
    final_max = user_max if user_max is not None else zoom_max
 
    if logradius and final_min is not None and final_min <= 0:
        final_min = float(np.min(rad_native[rad_native > 0])) if np.any(rad_native > 0) else None
 
    if final_min is not None:
        ax.set_rmin(_to_log_radius(final_min) if logradius else final_min)
    if final_max is not None:
        ax.set_rmax(_to_log_radius(final_max) if logradius else final_max)
 
    if logradius:
        plot_rmin = final_min if final_min is not None else float(rad_native.min())
        plot_rmax = final_max if final_max is not None else float(rad_native.max())
        lo_plot, hi_plot = sorted((_to_log_radius(plot_rmin), _to_log_radius(plot_rmax)))
        tick_locs = _radial_ticks(lo_plot, hi_plot, logradius=True)
        ax.set_rticks(tick_locs)
        ax.set_yticklabels([f"{10**t:g}" for t in tick_locs])
 
    return mesh
 

 
def _draw_1d_spectrum(
    ax, da, dim: str, kind: str, radius: str = "frequency",
    color: str = "#0463d7", alpha: float = 0.4,
):
    """kind: 'freq' or 'dir'. Draw a 1D spectrum slice (line + fill).
    No log-scale handling here -- a filled line plot has no color-mapped
    value/colorbar; log_scale only applies to the 2D case above."""
    x = da.coords[dim].values.astype(float)
    if kind == "freq" and radius == "period":
        x = 1.0 / x
    ax.plot(x, da.values, color=color)
    ax.fill_between(x, da.values, color=color, alpha=alpha)
    ax.set_xlabel(("Period [s]" if radius == "period" else "Frequency [Hz]")
                  if kind == "freq" else "Direction [deg]")
    ax.set_ylabel(_colorbar_label(da))
    return None  # no mappable to attach a colorbar to
 
def _is_all_nan(da: xr.DataArray) -> bool:
    """True if every value in this panel's data is NaN (nothing to draw)."""
    return da.dtype.kind == "f" and bool(np.all(np.isnan(da.values)))


def _draw_one_panel(
    ax, da, freq_dim: str | None, dir_dim: str | None,
    radius: str = "frequency", plot_type: str = "pcolormesh",
    cmap=cmocean.cm.thermal, vmin: float | None = None, vmax: float | None = None,
    dir_letters: bool = False, log_scale: bool = True,
    r_zoom: tuple[float | None, float | None] = (None, None),
    logradius: bool = True,
    r_limits: tuple[float | None, float | None] = (None, None),
    mask_below_vmin: bool = False,
):
    if _is_all_nan(da):
        ax.set_axis_off()
        return None

    if freq_dim and dir_dim:
        return _draw_polar_spectrum(ax, da, freq_dim, dir_dim, radius, plot_type, cmap, vmin, vmax, dir_letters, log_scale, r_zoom, logradius, r_limits, mask_below_vmin)
    elif freq_dim:
        return _draw_1d_spectrum(ax, da, freq_dim, "freq", radius)
    elif dir_dim:
        return _draw_1d_spectrum(ax, da, dir_dim, "dir", radius)
    else:
        ax.text(0.5, 0.5, "No spectral dimension in this slice", ha="center", va="center")
        ax.set_axis_off()
        return None

def _group_count_suffix(ds: xr.Dataset, sel: dict) -> str:
    """
    If `ds` has a `_group_counts` variable (as produced by
    SpectralTimeSeries.aggregate()/groupby().aggregate()), look up the
    count for this panel's selection and format it as a second title
    line -- "group entries: N". Returns "" if the variable isn't
    present, or if `sel` doesn't fully resolve to a single count (e.g.
    a leftover dim not covered by sel).
    """
    if "_group_counts" not in ds.data_vars:
        return ""
    try:
        count_da = ds["_group_counts"].sel(sel)
        if count_da.size != 1:
            return ""
        count_val = float(count_da.values)
        count = 0 if np.isnan(count_val) else int(count_val)
        return f"\ngroup entries: {count}"
    except (KeyError, ValueError):
        return ""

def plot_spectra(
    obj,
    var: str | None = None,
    time=None,
    row: str | None = None,
    col: str | None = None,
    col_wrap: int | None = None,
    max_cols: int = 4,
    errors: Literal["ignore", "raise"] = "raise",
    radius: str = "frequency",
    plot_type: str = "pcolormesh",
    cmap: str = cmocean.cm.thermal,
    vmin: float | None = None,
    vmax: float | None = None,
    log_scale: bool = True,
    dir_letters: bool = False,
    panel_size: float = 3.0,
    zoom: bool = True,
    logradius: bool = True,
    rmin: float | None = None,
    rmax: float | None = None,
    mask_below_vmin: bool = False,
    normalize: bool = False,
):
    """
    Flexible faceted spectrum plot. Works identically on:
      - a SpectralTimeSeries -- its own var_map/dim_map is used directly.
      - the raw xr.Dataset output of .aggregate() / .groupby().aggregate().
 
    If row=/col= are both omitted and more than one non-spectral
    dimension is left after time= selection, every leftover dimension is
    auto-faceted into a grid. If exactly one of row=/col= is given and
    dimensions are still left over, those are flattened into extra
    columns (row= given) or extra rows (col= given). Passing both row=
    and col= still requires every dimension to be accounted for
    explicitly.
 
    Color scale is logarithmic by default (log_scale=True). `vmax` (and,
    for log scale, the implied floor `vmin`) are always computed GLOBALLY
    across every panel that will be drawn -- not per panel -- so panels
    stay honestly comparable on one shared color scale; passing an
    explicit `vmin`/`vmax` overrides the computed one but is still
    applied uniformly to every panel. Colorbar ticks are placed at clean
    decade values for every plot_type (see _draw_polar_spectrum), not
    just pcolormesh.
 
    When zoom=True (default) and the panel is a 2D polar spectrum, the
    radial extent is also clipped GLOBALLY across every panel: the
    radial limits are set to the lowest/highest frequency (equivalently
    highest/lowest period, for radius="period") at which ANY panel still
    has data above `vmin`. `rmin`/`rmax` (native units -- Hz or seconds)
    override the auto-computed zoom on whichever side(s) are given,
    exactly like wavespectra's rmin=/rmax=.
 
    logradius=True (default) plots the radial axis on a log10 scale.
    `rmin`/`rmax` are always given in native units either way; the log
    transform is applied internally.
 
    The directional axis is periodically closed for every 2D panel (see
    _draw_polar_spectrum / _close_direction) so there's no undrawn wedge
    between the last and first direction bins.
 
    Parameters
    ----------
    obj : SpectralTimeSeries or xr.Dataset
    var : str, optional
    time : optional
        Selector applied via `.sel()` against the actual time dimension.
        A partial/date-only string (e.g. "2017-01-01") selects every
        timestamp on that date, not one -- standard xarray/pandas
        datetime-string indexing. Pass an exact timestamp (e.g.
        "2017-01-01T00:00") to select a single step.
    row, col : str, optional
    col_wrap : int, optional
    max_cols : int, default 4
    errors : {"ignore", "raise"}, default "raise"
        Governs auto-facet/flatten panel-count warnings/errors (not the
        fully-explicit row=+col= case). "raise": warn above
        _AUTO_FACET_WARN_THRESHOLD (25), raise above
        _AUTO_FACET_RAISE_THRESHOLD (100). "ignore": never warn/raise.
    radius : {"frequency", "period"}, default "frequency"
    plot_type : {"pcolormesh", "contour", "contourf"}, default "pcolormesh"
        "contour" draws unfilled level lines; "contourf" is the filled
        version (wavespectra's default kind).
    cmap : str, default "Blues"
    vmin : float, optional
        Shared colour scale floor across all panels, and (when
        zoom=True) the threshold used to decide the radial zoom extent.
        Computed as 1e-2 * vmax if omitted.
    vmax : float, optional
        Shared colour scale ceiling across all panels. Computed as the
        global max across every panel to be drawn if omitted.
    log_scale : bool, default True
        Color on a log scale (LogNorm). Set False for a linear scale.
    dir_letters : bool, default False
    panel_size : float, default 3.0
    zoom : bool, default True
        Zoom the radial (frequency/period) axis of 2D polar panels in on
        the region where data exceeds `vmin`, computed globally across
        all panels. Has no effect on 1D panels. Set False to always show
        the full frequency/period range. Ignored on whichever side(s)
        rmin/rmax explicitly set.
    logradius : bool, default True
        Plot the radial (frequency/period) axis on a log10 scale rather
        than linear. Has no effect on 1D panels.
    rmin, rmax : float, optional
        Explicit radial-axis limits in NATIVE units (Hz for
        radius="frequency", seconds for radius="period"). Each,
        independently, overrides the auto-computed zoom on that side.
        Has no effect on 1D panels.
 
    Returns
    -------
    matplotlib.axes.Axes  (single panel)
    (matplotlib.figure.Figure, np.ndarray of Axes)  (grid)
    """

    is_sts_like = hasattr(obj, "ds") and hasattr(obj, "var_map") and hasattr(obj, "dim_map")

    if isinstance(obj, xr.DataArray):
        if var is not None and var != (obj.name or var):
            raise ValueError(
                f"var={var!r} was given but obj is already a DataArray named "
                f"{obj.name!r}. Pass the Dataset instead if you want to select "
                f"a different variable."
            )
        da_name = obj.name or "__spectrum__"
        ds = obj.rename(da_name).to_dataset()
        spec_var, freq_dim, dir_dim = _resolve_spectral_target(ds, da_name)
        time_dim = None
        if time is not None:
            raise ValueError(
                "time= only applies to a SpectralTimeSeries. For a plain "
                "xr.DataArray/Dataset (e.g. aggregate() output), select "
                "directly with row=/col=, or call .sel(...)/.isel(...) first."
            )
    elif isinstance(obj, xr.Dataset):
        ds = obj
        spec_var, freq_dim, dir_dim = _resolve_spectral_target(ds, var)
        time_dim = None
        if time is not None:
            raise ValueError(
                "time= only applies to a SpectralTimeSeries. For a plain "
                "xr.Dataset (e.g. aggregate() output), select directly "
                "with row=/col=, or call ds.sel(...)/.isel(...) first."
            )
    elif is_sts_like:
        ds = obj.ds
        spec_var = var or obj.var_map["S"] or obj.var_map["E"] or obj.var_map["D"]
        if spec_var is None:
            raise ValueError("This object has no spectral variable to plot.")
        freq_dim = obj.dim_map[_ROLE_FREQ] if obj.dim_map[_ROLE_FREQ] in ds[spec_var].dims else None
        dir_dim  = obj.dim_map[_ROLE_DIR]  if obj.dim_map[_ROLE_DIR]  in ds[spec_var].dims else None
        time_dim = obj.dim_map[_ROLE_TIME]
    else:
        raise TypeError(
            f"obj must be a SpectralTimeSeries, xr.Dataset, or xr.DataArray, "
            f"got {type(obj).__name__}."
        )

    da = ds[spec_var]


    if time is not None and time_dim is not None:
        da = da.sel({time_dim: time})
        # A bare/partial date string is a label-slice, not a point-select
        # -- matches every timestamp that day. Left multi-valued here, it
        # flows into the same leftover-dim handling (batch_dims / Case
        # A-D below) as any other dimension. An exact single match still
        # needs isel(0) since .sel()'s label-slice path doesn't
        # auto-drop a size-1 dimension the way an exact scalar match does.
        if time_dim in da.dims and da.sizes[time_dim] == 1:
            da = da.isel({time_dim: 0})
 
    spectral_dims = tuple(d for d in (freq_dim, dir_dim) if d is not None)
    batch_dims = _leftover_dims(da, *spectral_dims)
 
    for d in list(batch_dims):          # auto-drop already-size-1 dims
        if da.sizes[d] == 1:
            da = da.isel({d: 0})
    batch_dims = _leftover_dims(da, *spectral_dims)
 
    requested = {d for d in (row, col) if d is not None}
    unaccounted = [d for d in batch_dims if d not in requested]

    if normalize:
            da = da / da.max(dim=spectral_dims, skipna=True)

    # --- GLOBAL vmax/vmin, computed once across every panel that could
    # possibly be drawn (i.e. over the whole of `da` as it stands right
    # now, before any per-panel .sel() narrows it further) -- this is
    # what guarantees every panel shares one honestly-comparable scale.
    vmax = vmax if vmax is not None else float(np.nanmax(da.values))
    if vmin is None:
        vmin = max(vmax * 1e-2, 1e-12) if log_scale else vmax * 1e-2
 
    # --- GLOBAL radial zoom limit, computed once across every panel,
    # same rationale as vmin/vmax above. Explicit rmin=/rmax= (native
    # units, like wavespectra) are passed through separately as
    # r_limits and override the computed zoom per-side inside
    # _draw_polar_spectrum.
    r_zoom = _global_radial_zoom(da, freq_dim, dir_dim, vmin, radius) if zoom else (None, None)
    r_limits = (rmin, rmax)
 
    fig_kw = dict(subplot_kw=dict(projection="polar")) if dir_dim else {}
    fig_kw["constrained_layout"] = True
    cbar_label = _colorbar_label(da) if dir_dim and freq_dim else None 

    def _draw_grid(nrows, ncols, cell_fn, n_real_panels):
        """cell_fn(panel_index) -> (panel_da, title)."""
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows), **fig_kw
        )
        axes_flat = np.atleast_1d(axes).ravel()
        last_mesh = None
        for i, ax in enumerate(axes_flat):
            if i >= n_real_panels:
                ax.set_visible(False)
                continue
            panel_da, title = cell_fn(i)
            mesh = _draw_one_panel(ax, panel_da, freq_dim, dir_dim, radius, plot_type, cmap, vmin, vmax, dir_letters, log_scale, r_zoom, logradius, r_limits, mask_below_vmin)
            last_mesh = mesh if mesh is not None else last_mesh
            ax.set_title(title, fontsize=9)
        if last_mesh is not None:
            fig.colorbar(
                last_mesh, ax=axes_flat.tolist(), pad=0.02, shrink=0.6, label=cbar_label,
                ticks=getattr(last_mesh, "_log_cbar_ticks", None),
            )
        return fig, axes
 
    # --- Case A: single panel, nothing to facet at all --------------------- #
    if not requested and not unaccounted:
        fig, ax = plt.subplots(figsize=(panel_size + 2, panel_size + 2), **fig_kw)
        mesh = _draw_one_panel(ax, da, freq_dim, dir_dim, radius, plot_type, cmap, vmin, vmax, dir_letters, log_scale, r_zoom, logradius, r_limits, mask_below_vmin)
        if mesh is not None:
            fig.colorbar(
                mesh, ax=ax, pad=0.1, shrink=0.8, label=cbar_label,
                ticks=getattr(mesh, "_log_cbar_ticks", None),
            )
        return ax
 
    # --- Case B: exactly one of row=/col= given, dims still left over ----- #
    if (row is not None) != (col is not None) and unaccounted:
        primary, primary_is_row = (row, True) if row is not None else (col, False)
        primary_labels = list(da.coords[primary].values)
 
        flat_combos = list(itertools.product(*(da.coords[d].values for d in unaccounted)))
        n_flat = len(flat_combos)
        n_panels = len(primary_labels) * n_flat
        _check_panel_count(n_panels, unaccounted, errors)
 
        wrap = col_wrap or max_cols
        n_secondary = min(n_flat, wrap)
        bands_per_primary = int(np.ceil(n_flat / n_secondary))
 
        if primary_is_row:
            nrows = len(primary_labels) * bands_per_primary
            ncols = n_secondary
        else:
            nrows = n_secondary
            ncols = len(primary_labels) * bands_per_primary
 
        def cell_fn(i):
            primary_idx, rem = divmod(i, n_flat)
            combo = flat_combos[rem]
            sel = {primary: primary_labels[primary_idx], **dict(zip(unaccounted, combo))}
            panel_da = da.sel(sel)
            title = ", ".join(f"{k}={_format_panel_value(v)}" for k, v in sel.items())
            title += _group_count_suffix(ds, sel)
            return panel_da, title
 
        return _draw_grid(nrows, ncols, cell_fn, n_panels)
 
    # --- Case C: neither row= nor col= given, dims left over --------------- #
    if not requested and unaccounted:
        combos = list(itertools.product(*(da.coords[d].values for d in unaccounted)))
        n_panels = len(combos)
        _check_panel_count(n_panels, unaccounted, errors)
 
        nrows, ncols = _grid_shape(n_panels, max_cols=max_cols)
 
        def cell_fn(i):
            sel = dict(zip(unaccounted, combos[i]))
            title = ", ".join(f"{k}={_format_panel_value(v)}" for k, v in sel.items())
            title += _group_count_suffix(ds, sel)
            return da.sel(sel), title
 
        return _draw_grid(nrows, ncols, cell_fn, n_panels)
 
    # --- Case D: both row= and col= given -- everything explicit ----------- #
    if unaccounted:
        raise ValueError(
            f"Dimension(s) {unaccounted} have more than one value and "
            "aren't handled by time=/row=/col=. Select down to a single "
            "value, or pass them to row=/col= to facet over them. "
            f"Dims present: {list(da.dims)}."
        )
 
    row_labels = list(da.coords[row].values) if row else [None]
    col_labels = list(da.coords[col].values) if col else [None]
 
    if row and not col and col_wrap:
        combos = [(v,) for v in row_labels]
        ncols = min(col_wrap, len(combos))
        nrows = int(np.ceil(len(combos) / ncols))
        row_key, col_key = row, None
    else:
        nrows, ncols = len(row_labels), len(col_labels)
        combos = [(r, c) for r in row_labels for c in col_labels]
        row_key, col_key = row, col
 
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows), **fig_kw)
    axes_flat = np.atleast_1d(axes).ravel()
 
    last_mesh = None
    for ax, combo in zip(axes_flat, combos):
        sel = {}
        if row_key:
            sel[row_key] = combo[0]
        if col_key and len(combo) > 1:
            sel[col_key] = combo[1]
        panel_da = da.sel(sel)
        mesh = _draw_one_panel(ax, panel_da, freq_dim, dir_dim, radius, plot_type, cmap, vmin, vmax, dir_letters, log_scale, r_zoom, logradius, r_limits, mask_below_vmin)
        last_mesh = mesh if mesh is not None else last_mesh
        title = ", ".join(f"{k}={_format_panel_value(v)}" for k, v in sel.items())
        title += _group_count_suffix(ds, sel)
        ax.set_title(title, fontsize=9)
 
    for ax in axes_flat[len(combos):]:
        ax.set_visible(False)
 
    if last_mesh is not None:
        fig.colorbar(
            last_mesh, ax=axes_flat.tolist(), pad=0.02, shrink=0.6, label=cbar_label,
            ticks=getattr(last_mesh, "_log_cbar_ticks", None),
        )
 
    return fig, axes
 

def plot_rose(
    magnitude: "np.ndarray",
    direction: "np.ndarray",
    ax: "plt.Axes",
    *,
    n_dir_bins:     int   = 16,
    mag_bins:       "np.ndarray | None" = None,
    mag_step:       "float | None"      = None,
    cmap:           str   = "YlOrRd",
    r_max:          "float | None" = None,
    calm_threshold: float = 0.0,
    title:          str   = "",
    legend:         bool  = True,
    mag_label:      str   = "magnitude",
    mag_unit:       str   = "",
) -> None:
    """
    Draw a single wind / wave rose onto a polar Axes.

    This is a standalone utility intended for ``utils.py``. It has no
    dependency on ``TimeSeriesValidation`` — pass plain numpy arrays and a
    pre-created polar Axes. The companion ``TimeSeriesValidation.plot_rose``
    method calls this twice with shared ``mag_bins`` and ``r_max`` so both
    panels are directly comparable.

    Parameters
    ----------
    magnitude : np.ndarray
        1-D array of magnitudes (wind speed, wave height, current speed …).
    direction : np.ndarray
        1-D array of directions in degrees, meteorological convention
        (direction *from*, 0° = North, clockwise positive). Must be the same
        length as ``magnitude``.
    ax : plt.Axes
        A polar Axes (created with ``subplot_kw={"projection": "polar"}``).
        The function draws onto this axes and returns None so the caller
        controls figure layout.
    n_dir_bins : int, default 16
        Number of directional sectors. 16 → 22.5° sectors, 8 → 45° sectors.
    mag_bins : np.ndarray, optional
        Explicit bin edges, e.g. ``np.arange(0, 15, 2)``. Overrides both
        auto-binning and ``mag_step``. Pass the same array to both
        ``plot_rose`` calls to ensure identical colouring between panels.
    mag_step : float, optional
        Force a specific bin step size, overriding auto-selection while still
        starting at 0 and extending to the 99th percentile. Ignored if
        ``mag_bins`` is supplied.
    cmap : str, default "YlOrRd"
        Matplotlib colormap used to colour magnitude classes (lowest → highest).
    r_max : float, optional
        Maximum radial extent as a fraction of total observations (0–1 scale).
        If None, derived from the data. Pass the same value to both calls for
        a shared radial scale.
    calm_threshold : float, default 0.0
        Observations with magnitude <= this value are counted as "calm" and
        excluded from the directional bins (but noted in the title).
    title : str, default ""
        Axes title.
    legend : bool, default True
        Whether to draw a magnitude legend below the axes.
    mag_label : str, default "magnitude"
        Human-readable name for the magnitude variable, used in the legend.
    mag_unit : str, default ""
        Unit string appended to legend labels, e.g. "m/s" or "m".

    Notes
    -----
    - Directions are converted from meteorological (from-direction, CW from N)
      to the polar Axes convention (CCW from East) internally.
    - The radial axis shows percentage of total observations (including calm).
    - ``r_max`` is expressed as a fraction (0–1); the axes tick labels are
      converted to percentages automatically.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as mcm
    import matplotlib.patches as mpatches

    # ------------------------------------------------------------------ #
    # Input coercion and validation                                        #
    # ------------------------------------------------------------------ #
    magnitude = np.asarray(magnitude, dtype=float).ravel()
    direction = np.asarray(direction, dtype=float).ravel()

    if magnitude.shape != direction.shape:
        raise ValueError(
            f"magnitude and direction must have the same length, "
            f"got {magnitude.shape} and {direction.shape}."
        )

    valid     = np.isfinite(magnitude) & np.isfinite(direction)
    magnitude = magnitude[valid]
    direction = direction[valid]
    n_total   = len(magnitude)

    if n_total == 0:
        raise ValueError("No finite magnitude/direction pairs to plot.")

    # ------------------------------------------------------------------ #
    # Calm separation                                                      #
    # ------------------------------------------------------------------ #
    active     = magnitude > calm_threshold
    mag_active = magnitude[active]
    dir_active = direction[active]
    n_calm     = int((~active).sum())

    # ------------------------------------------------------------------ #
    # Magnitude bins                                                       #
    # ------------------------------------------------------------------ #
    if mag_bins is None:
        p99 = float(np.percentile(mag_active, 99)) if len(mag_active) else 1.0
        if mag_step is None:
            import pandas as pd
            mag_step = infer_step(pd.Series(mag_active), target=7)
        mag_bins = np.arange(0, p99 + mag_step, mag_step)

    mag_bins = np.asarray(mag_bins, dtype=float)
    n_mag    = len(mag_bins) - 1
    cm_obj   = mcm.get_cmap(cmap, n_mag)
    colors   = [cm_obj(i / n_mag) for i in range(n_mag)]

    # ------------------------------------------------------------------ #
    # Directional bins                                                     #
    # ------------------------------------------------------------------ #
    sector_width    = 360.0 / n_dir_bins
    dir_edges       = np.linspace(
        -sector_width / 2, 360 - sector_width / 2, n_dir_bins + 1
    )
    dir_centres_met = np.linspace(0, 360 - sector_width, n_dir_bins)

    # Pass meteorological directions directly in radians.
    # set_theta_zero_location("N") and set_theta_direction(-1) already
    # configure the axes to use 0° = North, clockwise — so no manual
    # conversion is needed. Applying 90 - dir would double-transform.
    theta = np.deg2rad(dir_centres_met)

    # ------------------------------------------------------------------ #
    # Bin counts: (n_dir_bins × n_mag) matrix                             #
    # ------------------------------------------------------------------ #
    dir_norm = dir_active % 360.0
    counts   = np.zeros((n_dir_bins, n_mag), dtype=float)

    for m_idx in range(n_mag):
        m_lo   = mag_bins[m_idx]
        m_hi   = mag_bins[m_idx + 1]
        in_mag = (mag_active >= m_lo) & (mag_active < m_hi)

        for d_idx in range(n_dir_bins):
            d_lo = dir_edges[d_idx] % 360
            d_hi = dir_edges[d_idx + 1] % 360
            if d_lo < d_hi:
                in_dir = (dir_norm >= d_lo) & (dir_norm < d_hi)
            else:
                in_dir = (dir_norm >= d_lo) | (dir_norm < d_hi)
            counts[d_idx, m_idx] = np.sum(in_mag & in_dir)

    # Express as fraction of ALL observations (including calm)
    fractions = counts / n_total

    # ------------------------------------------------------------------ #
    # Radial scale                                                         #
    # ------------------------------------------------------------------ #
    sector_totals = fractions.sum(axis=1)

    if r_max is None:
        r_max = float(sector_totals.max()) * 1.15
    r_max = max(r_max, 1e-9)

    # ------------------------------------------------------------------ #
    # Draw bars                                                            #
    # ------------------------------------------------------------------ #
    bar_width = np.deg2rad(sector_width) * 0.9

    for m_idx in range(n_mag):
        bottoms = fractions[:, :m_idx].sum(axis=1)
        heights = fractions[:, m_idx]
        ax.bar(
            theta,
            heights,
            width=bar_width,
            bottom=bottoms,
            color=colors[m_idx],
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )

    # ------------------------------------------------------------------ #
    # Axes cosmetics                                                       #
    # ------------------------------------------------------------------ #
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.set_ylim(0, r_max)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, _: f"{val * 100:.0f}%")
    )
    # Nice radial ticks in percentage space — same snap logic as infer_step
    import pandas as pd
    r_max_pct  = r_max * 100
    tick_step  = infer_step(pd.Series([0.0, r_max_pct]), target=4)
    tick_vals  = np.arange(tick_step, r_max_pct + tick_step * 0.5, tick_step)
    ax.set_yticks(tick_vals / 100)   # back to fraction for the axes
    ax.tick_params(axis="y", labelsize=7, pad=2)

    compass = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(compass, fontsize=8)

    # Grid: draw on top of the background but below the bars
    ax.set_facecolor("#f0f0f0")
    ax.grid(True, color="#b0b8c8", linewidth=0.7, linestyle="-", zorder=2)
    ax.spines["polar"].set_visible(False)

    calm_pct  = 100 * n_calm / n_total if n_total > 0 else 0.0
    unit_str  = f" {mag_unit}" if mag_unit else ""
    calm_note = f"calm (≤{calm_threshold:g}{unit_str}): {calm_pct:.1f}%"
    full_title = f"{title}\n{calm_note}" if title else calm_note
    ax.set_title(full_title, pad=12, fontsize=9)

    # ------------------------------------------------------------------ #
    # Legend                                                               #
    # ------------------------------------------------------------------ #
    if legend:
        unit_str = f" {mag_unit}" if mag_unit else ""
        patches  = [
            mpatches.Patch(
                facecolor=colors[m_idx],
                label=f"{mag_bins[m_idx]:g}–{mag_bins[m_idx + 1]:g}{unit_str}",
            )
            for m_idx in range(n_mag)
        ]
        ax.legend(
            handles=patches,
            title=mag_label,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=min(n_mag, 5),
            fontsize=7,
            title_fontsize=7,
            frameon=False,
            handlelength=1.2,
        )

def plot_scatter_diagram(
        data: pd.DataFrame, 
        var1 = "TP",
        step_var1 = 1,
        var2 = "HS",
        step_var2 = 1,
        density_joint = False,
        density_marginal = True,
        format_joint = ".0f",
        format_marginal = ".0f",
        format_xticks = ".1f",
        format_yticks = ".1f",
        from_origin = True,
        xlim = None,
        ylim = None,
        annot_cells = "nonzero",
        annot_margin = True,
        percent_sign_cells = False,
        percent_sign_margin = True,
        norm = mcolors.LogNorm(),
        cbar = False,
        cmap = "Blues",
        **kwargs
        ):
    """
    Plot scatter diagram (heatmap) of two variables (e.g, var1='hs', var2='tp')
    
    Parameters
    -----------
    data : pd.DataFrame
        The data containing the variables as columns.
    var1 : str
        The first (y-axis) variable as a column name.
    step_var1 : float
        Interval of bins of the first variable.
    var2 : str
        The second (x-axis) variable as a column name.
    step_var2 : float
        Interval of bins of the second variable.
    density_joint : bool, default False
        Display the joint distribution as percentage values.
    density_marginal : bool, default True
        Display the marginal distribution as percentage values. 
    format_joint : str
        Formatting string of the joint (cell) values
    format_marginal : str
        Formatting string of the marginal values
    format_xticks : str
        Formatting string of the x-tick values
    format_yticks : str
        Formatting string of the y-tick values
    from_origin : bool
        Control whether the histogram should start at the origin
        (default) or at the first observed data points.
    xlim : tuple(float,float)
        Manually specify start and end of the x-axis.
        This should be a multiple of step_var2.
    ylim : tuple(float,float)
        Manually specify start and end of the y-axis.
        This should be a multiple of step_var1.
    annot_cells : str
        Controls which cells will be annotated with a value:
         - "all": all cells are annotated
         - "nonzero": all cells with any occurance are annotated
         - "rounded": only cells which still have a nonzero value 
         after applying the cell number formatting are annotated
         - "off": no cells are annotated
    annot_margin : bool, default True
        Add the marginal distributions of values.
    percent_sign_cells : bool, default False
        Add a percent sign after the cell values, if using density.
    percent_sign_marginal: bool, default True
        Add a percent sign after the marginal values, if using density.
    norm : matplotlib color norm, default LogNorm()
        A colormap norm.
    cbar : bool, default False
        Include a colorbar.
    **kwargs
        Any keyword arguments for seaborn heatmap.
        For example: cbar_kws = {"anchor":(x, y)} 
        will adjust position of the colorbar.

    Returns
    ----------
    matplotlib axis

    Notes
    -------
    The function is written by efvik.
    """

    valid_annot = ["all","nonzero","rounded","off"]
    if annot_cells not in valid_annot:
        raise ValueError(f"Keyword annot_cells must be one of {valid_annot}.")

    data = data[[var1,var2]]
    if np.any(np.isnan(data)):
        print("Warning: Removing NaN rows.")
        data = data.dropna(how="any")

    # Initial min and max
    xmin = data[var2].values.min()
    ymin = data[var1].values.min()
    xmax = data[var2].values.max()
    ymax = data[var1].values.max()

    # Change min (max) to zero only if all values are above (below) and from_origin=True
    xmin = 0 if (from_origin and (xmin>0)) else np.floor(xmin/step_var2)*step_var2
    ymin = 0 if (from_origin and (ymin>0)) else np.floor(ymin/step_var1)*step_var1

    xmax = 0 if (from_origin and (xmax<0)) else np.ceil(xmax/step_var2)*step_var2
    ymax = 0 if (from_origin and (ymax<0)) else np.ceil(ymax/step_var1)*step_var1

    # ylim and xlim can be manually specified
    if xlim is not None:
        xmin = xlim[0] if xlim[0] is not None else xmin
        xmax = xlim[1] if xlim[1] is not None else xmax
        if (l:=data[(data[var2]<xmin)|(data[var2]>xmax)].size):
            print(f"WARNING: {l} points excluded by chosen xlim {xlim}")
        diff = xmax-xmin
    if ylim is not None:
        ymin = ylim[0] if ylim[0] is not None else ymin
        ymax = ylim[1] if ylim[1] is not None else ymax
        if (l:=data[(data[var1]<ymin)|(data[var1]>ymax)].size):
            print(f"WARNING: {l} points excluded by chosen ylim {ylim}")

    # Define bins and get histogram
    n_bins_x = int(np.round((xmax - xmin) / step_var2)) + 1
    n_bins_y = int(np.round((ymax - ymin) / step_var1)) + 1
    x = np.linspace(xmin,xmax,n_bins_x)
    y = np.linspace(ymin,ymax,n_bins_y)

    hist, y, x = np.histogram2d(data.values[:,0],data.values[:,1],bins=(y,x))

    xlabels = [f"{i:{format_xticks}}" for i in x]
    ylabels = [f"{i:{format_yticks}}" for i in y]

    sum_x = hist.sum(axis=0)
    sum_y = hist.sum(axis=1)

    suffix_cell = ""
    if density_joint:
        hist = 100*hist/hist.sum()
        if percent_sign_cells:
            suffix_cell = "%"

    suffix_margin = ""
    if density_marginal:
        sum_x = 100*sum_x/sum_x.sum()
        sum_y = 100*sum_y/sum_y.sum()
        if percent_sign_margin:
            suffix_margin = "%"

    if annot_cells == "rounded":
        text = [[f"{h:{format_joint}}"+suffix_cell for h in row] for row in hist]
        zero = f"{0:{format_joint}}"
        text = [[t if t!=zero else "" for t in row] for row in text]
    if annot_cells == "nonzero":
        text = [[f"{h:{format_joint}}"+suffix_cell if h>0 else "" for h in row] for row in hist]
    if annot_cells == "all":
        text = [[f"{h:{format_joint}}"+suffix_cell for h in row] for row in hist]
    if annot_cells == "off":
        text =  [["" for h in row] for row in hist]

    if hasattr(norm,"vmin") and (norm.vmin == None):
        norm.vmin = hist[hist>0].min()/10

    ax = sns.heatmap(data=np.where(hist,hist, 1e-16),annot=text,fmt="",
                    xticklabels=False,yticklabels=False,
                    cbar=cbar,norm=norm,cmap=cmap,**kwargs)

    sum_x = [[f"{i:{format_marginal}}"+suffix_margin for i in sum_x]]
    sum_y = [[f"{i:{format_marginal}}"+suffix_margin] for i in sum_y[::-1]]

    if annot_margin:
        ax.table(sum_x,loc="top",cellLoc="center")
        ax.table(sum_y,loc="right",cellLoc="center",bbox=(1,0,(1/hist.shape[1]),1))

    xticks = np.arange(0,hist.shape[1]+1)
    yticks = np.arange(0,hist.shape[0]+1)

    _=ax.set_xticks(xticks,xlabels)
    _=ax.set_yticks(yticks,ylabels)

    ax.set_xlabel(var2)
    ax.set_ylabel(var1)
    ax.invert_yaxis()
    return ax
