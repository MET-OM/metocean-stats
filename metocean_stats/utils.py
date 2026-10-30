"""
Shared helpers for grouping, binning, figure layout and vector conversion.
"""

import re

import numpy as np
import pandas as pd

DEFAULT_SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _decimals_for_step(step: float) -> int:
    """
    Smallest number of decimal places that renders `step` exactly (up to
    float noise) -- so every edge derived from it (lo, lo+step, lo+2*step,
    ...) can share one fixed precision, rather than each edge
    independently stripping its own trailing zeros (e.g. step=0.5 giving
    "0.5-1-1.5-2" instead of "0.5-1.0-1.5-2.0").
    """
    step = abs(float(step))
    if step == 0:
        return 0
    for d in range(8):
        scaled = step * 10**d
        if abs(scaled - round(scaled)) < 1e-6 * max(1.0, scaled):
            return d
    return 8


def bin_range_labels(edges, unit: str = "", wrap: float = None, fmt: str = "{:.10g}") -> list[str]:
    """
    Format consecutive bin edges as "lo-hi" range labels.

    Parameters
    ----------
    edges : array-like
        Bin edges. For n bins, n + 1 values.
    unit : str, optional
        Suffix appended to each label, e.g. " m". Include any separating space.
    wrap : float, optional
        If given, edges are reduced modulo `wrap` before formatting. Use 360
        for direction.
    fmt : str, default "{:.10g}"
        Format applied to each edge value.

    Returns
    -------
    list of str
        One label per bin.
    """
    edges = np.asarray(edges, dtype=float)
    los, his = edges[:-1], edges[1:]
    if wrap is not None:
        los = los % wrap
        his = his % wrap
    return [f"{fmt.format(lo)}-{fmt.format(hi)}{unit}" for lo, hi in zip(los, his)]


def _empty_grouped_result() -> dict[str, list]:
    return {k: [] for k in (
        "data", "label", "left_label", "center_label", "right_label",
        "left_tick", "center_tick", "right_tick",
    )}


def _append_group(result, data, label, left_label, center_label, right_label,
                   left_tick, center_tick, right_tick) -> None:
    result["data"].append(data)
    result["label"].append(label)
    result["left_label"].append(left_label)
    result["center_label"].append(center_label)
    result["right_label"].append(right_label)
    result["left_tick"].append(left_tick)
    result["center_tick"].append(center_tick)
    result["right_tick"].append(right_tick)


def groupby_with_metadata(
        data: pd.DataFrame,
        by: str,
        var: str = None,
        var_dir: str = None,
        sectors: int | list[float] = 12,
        seasons: dict[str, list[int]] = None,
        step: float = None,
        observed: bool = False,
        ) -> dict[str, list]:
    """
    Group a DataFrame by a calendar period, direction sector, or value bin.

    Each group carries the geometry of its bin, computed from the boundaries
    used for grouping. Labels are bare (e.g. "0-2" or "345"). Units are the
    caller's responsibility.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data. Calendar groupings use its DatetimeIndex.
    by : {"month", "season", "direction", "year", "week", "day", "hour"} or str
        Grouping. Any other string is a column of `data`, binned by value.
    var : str, optional
        Column returned for each group. If None, each group is a DataFrame.
    var_dir : str, optional
        Direction column in degrees. Required when ``by="direction"``.
    sectors : int or list of float, default 12
        Number of equal direction sectors centred on north, or explicit sector
        edges.
    seasons : dict of str to list of int, optional
        Season names mapped to month numbers. Default DJF, MAM, JJA, SON.
    step : float, optional
        Bin width when ``by`` is a column. Inferred if None.
    observed : bool, default False
        If False, every category of the full domain is returned, empty ones
        included: 12 months, 4 seasons, weeks 1-53, days 1-366, hours 0-23,
        every sector, and every bin between the data minimum and maximum. If
        True, empty categories are dropped. For ``by="year"``, False gives
        every year from the data minimum to maximum, and True gives only the
        years present.

    Returns
    -------
    dict of list
        One list per key, all of equal length and in the same order:

        ``"data"``
            Group values: a Series if `var` is given, otherwise a DataFrame.
        ``"label"``, ``"left_label"``, ``"center_label"``, ``"right_label"``
            Display text for each bin.
        ``"left_tick"``, ``"center_tick"``, ``"right_tick"``
            Numeric bin edges and centre, or None for a categorical bin such as
            a season. Direction ticks are unwrapped, so they stay monotonic:
            the north sector runs from -15 to 15.

    Raises
    ------
    ValueError
        If ``by="direction"`` is given without `var_dir`, or if `by` is
        neither a grouping nor a column of `data`.
    """
    if by == "month":
        result = _empty_grouped_result()
        months = sorted(data.index.month.unique()) if observed else list(range(1, 13))
        for m in months:
            mask = data.index.month == m
            g = data.loc[mask, var] if var else data.loc[mask]
            label = MONTH_ABBR[m - 1]
            next_label = MONTH_ABBR[m] if m < 12 else ""
            _append_group(result, g, label, label, label, next_label,
                          float(m), m + 0.5, float(m + 1))
        return result

    if by == "season":
        result = _empty_grouped_result()
        for label, months in (seasons or DEFAULT_SEASONS).items():
            mask = data.index.month.isin(months)
            g = data.loc[mask, var] if var else data.loc[mask]
            if observed and len(g) == 0:
                continue
            _append_group(result, g, label, label, label, label, None, None, None)
        return result

    if by == "year":
        result = _empty_grouped_result()
        if observed:
            years = sorted(data.index.year.unique())
        else:
            years = range(data.index.year.min(), data.index.year.max() + 1)
        for y in years:
            mask = data.index.year == y
            g = data.loc[mask, var] if var else data.loc[mask]
            label = str(y)
            _append_group(result, g, label, label, label, str(y + 1),
                          float(y), y + 0.5, float(y + 1))
        return result

    if by == "week":
        result = _empty_grouped_result()
        weeks = sorted(data.index.isocalendar().week.unique()) if observed else range(1, 54)
        for w in weeks:
            mask = data.index.isocalendar().week == w
            g = data.loc[mask, var] if var else data.loc[mask]
            label = str(int(w))
            _append_group(result, g, label, label, label, str(int(w) + 1),
                          float(w), w + 0.5, float(w) + 1.0)
        return result

    if by in ("day", "dayofyear"):
        result = _empty_grouped_result()
        days = sorted(data.index.day_of_year.unique()) if observed else range(1, 367)
        for d in days:
            mask = data.index.day_of_year == d
            g = data.loc[mask, var] if var else data.loc[mask]
            label = str(int(d))
            _append_group(result, g, label, label, label, str(int(d) + 1),
                          float(d), d + 0.5, float(d) + 1.0)
        return result

    if by == "hour":
        result = _empty_grouped_result()
        for h in range(24):
            mask = data.index.hour == h
            if observed and not mask.any():
                continue
            g = data.loc[mask, var] if var else data.loc[mask]
            label = str(h)
            _append_group(result, g, label, label, label, str(h + 1),
                          float(h), h + 0.5, float(h + 1))
        return result

    if by == "direction":
        if var_dir is None:
            raise ValueError("by='direction' requires a direction column (var_dir).")
        if not hasattr(sectors, "__len__"):
            sectors = np.linspace(0, 360, sectors + 1, dtype=int)
        sectors = np.asarray(sectors, dtype=float)
        dir_offset = (sectors[1] - sectors[0]) / 2

        # Unwrapped edges (e.g. the north sector is -15..15, not the
        # wrapped 345..15) so ticks stay monotonic across every sector;
        # labels use the wrapped, human-readable 0-360 form.
        lo_unwrapped = sectors[:-1] - dir_offset
        hi_unwrapped = sectors[1:] - dir_offset
        lo_wrapped = lo_unwrapped % 360
        hi_wrapped = hi_unwrapped % 360
        labels = [f"{lo:.0f}-{hi:.0f}" for lo, hi in zip(lo_wrapped, hi_wrapped)]

        bins_col = pd.cut((data[var_dir] + dir_offset) % 360, bins=sectors,
                           labels=labels, right=False)

        result = _empty_grouped_result()
        for label, g in data.groupby(bins_col, observed=observed):
            i = labels.index(str(label))
            _append_group(
                result, g[var] if var else g, label,
                f"{lo_wrapped[i]:.0f}", label, f"{hi_wrapped[i]:.0f}",
                float(lo_unwrapped[i]), float((lo_unwrapped[i] + hi_unwrapped[i]) / 2),
                float(hi_unwrapped[i]),
            )
        return result

    if by not in data.columns:
        raise ValueError(
            "'by' must be one of 'month', 'season', 'direction', 'year', "
            f"'week', 'day', 'hour', or a column name in `data`. Got '{by}'."
        )

    s_by  = data[by].dropna()
    _step = step or infer_step(s_by)
    edges = bin_edges(s_by, _step)
    fmt = "{:." + str(_decimals_for_step(_step)) + "f}"
    labels = bin_range_labels(edges, fmt=fmt)
    bins_col = pd.cut(data[by], bins=edges, labels=labels, right=False)

    result = _empty_grouped_result()
    for label, g in data.groupby(bins_col, observed=observed):
        i = labels.index(str(label))
        lo, hi = edges[i], edges[i + 1]
        _append_group(
            result, g[var] if var else g, label,
            fmt.format(lo), label, fmt.format(hi),
            float(lo), float((lo + hi) / 2), float(hi),
        )
    return result


def get_n_axes(n_intervals: int, max_cols: int = 4, sharex=None, sharey=None):
    """
    Create a figure with a grid of at least `n_intervals` axes.

    The grid is as square as possible, with at most `max_cols` columns.

    Parameters
    ----------
    n_intervals : int
        Number of panels. Must be positive.
    max_cols : int, default 4
        Maximum number of columns.
    sharex, sharey : bool or str, optional
        Passed to `matplotlib.pyplot.subplots`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray
        Flat array of length ``nrows * ncols``, at least `n_intervals`.

    Raises
    ------
    ValueError
        If `n_intervals` is not positive.
    """
    import matplotlib.pyplot as plt

    if n_intervals <= 0:
        raise ValueError(f"n_intervals should be a positive integer, but got {n_intervals}.")
    elif n_intervals < max_cols**2:
        table = np.array(
            [
                (rows, cols)
                for cols in range(1, max_cols + 1)
                for rows in range(1, cols + 1)
            ]
        )
        table = table[np.prod(table, axis=1) >= n_intervals]
        (nrows, ncols) = table[np.argmin(np.prod(table, axis=1) - n_intervals)]
    else:
        nrows = np.ceil(n_intervals / max_cols).astype(int)
        ncols = max_cols

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, squeeze=False
    )
    return fig, axes.ravel()


def resolve_axes_grid(n_panels: int, ax=None, ncols: int = 3, nfigs: int = 1):
    """
    Return axes for `n_panels` panels, either the supplied ones or new figures.

    Parameters
    ----------
    n_panels : int
        Number of panels to draw.
    ax : matplotlib.axes.Axes or array-like, optional
        Existing axes to use. If given, no figure is created.
    ncols : int, default 3
        Maximum number of columns per figure.
    nfigs : int, default 1
        Number of figures to split the panels across. Ignored if `ax` is given.

    Returns
    -------
    figures : numpy.ndarray
        New figures. Empty if `ax` was given.
    axes : numpy.ndarray
        Flat array of at least `n_panels` axes.

    Raises
    ------
    ValueError
        If `ax` provides fewer than `n_panels` axes.
    """
    if ax is not None:
        axes = np.asarray(ax).ravel()
        figures = np.array([])
    elif nfigs == 1:
        fig, axes = get_n_axes(n_panels, max_cols=ncols)
        figures = np.array([fig])
    else:
        base, rem = divmod(n_panels, nfigs)
        sizes = [base + 1] * rem + [base] * (nfigs - rem)
        subplots = [get_n_axes(n, max_cols=ncols) for n in sizes]
        figures = np.array([s[0] for s in subplots])
        axes = np.concatenate([s[1] for s in subplots])

    if len(axes) < n_panels:
        raise ValueError(f"Expected ax to contain {n_panels} axes, got {len(axes)}.")
    return figures, axes


def infer_step(s: pd.Series, target: int = 10) -> float:
    """Infer a round step size from the data range, snapped to a nice value.

    Parameters
    ----------
    s : pandas.Series
        Data to infer the step from.
    target : int, default 10
        Approximate number of bins across the data range.

    Returns
    -------
    float
        Step size from the set 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, scaled by a power of ten.
    """
    NICE_STEPS = [0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10]
    raw        = (s.max() - s.min()) / target
    magnitude  = 10 ** np.floor(np.log10(raw))
    scaled     = raw / magnitude
    nice       = min(NICE_STEPS, key=lambda x: abs(x - scaled))
    return nice * magnitude

def bin_edges(s: pd.Series, step: float) -> np.ndarray:
    """
    Compute bin edges for a series at a given step.

    Edges start at zero when all values are non-negative, and otherwise at
    the nearest step below the minimum. The upper edge is the nearest step
    at or above the maximum.

    Parameters
    ----------
    s : pandas.Series
        Data to bin.
    step : float
        Bin width.

    Returns
    -------
    numpy.ndarray
        Evenly spaced edges.
    """
    lo = 0.0 if s.min() >= 0 else np.floor(s.min() / step) * step
    hi = np.ceil(s.max() / step) * step
    n  = int(round((hi - lo) / step)) + 1
    return np.linspace(lo, hi, n)


_STAT_ALIASES = {
    "count": "count",
    "min": "min",
    "minimum": "min",
    "max": "max",
    "maximum": "max",
    "mean": "mean",
    "std": "std",
    "median": "median",
}

# Sort position of each non-percentile kind, chosen so a mixed list of stats
# comes out "count, %/density, min, ...percentiles/median/mean/std by value...,
# max" regardless of the order they were given in.
_KIND_SORT_GROUP = {"count": -2, "percent_of_total": -1, "min": 0, "max": 1}
_KIND_SORT_VALUE = {"min": 0.0, "median": 50.0, "mean": 50.5, "std": 50.6}

_PERCENT_RE = re.compile(r"^(\d+(?:\.\d+)?)%$")
_P_RE = re.compile(r"^p(\d+(?:\.\d+)?)$")


def _parse_stat(name: str) -> tuple[str, float | None]:
    """
    Parse one statistic token into (kind, percentile).

    `kind` is one of "count", "min", "max", "mean", "std", "median",
    "percentile", or "percent_of_total" (``"%"`` or ``"density"``).
    `percentile` is the matched percentile (0-100) for kind "percentile",
    otherwise None. Matching is case-insensitive.
    """
    key = name.strip().lower()

    if key in ("%", "density"):
        return "percent_of_total", None
    if key in _STAT_ALIASES:
        return _STAT_ALIASES[key], None

    m = _PERCENT_RE.match(key) or _P_RE.match(key)
    if m:
        value = float(m.group(1))
        if not (0.0 <= value <= 100.0):
            raise ValueError(
                f"Percentile in '{name}' must be between 0 and 100, got {value}."
            )
        return "percentile", value

    raise ValueError(
        f"Unrecognised statistic '{name}'. Expected one of 'count', 'min' "
        "('minimum'), 'max' ('maximum'), 'mean', 'std', 'median', '%' "
        "('density'), or a percentile such as 'P90' or '90%' "
        "(case-insensitive)."
    )


def _stat_sort_key(parsed: tuple[str, float | None]) -> tuple[float, float]:
    kind, pct = parsed
    group = _KIND_SORT_GROUP.get(kind, 0)
    value = pct if kind == "percentile" else _KIND_SORT_VALUE.get(kind, 0.0)
    return (group, value)


def aggregate_statistics(
        series: pd.Series,
        stats: list[str],
        total: int | None = None,
    ) -> dict:
    """
    Compute named statistics of a series.

    Parameters
    ----------
    series : pandas.Series
        Data to summarise.
    stats : list of str
        Statistic names, case-insensitive:

        - ``"count"``: number of valid (non-NaN) values.
        - ``"min"`` / ``"minimum"``, ``"max"`` / ``"maximum"``.
        - ``"mean"``, ``"std"``, ``"median"``.
        - A percentile, given as ``"P<value>"`` or ``"<value>%"`` (e.g.
          ``"P90"``, ``"p05"``, ``"P99.9"``, ``"95%"``, ``"0.01%"``).
          ``"P90"`` and ``"90%"`` are equivalent. A bare number with
          neither a leading ``P`` nor a trailing ``%`` is rejected, since
          it would be ambiguous between a fraction and a percentage.
        - ``"%"`` / ``"density"``: `series`'s count as a percentage of
          `total`. Requires `total` to be given.
    total : int, optional
        Denominator for ``"%"`` / ``"density"``. Required only if `stats`
        includes one of those tokens.

    Returns
    -------
    dict
        Maps each entry of `stats` to its value, in a standardised,
        sorted order (count, then "%"/density, then min/percentiles/
        median/mean/std ordered by position in the distribution, then
        max) regardless of the order `stats` was given in. Keys are
        exactly the strings given in `stats`, so the display format
        (e.g. ``"P90"`` vs ``"p90"`` vs ``"90%"``) is preserved.

    Raises
    ------
    ValueError
        If a name is not recognised, a percentile is out of [0, 100], or
        ``"%"``/``"density"`` is requested without `total`.
    """
    parsed = {name: _parse_stat(name) for name in stats}
    ordered_names = sorted(parsed, key=lambda name: _stat_sort_key(parsed[name]))

    result = {}
    for name in ordered_names:
        kind, pct = parsed[name]
        if kind == "count":
            result[name] = series.count()
        elif kind == "min":
            result[name] = series.min()
        elif kind == "max":
            result[name] = series.max()
        elif kind == "mean":
            result[name] = series.mean()
        elif kind == "std":
            result[name] = series.std()
        elif kind == "median":
            result[name] = series.median()
        elif kind == "percentile":
            result[name] = series.quantile(pct / 100.0)
        elif kind == "percent_of_total":
            if total is None:
                raise ValueError(
                    f"'{name}' requires `total` to be given (count of "
                    "`series` as a percentage of `total`)."
                )
            result[name] = 100 * series.count() / total if total else np.nan
    return result


def table_to_latex(
        table: pd.DataFrame,
        filename: str,
        float_format,
        column_fmt: str = None,
    ):
    """
    Write a DataFrame to a LaTeX tabular file.

    The top-left cell is the index name if set, otherwise the column name if
    set, otherwise empty.

    Parameters
    ----------
    table : pandas.DataFrame
        Table to write.
    filename : str
        Output path. ``.tex`` is appended if missing.
    float_format : str or sequence of str
        printf-style format for values. A sequence needs one entry for the
        index followed by one per column.
    column_fmt : str, optional
        Format applied to numeric column headers.

    Raises
    ------
    ValueError
        If `float_format` is a sequence of the wrong length.
    """
    if not filename.endswith(".tex"): filename += ".tex"

    if column_fmt is None and isinstance(float_format, str):
        column_fmt = float_format

    def _fmt_column(c):
        if isinstance(c, str):
            return c.replace("%", r"\%")
        if column_fmt is not None:
            return f"{c:{column_fmt}}"
        return str(c)

    with open(filename, "w") as f:
        f.write("\\begin{tabular}{l"+"r"*len(table.columns)+"}\n")
        f.write("\\toprule \n")
        if table.index.name: f.write(f"{table.index.name}")
        elif table.columns.name: f.write(f"{table.columns.name}")
        columns = [_fmt_column(c) for c in table.columns]
        f.write(" & " + " & ".join(columns)+" \\\\\n")
        f.write("\\midrule\n")
        for index, row in table.iterrows():
            if isinstance(float_format,str):
                if isinstance(index,str):
                    f.write(f"{index}")
                else:
                    f.write(f"{index:{float_format}}")
                for i,x in enumerate(row.values):
                    if isinstance(x, str):
                        f.write(f" & {x}")
                    elif np.isnan(x):
                        f.write(" & ")
                    else:
                        f.write(f" & {x:{float_format}}")
                f.write(" \\\\\n")
            elif hasattr(float_format,"__len__") and (len(float_format)==len(columns)+1):
                f.write(f"{index:{float_format[0]}}")
                for i,x in enumerate(row.values):
                    if isinstance(x, str):
                        f.write(f" & {x}")
                    elif np.isnan(x):
                        f.write(" & ")
                    else:
                        f.write(f" & {x:{float_format[i+1]}}")
                f.write(" \\\\\n")
            else: raise ValueError("float_format has incorrect length: "
                f"Expected {len(columns)+1} and got {len(float_format)}.")
        f.write("\\bottomrule \n")
        f.write("\\end{tabular}")


def dirmag_to_uv(wind_direction, wind_speed, going_to=True):
    """
    Convert speed and direction to eastward and northward components.

    Parameters
    ----------
    wind_direction : numpy.ndarray
        Direction in degrees.
    wind_speed : numpy.ndarray
        Speed, in any unit. The components are returned in the same unit.
    going_to : bool, default True
        Direction convention. True is oceanographic ("going to"), False is
        meteorological ("coming from").

    Returns
    -------
    u, v : numpy.ndarray
        Eastward and northward components.
    """
    
    wind_direction = np.radians(wind_direction)
    
    if not going_to: 
        wind_direction = (wind_direction+np.pi)%(2*np.pi)
    
    u = wind_speed*np.sin(wind_direction)
    v = wind_speed*np.cos(wind_direction)
    return u,v

def uv_to_dirmag(u, v, going_to=True):
    """
    Convert eastward and northward components to direction and magnitude.

    Parameters
    ----------
    u : numpy.ndarray
        Eastward component.
    v : numpy.ndarray
        Northward component.
    going_to : bool, default True
        Direction convention. True is oceanographic ("going to"), False is
        meteorological ("coming from").

    Returns
    -------
    direction : numpy.ndarray
        Direction in degrees, in [0, 360).
    magnitude : numpy.ndarray
        Magnitude, in the units of `u` and `v`.
    """
    direction = np.degrees(np.arctan2(u,v))%360
    
    if not going_to:
        direction = (direction+180)%360
    
    magnitude = np.sqrt(u**2+v**2)
    return direction, magnitude

def _extend_xlim_to_zero(ax: "plt.Axes") -> None:
    """
    Extend the x-axis limits to include 0.
    """
    xlim = ax.get_xlim()
    if xlim[0] > 0 and xlim[1] > 0:
        ax.set_xlim([0, xlim[1]])
    elif xlim[0] < 0 and xlim[1] < 0:
        ax.set_xlim([xlim[0], 0])

def _extend_ylim_to_zero(ax: "plt.Axes") -> None:
    """
    Extend the x-axis limits to include 0.
    """
    ylim = ax.get_ylim()
    if ylim[0] > 0 and ylim[1] > 0:
        ax.set_ylim([0, ylim[1]])
    elif ylim[0] < 0 and ylim[1] < 0:
        ax.set_ylim([ylim[0], 0])