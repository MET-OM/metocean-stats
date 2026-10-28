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
    Format consecutive bin edges as "lo-hiunit" range labels, e.g. "0-2 m"
    or "30-60°" -- the single labelling convention shared by every
    range-based (non-categorical) binning in the package, whether the bins
    come from a plain variable (e.g. Hs) or a cyclic one (direction).

    Parameters
    ----------
    edges : array-like
        Bin edges, length n+1 for n bins (edges[i] and edges[i+1] are the
        low/high bound of bin i).
    unit : str, optional
        Suffix appended after each label, e.g. " m" or "°". Include any
        separating space yourself; none is added automatically.
    wrap : float, optional
        If given (e.g. 360), edges are wrapped with ``% wrap`` before
        formatting -- for cyclic axes like compass direction.
    fmt : str, default "{:.10g}"
        Format spec applied to each edge value.
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
    The package's single shared grouping primitive -- every class's
    `_get_groups` (and the EVA/CMA classes' own direct grouping) delegates
    here. Groups `data` by `by`, returning the groups alongside bin-edge
    metadata computed directly from the actual bin boundaries used at
    grouping time -- not reconstructed later by parsing a label string
    back into numbers, which breaks down for things like direction's
    wrap-around sector.

    Grouping only ever returns groups and their bin geometry -- never
    units or Variable metadata, which stay the calling class's job (it
    knows the Variable; this function never does). Labels are therefore
    always bare, e.g. "0-2" or "345", not "0-2 m" or "345°".

    Parameters
    ----------
    by : {"month", "season", "direction", "year", "week", "day", "hour"}
        or a column name in `data` (binned by value, step controlled by
        `step`).
    var : str, optional
        Column to pull out of each group. Groups are DataFrames if not given.
    var_dir, sectors, seasons, step
        Direction column name/sector count/season map/bin step, used only
        by the matching `by` value.
    observed : bool, default False
        Mirrors pandas' own `groupby(observed=...)`. False (the default)
        means every category in the bin's full domain is represented,
        even if empty (empty Series/DataFrame in "data"): all 12 months,
        all 4 seasons, weeks 1-53, days 1-366, hours 0-23, every sector,
        every variable bin between the data's min and max. True drops
        categories with no observations. Consistent across every `by`
        -- except "year", which has no fixed category domain to begin
        with: `observed=False` there means every year in
        [data.index.year.min(), data.index.year.max()], `observed=True`
        means only years actually present.

    Returns
    -------
    dict[str, list], one list per key below, all the same length (one
    entry per bin, in the same order) -- so a caller picks out only the
    keys it needs, and more keys can be added later without breaking any
    existing caller:

    - "data" : the grouped Series/DataFrame for that bin.
    - "label", "left_label", "center_label", "right_label" : display text.
    - "left_tick", "center_tick", "right_tick" : float tick positions, or
      None for a bin with no meaningful numeric position (season). The
      direction axis's ticks are the *unwrapped* sector edges (e.g. the
      north sector is -15..15, not the wrapped 345..15) so they stay
      monotonic across every sector; its labels still show the wrapped,
      human-readable 0-360 form (e.g. "345").
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
    Lay out `n_intervals` panels in a single figure as squarely as
    possible (up to `max_cols` columns, then widening rows), via one
    `plt.subplots()` call. Shared by every class that plots one subplot
    per group (EVA, CMA, ProfileEVA).

    Returns
    -------
    (fig, axes) : tuple[plt.Figure, np.ndarray]
        `axes` is always a flat 1-D array, length nrows*ncols (>= n_intervals).
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
    Resolve the axes to plot `n_panels` group-subplots on: either validate
    caller-supplied `ax`, or create one or more new figures via
    `get_n_axes`, splitting panels evenly across `nfigs` figures when more
    than one is requested. Shared "one subplot per group, across one or
    more figures" layout used by EVA/CMA's per-group plot methods.

    Returns
    -------
    (figures, axes) : tuple[np.ndarray, np.ndarray]
        `figures` is every newly-created Figure (empty if `ax` was
        supplied -- nothing to adjust() in that case, since the caller
        owns those figures). `axes` is a flat array of at least
        `n_panels` Axes.
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
    """Infer a round step size from the data range, snapped to a nice value."""
    NICE_STEPS = [0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10]
    raw        = (s.max() - s.min()) / target
    magnitude  = 10 ** np.floor(np.log10(raw))
    scaled     = raw / magnitude
    nice       = min(NICE_STEPS, key=lambda x: abs(x - scaled))
    return nice * magnitude

def bin_edges(s: pd.Series, step: float) -> np.ndarray:
    """
    Compute bin edges for a series snapped to a given step size.
    Starts from zero if all values are non-negative, otherwise
    fits outer edges to the data range.
    """
    lo = 0.0 if s.min() >= 0 else np.floor(s.min() / step) * step
    hi = np.ceil(s.max() / step) * step
    n  = int(round((hi - lo) / step)) + 1
    return np.linspace(lo, hi, n)


def aggregate_statistics(series: pd.Series, func: list[str]) -> dict:
    """
    Apply a list of statistic names to a series and return a dict of results.
    Accepts: "min", "mean", "max", "std", "count", and percentiles as "p90", "P90", "90%" etc.
    """
    result = {}
    for f in func:
        fl = f.lower().strip("%")
        if fl == "min":
            result[f] = series.min()
        elif fl == "mean":
            result[f] = series.mean()
        elif fl == "max":
            result[f] = series.max()
        elif fl == "std":
            result[f] = series.std()
        elif fl == "count":
            result[f] = series.count()
        elif fl.startswith("p") and fl[1:].isdigit():
            result[f] = series.quantile(int(fl[1:]) / 100)
        elif fl.isdigit():
            result[f] = series.quantile(int(fl) / 100)
        else:
            raise ValueError(f"Unknown aggregation function: '{f}'")
    return result


def table_to_latex(
        table: pd.DataFrame,
        filename: str,
        float_format,
        column_fmt: str = None,
    ):
    """
    Script to write a pandas dataframe to a latex .tex file.
    Allows setting a per-column float format.
    The top left cell will be the index name, if set,
    otherwise the column name, if set, otherwise empty.
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
    '''
    Get wind x (east) and y (north) component 
    from speed and direction (degrees, default: going to).

    Parameters
    ---------
    wind_direction : np.ndarray
        Wind direction (degrees)
    wind_speed : np.ndarray
        Wind Speed (degrees)
    going_to : bool, default True (oceanographic)
        Controls direction convention, False gives "from" direction.
    '''
    
    wind_direction = np.radians(wind_direction)
    
    if not going_to: 
        wind_direction = (wind_direction+np.pi)%(2*np.pi)
    
    u = wind_speed*np.sin(wind_direction)
    v = wind_speed*np.cos(wind_direction)
    return u,v

def uv_to_dirmag(u, v, going_to=True):
    '''
    Get direction (degrees, default: going to) and magnitude 
    from u (east) and v (north) components.

    Parameters
    ---------
    u : np.ndarray
        Eastwards velocity
    v : np.ndarray
        Northwards velocity
    going_to : bool, default True
        Controls direction convention, False gives "from" direction.
    '''
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