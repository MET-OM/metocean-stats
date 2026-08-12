"""
TimeSeriesValidation
--------------------
Validation module for comparing metocean data from multiple sources —
model output, satellite retrievals, in-situ measurements, or any
combination thereof.

Design principles
-----------------
- ``sources`` is a dict of named DataFrames (or TimeSeries objects).
  There is no hardcoded "model vs obs" binary — any source can be
  compared against any other.
- Variable metadata (units, symbols, names) is optional. If a source
  is a TimeSeries, metadata is pulled from it automatically. For plain
  DataFrames, column names are used as fallback.
- Distributional comparisons (A) do not require temporal alignment and
  work on the raw source data.
- Time-based comparisons (B) require an explicit call to ``.align()``,
  which populates ``self.aligned``. Any method that requires alignment
  will raise a clear error if called before ``.align()``.

Example usage
-------------
>>> tsv = TimeSeriesValidation(
...     sources={
...         "model": ts,        # TimeSeries
...         "sat":   sat_df,    # pd.DataFrame
...         "buoy":  buoy_df,   # pd.DataFrame
...     },
...     variables={
...         "sat":  [Variable("hs", "Significant wave height", "H_s", "m")],
...         "buoy": [Variable("hs", "Significant wave height", "H_s", "m")],
...     },
... )
>>>
>>> # Distributional comparison — no alignment needed
>>> tsv.plot_cdf(var=("model", "hs"), ref=("sat", "hs"))
>>> tsv.summary_table(vars=[("model", "hs"), ("sat", "hs"), ("buoy", "hs")])
>>>
>>> # Time-based comparison — alignment required first
>>> tsv.align(reference="model", method="nearest", tolerance="30min")
>>> tsv.plot_scatter(x=("model", "hs"), y=("sat", "hs"))
>>> tsv.rmse(x=("model", "hs"), y=("buoy", "hs"))
"""

from __future__ import annotations

import warnings
from typing import Union, Literal

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .timeseries import TimeSeries, Variable
from .plots import plot_rose
from .utils import infer_step, bin_edges

def _fmt_timedelta(td: pd.Timedelta) -> str:
    """Format a Timedelta into a human-readable string."""
    if pd.isna(td):
        return "N/A"
    total_seconds = int(td.total_seconds())
    if total_seconds % 3600 == 0:
        return f"{total_seconds // 3600}h"
    if total_seconds % 60 == 0:
        return f"{total_seconds // 60}min"
    return f"{total_seconds}s"

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Source = Union[TimeSeries, pd.DataFrame]


# ---------------------------------------------------------------------------
# TimeSeriesValidation
# ---------------------------------------------------------------------------

class TimeSeriesValidation:
    """
    Multi-source metocean validation container.

    Parameters
    ----------
    sources : dict[str, TimeSeries | pd.DataFrame]
        Named data sources to compare. Keys are short identifiers used
        throughout the API, e.g. ``"model"``, ``"sat"``, ``"buoy"``.
        At least two sources are required.

        Each value may be either:
        - A ``TimeSeries`` object: Variable metadata is extracted automatically.
        - A ``pd.DataFrame``: Index must be (or be convertible to) a
          DatetimeIndex. Variable metadata falls back to column names unless
          supplied via ``variables``.

    variables : dict[str, list[Variable]], optional
        Variable metadata for plain DataFrame sources. Keys must match
        keys in ``sources``. TimeSeries sources already carry their own
        metadata and will ignore any entry here (a warning is issued if
        an entry is provided for a TimeSeries source, to avoid confusion).

        If a column in a DataFrame source has no corresponding Variable
        entry, the column name is used as a fallback label in plots and
        tables.

    Notes
    -----
    Temporal alignment is not performed on construction. Call ``.align()``
    explicitly before using any method that compares sources in time (scatter
    plots, correlation, RMSE, etc.). Methods that only require distributional
    data (CDFs, summary tables, roses) work without alignment.

    The ``(source_key, column_name)`` tuple is the standard way to identify
    a variable across the API::

        tsv.plot_scatter(x=("model", "hs"), y=("sat", "hs"))
    """

    def __init__(
        self,
        sources: dict[str, Source],
        variables: dict[str, list[Variable]] | None = None,
    ):
        # ------------------------------------------------------------------ #
        # Validate sources argument type                                       #
        # ------------------------------------------------------------------ #
        if not isinstance(sources, dict):
            raise TypeError(
                f"'sources' must be a dict mapping source names to TimeSeries "
                f"or DataFrames, got {type(sources).__name__}."
            )

        if len(sources) < 2:
            raise ValueError(
                f"'sources' must contain at least two entries to enable "
                f"comparison, got {len(sources)}."
            )

        # ------------------------------------------------------------------ #
        # Validate source keys                                                 #
        # ------------------------------------------------------------------ #
        for key in sources:
            if not isinstance(key, str):
                raise TypeError(
                    f"All keys in 'sources' must be strings, "
                    f"got {type(key).__name__!r} for key {key!r}."
                )
            if not key.strip():
                raise ValueError(
                    "'sources' keys must be non-empty, non-whitespace strings."
                )

        # ------------------------------------------------------------------ #
        # Validate source values and extract DataFrames                        #
        # ------------------------------------------------------------------ #
        dataframes: dict[str, pd.DataFrame] = {}

        for key, src in sources.items():
            if isinstance(src, TimeSeries):
                dataframes[key] = src.data
            elif isinstance(src, pd.DataFrame):
                if src.empty:
                    raise ValueError(
                        f"Source '{key}' is an empty DataFrame."
                    )
                dataframes[key] = src.copy()
            else:
                raise TypeError(
                    f"Source '{key}' must be a TimeSeries or pd.DataFrame, "
                    f"got {type(src).__name__!r}."
                )

        # ------------------------------------------------------------------ #
        # Validate and coerce DatetimeIndex                                  #
        # ------------------------------------------------------------------ #
        for key, df in dataframes.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                raise TypeError(
                    f"Source '{key}' must have a DatetimeIndex, "
                    f"got {type(df.index).__name__!r}. "
                )

            if df.index.hasnans:
                raise ValueError(
                    f"Source '{key}': DatetimeIndex contains NaT values. "
                    f"Drop or fill missing timestamps before constructing "
                    f"TimeSeriesValidation, e.g.:\n"
                    f"    df = df[df.index.notna()]"
                )

            if not df.index.is_monotonic_increasing:
                warnings.warn(
                    f"Source '{key}': index is not sorted in ascending order. "
                    f"It will be sorted automatically.",
                    UserWarning,
                    stacklevel=2,
                )
                dataframes[key] = df.sort_index()

        # ------------------------------------------------------------------ #
        # Require consistent timezones across all sources                      #
        # ------------------------------------------------------------------ #
        timezones = {key: df.index.tz for key, df in dataframes.items()}
        unique_tzs = set(str(tz) for tz in timezones.values())

        if len(unique_tzs) > 1:
            tz_summary = "\n".join(
                f"    '{k}': {tz if tz is not None else 'timezone-naive'}"
                for k, tz in timezones.items()
            )
            raise ValueError(
                f"Sources have inconsistent timezone information:\n{tz_summary}\n\n"
                f"Harmonize all sources to the same timezone before constructing "
                f"TimeSeriesValidation. For example, to convert everything to UTC:\n"
                f"    df.index = df.index.tz_localize('UTC')   # naive → UTC\n"
                f"    df.index = df.index.tz_convert('UTC')    # aware, other tz → UTC"
            )

        # ------------------------------------------------------------------ #
        # Validate variables argument type                                     #
        # ------------------------------------------------------------------ #
        variables = variables or {}

        if not isinstance(variables, dict):
            raise TypeError(
                f"'variables' must be a dict mapping source names to lists of "
                f"Variable objects, got {type(variables).__name__}."
            )

        for key, varlist in variables.items():
            if key not in sources:
                raise KeyError(
                    f"'variables' contains key '{key}' which is not present in "
                    f"'sources'. Valid source keys are: "
                    f"{list(sources.keys())}."
                )
            if isinstance(sources[key], TimeSeries):
                warnings.warn(
                    f"'variables' entry for '{key}' will be ignored because "
                    f"'{key}' is a TimeSeries and already carries Variable "
                    f"metadata. Remove the entry to suppress this warning.",
                    UserWarning,
                    stacklevel=2,
                )
            if not isinstance(varlist, list):
                raise TypeError(
                    f"'variables[{key!r}]' must be a list of Variable objects, "
                    f"got {type(varlist).__name__}."
                )
            for i, v in enumerate(varlist):
                if not isinstance(v, Variable):
                    raise TypeError(
                        f"'variables[{key!r}][{i}]' must be a Variable instance, "
                        f"got {type(v).__name__!r}."
                    )

        # ------------------------------------------------------------------ #
        # Validate that declared Variable columns exist in source DataFrames  #
        # ------------------------------------------------------------------ #
        for key, varlist in variables.items():
            if isinstance(sources[key], TimeSeries):
                continue  # already warned above; skip column check
            df_cols = set(dataframes[key].columns)
            for v in varlist:
                if v.col not in df_cols:
                    raise KeyError(
                        f"Variable '{v.col}' declared for source '{key}' "
                        f"does not exist in that DataFrame. "
                        f"Available columns: {sorted(df_cols)}."
                    )

        # ------------------------------------------------------------------ #
        # Build variable metadata registry                                     #
        # ------------------------------------------------------------------ #
        # Registry: source_key -> {col_name: Variable}
        # TimeSeries sources: use their own _variables dict.
        # DataFrame sources: use supplied Variable list, fall back to col name.
        variable_registry: dict[str, dict[str, Variable]] = {}

        for key, src in sources.items():
            if isinstance(src, TimeSeries):
                variable_registry[key] = dict(src._variables)
            else:
                reg: dict[str, Variable] = {}
                supplied = {v.col: v for v in variables.get(key, [])}
                for col in dataframes[key].columns:
                    if col in supplied:
                        reg[col] = supplied[col]
                    else:
                        # Fallback: column name used as all labels
                        reg[col] = Variable(
                            col=col,
                            name=col,
                            symbol=col,
                            unit="",
                        )
                variable_registry[key] = reg

        # ------------------------------------------------------------------ #
        # Store public and private state                                       #
        # ------------------------------------------------------------------ #
        self.sources:   dict[str, Source]              = dict(sources)
        self._data:     dict[str, pd.DataFrame]        = dataframes
        self._variables: dict[str, dict[str, Variable]] = variable_registry
        self.aligned:   pd.DataFrame | None            = None
        self._align_reference: str | None              = None
        self._align_method:    str | None              = None
        self._align_tolerance: str | None              = None

        # ------------------------------------------------------------------ #
        # Coerce all source indexes to a common datetime resolution          #
        # ------------------------------------------------------------------ #
        for key, df in self._data.items():
            if df.index.tz is not None:
                self._data[key].index = df.index.astype(f"datetime64[ns, {df.index.tz}]")
            else:
                self._data[key].index = df.index.astype("datetime64[ns]")

        # ------------------------------------------------------------------ #
        # Print initialisation summary                                         #
        # ------------------------------------------------------------------ #
        self.summary()


    def summary(self) -> None:
        """
        Print a diagnostic summary for each source dataset.

        Includes per-source:
        - Time period (start / end)
        - Number of timestamps
        - Mean and median timestep
        - Number of columns
        - Percentage of valid (non-NaN) values per column
        """
        SEP   = "─" * 60
        SEP_S = "─" * 40

        # Alignment status
        if self.aligned is not None:
            align_info = (
                f"  Aligned  : yes  (reference='{self._align_reference}', "
                f"method='{self._align_method}', tolerance='{self._align_tolerance}', "
                f"n={len(self.aligned)})"
            )
        else:
            align_info = "  Aligned  : no  (call .align() before time-based methods)"

        print(SEP)
        print("TimeSeriesValidation — source summary")
        print(align_info)
        print(SEP)

        for key, df in self._data.items():
            n       = len(df)
            start   = df.index.min()
            end     = df.index.max()
            ncols   = len(df.columns)

            # Timestep statistics — only meaningful if more than one timestamp
            if n > 1:
                steps       = df.index.to_series().diff().dropna()
                mean_step   = steps.mean()
                median_step = steps.median()

                # Flag irregular sampling
                unique_steps = steps.nunique()
                regular_flag = "" if unique_steps == 1 else "  ⚠ irregular sampling"
            else:
                mean_step    = median_step = pd.NaT
                regular_flag = ""

            src_type = type(self.sources[key]).__name__

            print(f"  [{key}]  ({src_type})")
            print(f"  {'Start':<18}: {start}")
            print(f"  {'End':<18}: {end}")
            print(f"  {'N timestamps':<18}: {n:,}")
            print(f"  {'Mean timestep':<18}: {_fmt_timedelta(mean_step)}{regular_flag}")
            print(f"  {'Median timestep':<18}: {_fmt_timedelta(median_step)}")
            print(f"  {'N columns':<18}: {ncols}")
            print(f"  {SEP_S}")
            print(f"  {'Column':<20}  {'N valid':>8}  {'% valid':>8}")
            print(f"  {SEP_S}")

            for col in df.columns:
                n_valid   = df[col].notna().sum()
                pct_valid = 100 * n_valid / n if n > 0 else 0.0
                flag      = "  ⚠" if pct_valid < 90 else ""
                print(f"  {col:<20}  {n_valid:>8,}  {pct_valid:>7.1f}%{flag}")

            print(SEP)

        print()

    # ---------------------------------------------------------------------- #
    # Alignment                                                                #
    # ---------------------------------------------------------------------- #

    def align(
        self,
        reference: str,
        method: str = "nearest",
        tolerance: str = "30min",
        resample_freq: str | None = None,
    ) -> None:
        """
        Temporally align all sources to a common time axis and populate
        ``self.aligned``.

        Must be called explicitly before any method that performs time-based
        comparison (scatter, correlation, RMSE, etc.).

        Parameters
        ----------
        reference : str
            Key of the source whose timestamps are used as the alignment axis.
            All other sources are mapped onto these timestamps.
            Must be a key in ``self.sources``.
        method : {"nearest", "interpolate", "resample"}, default "nearest"
            Alignment strategy:

            - ``"nearest"``     — each reference timestamp is matched to the
              nearest observation within ``tolerance``. Suitable for satellite
              vs model.
            - ``"interpolate"`` — source values are linearly interpolated to
              reference timestamps. Suitable for dense buoy vs coarser model.
            - ``"resample"``    — both reference and all sources are resampled
              to ``resample_freq`` before alignment. Requires ``resample_freq``.

        tolerance : str, default "30min"
            Maximum allowed time difference for ``method="nearest"``.
            Pandas offset string, e.g. ``"30min"``, ``"1h"``, ``"3h"``.
            Ignored for other methods.
        resample_freq : str, optional
            Target frequency for ``method="resample"``, e.g. ``"1h"``, ``"3h"``.
            Required when ``method="resample"``, ignored otherwise.

        Raises
        ------
        KeyError
            If ``reference`` is not a key in ``self.sources``.
        ValueError
            If ``method`` is not one of the supported strategies, or if
            ``method="resample"`` is used without supplying ``resample_freq``.

        Notes
        -----
        Calling ``.align()`` a second time with different parameters will
        overwrite ``self.aligned``. A warning is issued if this happens.
        """
        # ------------------------------------------------------------------ #
        # Validate reference                                                   #
        # ------------------------------------------------------------------ #
        if reference not in self.sources:
            raise KeyError(
                f"Alignment reference '{reference}' is not a key in 'sources'. "
                f"Valid keys are: {list(self.sources.keys())}."
            )

        # ------------------------------------------------------------------ #
        # Validate method                                                      #
        # ------------------------------------------------------------------ #
        valid_methods = {"nearest", "interpolate", "resample"}
        if method not in valid_methods:
            raise ValueError(
                f"Alignment method '{method}' is not supported. "
                f"Choose one of: {sorted(valid_methods)}."
            )

        if method == "resample" and resample_freq is None:
            raise ValueError(
                "method='resample' requires 'resample_freq' to be specified, "
                "e.g. resample_freq='1h'."
            )

        # ------------------------------------------------------------------ #
        # Validate tolerance is a parseable offset string                     #
        # ------------------------------------------------------------------ #
        if method == "nearest":
            try:
                pd.tseries.frequencies.to_offset(tolerance)
            except ValueError:
                raise ValueError(
                    f"'tolerance' must be a valid pandas offset string "
                    f"(e.g. '30min', '1h'), got {tolerance!r}."
                )

        # ------------------------------------------------------------------ #
        # Warn if overwriting a previous alignment                             #
        # ------------------------------------------------------------------ #
        if self.aligned is not None:
            warnings.warn(
                f"Overwriting existing alignment "
                f"(reference='{self._align_reference}', "
                f"method='{self._align_method}', "
                f"tolerance='{self._align_tolerance}'). "
                f"Any cached results from the previous alignment are discarded.",
                UserWarning,
                stacklevel=2,
            )

        # ------------------------------------------------------------------ #
        # Perform alignment                                                    #
        # ------------------------------------------------------------------ #
        self.aligned = self._do_align(reference, method, tolerance, resample_freq)
        self._align_reference = reference
        self._align_method    = method
        self._align_tolerance = tolerance

        n_ref   = len(self._data[reference])
        n_align = len(self.aligned)
        coverage = 100 * n_align / n_ref if n_ref > 0 else 0.0

        print(
            f"Alignment complete. Reference: '{reference}' ({n_ref} timestamps). "
            f"Collocations retained: {n_align} ({coverage:.1f}% coverage)."
        )

    def _do_align(
        self,
        reference: str,
        method: str,
        tolerance: str,
        resample_freq: str | None,
    ) -> pd.DataFrame:
        """
        Internal alignment implementation. Aligns each source independently
        against the reference index, then assembles a MultiIndex DataFrame.
        Columns are (source_key, col_name) tuples throughout.
        """
        # ------------------------------------------------------------------ #
        # Prepare reference index (may change if resampling)                  #
        # ------------------------------------------------------------------ #
        if method == "resample":
            ref_df   = self._data[reference].resample(resample_freq).mean()
            ref_index = ref_df.index
        else:
            ref_df    = self._data[reference].copy()
            ref_index = ref_df.index

        aligned_parts: dict[str, pd.DataFrame] = {reference: ref_df}

        # ------------------------------------------------------------------ #
        # Align each non-reference source independently                        #
        # ------------------------------------------------------------------ #
        for key, df in self._data.items():
            if key == reference:
                continue

            src = df.copy()

            if method == "nearest":
                tol = pd.Timedelta(tolerance)

                # Keep it flat: left = reference times, right = src + its own timestamps.
                # Never touch aligned_parts here — that avoids the MultiIndex clash.
                left = pd.DataFrame({"__ref_time__": ref_index})
                right = src.assign(__src_time__=src.index).reset_index(drop=True)

                merged = pd.merge_asof(
                    left,
                    right,
                    left_on="__ref_time__",
                    right_on="__src_time__",
                    direction="nearest",
                    tolerance=tol,
                )
                merged.index = ref_index

                time_diff = (merged["__src_time__"] - merged["__ref_time__"]).abs()
                merged = merged.drop(columns=["__ref_time__", "__src_time__"])
                merged["__time_diff__"] = time_diff

                aligned_parts[key] = merged

            elif method == "interpolate":
                src_reindexed = (
                    src
                    .reindex(src.index.union(ref_index))
                    .interpolate(method="time")
                    .reindex(ref_index)
                )
                aligned_parts[key] = src_reindexed

            elif method == "resample":
                aligned_parts[key] = (
                    src
                    .resample(resample_freq)
                    .mean()
                    .reindex(ref_index)   # align to reference resampled grid
                )

        # ------------------------------------------------------------------ #
        # Assemble MultiIndex DataFrame                                      #
        # ------------------------------------------------------------------ #
        frames = []
        for key, part_df in aligned_parts.items():
            part_df = part_df.copy()
            part_df.columns = pd.MultiIndex.from_tuples(
                [(key, c) for c in part_df.columns]
            )
            frames.append(part_df)

        result = pd.concat(frames, axis=1)

        # Drop rows where ALL non-reference columns are NaN
        # (obs outside tolerance window, or gaps in resampled data)
        non_ref_cols = [c for c in result.columns if c[0] != reference]
        if non_ref_cols:
            result = result.dropna(subset=non_ref_cols, how="all")

        return result

    # ---------------------------------------------------------------------- #
    # Guard                                                                    #
    # ---------------------------------------------------------------------- #

    def _require_alignment(self) -> None:
        """
        Raise a clear RuntimeError if ``.align()`` has not been called.
        Called at the top of any method that requires ``self.aligned``.
        """
        if self.aligned is None:
            raise RuntimeError(
                "This method requires temporal alignment. "
                "Call .align(reference=<source_key>, method=...) first."
            )

    # ---------------------------------------------------------------------- #
    # Variable metadata helpers                                                #
    # ---------------------------------------------------------------------- #

    def get_variable(self, source: str, col: str) -> Variable:
        """
        Return the Variable metadata for a (source, col) pair.

        Parameters
        ----------
        source : str
            Source key, must be present in self.sources.
        col : str
            Column name within that source.

        Raises
        ------
        KeyError
            If source or col is not found.
        """
        if source not in self._variables:
            raise KeyError(
                f"Source '{source}' not found. "
                f"Valid keys are: {list(self.sources.keys())}."
            )
        if col not in self._variables[source]:
            raise KeyError(
                f"Column '{col}' not found in source '{source}'. "
                f"Available columns: {sorted(self._variables[source].keys())}."
            )
        return self._variables[source][col]

    def _get_aligned(self, ref: tuple[str, str]) -> tuple[pd.Series, Variable]:
        """
        Retrieve a single (source, col) Series from the aligned DataFrame,
        along with its Variable metadata.

        Requires .align() to have been called first.

        Parameters
        ----------
        ref : tuple[str, str]
            A (source_key, column_name) pair, e.g. ("model", "hs").

        Returns
        -------
        series : pd.Series
        var    : Variable
        """
        self._require_alignment()
        self._validate_ref(ref)

        src, col = ref
        if (src, col) not in self.aligned.columns:
            raise KeyError(
                f"Column '{col}' not found in source '{src}' in the aligned "
                f"DataFrame. Available columns for '{src}': "
                f"{[c for s, c in self.aligned.columns if s == src]}."
            )

        return self.aligned[ref], self.get_variable(*ref)


    def _get_source(self, ref: tuple[str, str]) -> tuple[pd.Series, Variable]:
        """
        Retrieve a single (source, col) Series from the raw (non-aligned)
        source data, along with its Variable metadata.

        Does not require .align() to have been called. Use this for
        distributional methods (CDFs, summary tables, roses) that operate
        on the full original record.

        Parameters
        ----------
        ref : tuple[str, str]
            A (source_key, column_name) pair, e.g. ("model", "hs").

        Returns
        -------
        series : pd.Series
        var    : Variable
        """
        self._validate_ref(ref)

        src, col = ref
        if col not in self._data[src].columns:
            raise KeyError(
                f"Column '{col}' not found in source '{src}'. "
                f"Available columns: {sorted(self._data[src].columns)}."
            )

        return self._data[src][col], self.get_variable(*ref)


    def _validate_ref(self, ref: tuple[str, str]) -> None:
        """
        Shared input validation for a (source, col) reference tuple.
        """
        if (
            not isinstance(ref, tuple)
            or len(ref) != 2
            or not all(isinstance(p, str) for p in ref)
        ):
            raise TypeError(
                f"Variable reference must be a (source_key, column_name) tuple "
                f"of two strings, got {ref!r}."
            )

        src = ref[0]
        if src not in self.sources:
            raise KeyError(
                f"Source '{src}' is not in 'sources'. "
                f"Valid keys: {list(self.sources.keys())}."
            )





    # ---------------------------------------------------------------------- #
    # Dunder methods                                                         #
    # ---------------------------------------------------------------------- #

    def __repr__(self) -> str:
        source_summary = ", ".join(
            f"'{k}' ({type(v).__name__})" for k, v in self.sources.items()
        )
        aligned_status = (
            f"aligned on '{self._align_reference}' "
            f"({len(self.aligned)} collocations)"
            if self.aligned is not None
            else "not aligned"
        )
        return (
            f"TimeSeriesValidation(\n"
            f"  sources   : [{source_summary}]\n"
            f"  alignment : {aligned_status}\n"
            f")"
        )
    
    def plot_scatter(
        self,
        x: tuple[str, str],
        y: tuple[str, str],
        bin_size: float | None = None,
        cmap: str = "viridis",
        figsize: tuple[float, float] = (7, 7),
        ax: "plt.Axes | None" = None,
    ) -> "plt.Axes":
        """
        Scatter plot between two aligned sources rendered as a 2-D histogram
        (heatmap), with a 1:1 reference line, a quantile-quantile line, key
        performance metrics in a text box, and a logarithmic colorbar.

        Parameters
        ----------
        x : tuple[str, str]
            (source_key, column_name) for the x-axis (typically model / reference).
        y : tuple[str, str]
            (source_key, column_name) for the y-axis (typically observation / other).
        bin_size : float, optional
            Edge length of each square histogram bin in the units of the variable.
            Defaults to a "nice" value derived from the data range:
            the nearest value in {0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0} such
            that the axis spans roughly 40–80 bins.
        cmap : str, default "plasma"
            Any matplotlib colormap name. Applied on a log scale.
        figsize : tuple[float, float], default (7, 7)
            Figure size in inches. Ignored when ``ax`` is supplied.
        ax : plt.Axes, optional
            Existing axes to draw on. If None, a new figure is created.

        Returns
        -------
        ax : plt.Axes

        Raises
        ------
        RuntimeError
            If ``.align()`` has not been called.
        KeyError
            If either variable reference is not found in the aligned data.

        Notes
        -----
        - Requires temporal alignment (``.align()`` must be called first).
        - The histogram is built from paired, non-NaN observations only.
        - The Q-Q line connects the matching quantiles of x and y (p = 2 % … 98 %
        in 2 % steps), drawn as a thin line so it stays readable over the bins.
        - A dummy ``Patch`` in the median colormap colour is added to the legend
        so that the data label appears even though the histogram is not a
        standard Line2D artist.
        - The colorbar label states the bin size, e.g.
        "entries per 0.25 × 0.25 m bin".
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
        import matplotlib.ticker as mticker
        import numpy as np

        # ------------------------------------------------------------------ #
        # Guards and data retrieval                                            #
        # ------------------------------------------------------------------ #
        self._require_alignment()
        x_series, x_var = self._get_aligned(x)
        y_series, y_var = self._get_aligned(y)

        # Drop rows where either value is NaN
        mask   = x_series.notna() & y_series.notna()
        x_vals = x_series[mask].values.astype(float)
        y_vals = y_series[mask].values.astype(float)

        if len(x_vals) < 2:
            raise ValueError(
                "Fewer than 2 non-NaN collocated pairs — cannot produce scatter plot."
            )

        # ------------------------------------------------------------------ #
        # Metrics                                                              #
        # ------------------------------------------------------------------ #
        diff   = y_vals - x_vals
        bias   = float(np.mean(diff))
        mae    = float(np.mean(np.abs(diff)))
        rmse   = float(np.sqrt(np.mean(diff ** 2)))
        si     = rmse / float(np.mean(x_vals)) if np.mean(x_vals) != 0 else np.nan
        r      = float(np.corrcoef(x_vals, y_vals)[0, 1])
        n      = int(len(x_vals))

        # ------------------------------------------------------------------ #
        # Axis limits — equal, starting at zero                               #
        # ------------------------------------------------------------------ #
        data_max  = max(x_vals.max(), y_vals.max())
        axis_max  = data_max * 1.05
        axis_min  = 0.0

        # ------------------------------------------------------------------ #
        # Bin size selection                                                   #
        # ------------------------------------------------------------------ #
        _NICE = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]

        if bin_size is None:
            span = axis_max - axis_min
            # Pick the nice value that gives closest to 60 bins across the axis
            bin_size = min(_NICE, key=lambda b: abs(span / b - 150))

        # ------------------------------------------------------------------ #
        # 2-D histogram                                                        #
        # ------------------------------------------------------------------ #
        bins = np.arange(axis_min, axis_max + bin_size, bin_size)

        H, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=[bins, bins])
        H = H.T  # imshow/pcolormesh expects (row=y, col=x)

        # Mask empty bins so they stay white / transparent
        H_masked = np.ma.masked_where(H == 0, H)

        # ------------------------------------------------------------------ #
        # Figure / axes                                                        #
        # ------------------------------------------------------------------ #
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # ------------------------------------------------------------------ #
        # Draw histogram                                                       #
        # ------------------------------------------------------------------ #
        norm = mcolors.LogNorm(vmin=1, vmax=H_masked.max())
        cm   = plt.get_cmap(cmap)

        pcm = ax.pcolormesh(
            xedges, yedges, H_masked,
            cmap=cm,
            norm=norm,
            zorder=2,
        )

        # ------------------------------------------------------------------ #
        # Colorbar                                                             #
        # ------------------------------------------------------------------ #
        if x_var.unit: 
            unit_str = x_var.unit
        elif y_var.unit: 
            unit_str = y_var.unit
        else: 
            unit_str = "-"
        cbar_label = (
            f"entries per {bin_size:g} × {bin_size:g} {unit_str} bin"
        )

        cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label(cbar_label)
        cbar.ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

        # ------------------------------------------------------------------ #
        # 1:1 line                                                             #
        # ------------------------------------------------------------------ #
        ax.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            color="black",
            linewidth=1.2,
            linestyle="--",
            zorder=4,
            label="1:1",
        )

        # ------------------------------------------------------------------ #
        # Q-Q line (log-spaced quantiles for better tail resolution)          #
        # ------------------------------------------------------------------ #
        base       = 1.2
        n_levels   = int(np.floor(np.log(n) / np.log(base)))
        lower_half = [50 / base**k for k in range(n_levels + 1)]
        upper_half = [100 - p for p in lower_half if p != 50][::-1]
        percentiles = np.array(sorted(set(lower_half + upper_half)))

        qq_x = np.percentile(x_vals, percentiles, method="closest_observation")
        qq_y = np.percentile(y_vals, percentiles, method="closest_observation")

        ax.scatter(
            qq_x, qq_y,
            color="darkorange",
            s=22,
            zorder=5,
            linewidths=0,
            label="Q-Q",
        )

        # ------------------------------------------------------------------ #
        # Dummy patch for histogram legend entry                               #
        # ------------------------------------------------------------------ #
        mid_log   = np.sqrt(norm.vmin * norm.vmax)   # geometric mean
        mid_color = cm(norm(mid_log))
        data_patch = mpatches.Patch(
            facecolor=mid_color,
            edgecolor="none",
            label=f"Data  (N = {n:,})",
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=[data_patch] + handles,
            labels=[f"Data  (N = {n:,})"] + labels,
            loc="lower right",
            framealpha=0.85,
            edgecolor="0.7",
        )

        # ------------------------------------------------------------------ #
        # Metrics text box                                                     #
        # ------------------------------------------------------------------ #
        metrics_text = (
            f"Bias : {bias:+.3f} {x_var.unit}\n"
            f"MAE  : {mae:.3f} {x_var.unit}\n"
            f"RMSE : {rmse:.3f} {x_var.unit}\n"
            f"SI   : {si:.3f}\n"
            f"R    : {r:.3f}"
        )

        ax.text(
            0.03, 0.97,
            metrics_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="0.7",
                alpha=0.90,
            ),
            zorder=6,
        )

        # ------------------------------------------------------------------ #
        # Axis formatting                                                      #
        # ------------------------------------------------------------------ #
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_aspect("equal", adjustable="box")

        x_label = f"{x_var.name} [{x_var.unit}]" if x_var.unit else x_var.name
        y_label = f"{y_var.name} [{y_var.unit}]" if y_var.unit else y_var.name
        source_x, source_y = x[0], y[0]

        ax.set_xlabel(f"{source_x.capitalize()} — {x_label}")
        ax.set_ylabel(f"{source_y.capitalize()} — {y_label}")
        ax.set_title(
            f"{source_x} ${x_var.symbol}$ vs {source_y} ${y_var.symbol}$",
            pad=8,
        )

        ax.grid(True, linestyle=":", linewidth=0.6, color="0.75", zorder=1)

        return ax
    
    def plot_rose(
        self,
        magnitude_a:  tuple[str, str],
        direction_a:  tuple[str, str],
        magnitude_b:  tuple[str, str],
        direction_b:  tuple[str, str],
        *,
        n_dir_bins:     int         = 16,
        mag_step:       "float | None" = None,
        cmap:           str         = "viridis",
        calm_threshold: float       = 0.0,
        figsize:        tuple[float, float] = (12, 6),
    ) -> "plt.Figure":
        """
        Side-by-side rose plots for two (magnitude, direction) pairs.

        Both panels share identical magnitude bin edges, colormap, and radial
        scale so they are directly visually comparable.  The single-rose
        drawing logic lives in ``utils.plot_rose``; this method is responsible
        only for computing the shared scaling and dispatching.

        Parameters
        ----------
        magnitude_a, magnitude_b : tuple[str, str]
            (source_key, column_name) for the magnitude variable of each panel,
            e.g. ``("model", "ws")`` and ``("buoy", "ws")``.
        direction_a, direction_b : tuple[str, str]
            (source_key, column_name) for the direction variable of each panel,
            e.g. ``("model", "wdir")`` and ``("buoy", "wdir")``.
            Directions must be in meteorological convention (from-direction,
            degrees clockwise from North).
        n_dir_bins : int, default 16
            Number of directional sectors passed to ``utils.plot_rose``.
        mag_step : float, optional
            Force a specific bin step size (e.g. ``2.0`` for 0–2–4–6 … m/s).
            If None, a nice step is chosen automatically from the joint 99th
            percentile of both datasets, targeting ~7 bins.
        cmap : str, default "YlOrRd"
            Colormap for magnitude classes.
        calm_threshold : float, default 0.0
            Observations at or below this magnitude are classified as calm and
            excluded from directional statistics.
        figsize : tuple[float, float], default (12, 6)
            Figure size in inches.

        Returns
        -------
        fig : plt.Figure

        Notes
        -----
        - Does not require temporal alignment. Each panel is drawn from its
        raw source data so the full record length is used for each source.
        - Shared magnitude bins are derived from the combined 99th percentile
        of both datasets, ensuring neither panel clips its high-value tail.
        - Shared ``r_max`` is the maximum sector total across both datasets
        (with 15 % headroom), so a dominant direction in one panel does not
        visually exaggerate the other.
        """

        # ------------------------------------------------------------------ #
        # Retrieve raw series (no alignment required)                          #
        # ------------------------------------------------------------------ #
        mag_a, mag_var_a = self._get_source(magnitude_a)
        dir_a, _         = self._get_source(direction_a)
        mag_b, mag_var_b = self._get_source(magnitude_b)
        dir_b, _         = self._get_source(direction_b)

        mag_a = mag_a.dropna().values.astype(float)
        dir_a = dir_a.dropna().values.astype(float)
        mag_b = mag_b.dropna().values.astype(float)
        dir_b = dir_b.dropna().values.astype(float)

        # ------------------------------------------------------------------ #
        # Shared magnitude bins                                                #
        # Joint 99th percentile so neither panel clips its tail.              #
        # Nice step targeting ~7 bins, or user-supplied mag_step.             #
        # ------------------------------------------------------------------ #
        combined_active = np.concatenate([
            mag_a[mag_a > calm_threshold],
            mag_b[mag_b > calm_threshold],
        ])
        p99 = float(np.percentile(combined_active, 99)) if len(combined_active) else 1.0

        import pandas as pd
        if mag_step is None:
            mag_step = infer_step(pd.Series(combined_active), target=7)
        mag_bins = np.arange(0, p99 + mag_step, mag_step)

        # ------------------------------------------------------------------ #
        # Shared radial scale                                                  #
        # ------------------------------------------------------------------ #
        def _sector_max(mag, direction):
            n_total = len(mag)
            if n_total == 0:
                return 0.0
            sector_width = 360.0 / n_dir_bins
            dir_edges    = np.linspace(
                -sector_width / 2, 360 - sector_width / 2, n_dir_bins + 1
            )
            active   = mag > calm_threshold
            dir_norm = direction[active] % 360.0
            totals   = np.zeros(n_dir_bins)
            for d_idx in range(n_dir_bins):
                d_lo = dir_edges[d_idx] % 360
                d_hi = dir_edges[d_idx + 1] % 360
                if d_lo < d_hi:
                    in_dir = (dir_norm >= d_lo) & (dir_norm < d_hi)
                else:
                    in_dir = (dir_norm >= d_lo) | (dir_norm < d_hi)
                totals[d_idx] = in_dir.sum() / n_total
            return float(totals.max())

        r_max = max(_sector_max(mag_a, dir_a), _sector_max(mag_b, dir_b)) * 1.15

        # ------------------------------------------------------------------ #
        # Figure                                                               #
        # ------------------------------------------------------------------ #
        fig, axes = plt.subplots(
            1, 2,
            figsize=figsize,
            subplot_kw={"projection": "polar"},
        )

        src_a = magnitude_a[0]
        src_b = magnitude_b[0]

        plot_rose(
            mag_a, dir_a, axes[0],
            n_dir_bins=n_dir_bins,
            mag_bins=mag_bins,
            cmap=cmap,
            r_max=r_max,
            calm_threshold=calm_threshold,
            title=f"{src_a.capitalize()} — {mag_var_a.name}",
            legend=True,
            mag_label=mag_var_a.name,
            mag_unit=mag_var_a.unit,
        )

        plot_rose(
            mag_b, dir_b, axes[1],
            n_dir_bins=n_dir_bins,
            mag_bins=mag_bins,
            cmap=cmap,
            r_max=r_max,
            calm_threshold=calm_threshold,
            title=f"{src_b.capitalize()} — {mag_var_b.name}",
            legend=True,
            mag_label=mag_var_b.name,
            mag_unit=mag_var_b.unit,
        )

        fig.tight_layout(w_pad=4)
        return axes
    
    def plot_timeseries(
        self,
        vars: list[tuple[str, str]],
        aligned:  bool = False,
        xlim:     tuple[str, str] | None = None,
        ax:       "plt.Axes | None" = None,
        figsize:  tuple[float, float] = None,
        plot_kw:  dict[str,str] = {},
    ) -> "plt.Axes":
        """
        Plot one or more (source, column) time series on a single axes.

        Parameters
        ----------
        vars : list[tuple[str, str]]
            Any number of (source_key, column_name) pairs to plot, e.g.::

                tsv.plot_timeseries([
                    ("model", "hs"),
                    ("buoy",  "hs"),
                    ("sat",   "hs"),
                ])

        aligned : bool, default False
            If False, each series is drawn from its full raw source record
            (different lengths and time axes are fine).
            If True, all series are drawn from ``self.aligned``, which must
            have been populated by a prior call to ``.align()``. Useful for
            inspecting the collocated subset directly.
        xlim : tuple[str, str], optional
            Restrict the x-axis to a sub-period, given as a pair of
            ISO-8601 strings, e.g. ``("2015-05-01", "2018-08-30")``.
            The strings are parsed by ``pd.Timestamp`` so any unambiguous
            date/datetime format is accepted.  Data outside the window is
            not plotted (the series are sliced before drawing, so
            auto-scaling of the y-axis reflects only the visible period).
        ax : plt.Axes, optional
            Existing axes to draw on.  If None, a new figure is created.
            Pass an axes from a ``plt.subplots`` grid to compose multi-panel
            figures::

                fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
                tsv.plot_timeseries([("model", "hs")], ax=axes[0])
                tsv.plot_timeseries([("model", "ws")], ax=axes[1])
                tsv.plot_timeseries([("model", "tp")], ax=axes[2])

        figsize : tuple[float, float], default (12, 4)
            Figure size in inches.  Ignored when ``ax`` is supplied.

        Returns
        -------
        ax : plt.Axes

        Raises
        ------
        ValueError
            If ``vars`` is empty, or if ``aligned=True`` but ``.align()``
            has not been called.
        TypeError
            If any entry in ``vars`` is not a two-string tuple.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # ------------------------------------------------------------------ #
        # Validate vars                                                        #
        # ------------------------------------------------------------------ #
        if not vars:
            raise ValueError("'vars' must contain at least one (source, col) pair.")

        for ref in vars:
            self._validate_ref(ref)

        # ------------------------------------------------------------------ #
        # Alignment guard                                                      #
        # ------------------------------------------------------------------ #
        if aligned:
            self._require_alignment()

        # ------------------------------------------------------------------ #
        # Parse xlim                                                           #
        # ------------------------------------------------------------------ #
        t_lo: pd.Timestamp | None = None
        t_hi: pd.Timestamp | None = None

        if xlim is not None:
            if (
                not isinstance(xlim, tuple)
                or len(xlim) != 2
                or not all(isinstance(s, str) for s in xlim)
            ):
                raise TypeError(
                    "'xlim' must be a tuple of two ISO-8601 strings, "
                    f"e.g. (\"2015-01-01\", \"2018-12-31\"), got {xlim!r}."
                )
            try:
                t_lo = pd.Timestamp(xlim[0])
                t_hi = pd.Timestamp(xlim[1])
            except Exception as exc:
                raise ValueError(
                    f"Could not parse 'xlim' timestamps: {exc}"
                ) from exc

            # Match timezone of the first source so .loc slicing never hits a
            # tz-naive vs tz-aware mismatch.  The constructor already enforces
            # consistent timezones across all sources, so the first index is
            # representative for all of them.
            source_tz = next(iter(self._data.values())).index.tz
            if source_tz is not None:
                t_lo = t_lo.tz_localize(source_tz) if t_lo.tzinfo is None else t_lo.tz_convert(source_tz)
                t_hi = t_hi.tz_localize(source_tz) if t_hi.tzinfo is None else t_hi.tz_convert(source_tz)

            if t_lo >= t_hi:
                raise ValueError(
                    f"xlim start ({t_lo}) must be earlier than end ({t_hi})."
                )

        # ------------------------------------------------------------------ #
        # Figure / axes                                                        #
        # ------------------------------------------------------------------ #
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # ------------------------------------------------------------------ #
        # Retrieve and plot each series                                        #
        # ------------------------------------------------------------------ #
        for ref in vars:
            if aligned:
                series, var = self._get_aligned(ref)
            else:
                series, var = self._get_source(ref)

            # Slice to xlim window before plotting so y-autoscale is correct
            if t_lo is not None:
                series = series.loc[t_lo:t_hi]

            src   = ref[0]
            label = f"{src.capitalize()} — {var.name}"

            kw = {"linewidth":0.8} | plot_kw
            ax.plot(
                series.index,
                series.values,
                label=label,
                **kw
            )

        # ------------------------------------------------------------------ #
        # Axes formatting                                                      #
        # ------------------------------------------------------------------ #
        # x-axis: set limits explicitly if requested, otherwise let matplotlib
        # auto-scale from the plotted data
        if t_lo is not None:
            ax.set_xlim(t_lo, t_hi)

        # Sensible date tick formatting that adapts to the visible span
        # ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
        ax.figure.autofmt_xdate(rotation=30, ha="right")

        # y-axis label: ignore empty-string units (lazy callers who only
        # annotate one source). If all non-empty units agree, use that unit.
        # Conflict between two distinct non-empty units → stay silent.
        all_units  = [self.get_variable(*ref).unit for ref in vars]
        all_names  = [self.get_variable(*ref).name for ref in vars]
        nonempty_units = {u for u in all_units if u}
        unique_names   = list(dict.fromkeys(all_names))   # preserve order

        if len(nonempty_units) <= 1:
            unit     = next(iter(nonempty_units), "")
            unit_str = f" [{unit}]" if unit else ""
            name_str = unique_names[0] if len(unique_names) == 1 else ""
            ylabel   = f"{name_str}{unit_str}".strip()
            if ylabel:
                ax.set_ylabel(ylabel)

        ax.legend(
            loc="upper right",
            framealpha=0.85,
            edgecolor="0.7",
        )
        ax.grid(True, linestyle=":", linewidth=0.6, color="0.75")

        return ax

    def error_metrics(
        self,
        x: tuple[str, str],
        y: tuple[str, str],
        by: "Literal['month', 'season', 'year'] | str" = "month",
        func: "list[str] | None" = None,
        circular: bool = False,
        seasons: "dict[str, list[int]] | None" = None,
        step: float | None = None,
    ) -> "pd.DataFrame":
        """
        Compute pairwise error metrics between two aligned sources, grouped by
        month, season, year, or an arbitrary aligned variable — mirroring the
        ``TimeSeries.statistics`` interface.

        Requires ``.align()`` to have been called first.

        Parameters
        ----------
        x : tuple[str, str]
            (source_key, column_name) for the reference (e.g. observations).
        y : tuple[str, str]
            (source_key, column_name) for the candidate (e.g. model).
        by : {"month", "season", "year"} or tuple[str, str], default "month"
            Grouping strategy:

            - ``"month"``  — calendar months (Jan … Dec).
            - ``"season"`` — meteorological seasons, or custom via ``seasons``.
            - ``"year"``   — calendar years present in the aligned record.
            - ``(source_key, column_name)`` — bin by an arbitrary aligned
            variable (e.g. ``("model", "hs")``). Bin edges are derived
            automatically via ``infer_step``, or set explicitly with
            ``step``.

        func : list[str], optional
            Error metrics to compute per group. Defaults to
            ``["bias", "mae", "rmse", "si", "r", "count", "%"]``.
            Available tokens:

            - ``"bias"``  — mean signed error (y − x)
            - ``"mae"``   — mean absolute error
            - ``"rmse"``  — root mean square error
            - ``"si"``    — scatter index (RMSE / mean(x))
            - ``"r"``     — Pearson correlation coefficient
            - ``"count"`` — number of collocated pairs in the group
            - ``"%"``     — group count as % of total collocated pairs

        circular : bool, default False
            If True, treat the variable as a circular quantity (e.g. wave or
            wind direction in degrees). Differences are computed as::

                diff = ((y − x) + 180) % 360 − 180

            which wraps all differences into [−180°, 180°], so that e.g.
            350° vs 10° gives −20° rather than 340°. Note that ``"si"`` and
            ``"r"`` are suppressed when ``circular=True`` since they are not
            meaningful for angular errors.
        seasons : dict[str, list[int]], optional
            Custom season map, e.g.
            ``{"Winter": [12, 1, 2], "Summer": [6, 7, 8]}``.
            Only used when ``by="season"``.
        step : float, optional
            Bin width when grouping by a continuous variable (``by`` is a
            tuple). If None, derived automatically via ``infer_step``.

        Returns
        -------
        pd.DataFrame
            Groups as columns (always includes an ``"All"`` column),
            metrics as rows.

        Raises
        ------
        RuntimeError
            If ``.align()`` has not been called.
        KeyError
            If either variable reference is not found in the aligned data.
        TypeError
            If ``by`` is a tuple but does not resolve to an aligned column.

        Examples
        --------
        >>> tsv.align(reference="model")
        >>> tsv.error_metrics(x=("buoy", "hs"), y=("model", "hs"), by="month")
        >>> tsv.error_metrics(
        ...     x=("buoy", "hs"), y=("model", "hs"),
        ...     by="season",
        ...     seasons={"Winter": [12, 1, 2], "Summer": [6, 7, 8]},
        ...     func=["bias", "rmse", "r"],
        ... )
        >>> # Group by model Hs bins — e.g. to see how bias varies with sea state
        >>> tsv.error_metrics(
        ...     x=("buoy", "hs"), y=("model", "hs"),
        ...     by=("model", "hs"),
        ...     step=0.5,
        ... )
        >>> # Directional comparison
        >>> tsv.error_metrics(
        ...     x=("buoy", "mdir"), y=("model", "mdir"),
        ...     by="month",
        ...     circular=True,
        ... )
        """
        import numpy as np

        # ------------------------------------------------------------------ #
        # Guards and retrieval                                                 #
        # ------------------------------------------------------------------ #
        self._require_alignment()
        x_series, _ = self._get_aligned(x)
        y_series, _ = self._get_aligned(y)

        paired = pd.DataFrame({"x": x_series, "y": y_series}).dropna()
        if paired.empty:
            raise ValueError("No non-NaN collocated pairs found after alignment.")

        n_total = len(paired)

        # ------------------------------------------------------------------ #
        # Defaults                                                             #
        # ------------------------------------------------------------------ #
        if func is None:
            func = ["bias", "mae", "rmse", "si", "r", "count", "%"]
            if circular:
                # si and r are not meaningful for angular quantities
                func = ["bias", "mae", "rmse", "count"]

        # ------------------------------------------------------------------ #
        # Build groups                                                         #
        # ------------------------------------------------------------------ #
        groups: dict[str, pd.DataFrame] = {}

        if by == "month":
            month_names = {
                1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
                5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
                9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
            }
            for m, name in month_names.items():
                mask = paired.index.month == m
                if mask.any():
                    groups[name] = paired.loc[mask]
            col_label = "Month"

        elif by == "season":
            default_seasons = {
                "DJF": [12, 1, 2],
                "MAM": [3, 4, 5],
                "JJA": [6, 7, 8],
                "SON": [9, 10, 11],
            }
            season_map = seasons or default_seasons
            for name, months in season_map.items():
                mask = paired.index.month.isin(months)
                if mask.any():
                    groups[name] = paired.loc[mask]
            col_label = "Season"

        elif by == "year":
            for yr in sorted(paired.index.year.unique()):
                mask = paired.index.year == yr
                groups[str(yr)] = paired.loc[mask]
            col_label = "Year"

        elif isinstance(by, tuple):
            # ------------------------------------------------------------------ #
            # Continuous variable binning                                          #
            # Retrieve the grouping series from the aligned data, reindex onto    #
            # the paired index (dropna already removed NaTs), then bin with       #
            # infer_step / user-supplied step.                                    #
            # ------------------------------------------------------------------ #
            by_series, by_var = self._get_aligned(by)
            by_series = by_series.reindex(paired.index)

            _step = step if step is not None else infer_step(by_series.dropna())
            edges  = bin_edges(by_series.dropna(), _step)
            labels = [f"{e:.10g}" for e in edges[:-1]]
            bins   = pd.cut(by_series, bins=edges, labels=labels, right=False)

            for label, grp_idx in bins.groupby(bins, observed=False).groups.items():
                sub = paired.loc[paired.index.isin(grp_idx)]
                if not sub.empty:
                    groups[str(label)] = sub

            unit_str  = f" [{by_var.unit}]" if by_var.unit else ""
            col_label = f"{by_var.symbol}{unit_str}"

        else:
            raise ValueError(
                f"Unsupported grouping {by!r}. "
                f"Use 'month', 'season', 'year', or a (source, col) tuple."
            )

        groups["All"] = paired

        # ------------------------------------------------------------------ #
        # Metric computation                                                   #
        # ------------------------------------------------------------------ #
        def _compute_diff(df: "pd.DataFrame") -> "np.ndarray":
            raw = df["y"].values.astype(float) - df["x"].values.astype(float)
            if circular:
                return ((raw + 180) % 360) - 180
            return raw

        def _agg_row(df: "pd.DataFrame") -> dict:
            xv   = df["x"].values.astype(float)
            diff = _compute_diff(df)
            n    = len(df)

            result: dict[str, float] = {}

            for f in func:
                if f == "bias":
                    result["Bias"] = float(np.mean(diff))
                elif f == "mae":
                    result["MAE"] = float(np.mean(np.abs(diff)))
                elif f == "rmse":
                    result["RMSE"] = float(np.sqrt(np.mean(diff ** 2)))
                elif f == "si":
                    mx = float(np.mean(xv))
                    result["SI"] = float(np.sqrt(np.mean(diff ** 2)) / mx) if mx != 0 else np.nan
                elif f == "r":
                    yv = df["y"].values.astype(float)
                    result["R"] = float(np.corrcoef(xv, yv)[0, 1]) if n > 1 else np.nan
                elif f == "count":
                    result["Count"] = n
                elif f == "%":
                    result["%"] = 100 * n / n_total
                else:
                    raise ValueError(
                        f"Unknown metric '{f}'. Valid options: "
                        f"'bias', 'mae', 'rmse', 'si', 'r', 'count', '%'."
                    )

            return result

        # ------------------------------------------------------------------ #
        # Assemble output DataFrame                                            #
        # ------------------------------------------------------------------ #
        df = pd.DataFrame(
            {label: _agg_row(grp) for label, grp in groups.items()}
        )
        df.columns.name = col_label
        return df.T

