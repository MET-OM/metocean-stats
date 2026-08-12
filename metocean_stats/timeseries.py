"""
TimeSeries
----------
Core module for single-point metocean statistical analysis.

A TimeSeries wraps a pandas DataFrame with user-supplied metadata about
variables of interest, and provides a unified interface for:
  - Descriptive and directional statistics
  - Extreme value analysis (via UnivariateEVA)
  - Joint/conditional analysis (via BivariateEVA)
  - Persistence and operability
  - Seasonality and trends

Out of scope (handled by separate modules):
  - Multi-level profiles (wind shear, current profiles)
  - Spatial / map-based analysis

Example usage
-------------
>>> import pandas as pd
>>> from metocean_stats import TimeSeries, Variable, CMAConfig
>>> from metocean_stats.CMA import predefined
>>>
>>> df = pd.read_csv("hindcast.csv", index_col=0, parse_dates=True)
>>>
>>> hs  = Variable("hs",  "Significant wave height", "H_s",     "m")
>>> tp  = Variable("tp",  "Peak wave period",        "T_p",     "s")
>>> mwd = Variable("mwd", "Mean wave direction",     "\\theta", "°")
>>> ws  = Variable("ws",  "Wind speed",              "U_{10}", "m/s")
>>>
>>> ts = TimeSeries(
...     df=df,
...     primary=hs,
...     direction=mwd,
...     secondary=[tp, ws],
...     cma_pairs=[
...         CMAConfig(vars=("hs", "tp"), model=predefined.get_DNVGL_Hs_Tz),
...         CMAConfig(vars=("ws", "hs"), model=predefined.get_DNVGL_Hs_U),
...     ]
... )
>>>
>>> # Simple statistics
>>> ts.describe()
>>> ts.monthly_stats()
>>> ts.rose()
>>>
>>> # Extreme value analysis
>>> ts.EVA.get_extremes()
>>> ts.EVA.fit(AM_dist=["GEV"], POT_dist=["GP"])
>>> ts.EVA.plot_return_value_comparison("omni", return_periods=[1, 10, 50, 100])
>>>
>>> # Joint analysis
>>> ts.CMA["hs-tp"].fit()
>>> ts.CMA["hs-tp"].plot_isodensity_contours("omni", RVE=rve)
"""

from __future__ import annotations

import warnings
import typing
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .EVA import UnivariateEVA
from .CMA import BivariateEVA
from .utils import groupby_month, groupby_sector, groupby_season, infer_step, bin_edges, aggregate_statistics
from .plots import plot_rose

# ---------------------------------------------------------------------------
# Variable — semantics container for a single metocean variable
# ---------------------------------------------------------------------------

@dataclass
class Variable:
    """
    Metadata container for a single metocean variable.

    Parameters
    ----------
    col : str
        Column name in the DataFrame. Must match exactly.
    name : str
        Human-readable name used in table headers and axis titles.
        E.g. "Significant wave height".
    symbol : str
        LaTeX-compatible symbol (without $ delimiters).
        E.g. "H_s", "T_p", "U_{10}".
    unit : str
        Physical unit string. E.g. "m", "m/s", "s", "°".
    label : str, optional
        Short display label for legend entries or compact tables.
        Defaults to symbol if not provided.

    Examples
    --------
    >>> hs  = Variable("hs",  "Significant wave height", "H_s",     "m")
    >>> tp  = Variable("tp",  "Peak wave period",        "T_p",     "s")
    >>> mwd = Variable("mwd", "Mean wave direction",     "\\theta", "°")
    >>> ws  = Variable("ws",  "Wind speed",              "U_{10}", "m/s")
    """
    col: str
    name: str
    symbol: str
    unit: str
    label: str = None

    def __post_init__(self):
        if not self.col:
            raise ValueError("Variable.col must be a non-empty string.")
        if not self.name:
            raise ValueError(f"Variable '{self.col}': name must be a non-empty string.")
        if self.label is None:
            self.label = self.symbol

    @property
    def axis_label(self) -> str:
        """Formatted axis label: 'Name, $symbol$ (unit)'."""
        return f"{self.name}, ${self.symbol}$ ({self.unit})"

    @property
    def short_label(self) -> str:
        """Shorter variable label: '$symbol$ (unit)'."""
        return f"${self.symbol}$ ({self.unit})"

    @property
    def math_label(self) -> str:
        """Symbol wrapped in LaTeX math delimiters: '$symbol$'."""
        return f"${self.symbol}$"

    def __repr__(self) -> str:
        return (
            f"Variable(col='{self.col}', name='{self.name}', "
            f"symbol='{self.symbol}', unit='{self.unit}')"
        )


# ---------------------------------------------------------------------------
# CMAConfig — pairs a variable order with its joint model description
# ---------------------------------------------------------------------------

@dataclass
class CMAConfig:
    """
    Configuration for a single BivariateEVA (CMA) instance.

    Parameters
    ----------
    vars : tuple[str, str]
        Ordered pair of column names as they enter the model.
        Order matters: it must match the dependency structure of the model.
        E.g. ("hs", "tp") for a model where Hs is marginal and Tp|Hs is
        conditional. Both columns must exist in the TimeSeries (primary or
        secondary).
    model : callable
        A predefined model factory from metocean_stats.CMA.predefined,
        or any callable returning (dist_descriptions, fit_descriptions, semantics).
        E.g. predefined.get_DNVGL_Hs_Tz
    key : str, optional
        Custom key for ts.CMA dict lookup.
        Auto-generated as "var1-var2" if not provided.

    Examples
    --------
    >>> from metocean_stats.CMA import predefined
    >>> CMAConfig(vars=("hs", "tp"), model=predefined.get_DNVGL_Hs_Tz)
    >>> CMAConfig(vars=("ws", "hs"), model=predefined.get_DNVGL_Hs_U)
    >>> CMAConfig(vars=("hs", "tp"), model=my_model, key="hs-tp-custom")
    """
    vars: tuple[str, str]
    model: Callable
    key: str = None

    def __post_init__(self):
        if len(self.vars) != 2:
            raise ValueError(
                f"CMAConfig.vars must be a 2-tuple of column names, got {self.vars!r}"
            )
        if not callable(self.model):
            raise TypeError(
                f"CMAConfig.model must be callable, got {type(self.model)}"
            )
        if self.key is None:
            self.key = f"{self.vars[0]}-{self.vars[1]}"


# ---------------------------------------------------------------------------
# TimeSeries
# ---------------------------------------------------------------------------

class TimeSeries:
    """
    Single-point metocean time series wrapper.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. Index must be (or be convertible to) pandas DatetimeIndex.
    primary : Variable
        Primary metocean variable (e.g. Hs, wind speed).
    direction : Variable, optional
        Direction variable (0–360°). Many directional methods will raise a
        error if this is not provided.
    secondary : list[Variable], optional
        Secondary variables (e.g. Tp, Tm01, wind speed).
        Available for joint analysis via ts.CMA.
    sectors : int, default 12
        Number of directional sectors used in EVA and CMA sub-modules.
    cma : CMAConfig, optional
        Joint model configuration. An ordered variable pair and 
        its associated predefined model. If None, ts.CMA will be
        None, and no joint model is initialised.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        primary: Variable,
        direction: Variable = None,
        secondary: list[Variable] = None,
        sectors: int = 12,
        cma: CMAConfig = None,
    ):
        # ------------------------------------------------------------------ #
        # Variable validation                                                  #
        # ------------------------------------------------------------------ #
        primary   = TimeSeries._coerce_variable(primary)
        direction = TimeSeries._coerce_variable(direction) if direction is not None else None
        secondary = [TimeSeries._coerce_variable(s) 
                     for i, s in enumerate(secondary)] if secondary else []
 
        # Auto-register any extra df columns not already covered by a declared
        # variable. Uses the column name as name, symbol, and fallback label.
        declared_cols = (
            {primary.col}
            | ({direction.col} if direction else set())
            | {s.col for s in secondary}
        )
        for col in df.columns:
            if col not in declared_cols:
                secondary.append(
                    TimeSeries._coerce_variable(col)
                )
 
        all_vars = [primary]
        if direction is not None:
            all_vars.append(direction)
        if secondary:
            all_vars.extend(secondary)
 
        self._validate_variables(all_vars)
 
        # ------------------------------------------------------------------ #
        # Data ingestion                                                       #
        # ------------------------------------------------------------------ #
        self.data = self._parse_and_validate(
            df,
            primary_col=primary.col,
            direction_col=direction.col if direction else None,
            secondary_cols=[s.col for s in secondary] if secondary else None,
        )
 
        # ------------------------------------------------------------------ #
        # Metadata                                                             #
        # ------------------------------------------------------------------ #
        self.primary   = primary
        self.direction = direction
        self.secondary = list(secondary) if secondary else []
        self.sectors   = sectors
 
        # Convenience lookup: col name → Variable
        self._variables: dict[str, Variable] = {v.col: v for v in all_vars}
 
        # ------------------------------------------------------------------ #
        # Sub-module: Extreme Value Analysis                                   #
        # ------------------------------------------------------------------ #
        self.EVA = UnivariateEVA(
            data=self.data,
            var=self.primary.col,
            var_dir=self.direction.col if self.direction else None,
            var_name=self.primary.name,
            var_symbol=self.primary.symbol,
            var_unit=self.primary.unit,
            sectors=self.sectors,
        )
 
        # ------------------------------------------------------------------ #
        # Sub-module: Joint / Conditional Analysis                             #
        # ------------------------------------------------------------------ #
        self.CMA = self._init_cma(cma)

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                         #
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _coerce_variable(v: Variable | str) -> Variable:
        """Promote a bare column name string to a Variable with placeholder semantics."""
        if isinstance(v, Variable):
            return v
        if isinstance(v, str):
            return Variable(col=v, name=v, symbol=v, unit="")
        raise TypeError(
            f"Expected a Variable or a column name string, got {type(v)}. "
            "Example: Variable('hs', 'Significant wave height', 'H_s', 'm') or just 'hs'."
        )

    @staticmethod
    def _validate_variables(variables: list[Variable]) -> None:
        """Check that all entries are Variable instances and column names are unique."""
        for v in variables:
            if not isinstance(v, Variable):
                raise TypeError(
                    f"Expected a Variable instance, got {type(v)}. "
                    "Wrap your column name: Variable('hs', 'Significant wave height', 'H_s', 'm')"
                )
        cols = [v.col for v in variables]
        seen, dupes = set(), []
        for c in cols:
            if c in seen:
                dupes.append(c)
            seen.add(c)
        if dupes:
            raise ValueError(
                f"Duplicate column name(s) across Variable definitions: {dupes}. "
                "Each Variable must reference a unique column."
            )

    def _resolve_columns(
        self,
        columns: list[str] | None,
        default: list[Variable] | None = None,
    ) -> list[Variable]:
        """Resolve a list of column name strings to Variable instances."""
        if columns is None:
            return default if default is not None else [self.primary] + self.secondary
        resolved = []
        for col in columns:
            if col not in self._variables:
                raise ValueError(
                    f"Column '{col}' not found. "
                    f"Available columns: {list(self._variables.keys())}"
                )
            resolved.append(self._variables[col])
        return resolved

    def _parse_and_validate(
        self,
        df: pd.DataFrame,
        primary_col: str,
        direction_col: str | None,
        secondary_cols: list[str] | None,
    ) -> pd.DataFrame:
        """
        Validate inputs and return a clean, sorted DataFrame containing only
        the requested columns. Raises on critical errors, warns on recoverable
        issues.
        """
        # Index must be datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df = df.copy()
                df.index = pd.to_datetime(df.index)
            except Exception as exc:
                raise TypeError(
                    "DataFrame index could not be converted to DatetimeIndex."
                ) from exc

        df = df.sort_index()

        # Duplicate timestamps
        n_dupes = df.index.duplicated().sum()
        if n_dupes:
            warnings.warn(
                f"Index contains {n_dupes} duplicate timestamp(s). "
                "First occurrence kept; consider investigating before analysis."
            )
            df = df[~df.index.duplicated(keep="first")]

        # Required columns
        required = [primary_col]
        if direction_col is not None:
            required.append(direction_col)
        if secondary_cols:
            required.extend(secondary_cols)

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Column(s) not found in DataFrame: {missing}")

        # Direction range check
        if direction_col is not None:
            dir_min = df[direction_col].min()
            dir_max = df[direction_col].max()
            if dir_min < 0 or dir_max > 360:
                warnings.warn(
                    f"Direction column '{direction_col}' contains values outside "
                    f"[0, 360]: min={dir_min:.2f}, max={dir_max:.2f}."
                )

        # NaN check — warn only, do not drop (user's responsibility)
        for col in required:
            n_nan = df[col].isna().sum()
            if n_nan:
                pct = 100 * n_nan / len(df)
                warnings.warn(
                    f"Column '{col}' contains {n_nan} NaN value(s) ({pct:.1f}%). "
                    "Some methods may require gap-filling first."
                )

        return df[required].copy()

    def _init_cma(self, cma: CMAConfig | None) -> BivariateEVA | None:
        if cma is None:
            return None

        if not isinstance(cma, CMAConfig):
            raise TypeError(
                f"cma must be a CMAConfig instance, got {type(cma)}. "
                "Example: CMAConfig(vars=('hs', 'tp'), model=predefined.get_DNVGL_Hs_Tz)"
            )

        available = {self.primary.col} | {s.col for s in self.secondary}
        missing = [v for v in cma.vars if v not in available]
        if missing:
            raise ValueError(
                f"CMAConfig '{cma.key}': variable(s) {missing} not found in "
                f"TimeSeries. Available: {sorted(available)}"
            )

        if self.direction is None:
            warnings.warn(
                "No direction column provided — BivariateEVA will be initialised "
                "without directional splitting. Sector-based CMA methods will be unavailable."
            )

        return BivariateEVA(
            data=self.data,
            var1=cma.vars[0],
            var2=cma.vars[1],
            var_dir=self.direction.col if self.direction else None,
            sectors=self.sectors,
            model=cma.model,
        )

    def _require_direction(self, method_name: str) -> None:
        """Raise a clear error if direction was not provided at initialisation."""
        if self.direction is None:
            raise AttributeError(
                f"{method_name}() requires a direction variable, "
                "but none was provided at initialisation."
            )

    def _resolve_column(self, col: str | None, default: Variable) -> Variable:
        """
        Resolve a user-supplied column name string to a Variable.
        Falls back to `default` if col is None.
        Raises if col is provided but not known to this TimeSeries.
        """
        if col is None:
            return default
        if col not in self._variables:
            raise ValueError(
                f"Column '{col}' is not registered on this TimeSeries. "
                f"Known columns: {list(self._variables.keys())}"
            )
        return self._variables[col]

    def _get_groups(
        self,
        by: str,
        col: str,
        sectors: int = None,
        seasons: dict = None,
        step: float = None,
    ) -> dict:
        if by == "direction":
            data = self.data[[col, self.direction.col]].dropna()
            return groupby_sector(data, var_dir=self.direction.col,
                                sectors=sectors or self.sectors, var=col)

        data = self.data[[col]].dropna()

        if by == "month":
            return groupby_month(data, var=col)
        elif by == "season":
            return groupby_season(data, seasons=seasons, var=col)
        elif by == "year":
            return {str(y): g[col] for y, g in data.groupby(data.index.year)}
        elif by == "week":
            all_weeks = sorted(data.index.isocalendar().week.unique())
            return {
                str(int(w)): data.loc[data.index.isocalendar().week == w, col]
                for w in all_weeks
            }
        elif by == "day":
            all_days = sorted(data.index.day_of_year.unique())
            return {
                str(int(d)): data.loc[data.index.day_of_year == d, col]
                for d in all_days
            }
        elif by == "hour":
            return {
                str(h): data.loc[data.index.hour == h, col]
                for h in range(24)
                if (data.index.hour == h).any()
            }
        elif by in self._variables:
            s_by  = self.data[by].dropna()
            _step = step or infer_step(s_by)
            edges  = bin_edges(s_by, _step)
            labels = [f"{e:.10g}" for e in edges[:-1]]
            bins   = pd.cut(self.data[by], bins=edges, labels=labels, right=False)
            return {k: g[col].dropna() for k, g in self.data.groupby(bins, observed=False)}
        else:
            raise ValueError(
                f"'by' must be one of 'month', 'season', 'direction', 'year', "
                f"'week', 'day', 'hour', or a column name. Got '{by}'."
            )
    # ---------------------------------------------------------------------- #
    # Properties                                                               #
    # ---------------------------------------------------------------------- #

    @property
    def duration(self) -> pd.Timedelta:
        """Total record length as a pandas Timedelta."""
        return self.data.index[-1] - self.data.index[0]

    @property
    def timestep(self) -> pd.Timedelta:
        """Median time step inferred from the index."""
        return self.data.index.to_series().diff().median()

    @property
    def timestep_hours(self) -> float:
        """Median time step in decimal hours."""
        return self.timestep.total_seconds() / 3600.0

    @property
    def n_years(self) -> float:
        """Record length in fractional years."""
        return self.duration.total_seconds() / (365.2425 * 24 * 3600)

    @property
    def columns(self) -> list[str]:
        """All active column names (primary + direction + secondary)."""
        cols = [self.primary.col]
        if self.direction:
            cols.append(self.direction.col)
        cols.extend([s.col for s in self.secondary])
        return cols

    # ---------------------------------------------------------------------- #
    # Data utilities                                                           #
    # ---------------------------------------------------------------------- #

    def resample(
        self,
        rule: str,
        how: typing.Literal["mean", "max", "min", "median"] = "mean",
        inplace: bool = False,
    ) -> TimeSeries | None:
        """
        Resample the time series to a new frequency.

        Parameters
        ----------
        rule : str
            Pandas offset alias, e.g. "1h", "3h", "D".
        how : str
            Aggregation method: "mean", "max", "min", or "median".
        inplace : bool
            If True, modifies this instance. If False, returns a new TimeSeries.
        """
        raise NotImplementedError

    def fill_gaps(
        self,
        method: typing.Literal["interpolate", "ffill", "bfill"] = "interpolate",
        limit: int = None,
        inplace: bool = False,
    ) -> TimeSeries | None:
        """
        Fill missing values / irregular gaps in the time series.

        Parameters
        ----------
        method : str
            Gap-filling strategy: "interpolate", "ffill", or "bfill".
        limit : int, optional
            Maximum number of consecutive NaNs to fill.
        inplace : bool
            If True, modifies this instance. If False, returns a new TimeSeries.
        """
        raise NotImplementedError

    def trim(
        self,
        start: str | pd.Timestamp = None,
        end: str | pd.Timestamp = None,
        inplace: bool = False,
    ) -> TimeSeries | None:
        """
        Trim the time series to a given date range.

        Parameters
        ----------
        start : str or pd.Timestamp, optional
            Start of the desired range. Keeps from beginning if None.
        end : str or pd.Timestamp, optional
            End of the desired range. Keeps to end if None.
        inplace : bool
            If True, modifies this instance. If False, returns a new TimeSeries.
        """
        raise NotImplementedError

    # ---------------------------------------------------------------------- #
    # Descriptive statistics                                                   #
    # ---------------------------------------------------------------------- #

    def availability(self) -> pd.DataFrame:
        """
        Data availability summary.

        Returns a table with per-column counts of valid records, NaN records,
        and percentage availability, plus overall record start/end/duration.
        """
        raise NotImplementedError

    def describe(self) -> pd.DataFrame:
        """
        Extended descriptive statistics for all columns.

        Extends pandas .describe() with skewness, kurtosis, and common
        metocean percentiles. Direction is excluded (circular statistics
        are not meaningful here).

        Returns
        -------
        pd.DataFrame
            Statistics as rows, variables as columns. Column headers use
            the Variable symbol and unit for readability.
        """
        cols = [self.primary] + self.secondary
        percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

        records = {}
        for var in cols:
            s = self.data[var.col].dropna()
            records[var.short_label] = {
                "Count"   : s.count(),
                "Missing" : self.data[var.col].isna().sum(),
                "Mean"    : s.mean(),
                "Std"     : s.std(),
                "Skewness": s.skew(),
                "Kurtosis": s.kurt(),
                "Min"     : s.min(),
                **{f"p{int(p*100):02d}": s.quantile(p) for p in percentiles},
                "Max"     : s.max(),
            }

        df = pd.DataFrame(records)
        df.loc["Count"]   = df.loc["Count"].astype(int)
        df.loc["Missing"] = df.loc["Missing"].astype(int)
        return df

    def statistics(
        self,
        by: typing.Literal["month", "season", "direction", "year"] | str = "month",
        column: str = None,
        func: list[str] = None,
        percentiles: list[float] = None,
        sectors: int = None,
        seasons: dict[str, list[int]] = None,
        step: float = None,
    ) -> pd.DataFrame:
        """
        Aggregated statistics grouped by month, season, direction, or year.
        Summary statistics are followed by percentile rows.

        Parameters
        ----------
        by : {"month", "season", "direction", "year"}
            Grouping strategy. Defaults to "month".
        func : list[str], optional
            Aggregation functions.
            Defaults to ["mean", "std", "min", "max", "count", "availability"].
        percentiles : list[float], optional
            Percentiles to append below summary stats.
            Defaults to [1, 10, 50, 90, 99]. Pass [] to omit.
        sectors : int, optional
            Directional sectors. Only used when by="direction".
        seasons : dict[str, list[int]], optional
            Custom season map. Only used when by="season".

        Returns
        -------
        pd.DataFrame
            Groups as columns (+ "All"), statistics and percentiles as rows.
        """
        if by == "direction":
            self._require_direction("statistics")

        func        = func        or ["min", "mean", "max", "std", "count", "%"]
        percentiles = percentiles if percentiles is not None else [1, 10, 50, 90, 99]

        cols = self._resolve_columns(column, default=[self.primary])
        var  = cols[0].col

        # --- build groups ---------------------------------------------------- #
        groups        = self._get_groups(by, var, sectors=sectors, seasons=seasons, step=step)
        groups["All"] = self.data[var].dropna()

        # --- aggregation ----------------------------------------------------- #
        n_total = len(self.data)

        def _agg_row(s: pd.Series) -> dict:
            result = {}
            for p in percentiles:
                result[f"P{p:02d}"] = s.quantile(p / 100)
            for f in func:
                if   f == "mean":  result["Mean"]  = s.mean()
                elif f == "std":   result["Std"]   = s.std()
                elif f == "min":   result["Min"]   = s.min()
                elif f == "max":   result["Max"]   = s.max()
                elif f == "count": result["Count"] = len(s)
                elif f == "%":     result["%"]     = 100 * len(s) / n_total
            return result

        df = pd.DataFrame(
            {label: _agg_row(s) for label, s in groups.items()}
        )
        df.columns.name = self._variables[by].short_label if by in self._variables else None
        return df


    def nonexceedance(
        self,
        by: typing.Literal["month", "season", "direction", "year"] | str = "month",        
        thresholds: list[float] = None,
        step: float = None,
        column: str = None,
        sectors: int = None,
        seasons: dict[str, list[int]] = None,
        by_step: float = None,
    ) -> pd.DataFrame:
        """
        Non-exceedance table P(X ≤ x) in percent, grouped by month, season,
        direction, or year.

        Convention
        ----------
        by="direction" : values are P(dir=d AND X ≤ x), i.e. conditional CDF
            scaled by sector frequency. The last row of each sector column gives
            P(dir=d) — the sector frequency. "All" column is the marginal CDF.
        by="month" / "season" / "year" : values are P(X ≤ x | group), i.e.
            conditional CDF within each group. Last row is 100 for all columns.

        Parameters
        ----------
        by : {"month", "season", "direction", "year"}
            Grouping strategy. Defaults to "month".
        thresholds : list[float], optional
            Exceedance thresholds. Defaults to evenly spaced values from 0 (or
            min) to max, with step inferred from the data range.
        step : float, optional
            Step size for auto-generated thresholds. Ignored if thresholds is
            provided.
        column : str, optional
            Column to analyse. Defaults to primary.
        sectors : int, optional
            Number of directional sectors. Only used when by="direction".
        seasons : dict[str, list[int]], optional
            Custom season map. Only used when by="season".

        Returns
        -------
        pd.DataFrame
            Thresholds as index, groups + "All" as columns. Values in percent.
                """
        if by == "direction":
                self._require_direction("nonexceedance")

        var   = self._resolve_column(column, self.primary)
        s_all = self.data[var.col].dropna()
        n_all = len(s_all)

        # --- thresholds ------------------------------------------------------ #
        if thresholds is None:
            _step      = step or infer_step(s_all)
            thresholds = np.arange(0, s_all.max() + _step, _step)

        # --- groups ---------------------------------------------------------- #
        if by == var.col:
            raise ValueError("'by' column and 'column' cannot be the same variable.")

        groups = self._get_groups(by, var.col, sectors=sectors, seasons=seasons, step=by_step)

        # --- compute --------------------------------------------------------- #
        result = {}
        joint  = by == "direction" or by in self._variables

        for label, grp in groups.items():
            grp = grp.dropna()
            if joint:
                sector_freq   = len(grp) / n_all
                result[label] = [100 * (grp <= t).mean() * sector_freq for t in thresholds]
            else:
                result[label] = [100 * (grp <= t).mean() for t in thresholds]

        result["All"] = [100 * (s_all <= t).mean() for t in thresholds]

        df = pd.DataFrame(result, index=pd.Index(thresholds, name=var.short_label))
        df.columns.name = self._variables[by].short_label if by in self._variables else None
        return df

    def frequency_table(
        self,
        row: typing.Literal["month", "season", "direction", "year"] | str = None,
        col: typing.Literal["month", "season", "direction", "year"] | str = None,
        step_row: float = None,
        step_col: float = None,
        values: typing.Literal["count", "frequency"] = "frequency",
        margins: bool = True,
        sectors: int = None,
        seasons_row: dict[str, list[int]] = None,
        seasons_col: dict[str, list[int]] = None,
    ) -> pd.DataFrame:
        """
        Joint frequency table between two axes. Each axis can be either a
        grouping strategy ("month", "season", "direction", "year") or a
        column name (binned by value).

        Defaults to primary variable on rows and first secondary on columns
        (classic scatter diagram). If only one is a grouping keyword the other
        defaults to the primary variable.

        Parameters
        ----------
        row : str
            Row axis — grouping keyword or column name. Defaults to primary.
        col : str
            Column axis — grouping keyword or column name. Defaults to first
            secondary variable, or primary if none exists.
        step_row : float, optional
            Bin width for row axis when row is a column name.
        step_col : float, optional
            Bin width for col axis when col is a column name.
        values : {"count", "frequency"}
            Return raw counts or percentage of total. Defaults to "frequency".
        margins : bool
            Append "All" row and column. Defaults to True.
        sectors : int, optional
            Directional sectors. Used when row or col is "direction".
        seasons_row : dict[str, list[int]], optional
            Custom season map for row axis.
        seasons_col : dict[str, list[int]], optional
            Custom season map for col axis.

        Returns
        -------
        pd.DataFrame
            Row groups as index, column groups as columns, values as counts or %.
        """
        _KEYWORDS = {"month", "season", "direction", "year"}

        # --- defaults -------------------------------------------------------- #
        if row is None:
            row = self.primary.col
        if col is None:
            col = self.secondary[0].col if self.secondary else self.primary.col

        if row == col:
            raise ValueError("'row' and 'col' must be different.")

        if "direction" in (row, col):
            self._require_direction("frequency_table")

        # --- helper: label and get groups ------------------------------------ #
        def _groups_and_label(axis, step, seasons):
            if axis in _KEYWORDS:
                groups = self._get_groups(axis, self.primary.col,
                                        sectors=sectors, seasons=seasons)
                label  = axis.capitalize()
            else:
                var = self._resolve_column(axis, self.primary)
                groups = self._get_groups(axis, var.col, step=step)
                label  = var.short_label
            return groups, label

        row_groups, row_label = _groups_and_label(row, step_row, seasons_row)
        col_groups, col_label = _groups_and_label(col, step_col, seasons_col)

        # --- build matrix ---------------------------------------------------- #
        # For keyword groups the index is the timestamp index.
        # For variable bins the group values are series — use their index.
        def _idx(grp):
            return grp.index if hasattr(grp, "index") else grp.index

        n = len(self.data.dropna())

        records = {}
        for col_label_, col_grp in col_groups.items():
            records[col_label_] = {}
            for row_label_, row_grp in row_groups.items():
                records[col_label_][row_label_] = len(
                    col_grp.index.intersection(row_grp.index)
                )

        df = pd.DataFrame(records).fillna(0).astype(int)

        if margins:
            df["All"]      = df.sum(axis=1)
            df.loc["All"]  = df.sum(axis=0)

        if values == "frequency":
            df = (100 * df / n).round(2)
        elif values != "count":
            raise ValueError(
                f"'values' must be 'frequency' or 'count', got '{values}'"
            )

        df.index.name   = row_label
        df.columns.name = col_label

        return df
    
    def block_maxima(
        self,
        by: typing.Literal["year", "month", "season", "direction"] | str = "direction",
        column: str = None,
        func: list[str] = None,
        percentiles: list[float] = None,
        year_start: int = 1,
        min_coverage: float = 0.9,
        sectors: int = None,
        seasons: dict[str, list[int]] = None,
        step: float = None,
    ) -> pd.DataFrame:
        """
        Statistics of block maxima grouped by year, month, season, direction,
        or bins of another variable.

        Parameters
        ----------
        by : {"year", "month", "season", "direction"} or str
            Grouping strategy. Pass a column name to bin by value.
            Defaults to "year".
        column : str, optional
            Column to compute maxima of. Defaults to primary variable.
        func : list[str], optional
            Statistics to compute over the block maxima.
            Defaults to ["min", "mean", "max"].
        percentiles : list[float], optional
            Percentiles to compute over the block maxima.
            Defaults to None (no percentiles). Pass [50, 90, 99] for example.
        year_start : int, optional
            Month (1-12) that defines the start of a "year" block.
            Used for ALL grouping methods to ensure consistent year boundaries.
            Defaults to 1 (calendar year).
            Set to 6 for June-to-June blocks per DNV/NORSOK convention.
        min_coverage : float, optional
            Minimum fraction of expected timesteps required to include a
            year block. Blocks below this threshold are dropped.
            Defaults to 0.9.
        sectors : int, optional
            Number of directional sectors. Used when by="direction".
        seasons : dict[str, list[int]], optional
            Custom season map. Used when by="season".
        step : float, optional
            Bin width when by is a column name. Inferred if not provided.

        Returns
        -------
        pd.DataFrame
            Rows are groups, columns are statistics (min, mean, max, ...).
            An "All" row with overall statistics is appended.
            Dropped year blocks are listed in a warning.
        """
        func        = func or ["min", "mean", "max"]
        percentiles = percentiles if percentiles is not None else []
        var         = self._resolve_column(column, default=self.primary)
        series      = self.data[var.col]

        # Merge percentiles into func list
        combined_func = func.copy()
        for p in percentiles:
            combined_func.append(f"p{int(p):02d}")

        # ------------------------------------------------------------------ #
        # STEP 1: Always group by year first (respecting year_start)        #
        # ------------------------------------------------------------------ #
        _MONTH_ABBR = ["JAN","FEB","MAR","APR","MAY","JUN",
                    "JUL","AUG","SEP","OCT","NOV","DEC"]

        freq = f"YS-{_MONTH_ABBR[year_start - 1]}"
        
        # Create year labels and filter by coverage
        dt_h = self.timestep_hours
        year_groups = {}
        dropped = []
        
        for period, group in self.data.resample(freq):
            expected = (365.25 * 24) / dt_h
            coverage = group[var.col].count() / expected
            label = str(period.year if year_start == 1
                        else f"{period.year}-{period.year+1}")
            
            if coverage < min_coverage:
                dropped.append(f"{label} ({coverage:.0%})")
                continue
            
            year_groups[label] = group
        
        if dropped:
            warnings.warn(
                f"Dropped {len(dropped)} year block(s) below {min_coverage:.0%} "
                f"coverage: {', '.join(dropped)}"
            )

        # ------------------------------------------------------------------ #
        # STEP 2: Extract maxima per year, then group by secondary criterion#
        # ------------------------------------------------------------------ #
        if by == "year":
            # Simple case: just one maximum per year
            maxima_series = pd.Series(
                {year_label: year_df[var.col].max() 
                for year_label, year_df in year_groups.items()},
                name=var.col
            )
            
        else:
            # Complex case: extract max per (year, secondary_group) combination
            maxima_dict = {}
            
            for year_label, year_df in year_groups.items():
                # Get secondary groups within this year
                if by == "direction":
                    self._require_direction("block_maxima")
                    secondary_groups = groupby_sector(
                        year_df[[var.col, self.direction.col]].dropna(),
                        var_dir=self.direction.col,
                        sectors=sectors or self.sectors,
                        var=var.col
                    )
                elif by == "month":
                    secondary_groups = groupby_month(
                        year_df[[var.col]].dropna(),
                        var=var.col
                    )
                elif by == "season":
                    secondary_groups = groupby_season(
                        year_df[[var.col]].dropna(),
                        seasons=seasons,
                        var=var.col
                    )
                elif by in self._variables:
                    # Bin by another variable
                    s_by = year_df[by].dropna()
                    _step = step or infer_step(s_by)
                    edges = bin_edges(s_by, _step)
                    labels = [f"{e:.10g}" for e in edges[:-1]]
                    bins = pd.cut(year_df[by], bins=edges, labels=labels, right=False)
                    secondary_groups = {
                        k: g[var.col].dropna() 
                        for k, g in year_df.groupby(bins, observed=False)
                    }
                else:
                    raise ValueError(
                        f"'by' must be one of 'year', 'month', 'season', 'direction', "
                        f"or a column name. Got '{by}'."
                    )
                
                # Extract maximum for each secondary group
                for sec_label, sec_data in secondary_groups.items():
                    if len(sec_data) > 0:
                        maxima_dict[sec_label] = maxima_dict.get(sec_label, [])
                        maxima_dict[sec_label].append(sec_data.max())
            
            # Now maxima_dict contains {group_label: [max_year1, max_year2, ...]}
            # Convert to series of all maxima per group
            maxima_series = {
                label: pd.Series(values, name=var.col)
                for label, values in maxima_dict.items()
            }

        # ------------------------------------------------------------------ #
        # STEP 3: Aggregate statistics over maxima                           #
        # ------------------------------------------------------------------ #
        records = {}
        
        if by == "year":
            # Simple case: one value per year
            for label, max_val in maxima_series.items():
                records[label] = aggregate_statistics(pd.Series([max_val]), combined_func)
            records["All"] = aggregate_statistics(maxima_series, combined_func)
        else:
            # Complex case: multiple maxima per group
            for label, max_series in maxima_series.items():
                records[label] = aggregate_statistics(max_series, combined_func)
            
            # "All": concatenate all maxima across all groups
            all_maxima = pd.concat(list(maxima_series.values()))
            records["All"] = aggregate_statistics(all_maxima, combined_func)

        df = pd.DataFrame(records).T  # Transpose: groups as rows, stats as columns

        df.index.name   = var.short_label
        df.columns.name = None

        return df

    def persistence(
        self,
        threshold: float,
        above: bool = True,
        by: typing.Literal["month", "season", "direction", "year"] | str = "month",
        column: str = None,
        stats: list[str] = None,
        percentiles: list[float] = None,
        sectors: int = None,
        seasons: dict[str, list[int]] = None,
        by_step: float = None,
    ) -> pd.DataFrame:
        """
        Statistics of exceedance events: periods where the variable is
        continuously above (or below) a given threshold.

        Events are identified on the full time series before any grouping.
        An event is attributed to a group based on its *start* timestamp, so
        events that straddle group boundaries (e.g. a storm starting in January
        and ending in February) are counted once, in full, under the group
        where they began.

        Parameters
        ----------
        threshold : float
            The exceedance threshold, in the same units as the variable.
        above : bool, default True
            If True, events are periods where value > threshold.
            If False, events are periods where value < threshold.
        by : {"month", "season", "direction", "year"} or str
            Grouping strategy for attribution. Defaults to "month".
            Pass a column name to group by bins of another variable.
        column : str, optional
            Column to analyse. Defaults to the primary variable.
        stats : list[str], optional
            Duration statistics to report. Defaults to ["mean", "std", "max"].
            Supported: "mean", "std", "min", "max", "median", "count".
        percentiles : list[float], optional
            Additional percentiles of event duration to append as rows.
            E.g. [50, 90, 99]. Defaults to None.
        sectors : int, optional
            Number of directional sectors. Used only when by="direction".
        seasons : dict[str, list[int]], optional
            Custom season map. Used only when by="season".
        by_step : float, optional
            Bin width when by is a column name. Inferred if not provided.

        Returns
        -------
        pd.DataFrame
            Rows are groups plus a "Yearly" summary row. Columns are duration
            statistics plus a "Yearly" column (mean events per year).
            For by="year" the "Yearly" column contains the raw annual count
            per row; the "Yearly" summary row is always total events / n_years.
            Duration values are in hours.

        Notes
        -----
        Event identification assumes a regular time step. NaN values are
        treated as below-threshold, so gaps break any ongoing event.
        ``timestep_hours`` is used to convert event lengths to hours.

        Examples
        --------
        >>> ts.persistence(threshold=5.0, by="month")
        >>> ts.persistence(threshold=10.0, above=False, by="season")
        >>> ts.persistence(threshold=8.0, by="direction", sectors=8)
        >>> ts.persistence(threshold=5.0, by="year")
        >>> ts.persistence(threshold=5.0, column="ws", by="hs", by_step=1.0)
        """
        if by == "direction":
            self._require_direction("persistence")

        var         = self._resolve_column(column, self.primary)
        stats       = stats or ["mean", "std", "max"]
        percentiles = percentiles or []
        dt_h        = self.timestep_hours

        # ------------------------------------------------------------------ #
        # STEP 1: Build event mask and event-start on the full series         #
        # ------------------------------------------------------------------ #
        s    = self.data[var.col]
        mask = (s > threshold) if above else (s < threshold)
        mask = mask.fillna(False)

        # event_start: True at every False → True transition
        event_start = mask & ~mask.shift(1, fill_value=False)

        # Assign a monotonically increasing event ID to every in-event timestep
        event_id = event_start.cumsum().where(mask)  # NaN outside events

        # One row per event: start timestamp + duration
        start_timestamps = s.index[event_start]
        durations_h      = event_id.groupby(event_id).count() * dt_h

        events = pd.DataFrame({
            "start":    start_timestamps,
            "duration": durations_h.values,
        }).reset_index(drop=True)

        # ------------------------------------------------------------------ #
        # STEP 2: Build a timestamp → group-label lookup from _get_groups     #
        # ------------------------------------------------------------------ #
        groups_dict = self._get_groups(
            by, var.col, sectors=sectors, seasons=seasons, step=by_step
        )

        # ts_to_label maps every timestamp in self.data to its group label
        ts_to_label: dict[pd.Timestamp, str] = {}
        for label, grp_series in groups_dict.items():
            for ts in grp_series.index:
                ts_to_label[ts] = label

        # Attribute each event by its start timestamp
        if not events.empty:
            events["group"] = events["start"].map(ts_to_label)
            events = events[events["group"].notna()]   # drop unclassifiable starts
        else:
            events["group"] = pd.Series(dtype=str)

        # ------------------------------------------------------------------ #
        # STEP 3: Canonical group ordering                                    #
        # ------------------------------------------------------------------ #
        _MONTH_ORDER  = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"]
        _SEASON_ORDER = ["DJF","MAM","JJA","SON"]

        if by == "month":
            ordered_labels = [m for m in _MONTH_ORDER if m in groups_dict]
        elif by == "season":
            _smap = seasons or {"DJF":[12,1,2],"MAM":[3,4,5],
                                "JJA":[6,7,8],"SON":[9,10,11]}
            ordered_labels = [s for s in _SEASON_ORDER if s in _smap]
        else:
            ordered_labels = list(groups_dict.keys())

        # ------------------------------------------------------------------ #
        # STEP 4: Aggregate duration statistics per group                     #
        # ------------------------------------------------------------------ #
        stat_label_map = {
            "mean":   "Mean (h)",
            "std":    "Std (h)",
            "min":    "Min (h)",
            "max":    "Max (h)",
            "median": "Median (h)",
            "count":  "Count",
        }

        def _agg(durations: pd.Series) -> dict:
            row = {}
            if durations.empty:
                for st in stats:
                    row[stat_label_map.get(st, st)] = np.nan
                for p in percentiles:
                    row[f"p{int(p):02d}"] = np.nan
                return row
            for st in stats:
                lbl = stat_label_map.get(st, st)
                if   st == "mean":   row[lbl] = durations.mean()
                elif st == "std":    row[lbl] = durations.std()
                elif st == "min":    row[lbl] = durations.min()
                elif st == "max":    row[lbl] = durations.max()
                elif st == "median": row[lbl] = durations.median()
                elif st == "count":  row[lbl] = float(len(durations))
                else:
                    raise ValueError(
                        f"Unknown stat '{st}'. "
                        "Supported: 'mean', 'std', 'min', 'max', 'median', 'count'."
                    )
            for p in percentiles:
                row[f"p{int(p):02d}"] = durations.quantile(p / 100)
            return row

        records = {}
        event_counts = {}

        for label in ordered_labels:
            grp_durations        = events.loc[events["group"] == label, "duration"]
            records[label]       = _agg(grp_durations)
            event_counts[label]  = len(grp_durations)

        # ------------------------------------------------------------------ #
        # STEP 5: "Yearly" column — always total events / n_years (float)    #
        # ------------------------------------------------------------------ #
        n_years = max(self.n_years, 1e-9)

        if by == "year":
            # Each group IS a year: raw count for that row
            for label in ordered_labels:
                records[label]["Yearly"] = float(event_counts[label])
        else:
            for label in ordered_labels:
                records[label]["Yearly"] = event_counts[label] / n_years

        # "Yearly" summary row: always total events / n_years regardless of `by`
        yearly_row           = _agg(events["duration"])
        yearly_row["Yearly"] = len(events) / n_years
        records["Yearly"]    = yearly_row

        # ------------------------------------------------------------------ #
        # STEP 6: Assemble DataFrame                                          #
        # ------------------------------------------------------------------ #
        df = pd.DataFrame(records).T

        stat_cols = [stat_label_map.get(st, st) for st in stats
                     if stat_label_map.get(st, st) in df.columns]
        pct_cols  = [f"p{int(p):02d}" for p in percentiles
                     if f"p{int(p):02d}" in df.columns]
        df = df[stat_cols + pct_cols + ["Yearly"]]

        df.index.name = (
            self._variables[by].short_label if by in self._variables
            else by.capitalize()
        )
        direction_str    = "above" if above else "below"
        df.columns.name  = f"{var.short_label} {direction_str} {threshold} {var.unit}"
        return df

    def trends(
        self,
        by: str = "month",
        column: str = None,
        stat: str = "mean",
        method: "list[str] | None" = None,
        confidence: float = 0.95,
        intercept: bool = True,
        per: int = 100,
        sectors: "int | None" = None,
        seasons: "dict[str, list[int]] | None" = None,
        step: "float | None" = None,
        min_years: int = 5,
        min_entries_per_year: int = 100,
    ) -> "pd.DataFrame":
        """
        Inter-annual trend analysis grouped by month, season, direction, or a
        continuous variable.

        For each group defined by ``by``, the chosen statistic is computed per
        calendar year, and one or more linear trend models are fitted to those
        yearly values. This captures e.g. "is the mean January Hs increasing
        over time?".

        Parameters
        ----------
        by : {"month", "season", "direction"} or str, default "month"
            Grouping strategy. ``"year"`` is not valid here — trends are by
            definition computed over years. A continuous column name is also
            accepted and will be binned via ``infer_step`` / ``step``.
        column : str, optional
            Column to analyse. Defaults to the primary variable.
        stat : str, default "mean"
            Statistic aggregated per year within each group. Accepts
            ``"mean"``, ``"median"``, ``"std"``, ``"min"``, ``"max"``, or a
            percentile in the form ``"pNN"`` (e.g. ``"p90"``).
        method : list[str], optional
            Trend methods to apply. Any subset of
            ``["ols", "theil_sen", "kendall"]``. Defaults to all three.

            - ``"ols"``       — Ordinary least-squares. Returns slope,
            intercept (optional), and R².
            - ``"theil_sen"`` — Theil-Sen estimator. Returns median slope,
            intercept (optional), and lower/upper slope confidence bounds.
            - ``"kendall"``   — Kendall-tau test. Returns τ and p-value.

        confidence : float, default 0.95
            Confidence level for the Theil-Sen slope interval. Must be in
            [0.5, 1.0].
        intercept : bool, default True
            Whether to include the intercept in the output for OLS and
            Theil-Sen.
        per : int, default 100
            Scale slopes to this number of years. E.g. ``per=100`` reports
            [unit / 100 years], ``per=10`` reports [unit / decade]. Does not
            affect R², Kendall τ, or p-values.
        sectors : int, optional
            Number of directional sectors. Only used when ``by="direction"``.
        seasons : dict[str, list[int]], optional
            Custom season map, e.g. ``{"Winter": [12, 1, 2]}``.
            Only used when ``by="season"``.
        step : float, optional
            Bin width when grouping by a continuous variable.
            Inferred automatically if not provided.
        min_years : int, default 5
            Minimum number of years with valid data required to fit a trend.
            Groups below this threshold produce NaN metrics and a warning.
        min_entries_per_year : int, default 100
            Minimum number of non-NaN samples a given year must contain
            within the group to be included in the trend fit.

        Returns
        -------
        pd.DataFrame
            Groups as rows, metrics as columns (always includes an ``"All"``
            row for the pooled trend). Slope units are
            [variable unit / ``per`` years]. When grouping by a continuous
            variable, groups with all-NaN metrics are dropped.

        Raises
        ------
        ValueError
            If ``by="year"``, if ``stat`` is not a recognised token, or if
            any entry in ``method`` is unknown.
        AttributeError
            If ``by="direction"`` but no direction variable was provided.

        Examples
        --------
        >>> ts.trends(by="month")
        >>> ts.trends(by="season", stat="p90", method=["ols", "kendall"])
        >>> ts.trends(by="direction", column="hs", sectors=8)
        >>> ts.trends(by="month", stat="p90", method=["theil_sen"], confidence=0.9)
        >>> ts.trends(by="month", per=10)  # per decade
        >>> ts.trends(by="hs", step=0.5, min_samples_per_year=200)
        """
        from scipy.stats import linregress, theilslopes, kendalltau

        # ------------------------------------------------------------------ #
        # Validation                                                           #
        # ------------------------------------------------------------------ #
        if by == "year":
            raise ValueError(
                "by='year' is not supported — trends are computed over years "
                "within each group. Use by='month', 'season', or 'direction'."
            )

        if by == "direction":
            self._require_direction("trends")

        method = method or ["ols", "theil_sen", "kendall"]

        unknown_methods = [m for m in method if m not in {"ols", "theil_sen", "kendall"}]
        if unknown_methods:
            raise ValueError(
                f"Unknown method(s): {unknown_methods}. "
                f"Choose from 'ols', 'theil_sen', 'kendall'."
            )

        if not (0.5 <= confidence <= 1.0):
            raise ValueError(
                f"'confidence' must be between 0.5 and 1.0, got {confidence}."
            )

        var = self._resolve_columns(column, default=[self.primary])[0]
        col = var.col
        u   = var.unit

        # ------------------------------------------------------------------ #
        # Row labels (self-documenting units in the output index)             #
        # ------------------------------------------------------------------ #
        slope_unit = f"{u} / {per} yr" if u else f"/ {per} yr"

        row_names = {
            "ols_slope":      f"OLS slope [{slope_unit}]",
            "ols_intercept":  f"OLS intercept [{u}]" if u else "OLS intercept",
            "ols_r2":         "OLS R²",
            "ts_slope":       f"TS slope [{slope_unit}]",
            "ts_intercept":   f"TS intercept [{u}]" if u else "TS intercept",
            "ts_slope_lower": f"TS slope lower [{slope_unit}]",
            "ts_slope_upper": f"TS slope upper [{slope_unit}]",
            "kendall_tau":    "Kendall τ",
            "kendall_p":      "Kendall p",
        }

        # ------------------------------------------------------------------ #
        # Statistic helper                                                     #
        # ------------------------------------------------------------------ #
        def _apply_stat(s: "pd.Series") -> float:
            if stat == "mean":   return float(s.mean())
            if stat == "median": return float(s.median())
            if stat == "std":    return float(s.std())
            if stat == "min":    return float(s.min())
            if stat == "max":    return float(s.max())
            if stat.startswith("p"):
                try:
                    p = float(stat[1:]) / 100.0
                    if 0.0 <= p <= 1.0:
                        return float(s.quantile(p))
                except ValueError:
                    pass
            raise ValueError(
                f"Unknown stat '{stat}'. Use 'mean', 'median', 'std', "
                f"'min', 'max', or 'pNN' (e.g. 'p90')."
            )

        # ------------------------------------------------------------------ #
        # NaN row — returned for groups with insufficient data                #
        # ------------------------------------------------------------------ #
        def _nan_row() -> dict:
            row = {}
            if "ols" in method:
                row[row_names["ols_slope"]] = np.nan
                if intercept:
                    row[row_names["ols_intercept"]] = np.nan
                row[row_names["ols_r2"]] = np.nan
            if "theil_sen" in method:
                row[row_names["ts_slope"]] = np.nan
                if intercept:
                    row[row_names["ts_intercept"]] = np.nan
                row[row_names["ts_slope_lower"]] = np.nan
                row[row_names["ts_slope_upper"]] = np.nan
            if "kendall" in method:
                row[row_names["kendall_tau"]] = np.nan
                row[row_names["kendall_p"]]   = np.nan
            return row

        # ------------------------------------------------------------------ #
        # Per-group trend fitting                                              #
        # ------------------------------------------------------------------ #
        def _fit(label: str, s: "pd.Series") -> dict:
            # Aggregate per year, dropping years below the sample threshold
            yearly = (
                s.groupby(s.index.year)
                .apply(lambda g: _apply_stat(g) if g.count() >= min_entries_per_year else np.nan)
                .dropna()
            )

            if len(yearly) < min_years:
                warnings.warn(
                    f"Group '{label}' has only {len(yearly)} year(s) with at least "
                    f"{min_entries_per_year} valid samples "
                    f"(minimum required: {min_years}). "
                    f"Trend metrics set to NaN.",
                    UserWarning,
                    stacklevel=4,
                )
                return _nan_row()

            x   = yearly.index.values.astype(float)
            y   = yearly.values.astype(float)
            row = {}

            if "ols" in method:
                slope, interc, r_value, _, _ = linregress(x, y)
                row[row_names["ols_slope"]] = slope * per
                if intercept:
                    row[row_names["ols_intercept"]] = interc
                row[row_names["ols_r2"]] = r_value ** 2

            if "theil_sen" in method:
                slope, interc, lower, upper = theilslopes(y, x, confidence)
                row[row_names["ts_slope"]]       = slope * per
                if intercept:
                    row[row_names["ts_intercept"]] = interc
                row[row_names["ts_slope_lower"]] = lower * per
                row[row_names["ts_slope_upper"]] = upper * per

            if "kendall" in method:
                tau, p_value = kendalltau(x, y)
                row[row_names["kendall_tau"]] = tau
                row[row_names["kendall_p"]]   = p_value

            return row

        # ------------------------------------------------------------------ #
        # Build groups and assemble output                                     #
        # ------------------------------------------------------------------ #
        groups        = self._get_groups(by, col, sectors=sectors, seasons=seasons, step=step)
        groups["All"] = self.data[col].dropna()

        df = pd.DataFrame(
            {label: _fit(label, s) for label, s in groups.items()}
        ).T

        df.index.name = (
            self._variables[by].short_label if by in self._variables
            else by.capitalize()
        )

        # Drop all-NaN rows when grouping by a continuous variable — these
        # are sparse edge bins where no year meets the sample threshold
        if by in self._variables:
            df = df.dropna(how="all")

        return df

    # ---------------------------------------------------------------------- #
    # Figures — univariate                                                     #
    # ---------------------------------------------------------------------- #
    def plot_rose(
        self,
        magnitude: str = None,
        direction: str = None,
        by: typing.Literal["month", "season", "year", "all"] | str = "month",
        seasons: dict[str, list[int]] = None,
        step: float = None,
        n_dir_bins: int = 16,
        mag_step: float = None,
        cmap: str = "YlOrRd",
        calm_threshold: float = 0.0,
        min_entries: int = 100,
        ncols: int = 4,
        panels_per_fig: int | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> "plt.Figure | list[plt.Figure]":
        """
        Rose plot(s) of a magnitude/direction pair, optionally split into
        sub-panels by month, season, year, or bins of any other variable.

        All panels share identical magnitude bin edges and radial scale so
        they are directly visually comparable. Groups with fewer than
        ``min_entries`` observations are skipped with a warning.

        Parameters
        ----------
        magnitude : str, optional
            Column name of the magnitude variable. Defaults to the primary
            variable.
        direction : str, optional
            Column name of the direction variable. Defaults to
            ``self.direction``. Must be provided here or at construction.
        by : {"month", "season", "year", "all"} or str, default "month"
            Grouping strategy:

            - ``"month"``  — one panel per calendar month.
            - ``"season"`` — one panel per season (custom via ``seasons``).
            - ``"year"``   — one panel per year in the record.
            - ``"all"``    — a single panel for the full record.
            - any column name — panels for bins of that variable (step
            auto-inferred or set via ``step``).

        seasons : dict[str, list[int]], optional
            Custom season map. Only used when ``by="season"``.
        step : float, optional
            Bin step for numeric ``by`` grouping.
        n_dir_bins : int, default 16
            Number of compass sectors inside each rose panel.
        mag_step : float, optional
            Magnitude bin step. Auto-selected via ``infer_step`` if None.
        cmap : str, default "YlOrRd"
            Colormap for magnitude classes.
        calm_threshold : float, default 0.0
            Observations at or below this value are counted as calm.
        min_entries : int, default 100
            Minimum number of valid paired observations required to draw a
            panel. Groups below this threshold are skipped with a warning.
        ncols : int, default 4
            Maximum number of panels per row within each figure.
        panels_per_fig : int, optional
            If set, panels are distributed across multiple figures each
            holding at most ``panels_per_fig`` panels. Useful for fitting
            output into a report. When None, a single figure is returned.
        figsize : tuple[float, float], optional
            Figure size in inches. Defaults to matplotlib's rcParams
            ``figure.figsize``.

        Returns
        -------
        plt.Figure or list[plt.Figure]
            A single Figure when ``panels_per_fig`` is None, otherwise a
            list of Figures.

        Raises
        ------
        AttributeError
            If no direction column is available.
        ValueError
            If no groups survive the ``min_entries`` threshold.
        """
        # ------------------------------------------------------------------ #
        # Resolve magnitude and direction columns                              #
        # ------------------------------------------------------------------ #
        mag_var = self._resolve_column(magnitude, default=self.primary)

        if direction is not None:
            dir_var = self._resolve_column(direction, default=None)
        elif self.direction is not None:
            dir_var = self.direction
        else:
            raise AttributeError(
                "plot_rose() requires a direction column. "
                "Either provide direction= here or supply one at construction."
            )

        # ------------------------------------------------------------------ #
        # Build groups                                                         #
        # ------------------------------------------------------------------ #
        if by == "all":
            groups_mag = {"All": self.data[mag_var.col].dropna()}
        else:
            groups_mag = self._get_groups(
                by=by,
                col=mag_var.col,
                seasons=seasons,
                step=step,
            )

        # ------------------------------------------------------------------ #
        # Reformat labels when grouping by a variable column                 #
        # ------------------------------------------------------------------ #
        if by not in ("all", "month", "season", "year") and by in self._variables:
            by_var = self._variables[by]
            unit   = f" {by_var.unit}" if by_var.unit else ""
            keys   = list(groups_mag.keys())
            # Keys are already sorted left-edge strings — zip with the next key
            # to form "lo–hi" labels without any float conversion.
            new_keys = [
                f"${by_var.symbol}$ {lo}–{hi}{unit}"
                for lo, hi in zip(keys, keys[1:])
            ] + [f"${by_var.symbol}$ >{keys[-1]}{unit}"]
            groups_mag = dict(zip(new_keys, groups_mag.values()))

        # ------------------------------------------------------------------ #
        # Filter groups by min_entries                                         #
        # ------------------------------------------------------------------ #
        valid_groups: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for label, mag_series in groups_mag.items():
            idx      = mag_series.index
            dir_vals = self.data.loc[idx, dir_var.col]
            paired   = pd.DataFrame({"mag": mag_series, "dir": dir_vals}).dropna()
            n        = len(paired)

            if n < min_entries:
                warnings.warn(
                    f"plot_rose: group '{label}' has only {n} valid paired "
                    f"observations (min_entries={min_entries}) — skipped.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            valid_groups[label] = (
                paired["mag"].values.astype(float),
                paired["dir"].values.astype(float),
            )

        if not valid_groups:
            raise ValueError(
                f"No groups have enough data to plot "
                f"(min_entries={min_entries}). "
                f"Lower min_entries or check your data."
            )

        # ------------------------------------------------------------------ #
        # Shared magnitude bins                                                #
        # ------------------------------------------------------------------ #
        all_mag = np.concatenate([mv for mv, _ in valid_groups.values()])
        active  = all_mag[all_mag > calm_threshold]
        p99     = float(np.percentile(active, 99)) if len(active) else 1.0

        _mag_step = mag_step or infer_step(pd.Series(active), target=7)
        mag_bins  = np.arange(0, p99 + _mag_step, _mag_step)

        # ------------------------------------------------------------------ #
        # Shared radial scale                                                  #
        # ------------------------------------------------------------------ #
        def _sector_max(mag_vals, dir_vals):
            sector_width = 360.0 / n_dir_bins
            dir_edges    = np.linspace(
                -sector_width / 2, 360 - sector_width / 2, n_dir_bins + 1
            )
            n_total  = len(mag_vals)
            active_m = mag_vals > calm_threshold
            dir_norm = dir_vals[active_m] % 360.0
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

        r_max = max(
            _sector_max(mv, dv) for mv, dv in valid_groups.values()
        ) * 1.15

        # ------------------------------------------------------------------ #
        # Shared legend patches (built once, reused across figures)           #
        # ------------------------------------------------------------------ #
        n_mag    = len(mag_bins) - 1
        cm_obj   = plt.get_cmap(cmap, n_mag)
        colors   = [cm_obj(i / n_mag) for i in range(n_mag)]
        unit_str = f" {mag_var.unit}" if mag_var.unit else ""
        patches  = [
            mpatches.Patch(
                facecolor=colors[i],
                label=f"{mag_bins[i]:g}–{mag_bins[i+1]:g}{unit_str}",
            )
            for i in range(n_mag)
        ]

        # ------------------------------------------------------------------ #
        # Split panels across figures                                          #
        # ------------------------------------------------------------------ #
        panel_items  = list(valid_groups.items())
        n_panels     = len(panel_items)
        ppf          = panels_per_fig or n_panels   # all in one figure by default
        panel_chunks = [
            panel_items[i : i + ppf]
            for i in range(0, n_panels, ppf)
        ]

        _figsize = figsize or None

        # ------------------------------------------------------------------ #
        # Draw figures                                                         #
        # ------------------------------------------------------------------ #
        def _make_figure(chunk):
            n       = len(chunk)
            _ncols  = min(ncols, n)
            _nrows  = int(np.ceil(n / _ncols))

            fig, axes = plt.subplots(
                _nrows, _ncols,
                figsize=_figsize,
                subplot_kw={"projection": "polar"},
            )
            axes_flat = np.array(axes).ravel().tolist() if n > 1 else [axes]

            for ax, (label, (mag_vals, dir_vals)) in zip(axes_flat, chunk):
                plot_rose(
                    mag_vals, dir_vals, ax,
                    n_dir_bins=n_dir_bins,
                    mag_bins=mag_bins,
                    cmap=cmap,
                    r_max=r_max,
                    calm_threshold=calm_threshold,
                    title=f"{label}  (n={len(mag_vals):,})",
                    legend=False,
                    mag_label=mag_var.name,
                    mag_unit=mag_var.unit,
                )

            for ax in axes_flat[n:]:
                ax.set_visible(False)

            fig.legend(
                handles=patches,
                title=mag_var.name,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=min(n_mag, 6),
                frameon=False,
                handlelength=1.2,
            )
            fig.suptitle(f"{mag_var.name} rose by {by}", y=1.01)
            fig.tight_layout()
            return fig

        figures = [_make_figure(chunk) for chunk in panel_chunks]

        return figures[0] if panels_per_fig is None else figures

    def statistics_2d(
        self,
        row: typing.Literal["month", "season", "year", "direction"] | str = "month",
        col: typing.Literal["month", "season", "year", "direction"] | str = "direction",
        column: str = None,
        stat: str = "mean",
        step_row: float = None,
        step_col: float = None,
        margins: bool = True,
        sectors: int = None,
        seasons_row: dict[str, list[int]] = None,
        seasons_col: dict[str, list[int]] = None,
    ) -> pd.DataFrame:
        """
        2D summary statistics table: one variable aggregated across two
        independent grouping dimensions.

        Each cell contains a single statistic computed from all observations
        that fall into the corresponding (row group, col group) intersection.
        This is a double groupby — both axes are computed independently via
        ``_get_groups`` and then intersected.

        Parameters
        ----------
        row : str, default "month"
            Grouping for the row axis. Any value accepted by ``_get_groups``:
            ``"month"``, ``"season"``, ``"year"``, ``"direction"``, or any
            column name (binned by value, step controlled by ``step_row``).
        col : str, default "direction"
            Grouping for the column axis. Same options as ``row``.
        column : str, optional
            Column to aggregate. Defaults to the primary variable.
        stat : str, default "mean"
            Statistic to compute per cell. Supported:

            - ``"mean"``, ``"std"``, ``"min"``, ``"max"``
            - ``"count"`` — number of observations in the cell
            - ``"p{n}"``  — percentile, e.g. ``"p90"``, ``"p99"``

        step_row : float, optional
            Bin step for the row axis when ``row`` is a numeric column.
            Auto-inferred via ``infer_step`` if None.
        step_col : float, optional
            Bin step for the col axis when ``col`` is a numeric column.
            Auto-inferred via ``infer_step`` if None.
        margins : bool, default True
            If True, append an ``"All"`` row and column containing the
            statistic computed over the full marginal group.
        sectors : int, optional
            Number of directional sectors when ``row`` or ``col`` is
            ``"direction"``. Defaults to ``self.sectors``.
        seasons_row : dict[str, list[int]], optional
            Custom season map for the row axis when ``row="season"``.
        seasons_col : dict[str, list[int]], optional
            Custom season map for the col axis when ``col="season"``.

        Returns
        -------
        pd.DataFrame
            Row groups as index, column groups as columns. Cell values are
            the requested statistic. NaN where a cell has no observations.

        Examples
        --------
        >>> ts.statistics_2d()                                      # month × direction
        >>> ts.statistics_2d(row="month", col="tp", stat="mean")
        >>> ts.statistics_2d(row="year",  col="month", stat="p90")
        >>> ts.statistics_2d(row="hs",    col="tp",    stat="count", margins=False)
        """
        # ------------------------------------------------------------------ #
        # Resolve variable                                                     #
        # ------------------------------------------------------------------ #
        var = self._resolve_column(column, default=self.primary)

        # ------------------------------------------------------------------ #
        # Parse stat                                                           #
        # ------------------------------------------------------------------ #
        def _apply_stat(s: pd.Series) -> float:
            if s.empty:
                return np.nan
            if stat == "mean":   return s.mean()
            elif stat == "std":  return s.std()
            elif stat == "min":  return s.min()
            elif stat == "max":  return s.max()
            elif stat == "count": return float(len(s))
            elif stat.startswith("p"):
                try:
                    return s.quantile(float(stat[1:]) / 100)
                except ValueError:
                    pass
            raise ValueError(
                f"Unknown stat '{stat}'. Use 'mean', 'std', 'min', 'max', "
                f"'count', or 'p<n>' (e.g. 'p90')."
            )

        # ------------------------------------------------------------------ #
        # Build row and column groups via _get_groups                          #
        # ------------------------------------------------------------------ #
        row_groups = self._get_groups(
            by=row, col=var.col,
            sectors=sectors, seasons=seasons_row, step=step_row,
        )
        col_groups = self._get_groups(
            by=col, col=var.col,
            sectors=sectors, seasons=seasons_col, step=step_col,
        )

        if margins:
            row_groups["All"] = self.data[var.col].dropna()
            col_groups["All"] = self.data[var.col].dropna()

        # ------------------------------------------------------------------ #
        # Compute cell values: observations in row_group ∩ col_group           #
        # ------------------------------------------------------------------ #
        records = {}
        for row_label, row_series in row_groups.items():
            row_idx = row_series.index
            records[row_label] = {}
            for col_label, col_series in col_groups.items():
                shared_idx = row_idx.intersection(col_series.index)
                cell_vals  = self.data.loc[shared_idx, var.col].dropna()
                records[row_label][col_label] = _apply_stat(cell_vals)

        df = pd.DataFrame(records).T
        df.index.name   = row
        df.columns.name = col

        return df

    def plot_statistics(
            self,
            by: "typing.Literal['month', 'season', 'direction', 'year', 'week', 'day', 'hour'] | str" = "month",
            column: str = None,
            func: "list[str] | None" = None,
            percentiles: "list[float] | None" = None,
            sectors: "int | None" = None,
            seasons: "dict[str, list[int]] | None" = None,
            step: "float | None" = None,
            cmap: str = "viridis",
            table: bool = False,
            figsize: "tuple[float, float] | None" = None,
            ax: "plt.Axes | None" = None,
        ) -> "plt.Axes":
            """
            Line plot of ``statistics`` — aggregated statistics for one variable
            grouped by month, season, direction, year, week, day, or hour.

            Each requested statistic is drawn as a separate line. Markers are added
            when the number of groups is 12 or fewer. Lines are coloured using a
            colormap so they are visually distinct without relying on linestyle.
            An optional summary table can be appended below the axes.

            Parameters
            ----------
            by : {"month", "season", "direction", "year", "week", "day", "hour"} or str
                Grouping strategy. Passed directly to ``statistics``.
            column : str, optional
                Column to analyse. Defaults to the primary variable.
            func : list[str], optional
                Statistics to plot. Defaults to
                ``["mean", "p75", "p90", "p95", "p99"]``.
                Any token accepted by ``statistics`` is valid.
            percentiles : list[float], optional
                Passed to ``statistics``. If None, percentiles are inferred from
                ``func`` so that any ``pNN`` token in ``func`` is automatically
                included.
            sectors : int, optional
                Directional sectors. Only used when ``by="direction"``.
            seasons : dict[str, list[int]], optional
                Custom season map. Only used when ``by="season"``.
            step : float, optional
                Bin width for continuous variable grouping.
            cmap : str, default "viridis"
                Matplotlib colormap used to colour the stat lines.
            table : bool, default False
                If True, append a summary table below the plot. Raises a
                ValueError when ``by`` is ``"week"`` or ``"day"`` since the
                number of columns would be too large to render legibly.
            figsize : tuple[float, float], optional
                Figure size. Ignored when ``ax`` is supplied.
            ax : plt.Axes, optional
                Axes to draw on. If None, a new figure is created.

            Returns
            -------
            plt.Axes
            """
            # ------------------------------------------------------------------ #
            # Validate table request                                               #
            # ------------------------------------------------------------------ #
            if table and by in ("week", "day"):
                raise ValueError(
                    f"table=True is not supported for by='{by}' — "
                    f"the number of columns ({52 if by == 'week' else 365}) "
                    f"is too large to render legibly."
                )

            # ------------------------------------------------------------------ #
            # Defaults                                                             #
            # ------------------------------------------------------------------ #
            func = func or ["mean", "P75", "P90", "P95", "P99"]

            pct_from_func = []
            for f in func:
                if f.lower().startswith("p") and f[1:].isdigit():
                    pct_from_func.append(int(f[1:]))
            if percentiles is None:
                percentiles = pct_from_func

            # ------------------------------------------------------------------ #
            # Resolve grouping-by variable metadata (if by is a column name)      #
            # ------------------------------------------------------------------ #
            by_var = self._variables.get(by, None)   # Variable or None

            # ------------------------------------------------------------------ #
            # Compute statistics, drop "All"                                       #
            # ------------------------------------------------------------------ #
            df = self.statistics(
                by=by, column=column, func=func, percentiles=percentiles,
                sectors=sectors, seasons=seasons, step=step,
            )
            df = df.drop(columns="All", errors="ignore")

            def _row_label(f: str) -> str:
                mapping = {"mean": "Mean", "std": "Std", "min": "Min",
                        "max": "Max", "count": "Count"}
                if f in mapping:
                    return mapping[f]
                if f.lower().startswith("p") and f[1:].isdigit():
                    return f"P{int(f[1:]):02d}"
                return f

            row_labels = [_row_label(f) for f in func]
            row_labels = [r for r in row_labels if r in df.index]

            var  = self._resolve_columns(column, default=[self.primary])[0]

            # ------------------------------------------------------------------ #
            # Column labels: format bin edges as "lo–hi unit" when by is a var   #
            # ------------------------------------------------------------------ #
            raw_cols = list(df.columns)

            if by_var is not None:
                # raw_cols are bin-start strings produced by _get_groups.
                # Parse them back to floats and reconstruct "lo–hi unit" labels.
                try:
                    starts = [float(c) for c in raw_cols]
                    # Infer step from the spacing between bin starts
                    if len(starts) > 1:
                        _step = starts[1] - starts[0]
                    else:
                        _step = step or infer_step(self.data[by].dropna())
                    unit_str  = f" {by_var.unit}" if by_var.unit and by_var.unit != "-" else ""
                    display_cols = [
                        f"{lo:.10g}–{lo + _step:.10g}{unit_str}"
                        for lo in starts
                    ]
                except (ValueError, TypeError):
                    # Fallback: use raw labels if parsing fails
                    display_cols = raw_cols
            else:
                display_cols = raw_cols

            x = np.arange(len(raw_cols))

            # Groups with many categories: skip explicit xticks to avoid crowding
            _dense_by = {"day", "week"} | (
                {by} if by in self._variables else set()
            )
            dense_x = by in _dense_by or len(raw_cols) > 24
            markers = len(raw_cols) <= 12

            # ------------------------------------------------------------------ #
            # Figure                                                               #
            # ------------------------------------------------------------------ #
            if ax is None:
                _, ax = plt.subplots(figsize=figsize)

            # ------------------------------------------------------------------ #
            # One line per stat, coloured by colormap                             #
            # ------------------------------------------------------------------ #
            cm     = plt.get_cmap(cmap)
            n      = len(row_labels)
            colors = [cm(i / max(n - 1, 1)) for i in range(n)]
            marker = "o" if markers else None

            for label, color in zip(row_labels, colors):
                ax.plot(
                    x,
                    df.loc[label].values.astype(float),
                    color=color,
                    linewidth=1.4,
                    marker=marker,
                    markersize=4,
                    label=label,
                )

            # ------------------------------------------------------------------ #
            # Axes formatting                                                      #
            # ------------------------------------------------------------------ #
            if dense_x:
                ax.xaxis.set_major_formatter(
                    plt.FuncFormatter(
                        lambda v, _: display_cols[int(v)] if 0 <= int(v) < len(display_cols) else ""
                    )
                )
            else:
                ax.set_xticks(x)
                ax.set_xticklabels(
                    display_cols,
                    rotation=45 if len(display_cols) > 12 else 0,
                    ha="right" if len(display_cols) > 12 else "center",
                )

            ax.set_xlim(-0.5, len(raw_cols) - 0.5)
            ax.set_ylim(bottom=0)

            unit_str = f" [{var.unit}]" if var.unit else ""
            ax.set_ylabel(f"{var.name}{unit_str}")

            # x-axis label and title: use LaTeX symbol when by is a Variable
            if by_var is not None:
                col_axis_label = f"${by_var.symbol}$ ({by_var.unit})" if by_var.unit and by_var.unit != "-" else f"${by_var.symbol}$"
                title_by       = f"${by_var.symbol}$"
            else:
                col_axis_label = by.capitalize()
                title_by       = by

            ax.set_xlabel(col_axis_label)
            ax.set_title(f"{var.name} — statistics by {title_by}")

            ax.legend(
                loc="upper right",
                framealpha=0.85,
                edgecolor="0.7",
            )
            ax.grid(linestyle=":", linewidth=0.6, color="0.75", zorder=0)

            # ------------------------------------------------------------------ #
            # Optional table                                                       #
            # ------------------------------------------------------------------ #
            if table:
                cell_text = [
                    [
                        f"{df.loc[r, c]:.3g}" if not np.isnan(df.loc[r, c]) else "—"
                        for c in raw_cols
                    ]
                    for r in row_labels
                ]

                n_rows     = len(row_labels)
                row_height_inches = 0.25   # fixed physical height per row

                fig_h    = ax.figure.get_size_inches()[1]
                row_height = row_height_inches / fig_h

                tbl = ax.table(
                    cellText=cell_text,
                    rowLabels=row_labels,
                    colLabels=display_cols,
                    cellLoc="center",
                    rowLoc="center",
                    bbox=[0, -(row_height * (n_rows + 1)), 1, row_height * (n_rows + 1)],
                )

                # Colour row-label cells to match their line colour
                for i, color in enumerate(colors):
                    tbl[(i + 1, -1)].set_facecolor((*color[:3], 0.25))

                # Remove x tick labels and marks to avoid overlap with table
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False, length=0)

                # Expand figure bottom margin to accommodate the table
                ax.figure.subplots_adjust(bottom=row_height * (n_rows + 1) + 0.02)

            ax.grid(True,which="both")

            return ax


    def plot_statistics_2d(
            self,
            row: typing.Literal["month", "season", "year", "direction", "day", "hour"] | str = "month",
            col: typing.Literal["month", "season", "year", "direction", "day", "hour"] | str = "direction",
            column: str = None,
            stat: str = "mean",
            step_row: float = None,
            step_col: float = None,
            margins: bool = False,
            sectors: int = None,
            seasons_row: dict[str, list[int]] = None,
            seasons_col: dict[str, list[int]] = None,
            cmap: str = "viridis",
            annotate: bool = True,
            fmt: str = ".2g",
            ax: "plt.Axes | None" = None,
            figsize: "tuple[float, float] | None" = None,
        ) -> "plt.Axes":
            """
            Heatmap of ``statistics_2d`` — one variable aggregated across two
            grouping dimensions.

            Parameters
            ----------
            row : {"month", "season", "year", "direction", "day", "hour"} or str
                Row grouping strategy. Pass a column name to bin by value.
            col : {"month", "season", "year", "direction", "day", "hour"} or str
                Column grouping strategy. Pass a column name to bin by value.
            column : str, optional
                Variable to aggregate. Defaults to the primary variable.
            stat : str, default "mean"
                Aggregation statistic. Any token accepted by ``statistics_2d``
                is valid: ``"mean"``, ``"std"``, ``"min"``, ``"max"``,
                ``"count"``, or ``"p<n>"`` (e.g. ``"p90"``).
            step_row : float, optional
                Bin width for the row axis when ``row`` is a column name.
            step_col : float, optional
                Bin width for the column axis when ``col`` is a column name.
            margins : bool, default False
                If True, append an "All" row and column with overall statistics.
            sectors : int, optional
                Number of directional sectors. Used only when ``row`` or
                ``col`` is ``"direction"``.
            seasons_row : dict[str, list[int]], optional
                Custom season map for the row axis. Used only when
                ``row="season"``.
            seasons_col : dict[str, list[int]], optional
                Custom season map for the column axis. Used only when
                ``col="season"``.
            cmap : str, default "viridis"
                Matplotlib colormap.
            annotate : bool, default True
                If True, write the cell value inside each cell. Suppressed
                automatically when either axis has more than 24 groups, as
                annotations would be illegibly small.
            fmt : str, default ".2g"
                Format string for cell annotations, e.g. ``".1f"``, ``".0f"``.
            ax : plt.Axes, optional
                Axes to draw on. If None, a new figure is created.
            figsize : tuple[float, float], optional
                Figure size in inches. Ignored when ``ax`` is supplied.

            Returns
            -------
            plt.Axes

            Notes
            -----
            For keyword groupings (month, season, direction, day, hour) tick
            marks are placed at cell centres and every label is shown. For
            numeric bin groupings ticks sit at cell edges. When either axis
            has more than 12 groups only every N-th label is shown so that
            text does not overlap; the spacing N is chosen automatically.
            """
            # ------------------------------------------------------------------ #
            # Compute table                                                        #
            # ------------------------------------------------------------------ #
            df = self.statistics_2d(
                row=row, col=col, column=column, stat=stat,
                step_row=step_row, step_col=step_col, margins=margins,
                sectors=sectors, seasons_row=seasons_row, seasons_col=seasons_col,
            )

            var       = self._resolve_column(column, default=self.primary)
            _keywords = {"month", "season", "year", "direction", "day", "hour", "all"}

            # ------------------------------------------------------------------ #
            # Tick positions and labels                                            #
            # ------------------------------------------------------------------ #
            def _tick_info(labels, by, step):
                """
                Return (tick_positions, tick_labels, is_numeric).

                For keyword groupings: ticks at cell centres (0.5, 1.5, …).
                For numeric bins: ticks at cell edges (0, 1, 2, …, n).
                When there are more than 12 labels, only every N-th label is
                shown; intervening positions get an empty string so the tick
                mark still appears but does not crowd the axis.
                """
                if by in _keywords:
                    positions = np.arange(len(labels)) + 0.5
                    tick_labels = list(labels)
                    is_numeric  = False
                else:
                    # Numeric bins — reconstruct edges from left-edge strings
                    try:
                        edges = [float(l) for l in labels]
                        inferred_step = step or (
                            (edges[1] - edges[0]) if len(edges) > 1 else 1.0
                        )
                        edges.append(edges[-1] + inferred_step)
                        positions   = np.arange(len(edges))
                        tick_labels = [f"{e:g}" for e in edges]
                        is_numeric  = True
                    except (ValueError, IndexError):
                        positions   = np.arange(len(labels)) + 0.5
                        tick_labels = list(labels)
                        is_numeric  = False

                # Sparse labels when there are more than 12 entries
                if len(tick_labels) > 12:
                    # Choose step so we get roughly 8–10 visible labels
                    n      = len(tick_labels)
                    stride = max(1, round(n / 9))
                    tick_labels = [
                        lbl if i % stride == 0 else ""
                        for i, lbl in enumerate(tick_labels)
                    ]

                return positions, tick_labels, is_numeric

            x_ticks, x_labels, x_numeric = _tick_info(df.columns, col, step_col)
            y_ticks, y_labels, y_numeric = _tick_info(df.index,   row, step_row)

            # ------------------------------------------------------------------ #
            # Figure                                                               #
            # ------------------------------------------------------------------ #
            if ax is None:
                _, ax = plt.subplots(figsize=figsize)

            # ------------------------------------------------------------------ #
            # Heatmap                                                              #
            # ------------------------------------------------------------------ #
            data_arr   = df.values.astype(float)
            vmin, vmax = np.nanmin(data_arr), np.nanmax(data_arr)

            mesh = ax.pcolormesh(data_arr, cmap=cmap, vmin=vmin, vmax=vmax)

            # ------------------------------------------------------------------ #
            # Colorbar                                                             #
            # ------------------------------------------------------------------ #
            unit_str = f" [{var.unit}]" if var.unit else ""
            cbar = ax.figure.colorbar(mesh, ax=ax, pad=0.02)
            cbar.set_label(f"{stat}  {var.name}{unit_str}")

            # ------------------------------------------------------------------ #
            # Tick labels                                                          #
            # ------------------------------------------------------------------ #
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xticklabels(x_labels, rotation=45, ha="right" if x_numeric else "center")
            ax.set_yticklabels(y_labels)

            ax.tick_params(axis="x", length=4)
            ax.tick_params(axis="y", length=4)

            # ------------------------------------------------------------------ #
            # Axis labels and title                                                #
            # ------------------------------------------------------------------ #
            def _axis_label(by, by_var=None):
                if by_var is not None:
                    unit = by_var.unit
                    return f"${by_var.symbol}$" + (f" ({unit})" if unit and unit != "-" else "")
                return by.capitalize()

            row_var = self._variables.get(row, None)
            col_var = self._variables.get(col, None)

            ax.set_xlabel(_axis_label(col, col_var))
            ax.set_ylabel(_axis_label(row, row_var))

            title_col = f"${col_var.symbol}$" if col_var else col
            title_row = f"${row_var.symbol}$" if row_var else row
            ax.set_title(f"{var.name} — {stat} by {title_row} × {title_col}")

            # ------------------------------------------------------------------ #
            # Cell annotations (suppressed for dense grids)                       #
            # ------------------------------------------------------------------ #
            dense = len(df.index) > 24 or len(df.columns) > 24
            if annotate and not dense:
                cm_obj = plt.get_cmap(cmap)
                for i in range(len(df.index)):
                    for j in range(len(df.columns)):
                        val = data_arr[i, j]
                        if not np.isnan(val):
                            normed      = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                            r, g, b, _ = cm_obj(normed)
                            perceived   = 0.299*r + 0.587*g + 0.114*b
                            txt_color   = "white" if perceived < 0.5 else "black"
                            ax.text(
                                j + 0.5, i + 0.5,
                                format(val, fmt),
                                ha="center", va="center",
                                fontsize=8, color=txt_color,
                            )

            return ax

    def plot_timeseries(
        self,
        columns: list[str] = None,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Simple time series line plot of one or more columns.

        Parameters
        ----------
        columns : list[str], optional
            Columns to plot. Defaults to primary only.
        ax : plt.Axes, optional
            Existing axes to plot on.
        """
        raise NotImplementedError

    def plot_monthly_boxplot(
        self,
        column: str = None,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Box-and-whisker plot of a variable by calendar month.
        Useful for visualising seasonality.

        Parameters
        ----------
        column : str, optional
            Column to plot. Defaults to primary.
        ax : plt.Axes, optional
            Existing axes to plot on.
        """
        raise NotImplementedError

    def plot_exceedance(
        self,
        column: str = None,
        thresholds: list[float] = None,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Non-exceedance (or exceedance) cumulative distribution plot.

        Parameters
        ----------
        column : str, optional
            Column to plot. Defaults to primary.
        thresholds : list[float], optional
            Reference threshold lines to mark on the plot.
        ax : plt.Axes, optional
            Existing axes to plot on.
        """
        raise NotImplementedError

    def plot_histogram(
        self,
        column: str = None,
        bins: int | list = None,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Histogram of a variable.

        Parameters
        ----------
        column : str, optional
            Column to plot. Defaults to primary.
        bins : int or list, optional
            Number of bins or explicit bin edges.
        ax : plt.Axes, optional
            Existing axes to plot on.
        """
        raise NotImplementedError

    # ---------------------------------------------------------------------- #
    # Seasonality and trend                                                    #
    # ---------------------------------------------------------------------- #

    def climatology(
        self,
        column: str = None,
        freq: typing.Literal["month", "doy"] = "month",
    ) -> pd.DataFrame:
        """
        Long-term mean climatology profile.

        Parameters
        ----------
        column : str, optional
            Column to analyse. Defaults to primary.
        freq : str
            "month" → 12-point monthly climatology
            "doy"   → 365-point day-of-year climatology
        """
        raise NotImplementedError

    # ---------------------------------------------------------------------- #
    # Output helpers                                                           #
    # ---------------------------------------------------------------------- #

    def to_csv(
        self,
        path: str,
        columns: list[str] = None,
        **kwargs,
    ) -> None:
        """
        Export the underlying data (or a subset of columns) to CSV.

        Parameters
        ----------
        path : str
            Output file path.
        columns : list[str], optional
            Columns to export. Defaults to all.
        """
        raise NotImplementedError

    def to_excel(
        self,
        path: str,
        tables: dict[str, pd.DataFrame] = None,
        **kwargs,
    ) -> None:
        """
        Export one or more result tables to a formatted Excel workbook.
        Each entry in `tables` becomes a separate sheet.

        Parameters
        ----------
        path : str
            Output file path (.xlsx).
        tables : dict[str, pd.DataFrame], optional
            Named tables to write. If None, writes self.data.

        Example
        -------
        >>> ts.to_excel("results.xlsx", tables={
        ...     "Monthly stats":  ts.monthly_stats(),
        ...     "Exceedance":     ts.table_exceedance(),
        ...     "EVA parameters": ts.EVA.table_model_parameters("omni", "POT", "GP"),
        ... })
        """
        raise NotImplementedError

    @staticmethod
    def save_figure(
        fig: plt.Figure,
        path: str,
        dpi: int = 150,
        bbox_inches: str = "tight",
        **kwargs,
    ) -> None:
        """
        Consistent figure export helper.

        Parameters
        ----------
        fig : plt.Figure
            Figure to save.
        path : str
            Output path. Extension determines format (.png, .pdf, .svg, ...).
        dpi : int
            Resolution for raster formats.
        bbox_inches : str
            Passed directly to matplotlib savefig.
        """
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)

    # ---------------------------------------------------------------------- #
    # Dunder methods                                                           #
    # ---------------------------------------------------------------------- #

    def __repr__(self) -> str:
        dir_str = repr(self.direction) if self.direction else "None"
        sec_str = [s.col for s in self.secondary] if self.secondary else []
        cma_keys = list(self.CMA.keys()) if self.CMA else []
        return (
            f"TimeSeries(\n"
            f"  primary   = {self.primary!r}\n"
            f"  direction = {dir_str}\n"
            f"  secondary = {sec_str}\n"
            f"  period    = {self.data.index[0].date()} → "
            f"{self.data.index[-1].date()} ({self.n_years:.1f} years)\n"
            f"  timestep  = {self.timestep}\n"
            f"  EVA       = {self.EVA.__class__.__name__}\n"
            f"  CMA pairs = {cma_keys}\n"
            f")"
        )

    def __len__(self) -> int:
        """Number of records in the time series."""
        return len(self.data)