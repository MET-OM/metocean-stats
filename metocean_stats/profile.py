"""
ProfileTimeSeries
------------------
Single-point (or single-location) metocean profile data: any number of
variables that vary across "levels" (height above / depth below a
reference) and time -- wind speed, temperature, currents, salinity, etc.

Design
~~~~~~
Built from a *single* ``xr.Dataset`` (or ``xr.DataArray``, promoted to a
one-variable Dataset first). As with ``SpectralTimeSeries``, nothing in
the user's dataset is renamed, and metadata is optional -- if the input
already carries useful attrs they are preserved; anything undescribed
gets a bare placeholder.

Unlike ``SpectralTimeSeries``, no dimension-name aliasing is needed: a
profile variable has only ever got two candidate dims (time and
"the other one"), so whichever dim isn't time simply *is* a level
dimension, whatever it happens to be called. Different variables are
explicitly allowed to carry *different* level dimensions/grids (e.g. a
wind mast on one set of heights and an ADCP on a different set of
depths, in the same dataset) -- there is no requirement that profile
variables share one common grid. What *is* required is that every
level-like dimension in the dataset agrees on sign convention (see
Rule 4 below); mixing height-positive-up data with depth-positive-down
data in one instance is not supported and will raise.

Initialization pipeline (mirrors the identify -> standardize -> describe
shape used by SpectralTimeSeries, always run in full, never partial):

  1. **Squeeze.** Any dimension of length 1 is dropped (``ds.squeeze()``),
     demoting it to a scalar coordinate. This clears out things like a
     lingering ``lat``/``lon`` dimension left over from a single-point
     selection upstream. Nothing else is touched at this stage.

  2. **Dimensionality check.** After squeezing, no variable may have more
     than 2 dimensions. A variable with 3+ dims (e.g. an un-squeezed
     ``station`` axis) is a structural problem this class does not try
     to guess its way around, and raises immediately.

  3. **Time identification.** Every 2D variable must share exactly one
     dimension in common: the time dimension. That shared dimension's
     coordinate must be (or be cleanly convertible to) ``datetime64``.
     If 2D variables disagree on which dimension is shared, or the
     shared dimension isn't datetime-like, this raises -- there is no
     safe default to fall back on.

  4. **Level sign-convention check.** The "other" dimension of every 2D
     variable is a level dimension. In addition, any *1D* variable whose
     single dimension is not the time dimension is also carrying a level
     dimension (a "level-only" variable, e.g. a static roughness-length
     profile with no time axis) and is included in this check too. Every
     level dimension found this way must be single-signed: all its
     coordinate values must be >= 0, or all must be <= 0 (zero is
     compatible with either). If different level dimensions disagree in
     sign with each other, or a single level dimension mixes signs
     internally, this raises -- it very likely indicates a sigma/s-level
     (relative, dimensionless) coordinate, a mix of height and depth
     conventions, or corrupted data, none of which this class will
     silently guess its way through.

  5. **Level sign normalization.** Once convention is confirmed
     consistent, every level coordinate's values are replaced by their
     absolute value, and the *meaning* of that magnitude (pointing up
     from a reference, or down from a reference) is recorded via the
     required, user-declared ``level_convention`` parameter
     ("height" or "depth") -- exactly as ``dir_convention`` records
     the meaning of a directional value in SpectralTimeSeries, rather
     than being inferred from the data.

After these checks, every remaining variable is classified by its
dimension signature into one of three buckets:

    {time, level}  -> profile variable   (any number allowed, each may
                                           carry its own level dim/grid)
    {time}         -> scalar variable    (varies only in time)
    {level}        -> level-only variable (varies only across a level
                                           dim, no time axis)
    {}             -> static variable     (0D, kept but otherwise ignored)

As with the other classes in this family, ``self.ds`` is the *only*
copy of the data. It is never split into a "classified" object plus an
"auxiliary" leftover, and nothing beyond the squeeze / sign-normalize
steps above is ever mutated without being explicitly asked for later
(e.g. interpolation is always an output-time operation on a fresh
result, never a rewrite of ``self.ds``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from .utils import groupby_month, groupby_season, groupby_sector, aggregate_statistics
from .EVA import UnivariateEVA

# ---------------------------------------------------------------------------
# Signature buckets, expressed structurally (roles, not names) since there
# are only ever two candidate roles here: "time" and "level".
# ---------------------------------------------------------------------------

_ROLE_TIME = "time"
_ROLE_LEVEL = "level"

_SIGNATURE_LABELS = {
    frozenset({_ROLE_TIME, _ROLE_LEVEL}): "profile variable",
    frozenset({_ROLE_TIME}): "scalar variable",
    frozenset({_ROLE_LEVEL}): "level-only variable",
    frozenset(): "static variable",
}

# ---------------------------------------------------------------------------
# Variable -- shared metadata container, matching TimeSeries.Variable so the
# two modules present the same interface for describing a column/variable.
# ---------------------------------------------------------------------------

@dataclass
class Variable:
    """
    Metadata container for a single metocean variable.

    Parameters
    ----------
    col : str
        Variable name in the source ``xr.Dataset``. Must match exactly.
    name : str
        Human-readable name used in table headers and axis titles.
        E.g. "Wind speed".
    symbol : str
        LaTeX-compatible symbol (without $ delimiters). E.g. "U_{10}".
    unit : str
        Physical unit string. E.g. "m/s", "degC".
    label : str, optional
        Short display label for legend entries or compact tables.
        Defaults to symbol if not provided.

    Notes
    -----
    Deliberately identical in shape to ``TimeSeries.Variable`` -- the two
    classes describe variables the same way, so metadata built for one
    reads naturally as documentation for the other.
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


def _coerce_variable(v: "Variable | str") -> "Variable":
    """
    Promote a bare variable-name string to a Variable with placeholder
    semantics (name = symbol = col, unit = ""). Matches
    TimeSeries._coerce_variable exactly, so a string handed to either
    class is treated the same way.
    """
    if isinstance(v, Variable):
        return v
    if isinstance(v, str):
        return Variable(col=v, name=v, symbol=v, unit="")
    raise TypeError(
        f"Expected a Variable or a column name string, got {type(v)}. "
        "Example: Variable('ws', 'Wind speed', 'U_{10}', 'm/s') or just 'ws'."
    )


def _default_attrs_for(name: str) -> dict:
    """Bare placeholder metadata for a variable/coordinate nobody described."""
    return dict(name=name, long_name=name)


def _apply_default_attrs(da: xr.DataArray, name: str) -> None:
    """
    Set 'name' and 'long_name' only if not already meaningfully present.
    Never overwrites existing attrs -- matches the rest of the family.
    """
    if not da.attrs.get("name"):
        meta = _default_attrs_for(name)
        da.attrs["name"] = meta["name"]
        da.attrs.setdefault("long_name", meta["long_name"])


def _apply_variable_attrs(da: xr.DataArray, variable: "Variable") -> None:
    """
    Same never-overwrite contract as _apply_default_attrs, but sourced
    from a resolved Variable (name/symbol/unit/label) rather than just a
    bare name -- used once every var has been coerced to a Variable,
    whether the user described it explicitly or it was auto-registered.
    """
    if not da.attrs.get("name"):
        da.attrs["name"] = variable.name
        da.attrs.setdefault("long_name", variable.name)
    da.attrs.setdefault("symbol", variable.symbol)
    da.attrs.setdefault("units", variable.unit)
    da.attrs.setdefault("label", variable.label)


class ProfileTimeSeries:
    """
    Single-source metocean profile data: any number of variables varying
    across levels (height/depth) and time.

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        Source data. A bare DataArray is promoted to a one-variable
        Dataset first (using its own ``.name``, or "value" if unnamed),
        then subjected to the same checks as a Dataset input. Dimension
        names are never renamed -- whichever dimension isn't the shared
        time dimension is treated as a level dimension, whatever it is
        called in the source data. Different variables may use different
        level dimensions/grids; there is no requirement that profile
        variables share one common grid. See module docstring for the
        full initialization pipeline (squeeze -> dimensionality check ->
        time identification -> level sign check -> sign normalization).
    level_convention : {"height", "depth"}
        Declares what the (now positive) level values mean:
        "height" - distance above a reference (e.g. sea surface), values
                   increase upward.
        "depth"  - distance below a reference (e.g. sea surface or
                   seabed), values increase downward.
        This is metadata the class trusts, not something inferred from
        the data's original sign -- exactly as ``dir_convention`` works
        in SpectralTimeSeries. Required: mixing height- and
        depth-convention data in a single instance is not supported.
    primary : Variable | str
        The primary profile variable (e.g. wind speed, current speed,
        temperature) -- the one EVA and other single-variable methods
        default to when no `var=` is given explicitly. A bare string is
        promoted to a placeholder Variable(name=symbol=the string,
        unit=""); pass a full Variable to supply proper name/symbol/unit
        metadata instead. Must resolve to a profile variable ({time,
        level} signature) once the dataset has been classified --
        checked after classification, since which variables qualify as
        "profile variables" isn't known until then.
    direction : Variable | str, optional
        The profile variable to use as the default direction source for
        direction-based grouping (statistics(by="direction")) and for
        EVA's sector splitting. Must itself be a profile variable, and
        must share `primary`'s exact level dimension -- pairing a
        direction series on one grid with data on another would silently
        misalign which direction value applies to which level, so this
        is checked at construction and raises rather than guessing. If
        not set here, direction-based grouping/EVA-by-sector can still
        be supplied per-call; if neither is given, those features raise.
    level_convention : {"height", "depth"}
        Declares what the (now positive) level values mean:
        "height" - distance above a reference (e.g. sea surface), values
                   increase upward.
        "depth"  - distance below a reference (e.g. sea surface or
                   seabed), values increase downward.
        This is metadata the class trusts, not something inferred from
        the data's original sign -- exactly as ``dir_convention`` works
        in SpectralTimeSeries. Required: mixing height- and
        depth-convention data in a single instance is not supported.
    name : str, optional
        Human-readable label for this record (site name, model run,
        instrument, etc.).

    Attributes
    ----------
    ds : xr.Dataset
        The full source dataset after squeezing size-1 dims and
        normalizing level signs. All original variable/coordinate names
        and attrs are preserved except for the deliberate level-sign
        normalization described above.
    level_convention : {"height", "depth"}
    time_dim : str
        Name of the shared time dimension, whatever it was called in the
        input.
    level_dims : list[str]
        Names of every distinct level dimension found across all
        profile and level-only variables (may be more than one).
    profile_vars : list[str]
        Variables with signature {time, level}.
    scalar_vars : list[str]
        Variables with signature {time} only.
    level_only_vars : list[str]
        Variables with signature {level} only (no time axis).
    static_vars : list[str]
        0D variables, kept but otherwise unexamined.
    primary : Variable
        Resolved primary-variable metadata.
    direction : Variable | None
        Resolved direction-variable metadata, or None if not given.
    EVA : ProfileEVA
        Extreme value analysis broadcast across every level of
        `primary`'s level dimension -- one `UnivariateEVA` per level,
        built lazily and cached the first time each level is touched.
        Only ever scoped to `primary`; EVA on a different profile
        variable means constructing a new ProfileTimeSeries with that
        variable as `primary` (cheap -- EVA itself is what's expensive,
        and it isn't run until you call EVA.get_extremes()/.fit()).

    Notes
    -----
    There is no "unstandardized" object you can hold and forget to fix --
    if construction succeeds, the squeeze/dimensionality/time/sign checks
    have all passed and level values are already positive magnitudes.
    Anything this class cannot safely resolve on its own (ambiguous time
    dimension, mixed level sign convention, >2D variables, a primary/
    direction pair on mismatched level grids) raises during construction
    rather than being guessed at.
    """

    def __init__(
        self,
        data: xr.Dataset | xr.DataArray,
        level_convention: Literal["height", "depth"],
        primary: Variable | str,
        direction: Variable | str | None = None,
        name: str | None = None,
    ):
        if isinstance(data, xr.DataArray):
            ds = data.to_dataset(name=data.name or "value")
        elif isinstance(data, xr.Dataset):
            ds = data
        else:
            raise TypeError(
                f"data must be an xr.Dataset or xr.DataArray, got "
                f"{type(data).__name__}. Load your source data with xarray "
                "first, e.g.:\n"
                "    ds = xr.open_dataset('file.nc')"
            )

        if level_convention not in ("height", "depth"):
            raise ValueError(
                f"level_convention must be 'height' or 'depth', got "
                f"{level_convention!r}."
            )

        # ------------------------------------------------------------------ #
        # Variable validation -- coerce primary/direction, but don't check   #
        # they're actually profile variables yet: classification (which     #
        # determines self.profile_vars) hasn't run. Checked below, once it  #
        # has.                                                               #
        # ------------------------------------------------------------------ #
        primary = _coerce_variable(primary)
        direction = _coerce_variable(direction) if direction is not None else None

        self.level_convention = level_convention
        self.name = name

        title = f"ProfileTimeSeries: {name}" if name else "ProfileTimeSeries"
        print(title)
        print("=" * len(title))

        # ------------------------------------------------------------------ #
        # 1. Squeeze -- drop any length-1 dimension entirely.                #
        # ------------------------------------------------------------------ #
        squeezed_dims = [d for d, n in ds.sizes.items() if n == 1]
        if squeezed_dims:
            ds = ds.squeeze(dim=squeezed_dims, drop=False)
            print(f"Squeezed length-1 dimension(s): {squeezed_dims}")

        # ------------------------------------------------------------------ #
        # 2. Dimensionality check -- no variable may exceed 2 dims.          #
        # ------------------------------------------------------------------ #
        self._check_max_dimensionality(ds)

        # ------------------------------------------------------------------ #
        # 3. Time identification -- the one dim shared by all 2D vars.       #
        # ------------------------------------------------------------------ #
        time_dim = self._identify_time_dim(ds)
        self._check_time_is_datetime(ds, time_dim)

        # ------------------------------------------------------------------ #
        # 4. Level sign-convention check across every level dimension.       #
        # ------------------------------------------------------------------ #
        level_dims = self._identify_level_dims(ds, time_dim)
        self._check_level_sign_consistency(ds, level_dims, level_convention)

        # ------------------------------------------------------------------ #
        # 5. Normalize -- make every level coordinate's values positive.     #
        # ------------------------------------------------------------------ #
        ds = self._normalize_level_signs(ds, level_dims)

        # ------------------------------------------------------------------ #
        # Classify every variable by dimension signature.                   #
        # ------------------------------------------------------------------ #
        profile_vars, scalar_vars, level_only_vars, static_vars = (
            self._classify_variables(ds, time_dim, level_dims)
        )

        if not profile_vars:
            raise ValueError(
                "No profile variable found. Expected at least one data "
                f"variable with dims (time, level) -- i.e. ('{time_dim}', "
                "<some other dim>). Variables present: "
                f"{list(ds.data_vars)}."
            )

        # ------------------------------------------------------------------ #
        # Now that profile_vars is known: validate primary/direction        #
        # actually resolve to profile variables, and that direction (if     #
        # given) shares primary's level dimension. A direction series on a  #
        # different grid than primary can't be meaningfully paired with it  #
        # level-for-level, so this is checked explicitly rather than left   #
        # to fail confusingly deep inside EVA or statistics(by="direction").#
        # ------------------------------------------------------------------ #
        if primary.col not in profile_vars:
            raise ValueError(
                f"primary='{primary.col}' is not a recognised profile "
                f"variable (signature {{time, level}}). Available profile "
                f"variables: {profile_vars}."
            )
        primary_level_dim = [d for d in ds[primary.col].dims if d != time_dim][0]

        if direction is not None:
            if direction.col not in profile_vars:
                raise ValueError(
                    f"direction='{direction.col}' is not a recognised "
                    f"profile variable (signature {{time, level}}). "
                    f"Available profile variables: {profile_vars}."
                )
            direction_level_dim = [
                d for d in ds[direction.col].dims if d != time_dim
            ][0]
            if direction_level_dim != primary_level_dim:
                raise ValueError(
                    f"direction='{direction.col}' varies over level "
                    f"dimension '{direction_level_dim}', but primary="
                    f"'{primary.col}' varies over '{primary_level_dim}'. "
                    "direction must share primary's exact level dimension "
                    "-- pairing values from two different level grids "
                    "would silently misalign which direction applies to "
                    "which level. Regrid one of these variables onto the "
                    "other's level dimension before constructing "
                    "ProfileTimeSeries, or choose a different direction "
                    "variable that already shares primary's grid."
                )

        # ------------------------------------------------------------------ #
        # Store.                                                             #
        # ------------------------------------------------------------------ #
        self.ds = ds
        self.time_dim = time_dim
        self.level_dims = level_dims
        self.profile_vars = profile_vars
        self.scalar_vars = scalar_vars
        self.level_only_vars = level_only_vars
        self.static_vars = static_vars
        self.primary = primary
        self.direction = direction

        self.ds.attrs["level_convention"] = level_convention
        if name is not None:
            self.ds.attrs["name"] = name

        # ------------------------------------------------------------------ #
        # Describe -- attrs only, never overwriting existing metadata.      #
        # Every variable is registered as a Variable, whether or not the    #
        # user described it: primary/direction use the metadata given      #
        # above, everything else gets an auto-registered placeholder        #
        # Variable (name=symbol=col, unit=""), matching TimeSeries's        #
        # "auto-register any column not already covered by a declared      #
        # Variable" behaviour. There is no notion of "undeclared" variables #
        # being dropped -- every variable in the dataset is kept and gets   #
        # an entry.                                                         #
        # ------------------------------------------------------------------ #
        declared = {primary.col: primary}
        if direction is not None:
            declared[direction.col] = direction

        self._variables: dict[str, Variable] = {}
        for var_name in ds.data_vars:
            self._variables[var_name] = declared.get(
                var_name, _coerce_variable(var_name)
            )

        self._apply_default_attrs_all(ds, time_dim, level_dims, self._variables)

        # ------------------------------------------------------------------ #
        # Sub-module: Extreme Value Analysis, scoped to `primary`.           #
        # Cheap to construct -- no UnivariateEVA is actually built until    #
        # EVA.get_extremes()/.fit() is called for a given level.            #
        # ------------------------------------------------------------------ #
        self.EVA = ProfileEVA(self, primary.col)

        print()
        self._print_variable_table()
        print(f"\nlevel_convention : {self.level_convention}")
        print(f"time_dim         : '{self.time_dim}'")
        print(f"level_dim(s)     : {self.level_dims}")
        print(f"primary          : '{self.primary.col}'")
        if self.direction is not None:
            print(f"direction        : '{self.direction.col}'")

    # ---------------------------------------------------------------------- #
    # Init helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _check_max_dimensionality(ds: xr.Dataset) -> None:
        """Rule 2: after squeezing, no variable may have more than 2 dims."""
        offenders = {
            var_name: da.dims
            for var_name, da in ds.data_vars.items()
            if len(da.dims) > 2
        }
        if offenders:
            details = "\n".join(
                f"  '{v}': dims {d}" for v, d in offenders.items()
            )
            raise ValueError(
                "The following variable(s) have more than 2 dimensions "
                "after squeezing out length-1 dims:\n" + details + "\n"
                "ProfileTimeSeries only supports variables varying over "
                "(time, level) at most. Select or drop the extra "
                "dimension(s) before constructing ProfileTimeSeries, e.g.:\n"
                "    ds = ds.isel(station=0)"
            )

    @staticmethod
    def _identify_time_dim(ds: xr.Dataset) -> str:
        """
        Rule 3: every 2D variable must share exactly one dimension in
        common. That shared dimension is the time dimension.

        Note that with only two dims per variable, "the dims common to
        every 2D variable" can itself contain more than one candidate --
        e.g. if every single 2D variable happens to be (time, height),
        both 'time' and 'height' are technically common to all of them.
        The tie is broken by requiring the dimension to also be
        datetime-like; if more than one candidate is datetime-like (or
        none is), that is a genuine ambiguity and raises rather than
        guessing.
        """
        two_d_vars = {
            var_name: da.dims for var_name, da in ds.data_vars.items()
            if len(da.dims) == 2
        }

        if not two_d_vars:
            raise ValueError(
                "No 2D (time, level) variable found in the dataset -- "
                "cannot identify a time dimension. Variables present: "
                f"{ {v: da.dims for v, da in ds.data_vars.items()} }."
            )

        dim_sets = {var_name: set(dims) for var_name, dims in two_d_vars.items()}
        common = set.intersection(*dim_sets.values())

        def _is_datetime_like(dim: str) -> bool:
            if dim not in ds.coords:
                return False
            vals = ds.coords[dim].values
            if np.issubdtype(vals.dtype, np.datetime64):
                return True
            # Deliberately NOT calling pd.to_datetime() on numeric dtypes:
            # pd.to_datetime([10, 50, 100]) "succeeds" by treating them as
            # epoch nanoseconds, which is never what's meant for a level
            # coordinate that happens to hold small numbers. Only trust
            # non-numeric (string / object / cftime-like) coordinates to
            # the conversion attempt.
            if np.issubdtype(vals.dtype, np.number):
                return False
            try:
                pd.to_datetime(vals)
                return True
            except Exception:
                return False

        candidates = [d for d in common if _is_datetime_like(d)]

        if len(candidates) != 1:
            details = "\n".join(
                f"  '{v}': dims {tuple(d)}" for v, d in two_d_vars.items()
            )
            raise ValueError(
                "Could not identify a single shared time dimension across "
                f"the 2D variables below (dimension(s) common to all: "
                f"{sorted(common)}; datetime-like among those: "
                f"{sorted(candidates)}):\n" + details
                + "\nEvery (time, level) variable must share exactly one "
                "common dimension, and that dimension's coordinate must "
                "be datetime-like. Restructure the source data so this "
                "is unambiguous."
            )

        return candidates[0]

    @staticmethod
    def _check_time_is_datetime(ds: xr.Dataset, time_dim: str) -> None:
        """
        Rule 3 continued: the shared dimension's coordinate must be (or
        cleanly convert to) datetime64. Structural check, raised
        immediately rather than deferred -- a non-datetime time axis
        means the wrong dimension was identified, or the input itself is
        malformed.
        """
        if time_dim not in ds.coords:
            raise ValueError(
                f"Identified '{time_dim}' as the shared time dimension, but "
                "it has no associated coordinate values to check for "
                "datetime type. Assign coordinates for this dimension "
                "before constructing ProfileTimeSeries."
            )

        time_vals = ds.coords[time_dim].values
        if np.issubdtype(time_vals.dtype, np.datetime64):
            return

        # As in _identify_time_dim: don't let pd.to_datetime() coerce a
        # plain numeric coordinate (it would silently reinterpret the
        # numbers as epoch nanoseconds rather than actually failing).
        if np.issubdtype(time_vals.dtype, np.number):
            raise ValueError(
                f"Identified '{time_dim}' as the shared time dimension, but "
                f"its coordinate values are numeric (dtype {time_vals.dtype}), "
                "not datetime64. Convert this coordinate to datetime before "
                "constructing ProfileTimeSeries, e.g. with pd.to_datetime() "
                "if the numbers are a recognised epoch/unit, or xr.decode_cf() "
                "if they are CF-encoded time values."
            )

        try:
            pd.to_datetime(time_vals)
        except Exception as exc:
            raise ValueError(
                f"Identified '{time_dim}' as the shared time dimension, but "
                f"its coordinate values (dtype {time_vals.dtype}) are not "
                "datetime64 and could not be converted with "
                "pd.to_datetime(). Convert this coordinate to datetime "
                "before constructing ProfileTimeSeries."
            ) from exc

    @staticmethod
    def _identify_level_dims(ds: xr.Dataset, time_dim: str) -> list[str]:
        """
        Rule 4 setup: collect every distinct level dimension. This is the
        "other" dim of every 2D variable, plus the single dim of any 1D
        variable whose dim is not the time dimension.
        """
        level_dims: list[str] = []
        for da in ds.data_vars.values():
            dims = da.dims
            if len(dims) == 2:
                other = [d for d in dims if d != time_dim]
                # _identify_time_dim already guarantees time_dim is shared
                # by every 2D var, so `other` has exactly one entry here.
                level_dims.extend(other)
            elif len(dims) == 1 and dims[0] != time_dim:
                level_dims.append(dims[0])

        # de-duplicate, preserve first-seen order
        seen = set()
        ordered = []
        for d in level_dims:
            if d not in seen:
                seen.add(d)
                ordered.append(d)
        return ordered

    @staticmethod
    def _check_level_sign_consistency(
        ds: xr.Dataset, level_dims: list[str], level_convention: str
    ) -> None:
        """
        Rule 4: every level dimension's coordinate values must be
        internally single-signed (all >= 0 or all <= 0; zero is
        compatible with either), AND all level dimensions must agree
        with each other in sign. A violation most likely indicates a
        sigma/s-level (relative, dimensionless) coordinate, a mix of
        height/depth conventions, or corrupted data -- none of which is
        safely fixable automatically, so this raises rather than guesses.
        """
        if not level_dims:
            raise ValueError(
                "No level dimension found -- every variable is either 0D "
                "or varies only along the time dimension. "
                "ProfileTimeSeries requires at least one variable varying "
                "across a level dimension."
            )

        signs: dict[str, str] = {}
        mixed: dict[str, tuple[float, float]] = {}

        for dim in level_dims:
            if dim not in ds.coords:
                raise ValueError(
                    f"Level dimension '{dim}' has no associated coordinate "
                    "values, so its sign convention cannot be checked. "
                    "Assign coordinate values for this dimension before "
                    "constructing ProfileTimeSeries."
                )
            vals = ds.coords[dim].values.astype(float)
            has_pos = np.any(vals > 0)
            has_neg = np.any(vals < 0)

            if has_pos and has_neg:
                mixed[dim] = (float(np.nanmin(vals)), float(np.nanmax(vals)))
                continue

            signs[dim] = "positive" if has_pos else ("negative" if has_neg else "zero")

        if mixed:
            details = "\n".join(
                f"  '{d}': range=[{lo:.4g}, {hi:.4g}]" for d, (lo, hi) in mixed.items()
            )
            raise ValueError(
                "The following level dimension(s) mix positive and negative "
                "values, so no single sign convention applies:\n" + details + "\n"
                "This often indicates a sigma/s-level (relative,  "
                "dimensionless, typically in [-1, 0] or [0, 1] *relative* "
                "to local depth) coordinate rather than actual height/depth "
                "values. ProfileTimeSeries requires real z-values (e.g. "
                "metres above/below a fixed reference) -- convert sigma "
                "coordinates to actual depths/heights before constructing "
                "ProfileTimeSeries."
            )

        # A single-signed dimension can still plausibly be a sigma/s-level
        # coordinate -- e.g. [-1.0, -0.66, -0.33, 0.0] is entirely <= 0 and
        # passes the mixed-sign check above just fine. Sigma coordinates
        # are, by convention, bounded within roughly [-1, 1] (relative to
        # local water depth). This is only ever a *hint*, not something
        # worth guessing at with more elaborate heuristics (unit attrs,
        # naming, etc.) -- so rather than raising or silently deciding,
        # just warn per affected variable and let the user judge.
        _SIGMA_LIKE_BOUND = 1.0
        for dim in level_dims:
            vals = ds.coords[dim].values.astype(float)
            if np.all(np.abs(vals) <= _SIGMA_LIKE_BOUND):
                sample = ", ".join(f"{v:.4g}" for v in vals[:5])
                if len(vals) > 5:
                    sample += ", ..."
                warnings.warn(
                    f"Detected very small {level_convention} values for "
                    f"dimension '{dim}': {sample}. If these are sigma-levels "
                    "(relative to local water depth) rather than real "
                    f"{level_convention} values in metres, please convert "
                    "them to metres manually before constructing "
                    "ProfileTimeSeries -- this class cannot tell the "
                    "difference from values alone.",
                    UserWarning,
                    stacklevel=3,
                )

        # "zero" is compatible with either convention, so only dims with a
        # definite sign need to agree with each other.
        definite = {d: s for d, s in signs.items() if s != "zero"}
        unique_signs = set(definite.values())
        if len(unique_signs) > 1:
            details = "\n".join(f"  '{d}': {s}" for d, s in definite.items())
            raise ValueError(
                "Level dimensions disagree on sign convention -- all level "
                "data in a single ProfileTimeSeries must use the same "
                "convention (all values >= 0, or all values <= 0):\n"
                + details + "\n"
                "Fix the sign convention of one of these dimensions before "
                "constructing ProfileTimeSeries."
            )

    @staticmethod
    def _normalize_level_signs(ds: xr.Dataset, level_dims: list[str]) -> xr.Dataset:
        """Rule 5: replace every level coordinate's values with abs()."""
        updates = {}
        for dim in level_dims:
            vals = ds.coords[dim].values.astype(float)
            abs_vals = np.abs(vals)
            if not np.array_equal(vals, abs_vals):
                updates[dim] = abs_vals

        if updates:
            ds = ds.assign_coords(updates)
            for dim, vals in updates.items():
                print(f"  '{dim}' level values normalized to positive magnitude")

        return ds

    @staticmethod
    def _classify_variables(
        ds: xr.Dataset, time_dim: str, level_dims: list[str]
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """
        Bucket every variable by its dimension signature:
            {time, level} -> profile_vars   (any number allowed)
            {time}        -> scalar_vars
            {level}       -> level_only_vars
            {}            -> static_vars
        """
        profile_vars, scalar_vars, level_only_vars, static_vars = [], [], [], []

        for var_name, da in ds.data_vars.items():
            dims = set(da.dims)
            if dims == set():
                static_vars.append(var_name)
            elif dims == {time_dim}:
                scalar_vars.append(var_name)
            elif len(dims) == 1 and dims.issubset(set(level_dims)):
                level_only_vars.append(var_name)
            elif dims == {time_dim} | (dims & set(level_dims)) and len(dims) == 2:
                profile_vars.append(var_name)
            else:
                # Shouldn't be reachable given the dimensionality/time
                # checks above, but fail loudly rather than silently drop
                # a variable if it somehow gets here.
                raise ValueError(
                    f"Variable '{var_name}' with dims {da.dims} did not "
                    "match any recognised signature (time+level, time, "
                    "level, or static). This should not happen after the "
                    "earlier structural checks -- please report this."
                )

        return profile_vars, scalar_vars, level_only_vars, static_vars

    @staticmethod
    def _apply_default_attrs_all(
        ds: xr.Dataset,
        time_dim: str,
        level_dims: list[str],
        variables: dict[str, "Variable"],
    ) -> None:
        """
        Describe phase: attrs sourced from each variable's resolved
        Variable (name/symbol/unit/label), never overwriting existing
        attrs. Every data variable has an entry in `variables` by this
        point (primary/direction's real metadata, everything else an
        auto-registered placeholder), so this no longer falls back to
        the bare _apply_default_attrs path for data variables -- only
        coordinates (time_dim, level_dims), which aren't Variables, still
        use it.
        """
        for var_name, da in ds.data_vars.items():
            _apply_variable_attrs(da, variables[var_name])
        if time_dim in ds.coords:
            _apply_default_attrs(ds.coords[time_dim], time_dim)
        for dim in level_dims:
            if dim in ds.coords:
                _apply_default_attrs(ds.coords[dim], dim)

    # ---------------------------------------------------------------------- #
    # Display helpers
    # ---------------------------------------------------------------------- #

    def _signature_label(self, var_name: str) -> str:
        if var_name in self.profile_vars:
            return "profile variable"
        if var_name in self.scalar_vars:
            return "scalar variable"
        if var_name in self.level_only_vars:
            return "level-only variable"
        if var_name in self.static_vars:
            return "static variable"
        return "—"

    def _print_variable_table(self) -> None:
        headers = ["variable", "kind", "name", "units", "dims"]
        rows = []
        for var_name, da in self.ds.data_vars.items():
            rows.append([
                var_name,
                self._signature_label(var_name),
                da.attrs.get("name", var_name),
                da.attrs.get("units", "—"),
                "(" + ", ".join(da.dims) + ")" if da.dims else "()",
            ])

        widths = [
            max(len(headers[i]), max((len(r[i]) for r in rows), default=0)) + 2
            for i in range(len(headers))
        ]

        def fmt_row(cells: list[str]) -> str:
            return "".join(f"{c:<{widths[i]}}" for i, c in enumerate(cells))

        print(fmt_row(headers))
        print("-" * sum(widths))
        for row in rows:
            print(fmt_row(row))

    # ---------------------------------------------------------------------- #
    # Properties
    # ---------------------------------------------------------------------- #

    @property
    def time(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.ds.coords[self.time_dim].values)

    def levels(self, dim: str | None = None) -> np.ndarray:
        """
        Level coordinate values (always positive magnitudes; sign meaning
        given by ``self.level_convention``) for a given level dimension.
        If there is only one level dimension in the dataset, ``dim`` may
        be omitted.
        """
        if dim is None:
            if len(self.level_dims) != 1:
                raise ValueError(
                    "Multiple level dimensions present "
                    f"({self.level_dims}); specify which one via `dim=`."
                )
            dim = self.level_dims[0]
        elif dim not in self.level_dims:
            raise ValueError(
                f"'{dim}' is not a known level dimension. "
                f"Available: {self.level_dims}"
            )
        return self.ds.coords[dim].values

    @property
    def duration(self) -> pd.Timedelta:
        t = self.time
        return t[-1] - t[0]

    @property
    def timestep(self) -> pd.Timedelta:
        return pd.Series(self.time).diff().median()

    # ---------------------------------------------------------------------- #
    # Statistics
    # ---------------------------------------------------------------------- #

    def _require_direction(self, method_name: str, direction: str | None) -> str:
        """
        Resolve which profile variable to use as direction: explicit
        call-time argument first, falling back to the instance-level
        default (`self.direction.col`) set at __init__. Raises a clear,
        actionable error if neither is available.
        """
        resolved = direction if direction is not None else (
            self.direction.col if self.direction is not None else None
        )
        if resolved is None:
            raise ValueError(
                f"{method_name}(by='direction') requires a direction "
                "variable, but none was provided. Either pass "
                "direction='<column>' to this call, or set direction=... "
                "when constructing ProfileTimeSeries."
            )
        if resolved not in self.profile_vars:
            raise ValueError(
                f"direction='{resolved}' is not a recognised profile "
                f"variable. Available profile variables: {self.profile_vars}."
            )
        return resolved

    def _profile_var_as_dataframe(self, var: str, level_dim: str) -> pd.DataFrame:
        """
        Convert a (time, level_dim) profile variable into a plain
        time-indexed pd.DataFrame with one column per level, so the
        pandas-based groupby_month / groupby_season / groupby_sector
        utilities can be applied directly without reimplementing their
        logic for xarray.
        """
        da = self.ds[var].transpose(self.time_dim, level_dim)
        df = pd.DataFrame(
            da.values,
            index=pd.DatetimeIndex(self.ds[self.time_dim].values),
            columns=self.ds.coords[level_dim].values,
        )
        return df

    def _get_groups(
        self,
        by: str,
        var: str,
        level_dim: str,
        direction: str | None = None,
        sectors: int | list[float] = 12,
        seasons: dict[str, list[int]] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Build named groups of `var` (as a time-indexed DataFrame, one
        column per level) according to `by`, using the shared
        groupby_month / groupby_season / groupby_sector utilities
        directly -- not a reimplementation, so grouping conventions
        (month labels, season->month map, sector edges/centering/labels)
        always stay in sync with the rest of the package.
        """
        df = self._profile_var_as_dataframe(var, level_dim)

        if by == "direction":
            dir_var = self._require_direction("statistics", direction)
            dir_da = self.ds[dir_var]
            if level_dim not in dir_da.dims:
                raise ValueError(
                    f"direction variable '{dir_var}' does not vary over "
                    f"level dimension '{level_dim}', so it cannot be used "
                    f"to group '{var}' (which does) by direction. Grouping "
                    "by direction requires the direction variable to share "
                    "the same level dimension as the variable being "
                    "grouped."
                )
            # groupby_sector groups by ONE direction column against a
            # whole DataFrame. Since direction here is itself a profile
            # variable (one value per level, not one shared value), each
            # level must be grouped by *its own* direction column -- so
            # we call groupby_sector once per level and stitch the
            # per-level Series back into per-sector DataFrames.
            dir_df = self._profile_var_as_dataframe(dir_var, level_dim)
            per_level_groups: dict[str, dict] = {}
            for level in df.columns:
                merged = pd.DataFrame({"value": df[level], "dir": dir_df[level]}).dropna()
                sector_groups = groupby_sector(
                    merged, var_dir="dir", sectors=sectors, var="value"
                )
                per_level_groups[level] = sector_groups

            all_labels = sorted({
                label for lg in per_level_groups.values() for label in lg
            })
            groups = {}
            for label in all_labels:
                groups[label] = pd.DataFrame({
                    level: per_level_groups[level].get(label, pd.Series(dtype=float))
                    for level in df.columns
                })
            return groups

        if by == "month":
            return groupby_month(df)

        if by == "season":
            return groupby_season(df, seasons=seasons)

        if by == "year":
            return {str(y): g for y, g in df.groupby(df.index.year)}

        raise ValueError(
            f"'by' must be one of 'month', 'season', 'direction', 'year'. "
            f"Got '{by}'."
        )

    def statistics(
        self,
        var: str | None = None,
        by: Literal["all", "month", "season", "direction", "year"] = "all",
        stat: str | list[str] = ("mean", "P90", "P99"),
        direction: str | None = None,
        sectors: int | list[float] = 12,
        seasons: dict[str, list[int]] | None = None,
    ) -> pd.DataFrame:
        """
        Aggregated statistic(s) for one profile variable, with level
        kept as rows.

        Profile data has one more axis than TimeSeries data (level, in
        addition to time), so a single call cannot show every group and
        every statistic at once without producing a 3D result. Two modes
        are supported, chosen by `by`:

          - by="all" (the default): no grouping. `stat` may be a list
            (defaults to ["mean", "P90", "P99"]) -- the returned table
            has one column per requested statistic. This is the most
            common case: most variables are looked at as a whole first.
          - by="month"/"season"/"direction"/"year": grouped. `stat` must
            be a single string in this mode -- the returned table has
            one column per group (+ an "All" column). Passing a list of
            stats together with a non-"all" `by` raises, since the
            result would need a third axis to display validly; call this
            method once per statistic instead.

        Grouping and aggregation are delegated to the shared
        groupby_month / groupby_season / groupby_sector /
        aggregate_statistics utilities (the same ones TimeSeries uses),
        so conventions and accepted stat-name spellings ("mean", "p90",
        "P90", "90%", ...) are identical across the package.

        Parameters
        ----------
        var : str, optional
            Name of the profile variable (must be in `self.profile_vars`).
            Defaults to `self.primary.col` if not given.
        by : {"all", "month", "season", "direction", "year"}
            Grouping strategy. Defaults to "all" (no grouping).
        stat : str | list[str]
            Statistic(s), passed straight to aggregate_statistics --
            "mean", "std", "min", "max", "count", or a percentile in any
            of its accepted spellings ("p90", "P90", "90%", "90").
            Defaults to ["mean", "P90", "P99"]. Must be a single string
            (not a list) whenever `by` is not "all". Availability (%) is
            not one of aggregate_statistics's stats; compute it as
            100 * count / count("all") from two separate calls if needed.
        direction : str, optional
            Profile variable to use as direction when by="direction".
            Falls back to `self.direction` if not given; raises if
            neither is available.
        sectors : int or list[float], default 12
            Number of directional sectors, or explicit sector edges.
            Only used when by="direction".
        seasons : dict[str, list[int]], optional
            Custom season -> month-list map. Only used when by="season".

        Returns
        -------
        pd.DataFrame
            Index = level values for var's level dimension. Columns are
            statistic names when by="all", or group labels (+ "All")
            otherwise.
        """
        if var is None:
            var = self.primary.col
        if var not in self.profile_vars:
            raise ValueError(
                f"'{var}' is not a recognised profile variable. "
                f"Available: {self.profile_vars}"
            )

        stats = [stat] if isinstance(stat, str) else list(stat)

        if by != "all" and len(stats) != 1:
            raise ValueError(
                f"statistics(by={by!r}) requires a single `stat`, but got "
                f"{stats!r}. Multiple statistics can only be requested "
                "together when by='all' (no grouping) -- a grouped table "
                "already uses its columns for the groups, so a list of "
                "stats has nowhere to go without a third axis. Either "
                "pass stat='<one statistic>', e.g. stat='mean', or drop "
                "`by` to use the default by='all'."
            )

        level_dim = [d for d in self.ds[var].dims if d != self.time_dim][0]

        def _reduce(group_df: pd.DataFrame, stats_: list[str]) -> pd.DataFrame:
            """Reduce over time (rows) only, keep level (columns) as index."""
            rows = {}
            for level in self.ds.coords[level_dim].values:
                if level not in group_df.columns or group_df[level].dropna().empty:
                    rows[level] = {s: np.nan for s in stats_}
                    continue
                rows[level] = aggregate_statistics(group_df[level].dropna(), stats_)
            return pd.DataFrame(rows).T

        if by == "all":
            full_df = self._profile_var_as_dataframe(var, level_dim)
            table = _reduce(full_df, stats)
            table.index.name = level_dim
            table.columns.name = "stat"
            return table

        groups = self._get_groups(
            by, var, level_dim,
            direction=direction, sectors=sectors, seasons=seasons,
        )
        groups["All"] = self._profile_var_as_dataframe(var, level_dim)

        table = pd.DataFrame({
            label: _reduce(g, stats)[stats[0]] for label, g in groups.items()
        })
        table.index.name = level_dim
        table.columns.name = by
        return table

    # ---------------------------------------------------------------------- #
    # Plotting
    # ---------------------------------------------------------------------- #

    def _level_axis_label(self, level_dim: str) -> str:
        """
        Build the y-axis label for a level dimension: its described name
        if set, falling back to the raw dim name, always with unit
        metres appended (level values are always metres by convention
        in this class -- see Rule 5 / level_convention in the module
        docstring) and the height/depth convention noted only when not
        already implied by the label itself.
        """
        level_attrs = self.ds.coords[level_dim].attrs if level_dim in self.ds.coords else {}
        level_label = level_attrs.get("name", level_dim)
        convention_note = (
            "" if self.level_convention in level_label.lower()
            else f", {self.level_convention}"
        )
        return f"{level_label} (m{convention_note})"

    def plot_profile(
        self,
        time: str | pd.Timestamp | list[str | pd.Timestamp],
        var: str | None = None,
        ax: "plt.Axes | None" = None,
        **plot_kwargs,
    ) -> "plt.Axes":
        """
        Plot one profile variable's vertical profile at one or more
        timestamps: level on the y-axis, value on the x-axis (the
        standard meteorological/oceanographic profile-plot convention),
        one line per requested timestamp.

        Parameters
        ----------
        time : str | pd.Timestamp | list of these
            One timestamp, or a list of timestamps, to plot. Each is
            matched to the *nearest* available value on `var`'s time
            coordinate (exact matches are not required) -- the matched
            timestamp actually used is shown in the legend, so a loose
            match (e.g. "2015-01-01" matching an obs at 2015-01-01
            00:47) is never silently hidden.
        var : str, optional
            Name of the profile variable to plot (must be in
            `self.profile_vars`). Defaults to `self.primary.col`.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure/axes is created if omitted.
        **plot_kwargs
            Passed through to ax.plot() for every line (e.g. linestyle,
            marker, alpha). For per-line color/label control, call this
            method once per timestamp instead.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if var is None:
            var = self.primary.col

        if var not in self.profile_vars:
            raise ValueError(
                f"'{var}' is not a recognised profile variable. "
                f"Available: {self.profile_vars}"
            )

        level_dim = [d for d in self.ds[var].dims if d != self.time_dim][0]
        levels = self.ds.coords[level_dim].values

        times = [time] if isinstance(time, (str, pd.Timestamp)) else list(time)
        requested = pd.DatetimeIndex(sorted(pd.Timestamp(t) for t in times))

        da = self.ds[var]
        matched_idx = da.indexes[self.time_dim].get_indexer(requested, method="nearest")

        if ax is None:
            _, ax = plt.subplots()

        n = len(requested)
        cmap = plt.get_cmap("viridis")
        colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

        for req, idx, color in zip(requested, matched_idx, colors):
            matched_time = pd.Timestamp(da[self.time_dim].values[idx])
            values = da.isel({self.time_dim: idx}).values
            label = matched_time.strftime("%Y-%m-%d %H:%M")
            if abs((matched_time - req).total_seconds()) > 0:
                label += f" (nearest to {req.strftime('%Y-%m-%d %H:%M')})"
            style = dict(color=color) if n > 1 else {}
            style.update(plot_kwargs)
            ax.plot(values, levels, label=label, **style)

        var_attrs = da.attrs
        var_label = var_attrs.get("name", var)
        var_units = var_attrs.get("units")
        ax.set_xlabel(f"{var_label}" + (f" ({var_units})" if var_units else ""))

        ax.set_ylabel(self._level_axis_label(level_dim))
        if self.level_convention == "depth":
            ax.invert_yaxis()

        ax.legend()
        return ax

    def plot_statistics(
        self,
        var: str | None = None,
        by: Literal["all", "month", "season", "direction", "year"] = "all",
        stat: str | list[str] = ("mean", "P90", "P99"),
        direction: str | None = None,
        sectors: int | list[float] = 12,
        seasons: dict[str, list[int]] | None = None,
        ax: "plt.Axes | None" = None,
        **plot_kwargs,
    ) -> "plt.Axes":
        """
        Plot the table produced by statistics(var, by, stat, ...):
        level on the y-axis, statistic value on the x-axis, one line per
        column of that table. Accepts exactly the same grouping/
        statistic options as statistics() -- including the same
        by="all" + multi-stat default, and the same restriction that a
        list of stats is only allowed when by="all" -- since this is
        simply that table, plotted rather than returned as a DataFrame.

        Line coloring: lines are colored along a continuous colormap by
        their position in the table's columns -- in the order `stat`
        was given (by="all") or in statistics()'s natural group order
        (calendar order for month/season/year, ascending compass angle
        for direction). Never re-sorted, since that order is already
        meaningful. There is no separate "All" line unless by="all" --
        a grouped plot (by="month" etc.) only ever shows the individual
        groups, since the whole-record line would need to be requested
        as its own statistics(by="all") table/plot instead.

        Parameters
        ----------
        var : str, optional
            Name of the profile variable to plot. Defaults to
            `self.primary.col`, exactly as in statistics().
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure/axes is created if omitted.
        **plot_kwargs
            Passed through to ax.plot() for every line.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        # var resolution (None -> self.primary.col) is delegated to
        # statistics() so the two methods' defaulting behaviour stays
        # identical by construction. The resolved name is still needed
        # below (axis labels, ds[var].attrs), so mirror the same default
        # locally rather than re-deriving it some other way.
        if var is None:
            var = self.primary.col
        table = self.statistics(
            var=var, by=by, stat=stat, direction=direction,
            sectors=sectors, seasons=seasons,
        )

        if by != "all" and "All" in table.columns:
            table = table.drop(columns="All")

        if by == "direction":
            # groupby_sector labels sort alphabetically as plain strings
            # ("112-158°" < "22-68°") -- reorder by each sector's
            # starting compass angle so the legend/colormap progress
            # sensibly, rather than re-sorting arbitrarily.
            def _sector_start_angle(label: str) -> float:
                return float(label.split("-")[0])
            table = table[sorted(table.columns, key=_sector_start_angle)]

        if ax is None:
            _, ax = plt.subplots()

        n = table.shape[1]
        cmap = plt.get_cmap("viridis")
        colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

        for color, (label, series) in zip(colors, table.items()):
            style = dict(color=color) if n > 1 else {}
            style.update(plot_kwargs)
            ax.plot(series.values, series.index.values, label=str(label), **style)

        da = self.ds[var]
        var_attrs = da.attrs
        var_label = var_attrs.get("name", var)
        var_units = var_attrs.get("units")
        ax.set_xlabel(f"{var_label}" + (f" ({var_units})" if var_units else ""))

        level_dim = table.index.name
        ax.set_ylabel(self._level_axis_label(level_dim))
        if self.level_convention == "depth":
            ax.invert_yaxis()

        ax.legend(title=table.columns.name)
        return ax


def _get_n_axes(n_intervals: int, max_cols: int = 4):
    """
    Same subplot-grid layout UnivariateEVA uses for its per-group plots
    (plot_return_value_comparison, plot_return_value_confidence) --
    duplicated here rather than imported, since it's a small, dependency-
    free helper and ProfileEVA's plot_return_values uses the identical
    "one subplot per group, laid out as squarely as possible" logic for
    its per-month/per-sector subplots.
    """
    import matplotlib.pyplot as plt

    if n_intervals <= 0:
        raise ValueError(
            f"n_intervals should be a positive integer, but got {n_intervals}."
        )
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
        nrows=nrows, ncols=ncols, sharex=False, sharey=False, squeeze=False, layout="constrained"
    )
    return fig, axes.ravel()


class ProfileEVA:
    """
    Extreme value analysis for one profile variable, broadcast across
    every level of that variable's level dimension.

    Not constructed directly -- every ``ProfileTimeSeries`` builds its own
    ``self.EVA`` at ``__init__``, scoped to ``self.primary``. Requesting
    EVA on a different profile variable means constructing a new
    ``ProfileTimeSeries`` with that variable as ``primary`` (cheap: this
    class does no fitting work itself, and building it does not run
    ``UnivariateEVA`` for any level until ``get_extremes``/``fit`` is
    actually called for that level).

    One ``UnivariateEVA`` instance exists per level, indexed by that
    level's coordinate value in ``pts.levels(level_dim)``. Each is built
    lazily -- the first time a level is touched by ``get_extremes`` -- and
    cached in ``self.models`` from then on. ``fit`` reuses whatever
    ``UnivariateEVA`` instance already exists per level (it does not
    rebuild it), so the normal flow is always
    ``get_extremes()`` -> ``fit()`` -> one of the plot/table methods, in
    that order, exactly mirroring ``UnivariateEVA``'s own three-step
    shape -- just repeated once per level instead of once total.

    Parameters
    ----------
    pts : ProfileTimeSeries
        The parent instance this accessor is scoped to.
    var : str
        Name of the profile variable EVA is being run on (must be a
        profile variable in ``pts.profile_vars`` -- checked by
        ``ProfileTimeSeries.__init__`` before this is constructed, since
        that's where ``primary`` itself is validated).

    Attributes
    ----------
    var : str
    level_dim : str
        The level dimension `var` varies over.
    levels : np.ndarray
        Every level value on `level_dim`, in dataset order -- the full
        set `get_extremes`/`fit` broadcast over when no `levels=` subset
        is given.
    models : dict[float, UnivariateEVA]
        Cache of per-level UnivariateEVA instances, keyed by level value.
        Empty until `get_extremes` has been called for at least one
        level.
    """

    def __init__(self, pts: "ProfileTimeSeries", var: str):
        self._pts = pts
        self.var = var
        self.level_dim = [d for d in pts.ds[var].dims if d != pts.time_dim][0]
        self.levels = pts.ds.coords[self.level_dim].values
        self.models: dict[float, UnivariateEVA] = {}

        var_meta = pts._variables[var]
        self._var_name = var_meta.name
        self._var_symbol = var_meta.symbol
        self._var_unit = var_meta.unit

        self._dir_var = pts.direction.col if pts.direction is not None else None
        if self._dir_var is not None:
            # ProfileTimeSeries.__init__ already enforces that direction
            # shares primary's level dimension when direction is given at
            # all -- but EVA is scoped to whichever var was passed here,
            # which is *always* primary in practice (see class docstring),
            # so this should never actually fire. Kept as a genuine check
            # rather than an assert: if ProfileEVA is ever constructed for
            # a non-primary var in the future, silently pairing mismatched
            # level grids is exactly the failure mode that check exists to
            # prevent.
            dir_level_dim = [
                d for d in pts.ds[self._dir_var].dims if d != pts.time_dim
            ][0]
            if dir_level_dim != self.level_dim:
                raise ValueError(
                    f"direction='{self._dir_var}' varies over level "
                    f"dimension '{dir_level_dim}', but EVA is scoped to "
                    f"'{var}', which varies over '{self.level_dim}'. "
                    "EVA's per-level sector splitting requires direction "
                    "to share the same level dimension as the variable "
                    "being analysed."
                )

    def _level_dataframe(self, level) -> pd.DataFrame:
        """
        Build the (time-indexed, single-level) DataFrame UnivariateEVA
        expects: one column for `var`, plus a direction column if
        `direction` is set -- both sliced at `level`, dropping rows where
        either is NaN (UnivariateEVA has no NaN-handling of its own).
        """
        var_series = self._pts.ds[self.var].sel({self.level_dim: level}).to_series()
        var_series.name = self.var
        if self._dir_var is not None:
            dir_series = self._pts.ds[self._dir_var].sel(
                {self.level_dim: level}
            ).to_series()
            dir_series.name = self._dir_var
            df = pd.concat([var_series, dir_series], axis=1).dropna()
        else:
            df = var_series.to_frame().dropna()
        return df

    def _resolve_levels(self, levels) -> list:
        """
        `levels=None` (the default for both get_extremes and fit) means
        "every level" -- broadcasting over the full set is the common
        case (temperature, wind speed, current speed profiles are
        usually analysed level-by-level in full, not level-by-level on
        request), so no-argument calls do the whole profile rather than
        requiring the caller to enumerate every level themselves.
        """
        if levels is None:
            return list(self.levels)
        return list(levels)

    def get_extremes(
        self,
        levels: list[float] | None = None,
        th_omni: float | None = None,
        th_monthly: list[float] | None = None,
        th_sectors: list[float] | None = None,
        block_size: str = "365.2425D",
        min_last_block: float = 0.9,
        r: str = "48h",
        extremes_type: str = "high",
        errors: str = "raise",
        th_percentile: float = 0.98,
    ) -> "ProfileEVA":
        """
        Build (or rebuild) a UnivariateEVA for each requested level and
        run its `get_extremes(...)`. All keyword arguments are passed
        straight through to `UnivariateEVA.get_extremes` unchanged, so
        every level is extracted with the same threshold/blocking
        settings -- per-level threshold tuning is possible by calling
        this once per level with different kwargs, since it is keyed by
        level and safe to call again for a single level without
        disturbing the others.

        Parameters
        ----------
        levels : list[float], optional
            Subset of levels (values from `self.levels`) to process.
            Defaults to every level.
        th_omni, th_monthly, th_sectors, block_size, min_last_block, r,
        extremes_type, errors, th_percentile
            Passed straight through to `UnivariateEVA.get_extremes`. See
            that method for details -- thresholds left as None there are
            derived per level from `th_percentile`, independently for
            each level's own data.

        Returns
        -------
        ProfileEVA
            self, so this chains directly into `.fit(...)`.
        """
        for level in tqdm_levels(self._resolve_levels(levels), desc=f"EVA: {self.var}"):
            df = self._level_dataframe(level)
            model = UnivariateEVA(
                data=df,
                var=self.var,
                var_dir=self._dir_var,
                var_name=self._var_name,
                var_symbol=self._var_symbol,
                var_unit=self._var_unit,
                sectors=getattr(self._pts, "sectors", 12),
            )
            model.get_extremes(
                th_omni=th_omni,
                th_monthly=th_monthly,
                th_sectors=th_sectors,
                block_size=block_size,
                min_last_block=min_last_block,
                r=r,
                extremes_type=extremes_type,
                errors=errors,
                th_percentile=th_percentile,
            )
            self.models[level] = model
        return self

    def fit(
        self,
        levels: list[float] | None = None,
        AM_dist: list[str] = [],
        POT_dist: list[str] = [],
        IDM_dist: list[str] = [],
        errors: str = "raise",
    ) -> "ProfileEVA":
        """
        Fit every already-extracted level's UnivariateEVA. All keyword
        arguments are passed straight through to
        `UnivariateEVA.fit`. Raises per level if that level has no
        extremes yet -- `get_extremes` must be called (for that level,
        at least) before `fit`, matching UnivariateEVA's own
        `_got_extremes` check.

        Parameters
        ----------
        levels : list[float], optional
            Subset of levels to fit. Defaults to every level that has
            already had `get_extremes` run -- i.e. every key currently
            in `self.models` -- not `self.levels`, since a level with no
            extremes yet has nothing to fit.
        AM_dist, POT_dist, IDM_dist, errors
            Passed straight through to `UnivariateEVA.fit`.

        Returns
        -------
        ProfileEVA
            self, so this chains directly into a plot/table method.
        """
        target_levels = list(self.models) if levels is None else list(levels)
        missing = [lv for lv in target_levels if lv not in self.models]
        if missing:
            raise ValueError(
                f"No extremes found for level(s) {missing} -- call "
                "get_extremes(levels=...) for these levels before fit()."
            )

        for level in tqdm_levels(target_levels, desc=f"Fitting: {self.var}"):
            self.models[level].fit(
                AM_dist=AM_dist, POT_dist=POT_dist, IDM_dist=IDM_dist,
                errors=errors,
            )
        return self

    def table_return_values(
        self,
        return_period: float,
        grouping: Literal["omni", "monthly", "sectors"],
        method: str,
        dist: str,
        return_period_size: str = "365.2425D",
        levels: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Return-value table for one return period, with level on the
        index and month/sector on the columns.

        This is UnivariateEVA.table_return_values_final turned
        sideways: that method fixes one level (it only ever sees one
        series) and puts return period on the index, month/sector on
        the columns. Here there are multiple levels, so the axis that
        used to hold return periods now holds levels instead -- and
        return period becomes something you fix per call, since a
        record usually has far more months/sectors (12) than return
        periods worth tabulating (rarely more than 3-4), so "one table
        per return period" produces fewer, more readable tables than
        "one table per level" would.

        Built by calling `UnivariateEVA.table_return_values_final` on
        each level's model and pulling out the one requested return
        period's column -- not reimplemented, so the underlying
        Poisson-correction / return-value math always stays whatever
        UnivariateEVA does internally.

        Parameters
        ----------
        return_period : float
            The single return period (in years) to tabulate. Must
            appear, spelled exactly the way UnivariateEVA labels it
            ("{return_period}-year"), in the columns produced by
            grouping="omni"'s table -- in practice this just means
            passing the same number you'd pass in `return_periods` to
            UnivariateEVA.table_return_values_final.
        grouping : {"omni", "monthly", "sectors"}
            Passed straight through to
            `UnivariateEVA.table_return_values_final` for each level.
            "omni" produces a single-column table (no month/sector
            split) -- still valid here, just not very useful compared
            to calling statistics() directly.
        method : str
            "AM", "POT", or "IDM" -- passed straight through.
        dist : str
            Distribution name (shorthand or full scipy name) -- passed
            straight through.
        return_period_size : str, default "365.2425D"
            Passed straight through.
        levels : list[float], optional
            Subset of levels to include as rows. Defaults to every
            level that has a fitted model (`self.models` with
            `_fitted_models=True`) -- a level with no fitted model for
            this (method, dist) has nothing to look up and is skipped
            with a warning rather than raising, since a partially
            fitted profile (e.g. one level's fit failed and was
            skipped with errors="warn" during `.fit()`) shouldn't block
            tabulating the levels that did fit.

        Returns
        -------
        pd.DataFrame
            Index = level values (name = self.level_dim), columns =
            month/sector labels (or a single "Omni" column if
            grouping="omni"). Values are return-value estimates for the
            requested return period, sourced by column from the
            corresponding *grouping subset's* actual model, exactly as
            UnivariateEVA.table_return_values_final's own columns are.
        """
        rp_label = f"{return_period}-year"
        target_levels = list(self.models) if levels is None else list(levels)

        rows = {}
        skipped = []
        for level in target_levels:
            if level not in self.models or not self.models[level]._fitted_models:
                skipped.append(level)
                continue
            level_table = self.models[level].table_return_values_final(
                grouping=grouping,
                method=method,
                dist=dist,
                return_periods=[return_period],
                return_period_size=return_period_size,
            )
            if rp_label not in level_table.columns:
                raise ValueError(
                    f"return_period={return_period} produced column "
                    f"'{rp_label}', but that isn't in the table returned "
                    f"by UnivariateEVA for level={level!r} (columns: "
                    f"{list(level_table.columns)}). This should not "
                    "happen -- please report this."
                )
            rows[level] = level_table[rp_label]

        if skipped:
            warnings.warn(
                f"Skipped level(s) {skipped}: no fitted model for "
                f"({method}, {dist}) at this level. Call "
                "get_extremes(levels=...) and fit(levels=...) for these "
                "levels first if they should be included.",
                UserWarning,
                stacklevel=2,
            )
        if not rows:
            raise ValueError(
                "No level had a fitted model to tabulate -- call "
                "get_extremes() and fit() first."
            )

        table = pd.DataFrame(rows).T
        table.index.name = self.level_dim
        table.columns.name = "Omni" if grouping == "omni" else grouping
        return table


    def plot_return_values(
        self,
        return_periods: list[float],
        grouping: Literal["omni", "monthly", "sectors"],
        method: str,
        dist: str,
        return_period_size: str = "365.2425D",
        levels: list[float] | None = None,
        subplot_columns: int | None = None,
        plot_kwargs: dict | None = None,
        ax: "np.ndarray[plt.Axes] | None" = None,
    ) -> "np.ndarray[plt.Axes]":
        """
        One subplot per month/sector (laid out the same way
        `UnivariateEVA.plot_return_value_comparison` lays out its
        per-group subplots, via the same `_get_n_axes` grid logic), each
        showing the vertical profile -- level on the y-axis, return
        value on the x-axis, one line per requested return period.
        "omni" produces a single subplot (no month/sector split), same
        convention as `table_return_values`.

        Built on top of `table_return_values` -- one call per requested
        return period, then re-sliced by column (month/sector) to get
        each subplot's per-return-period lines -- so the same
        UnivariateEVA return-value math and the same skipped-level
        handling apply here too.

        Parameters
        ----------
        return_periods : list[float]
            Return periods (years) to plot as separate lines within
            each subplot.
        grouping : {"omni", "monthly", "sectors"}
            Which set of subplots to draw -- one subplot per month (12),
            one per sector, or a single "Omni" subplot with no
            month/sector split.
            For "monthly"/"sectors", the aggregate "Yearly"/"Omni"
            column that `table_return_values` includes for those
            groupings is dropped here -- every month/sector already gets
            its own subplot, so a 13th/extra aggregate subplot isn't
            wanted. For "omni" that column *is* the only column, so
            nothing is dropped.
        method : str
            "AM", "POT", or "IDM" -- passed straight through.
        dist : str
            Distribution name -- passed straight through.
        return_period_size : str, default "365.2425D"
            Passed straight through.
        levels : list[float], optional
            Subset of levels to include. Defaults to every fitted level,
            same as `table_return_values`.
        subplot_columns : int, optional
            Max columns in the subplot grid. Defaults to 3, matching
            `UnivariateEVA.plot_return_value_comparison`'s default.
        plot_kwargs : dict, optional
            Passed to `ax.plot()` for every line.
        ax : array of matplotlib.axes.Axes, optional
            Axes to draw on -- one per month/sector, already flattened
            or in any array shape. A new grid is created via the same
            `_get_n_axes` layout `UnivariateEVA` uses if omitted.

        Returns
        -------
        np.ndarray of matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        plot_kwargs = {} if plot_kwargs is None else dict(plot_kwargs)

        # One table per return period (table_return_values' own
        # reasoning: few return periods, many months/sectors), each
        # giving level-indexed values for every month/sector column.
        # The aggregate "Yearly"/"Omni" column is only dropped for
        # monthly/sectors -- for grouping="omni" it's the sole column
        # and must stay, since dropping it would leave an empty table.
        drop_cols = ["Yearly", "Omni"] if grouping in ("monthly", "sectors") else []
        tables = {
            rp: self.table_return_values(
                return_period=rp, grouping=grouping, method=method,
                dist=dist, return_period_size=return_period_size,
                levels=levels,
            ).drop(columns=drop_cols, errors="ignore")
            for rp in return_periods
        }

        # Every table shares the same columns (same grouping, same
        # skipped-level handling was applied identically to each) --
        # take the column order from the first table rather than
        # re-deriving or re-sorting it, since UnivariateEVA's own
        # column order (calendar order for monthly, sector-definition
        # order for sectors) is already meaningful.
        first_table = next(iter(tables.values()))
        columns = list(first_table.columns)

        if ax is None:
            if subplot_columns is None:
                subplot_columns = 3
            fig, ax_arr = _get_n_axes(len(columns), max_cols=subplot_columns)
        else:
            ax_arr = np.array(ax).ravel()
            if len(ax_arr) < len(columns):
                raise ValueError(
                    f"Expected ax to contain {len(columns)} axes "
                    f"(one per {grouping} group), got {len(ax_arr)}."
                )

        cmap = plt.get_cmap("viridis")
        n_rp = len(return_periods)
        colors = [cmap(i / max(n_rp - 1, 1)) for i in range(n_rp)]

        primary_var_meta = self._pts._variables[self.var]
        var_label = primary_var_meta.name
        var_units = primary_var_meta.unit

        for col_idx, column in enumerate(columns):
            axi = ax_arr[col_idx]
            for rp, color in zip(return_periods, colors):
                table = tables[rp]
                style = dict(color=color) if n_rp > 1 else {}
                style.update(plot_kwargs)
                axi.plot(
                    table[column].values, table.index.values,
                    label=f"{rp}-year", **style,
                )
            axi.set_title(str(column))
            axi.set_xlabel(
                f"{var_label}" + (f" ({var_units})" if var_units else "")
            )
            axi.set_ylabel(self._pts._level_axis_label(self.level_dim))
            if self._pts.level_convention == "depth":
                axi.invert_yaxis()
            axi.grid(True)
            axi.legend(title="Return period")

        return ax_arr



    # ---------------------------------------------------------------------- #
    # Turkstra profiles
    # ---------------------------------------------------------------------- #

    def _turkstra_series(self, month: str | None = None) -> pd.DataFrame:
        """
        Raw (time-indexed, one column per level) DataFrame for `self.var`
        at every level, optionally restricted to one month. This is the
        input the Turkstra correlation/regression works from -- always
        the *raw* series, never anything derived from a fitted
        UnivariateEVA, since Turkstra's cross-level correlation has
        nothing to do with the marginal extreme-value fit at any single
        level.

        Month filtering mirrors `groupby_month`'s own month-name
        resolution (full or abbreviated name, case-insensitive) rather
        than reimplementing a separate convention -- built directly here
        instead of calling `groupby_month` itself, since that groups
        into all 12 months and Turkstra only ever wants one.
        """
        df = self._pts._profile_var_as_dataframe(self.var, self.level_dim)
        if month is None:
            return df

        months_full = ['January', 'February', 'March', 'April', 'May', 'June',
                        'July', 'August', 'September', 'October', 'November', 'December']
        months_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_lower = month.lower()
        if month_lower in [m.lower() for m in months_full]:
            month_num = [m.lower() for m in months_full].index(month_lower) + 1
        elif month_lower in [m.lower() for m in months_abbr]:
            month_num = [m.lower() for m in months_abbr].index(month_lower) + 1
        else:
            raise ValueError(
                f"Unrecognised month '{month}'. Expected a full month name "
                f"(e.g. 'January') or abbreviation (e.g. 'Jan')."
            )
        return df[df.index.month == month_num]

    def _turkstra_matrix(
        self, df: pd.DataFrame, worst_case: pd.Series
    ) -> pd.DataFrame:
        """
        Core Turkstra computation, shared by both bases (percentile and
        return_period) -- the only thing that differs between them is
        how `worst_case` (the per-level anchor value) was obtained; the
        regression/clipping logic itself is identical either way.

        For each anchor level `d`, every other level `dd` is set to its
        conditional-expectation value given that level `d` sits at its
        worst-case value:

            value[dd, d] = mean[dd] + corr(d, dd) * std[dd]
                           * (worst_case[d] - mean[d]) / std[d]

        `value[d, d]` is exactly `worst_case[d]` by construction (level
        `d` is defined to be at its own worst case in its own column).

        Off-diagonal entries are clipped back down to `worst_case[dd]`
        whenever the regression overshoots it -- this is a real
        possibility, not just defensive rounding: the regression line is
        an unbounded linear extrapolation, and `worst_case[d]` sits far
        in the tail of level `d`'s distribution, so nothing about the
        formula guarantees `value[dd, d] <= worst_case[dd]`. Since
        Turkstra profiles exist specifically to avoid over-conservatism
        (vs. assuming every level is simultaneously at its own worst
        case), a profile value that exceeds a level's own independently
        -estimated worst case would undercut that purpose, so it's
        capped there. The diagonal is never touched by this clipping --
        it's the anchor itself.

        Parameters
        ----------
        df : pd.DataFrame
            Raw (time, level) series, one column per level, already
            restricted to whatever subset (e.g. month) is in scope.
            Column labels must match `worst_case`'s index.
        worst_case : pd.Series
            Anchor value per level, indexed the same way as `df`'s
            columns.

        Returns
        -------
        pd.DataFrame
            Index = level (rows, i.e. "value read off at this level"),
            columns = anchor level (i.e. "which level is pinned at its
            worst case"). `table_turkstra` adds the "Worst case" column
            on top of this.
        """
        levels = list(df.columns)
        means = df.mean(axis=0)
        stds = df.std(axis=0)
        corr = df.corr()

        matrix = pd.DataFrame(index=levels, columns=levels, dtype=float)
        for anchor in levels:
            if stds[anchor] == 0 or pd.isna(stds[anchor]):
                raise ValueError(
                    f"Level {anchor!r} has zero or undefined standard "
                    "deviation in the data used for this Turkstra "
                    "profile -- cannot standardize against it as an "
                    "anchor level."
                )
            z_anchor = (worst_case[anchor] - means[anchor]) / stds[anchor]
            for other in levels:
                if other == anchor:
                    matrix.loc[other, anchor] = worst_case[anchor]
                    continue
                value = means[other] + corr.loc[anchor, other] * stds[other] * z_anchor
                matrix.loc[other, anchor] = min(value, worst_case[other])

        return matrix

    def table_turkstra(
        self,
        basis: Literal["percentile", "return_period"],
        percentile: float | None = None,
        month: str | None = None,
        return_period: float | None = None,
        method: str | None = None,
        dist: str | None = None,
        return_period_size: str = "365.2425D",
        levels: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Turkstra combination-load profile table: for each level, the
        conditionally-expected value given that some *other* level is
        pinned at its own worst-case value -- one column per anchor
        level, plus a "Worst case" column giving each level's own
        independent worst-case value.

        Two ways to define "worst case" at a level, chosen by `basis`:

          - "percentile": worst case = the `percentile`-th percentile of
            the raw data at that level (optionally restricted to one
            `month`). No EVA required -- this basis only ever touches
            the raw time series in `self._pts.ds`.
          - "return_period": worst case = the `return_period`-year
            return value from a fitted `(method, dist)` model at that
            level, taken from `self.table_return_values(...,
            grouping="omni")` -- which means `get_extremes()` and
            `fit()` must already have been run for the levels involved
            (with that `method`/`dist` combination), exactly as
            `table_return_values` itself requires. Levels with no
            fitted model are skipped with the same warning
            `table_return_values` already gives.

        In both cases, the cross-level *correlation* used for the
        regression always comes from the raw data (optionally
        month-filtered for the percentile basis) -- correlation is a
        property of the joint distribution, not something a marginal
        EVA fit at a single level can supply, so the return_period
        basis still needs the raw series for that part even though the
        anchor values themselves come from EVA.

        Parameters
        ----------
        basis : {"percentile", "return_period"}
            Which way to define each level's own worst-case value.
        percentile : float, required if basis="percentile"
            Percentile (e.g. 99 for P99) defining the worst case at
            each level.
        month : str, optional
            Restrict the raw data (both the worst-case percentile and
            the correlation) to one month (full name or abbreviation,
            e.g. "January" or "Jan"). Only used for basis="percentile"
            -- return_period's worst case already comes from a
            whole-record EVA fit, so month-restricting only the
            correlation half while leaving the anchor whole-record
            would mix two different subsets of data into one profile;
            raises if given together with basis="return_period".
        return_period : float, required if basis="return_period"
            Return period (years) defining the worst case at each
            level.
        method : str, required if basis="return_period"
            "AM", "POT", or "IDM" -- passed to `table_return_values`.
        dist : str, required if basis="return_period"
            Distribution name -- passed to `table_return_values`.
        return_period_size : str, default "365.2425D"
            Passed to `table_return_values` when basis="return_period".
        levels : list[float], optional
            Subset of levels to include. Defaults to every level in
            `self.levels` (basis="percentile") or every fitted level
            (basis="return_period", same default as
            `table_return_values`).

        Returns
        -------
        pd.DataFrame
            Index = level values (name = self.level_dim), one column
            per anchor level plus a final "Worst case" column. Entry
            (row=`dd`, col=`d`) is the conditionally-expected value at
            level `dd` given level `d` is at its own worst case;
            diagonal entries equal that level's own "Worst case" value
            exactly.
        """
        if basis not in ("percentile", "return_period"):
            raise ValueError(
                f"basis must be 'percentile' or 'return_period', got {basis!r}."
            )

        if basis == "percentile":
            if percentile is None:
                raise ValueError("percentile is required when basis='percentile'.")
            if return_period is not None or method is not None or dist is not None:
                raise ValueError(
                    "return_period/method/dist are not used when "
                    "basis='percentile' -- did you mean basis='return_period'?"
                )

            df = self._turkstra_series(month=month)
            target_levels = list(self.levels) if levels is None else list(levels)
            missing = [lv for lv in target_levels if lv not in df.columns]
            if missing:
                raise ValueError(
                    f"Level(s) {missing} not found in the data for "
                    f"'{self.var}'. Available levels: {list(df.columns)}."
                )
            df = df[target_levels].dropna()
            if df.empty:
                raise ValueError(
                    "No overlapping non-NaN data across the requested "
                    "levels" + (f" for month={month!r}" if month else "")
                    + " -- cannot compute a Turkstra profile."
                )
            worst_case = df.apply(lambda col: np.percentile(col.dropna(), percentile))

        else:  # basis == "return_period"
            if month is not None:
                raise ValueError(
                    "month is only used with basis='percentile' -- "
                    "return_period's worst case comes from a whole-record "
                    "EVA fit, so it cannot be restricted to one month here."
                )
            if return_period is None or method is None or dist is None:
                raise ValueError(
                    "return_period, method, and dist are all required "
                    "when basis='return_period'."
                )

            rp_table = self.table_return_values(
                return_period=return_period, grouping="omni",
                method=method, dist=dist,
                return_period_size=return_period_size, levels=levels,
            )
            worst_case = rp_table["Omni"]
            target_levels = list(worst_case.index)

            df = self._turkstra_series(month=None)
            missing = [lv for lv in target_levels if lv not in df.columns]
            if missing:
                raise ValueError(
                    f"Level(s) {missing} (from fitted models) not found "
                    f"in the raw data for '{self.var}'. Available levels: "
                    f"{list(df.columns)}."
                )
            df = df[target_levels].dropna()
            if df.empty:
                raise ValueError(
                    "No overlapping non-NaN data across the fitted "
                    "levels -- cannot compute a Turkstra profile."
                )

        matrix = self._turkstra_matrix(df, worst_case)
        matrix["Worst case"] = worst_case.loc[matrix.index]
        matrix.index.name = self.level_dim
        matrix.columns.name = "anchor_level"
        return matrix

    def plot_turkstra(
        self,
        basis: Literal["percentile", "return_period"],
        percentile: float | None = None,
        month: str | None = None,
        return_period: float | None = None,
        method: str | None = None,
        dist: str | None = None,
        return_period_size: str = "365.2425D",
        levels: list[float] | None = None,
        plot_kwargs: dict | None = None,
        worst_case_kwargs: dict | None = None,
        ax: "plt.Axes | None" = None,
    ) -> "plt.Axes":
        """
        Plot the table produced by `table_turkstra`: level on the
        y-axis, value on the x-axis, one line per anchor-level column
        plus a bold "Worst case" line -- a single axes, unlike
        `plot_return_values`'s one-subplot-per-month/sector grid, since
        Turkstra profiles have no month/sector split of their own (the
        anchor-level columns already are the thing being compared
        within one plot, the way the old CCA plot did it).

        Accepts exactly the same basis/percentile/month/return_period/
        method/dist/return_period_size/levels arguments as
        `table_turkstra` -- this simply plots that table.

        Parameters
        ----------
        plot_kwargs : dict, optional
            Passed to `ax.plot()` for every anchor-level line.
        worst_case_kwargs : dict, optional
            Passed to `ax.plot()` for the "Worst case" line, on top of
            a bold-black default (`color="black", linewidth=2.5`) --
            matching the old CCA plot's convention of visually
            distinguishing the worst-case envelope from the individual
            anchor-level profiles.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure/axes is created if omitted.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        plot_kwargs = {} if plot_kwargs is None else dict(plot_kwargs)
        worst_case_kwargs = {} if worst_case_kwargs is None else dict(worst_case_kwargs)

        table = self.table_turkstra(
            basis=basis, percentile=percentile, month=month,
            return_period=return_period, method=method, dist=dist,
            return_period_size=return_period_size, levels=levels,
        )
        anchor_columns = [c for c in table.columns if c != "Worst case"]

        if ax is None:
            _, ax = plt.subplots()

        n = len(anchor_columns)
        cmap = plt.get_cmap("viridis")
        colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

        for anchor, color in zip(anchor_columns, colors):
            style = dict(color=color)
            style.update(plot_kwargs)
            ax.plot(
                table[anchor].values, table.index.values,
                label=str(anchor), **style,
            )

        wc_style = dict(color="black", linewidth=2.5)
        wc_style.update(worst_case_kwargs)
        ax.plot(
            table["Worst case"].values, table.index.values,
            label="Worst case", **wc_style,
        )

        primary_var_meta = self._pts._variables[self.var]
        var_label = primary_var_meta.name
        var_units = primary_var_meta.unit
        ax.set_xlabel(f"{var_label}" + (f" ({var_units})" if var_units else ""))
        ax.set_ylabel(self._pts._level_axis_label(self.level_dim))
        if self._pts.level_convention == "depth":
            ax.invert_yaxis()

        if basis == "percentile":
            title = f"Turkstra profile — P{percentile}"
            if month is not None:
                title += f" ({month})"
        else:
            title = f"Turkstra profile — {return_period}-year ({method}, {dist})"
        ax.set_title(title)

        ax.grid(True, color="lightgray", linestyle=":")
        ax.legend(title="Anchor level", loc="upper left", bbox_to_anchor=(1, 1))
        return ax


    def __getitem__(self, level) -> UnivariateEVA:
        """
        Drop down to the raw UnivariateEVA for one level -- the escape
        hatch for anything this accessor doesn't wrap directly (e.g.
        `table_model_parameters`, `plot_return_value_confidence`, or any
        future UnivariateEVA method). Raises a clear error rather than a
        bare KeyError if the level hasn't been extracted yet.
        """
        if level not in self.models:
            raise KeyError(
                f"No UnivariateEVA for level={level!r} yet -- call "
                f"get_extremes(levels=[{level!r}]) first. Levels "
                f"currently available: {list(self.models)}."
            )
        return self.models[level]

    def __repr__(self) -> str:
        n_fitted = sum(1 for m in self.models.values() if m._fitted_models)
        return (
            f"ProfileEVA(var='{self.var}', level_dim='{self.level_dim}', "
            f"levels={len(self.levels)}, extracted={len(self.models)}, "
            f"fitted={n_fitted})"
        )


def tqdm_levels(levels, desc=""):
    """
    Thin tqdm wrapper so ProfileEVA's per-level loops show progress the
    same way UnivariateEVA.fit's per-distribution loop already does,
    without hard-requiring tqdm at import time if it's ever missing --
    falls back to the plain iterable.
    """
    try:
        from tqdm import tqdm
        return tqdm(levels, desc=desc)
    except ImportError:
        return levels