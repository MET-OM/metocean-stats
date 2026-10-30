"""
ProfileTimeSeries
-----------------
Metocean profile data at a single location: one or more variables that vary
across levels (height above or depth below a reference) and time, such as
wind speed, temperature, currents or salinity.

Validation runs in full during construction, in this order:

1. Squeeze: length-1 dimensions are dropped.
2. Dimensionality: no variable may have more than two dimensions.
3. Time: every two-dimensional variable must share one datetime dimension,
   the time dimension.
4. Level sign: each level dimension must be single-signed (all values >= 0 or
   all <= 0). Mixed sign conventions are rejected.
5. Normalization: level values are replaced by their absolute values. Their
   meaning is recorded from the declared ``level_convention``.

Variables are then classified by dimension:

- ``profile_vars``: time and level.
- ``scalar_vars``: time only.
- ``level_only_vars``: level only, with no time axis.
- ``static_vars``: no dimensions.

Different variables may use different level grids. Dimension names are kept.
The dataset in ``ds`` is the only copy of the data. Interpolation always
returns a new result.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from .utils import aggregate_statistics, dirmag_to_uv, uv_to_dirmag, infer_step, _extend_xlim_to_zero, _extend_ylim_to_zero, groupby_with_metadata, get_n_axes
from .EVA import UnivariateEVA
from .plotting import plot_rose

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
from .timeseries import Variable, TimeSeries

def _coerce_variable(v: "Variable | str") -> "Variable":
    """
    Promote a bare variable-name string to a Variable with placeholder
    semantics (name = symbol = col, unit = ""). Accepts any object with
    the required attributes (col, name, symbol, unit), not just instances
    of this exact Variable class -- so a Variable-like object from
    another module works the same as one constructed here.
    """
    if isinstance(v, str):
        return Variable(col=v, name=v, symbol=v, unit="")

    required = ("col", "name", "symbol", "unit")
    missing = [attr for attr in required if not hasattr(v, attr)]
    if missing:
        raise TypeError(
            f"Expected a Variable or a column name string, got {type(v)} "
            f"missing attribute(s) {missing}. "
            "Example: Variable('ws', 'Wind speed', 'U_{10}', 'm/s') or just 'ws'."
        )
    return v


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
    Profile of one or more metocean variables over levels and time.

    Built from a single ``xarray.Dataset`` or ``xarray.DataArray``. See the
    module docstring for the validation steps.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        Source data. A DataArray is converted to a one-variable Dataset,
        named after its ``name`` or ``"value"``. Dimension names are kept.
        Variables may use different level dimensions.
    level_convention : {"height", "depth"}
        Meaning of the level values. "height" is distance above a reference,
        increasing upward. "depth" is distance below a reference, increasing
        downward. This is declared by the user and not inferred from the
        data. Mixing the two conventions in one instance is not supported.
    primary : Variable or str
        Primary profile variable, used as the default by the statistics,
        plotting and EVA methods. A string is converted to a placeholder
        `Variable`. Must be a profile variable.
    direction : Variable or str, optional
        Profile variable holding direction, used for direction-based grouping
        and sector EVA. Must share the level dimension of `primary`.
    name : str, optional
        Label for this record, such as a site, model run or instrument.

    Raises
    ------
    TypeError
        If `data` is not an xarray object.
    ValueError
        If the data fails a validation step, or `primary` or `direction` is
        not a suitable profile variable.

    Notes
    -----
    Validation runs in full during construction, so a successfully built
    instance is consistent. Ambiguous input is rejected rather than guessed
    at.
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

        #: Declared level convention, "height" or "depth".
        self.level_convention = level_convention
        #: Label for this record, or None.
        self.name = name

        # ------------------------------------------------------------------ #
        # 1. Squeeze -- drop any length-1 dimension entirely.                #
        # ------------------------------------------------------------------ #
        squeezed_dims = [d for d, n in ds.sizes.items() if n == 1]
        if squeezed_dims:
            ds = ds.squeeze(dim=squeezed_dims, drop=False)
            warnings.warn(
                f"Squeezed length-1 dimension(s): {squeezed_dims}.",
                UserWarning,
                stacklevel=2,
            )

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
        #: The validated dataset, with squeezed dimensions and absolute level values.
        self.ds = ds
        #: Name of the shared time dimension.
        self.time_dim = time_dim
        #: Names of all level dimensions.
        self.level_dims = level_dims
        #: Variables with signature {time, level}.
        self.profile_vars = profile_vars
        #: Variables with signature {time} only.
        self.scalar_vars = scalar_vars
        #: Variables with signature {level} only, with no time axis.
        self.level_only_vars = level_only_vars
        #: 0D variables.
        self.static_vars = static_vars
        #: Resolved primary-variable metadata.
        self.primary = primary
        #: Resolved direction-variable metadata, or None.
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
        #: Extreme value analysis for `primary`, see `ProfileEVA`.
        self.EVA = ProfileEVA(self, primary.col)

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
            for dim in updates:
                warnings.warn(
                    f"'{dim}' level values normalized to positive magnitude.",
                    UserWarning,
                    stacklevel=3,
                )

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

    def _variable_table_str(self) -> str:
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

        lines = [fmt_row(headers), "-" * sum(widths)]
        lines += [fmt_row(row) for row in rows]
        return "\n".join(lines)

    def __repr__(self) -> str:
        title = f"ProfileTimeSeries: {self.name}" if self.name else "ProfileTimeSeries"
        lines = [title, "=" * len(title), ""]
        lines.append(self._variable_table_str())
        lines.append(f"\nlevel_convention : {self.level_convention}")
        lines.append(f"time_dim         : '{self.time_dim}'")
        lines.append(f"level_dim(s)     : {self.level_dims}")
        lines.append(f"primary          : '{self.primary.col}'")
        if self.direction is not None:
            lines.append(f"direction        : '{self.direction.col}'")
        return "\n".join(lines)

    # ---------------------------------------------------------------------- #
    # Properties
    # ---------------------------------------------------------------------- #

    @property
    def time(self) -> pd.DatetimeIndex:
        """Time coordinate of the dataset."""
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
        """Time between the first and last timestamps."""
        t = self.time
        return t[-1] - t[0]

    @property
    def timestep(self) -> pd.Timedelta:
        """Median time between consecutive timestamps."""
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
        shared `groupby_with_metadata` utility can be applied directly
        without reimplementing its logic for xarray.
        """
        da = self.ds[var].transpose(self.time_dim, level_dim)
        df = pd.DataFrame(
            da.values,
            index=pd.DatetimeIndex(self.ds[self.time_dim].values),
            columns=self.ds.coords[level_dim].values,
        )
        return df

    def _resolve_profile_levels_uv(
        self,
        var: str,
        dir_var: str,
        level_dim: str,
        levels: list[float] | None,
        interp: str = "nearest",
        going_to: bool = True,
    ) -> dict[float, pd.DataFrame]:
        """
        Resolve a (magnitude, direction) profile pair onto requested output
        levels, interpolating through u/v (east/north) components rather
        than magnitude/direction directly.

        This is the vector-aware counterpart to `_resolve_level_axis`:
        direction is circular (10 degrees and 350 degrees are 20 degrees
        apart, not 340), so independently reindexing/interpolating a
        direction column the same way a plain numeric column is handled
        would silently produce wrong results whenever interp != "nearest"
        (e.g. linearly "interpolating" 350 deg and 10 deg gives 180 deg --
        the opposite direction). Converting to u/v first sidesteps this
        entirely: u and v are ordinary Cartesian numbers, so standard
        linear/quadratic/cubic interpolation across them is correct, and
        converting back afterwards recovers a direction that is the actual
        circular interpolation of the two bracketing directions.

        For interp="nearest", converting via u/v changes nothing numerically
        (nearest-neighbour selection doesn't blend anything), but the same
        code path is used for both cases rather than special-casing
        "nearest" to skip u/v -- one mechanism, not two to keep in sync.

        Parameters
        ----------
        var : str
            Magnitude profile variable name (e.g. current speed).
        dir_var : str
            Direction profile variable name. Must share `var`'s level
            dimension (same requirement enforced elsewhere for `direction`,
            e.g. at ProfileTimeSeries construction and in ProfileEVA).
        level_dim : str
            The shared level dimension of `var` and `dir_var`.
        levels : list[float], optional
            Requested output levels. None means "every level on
            `level_dim`, unchanged" -- no resolution/interpolation is
            attempted in that case.
        interp : {"nearest", "linear", "quadratic", "cubic", ...}
            Passed straight through to `_resolve_level_axis`'s underlying
            logic, applied to the u/v components. See that function for
            the full semantics, including how the resulting index is
            labelled ("nearest" reports the real matched level; anything
            else reports the requested level, since it is then a genuine
            new interpolated estimate).
        going_to : bool, default True
            Direction convention passed through to dirmag_to_uv/uv_to_dirmag
            (True = oceanographic "going to"; False = meteorological
            "coming from"). Must match the convention `dir_var`'s values
            are actually stored in.

        Returns
        -------
        dict[float, pd.DataFrame]
            One entry per resolved level, each a time-indexed DataFrame
            with columns "magnitude" and "direction" (NaNs dropped), ready
            to hand to `plot_rose` as its `magnitude`/`direction` arrays.
            Keyed by the resolved level values exactly as
            `_resolve_level_axis` would label them (see interp note above).
        """
        mag_df = self._profile_var_as_dataframe(var, level_dim).T   # level x time
        dir_df = self._profile_var_as_dataframe(dir_var, level_dim).T  # level x time

        if levels is None:
            resolved_levels = list(mag_df.index)
        else:
            u, v = dirmag_to_uv(dir_df.values, mag_df.values, going_to=going_to)
            u_df = pd.DataFrame(u, index=dir_df.index, columns=dir_df.columns)
            v_df = pd.DataFrame(v, index=dir_df.index, columns=dir_df.columns)

            u_resolved = _resolve_level_axis(u_df, levels, interp)
            v_resolved = _resolve_level_axis(v_df, levels, interp)

            dir_resolved, mag_resolved = uv_to_dirmag(
                u_resolved.values, v_resolved.values, going_to=going_to
            )
            mag_df = pd.DataFrame(mag_resolved, index=u_resolved.index, columns=u_resolved.columns)
            dir_df = pd.DataFrame(dir_resolved, index=u_resolved.index, columns=u_resolved.columns)
            resolved_levels = list(mag_df.index)

        out = {}
        for level in resolved_levels:
            level_df = pd.DataFrame({
                "magnitude": mag_df.loc[level],
                "direction": dir_df.loc[level],
            }).dropna()
            out[level] = level_df
        return out

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
        `groupby_with_metadata` primitive directly -- not a
        reimplementation, so grouping conventions (month labels,
        season->month map, sector edges/centering/labels) always stay in
        sync with the rest of the package.
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
            # groupby_with_metadata groups by ONE direction column against
            # a whole DataFrame. Since direction here is itself a profile
            # variable (one value per level, not one shared value), each
            # level must be grouped by *its own* direction column -- so
            # we call it once per level and stitch the per-level Series
            # back into per-sector DataFrames. observed=False (the
            # default) keeps the same fixed sector set for every level,
            # so no union-of-observed-labels bookkeeping is needed here.
            dir_df = self._profile_var_as_dataframe(dir_var, level_dim)
            per_level_meta: dict[str, dict] = {}
            for level in df.columns:
                merged = pd.DataFrame({"value": df[level], "dir": dir_df[level]}).dropna()
                per_level_meta[level] = groupby_with_metadata(
                    merged, by="direction", var="value", var_dir="dir", sectors=sectors,
                )

            labels = per_level_meta[df.columns[0]]["label"] if len(df.columns) else []
            groups = {}
            for i, label in enumerate(labels):
                groups[label] = pd.DataFrame({
                    level: per_level_meta[level]["data"][i] for level in df.columns
                })
            return groups

        if by in ("month", "season", "year"):
            meta = groupby_with_metadata(df, by=by, seasons=seasons)
            return dict(zip(meta["label"], meta["data"]))

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
        levels: list[float] | None = None,
        interp: str = "nearest",
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
        `groupby_with_metadata` / `aggregate_statistics` utilities (the
        same ones TimeSeries uses), so conventions and accepted
        stat-name spellings ("mean", "p90", "P90", "90%", ...) are
        identical across the package.

        Parameters
        ----------
        var : str, optional
            Name of the profile variable (must be in `self.profile_vars`).
            Defaults to `self.primary.col` if not given.
        by : {"all", "month", "season", "direction", "year"}
            Grouping strategy. Defaults to "all" (no grouping).
        stat : str | list[str]
            Statistic(s), passed straight to `utils.aggregate_statistics`:
            "count", "min"/"minimum", "max"/"maximum", "mean", "std",
            "median", or a percentile such as "P90" or "90%". Defaults to
            ["mean", "P90", "P99"]. Must be a single string (not a list)
            whenever `by` is not "all". "%"/"density" is not supported
            here, since this method has no well-defined grand total to
            compute it against; compute availability as
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
        levels : list[float], optional
            Levels to report, as any numeric values -- not necessarily
            ones present on the data's level coordinate. Resolved against
            the levels actually computed, according to `interp`. Defaults
            to every level on the variable's level dimension, unchanged.
        interp : {"nearest", "linear", "quadratic", "cubic", ...}
            How to resolve a requested level in `levels` that doesn't
            fall exactly on the data's level coordinate. "nearest" (the
            default) reports whichever computed level is closest;
            anything else interpolates numerically between computed
            levels via pandas/scipy. Ignored if `levels` is None. See
            `_resolve_level_axis` for details.

        Returns
        -------
        pd.DataFrame
            Index = level values for var's level dimension (or `levels`,
            if given). Columns are statistic names when by="all", or
            group labels (+ "All") otherwise.
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
            table.index.name = self._level_axis_label(level_dim)
            table.columns.name = "stat"
            return _resolve_level_axis(table, levels, interp)

        groups = self._get_groups(
            by, var, level_dim,
            direction=direction, sectors=sectors, seasons=seasons,
        )
        groups["All"] = self._profile_var_as_dataframe(var, level_dim)

        table = pd.DataFrame({
            label: _reduce(g, stats)[stats[0]] for label, g in groups.items()
        })
        table.index.name = self._level_axis_label(level_dim)
        table.columns.name = by
        return _resolve_level_axis(table, levels, interp)

    # ---------------------------------------------------------------------- #
    # Plotting
    # ---------------------------------------------------------------------- #

    def _level_axis_label(self, level_dim: str) -> str:
        """
        Build the y-axis label for a level dimension: just "Height (m)" or
        "Depth (m)" depending on `level_convention`.
        """
        return f"{self.level_convention.capitalize()} (m)"

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

        Each actual sampled level (the variable's level coordinate) is
        marked with a light dotted horizontal guide line and a marker on
        each plotted line, matching plot_statistics/plot_return_values's
        visual treatment.

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
            style = dict(color=color, marker=".") if n > 1 else dict(marker=".")
            style.update(plot_kwargs)
            ax.plot(values, levels, label=label, **style)

        # Mark the actual sampled levels with light dotted guides, same
        # treatment as plot_statistics/plot_return_values.
        for level_value in levels:
            ax.axhline(level_value, color="grey", linestyle=":",
                    linewidth=0.8, alpha=0.5, zorder=0)

        var_attrs = da.attrs
        var_label = var_attrs.get("name", var)
        var_units = var_attrs.get("units")
        ax.set_xlabel(f"{var_label}" + (f" ({var_units})" if var_units else ""))

        ax.set_ylabel(self._level_axis_label(level_dim))
        if self.level_convention == "depth":
            ax.invert_yaxis()

        ax.grid(True)
        ax.legend()

        _extend_xlim_to_zero(ax)
        _extend_ylim_to_zero(ax)
        return ax

    def plot_statistics(
        self,
        var: str | None = None,
        by: Literal["all", "month", "season", "direction", "year"] = "all",
        stat: str | list[str] = ("mean", "P90", "P99"),
        direction: str | None = None,
        sectors: int | list[float] = 12,
        seasons: dict[str, list[int]] | None = None,
        levels: list[float] | None = None,
        interp: str = "nearest",
        ncols: int = 3,
        plot_kwargs: dict | None = None,
        ax: "np.ndarray[plt.Axes] | None" = None,
    ) -> "np.ndarray[plt.Axes]":
        """
        Plot the table(s) produced by statistics(var, by, stat, ...):
        level on the y-axis, statistic value on the x-axis.

        Two layouts, chosen automatically by `by`:

        - by="all" (the default): a single Axes, one line per requested
            `stat` -- exactly as before. `stat` may be a list.
        - by="month"/"season"/"direction"/"year": one subplot per group
            (month, season, sector, or year), laid out the same way
            `ProfileEVA.plot_return_values` lays out its per-month/per-
            sector subplots (via the same `utils.get_n_axes` grid logic), each
            comparing the requested statistic(s) as separate lines within
            that group. `stat` may be a list here too -- unlike
            `statistics()` itself, which restricts `stat` to a single value
            whenever `by != "all"` (a single table has nowhere to put a
            third axis). Here, each requested stat gets its own
            `statistics()` call (one table per stat, level x group), and
            the subplot grid supplies the missing axis -- mirroring how
            `plot_return_values` calls `table_return_values` once per
            return period and uses subplots for the month/sector axis.

        This mirrors `plot_return_values` throughout: subplot grid,
        shared x/y limits across subplots, level markers and dotted level
        guides, and a single top-left legend, all behave identically --
        so reading a grouped `plot_statistics` figure and a grouped
        `plot_return_values` figure feels the same.

        Parameters
        ----------
        var : str, optional
            Name of the profile variable to plot. Defaults to
            `self.primary.col`, exactly as in statistics().
        by : {"all", "month", "season", "direction", "year"}
            Grouping strategy. Defaults to "all" (single Axes, no
            subplots).
        stat : str | list[str]
            Statistic(s) to plot as separate lines. Defaults to
            ["mean", "P90", "P99"]. Unlike `statistics()`, a list is
            accepted regardless of `by` -- see above.
        direction : str, optional
            Profile variable to use as direction when by="direction".
            Passed straight through to `statistics()`.
        sectors : int or list[float], default 12
            Directional sectors. Only used when by="direction". Passed
            straight through to `statistics()`.
        seasons : dict[str, list[int]], optional
            Custom season map. Only used when by="season". Passed
            straight through to `statistics()`.
        ncols : int, optional
            Max columns in the subplot grid, when by != "all". Defaults
            to 3, matching `plot_return_values`'s default. Unused for
            by="all" (always a single Axes).
        plot_kwargs : dict, optional
            Passed through to ax.plot() for every line.
        ax : matplotlib.axes.Axes or array of Axes, optional
            Axes to draw on. For by="all", a single Axes (a new one is
            created if omitted). For a grouped `by`, an array of Axes --
            one per group, already flattened or in any array shape (a new
            grid is created via `utils.get_n_axes` if omitted).

        Returns
        -------
        matplotlib.axes.Axes or np.ndarray of matplotlib.axes.Axes
            A single Axes for by="all"; an array of Axes otherwise.
        """
        import matplotlib.pyplot as plt

        if var is None:
            var = self.primary.col

        stats = [stat] if isinstance(stat, str) else list(stat)
        plot_kwargs = {} if plot_kwargs is None else dict(plot_kwargs)

        da = self.ds[var]
        var_attrs = da.attrs
        var_label = var_attrs.get("name", var)
        var_units = var_attrs.get("units")
        level_dim = [d for d in self.ds[var].dims if d != self.time_dim][0]
        level_label = self._level_axis_label(level_dim)
        invert = self.level_convention == "depth"

        # --------------------------------------------------------------- #
        # by="all": single Axes, one line per stat -- unchanged behaviour.#
        # --------------------------------------------------------------- #
        if by == "all":
            table = self.statistics(var=var, by="all", stat=stats, levels=levels, interp=interp)

            if ax is None:
                _, ax = plt.subplots()

            n = table.shape[1]
            cmap = plt.get_cmap("viridis")
            colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

            for color, (label, series) in zip(colors, table.items()):
                style = dict(color=color, marker=".") if n > 1 else dict(marker=".")
                style.update(plot_kwargs)
                ax.plot(series.values, series.index.values, label=str(label), **style)

            for level_value in table.index.values:
                ax.axhline(level_value, color="grey", linestyle=":",
                        linewidth=0.8, alpha=0.5, zorder=0)

            ax.set_xlabel(f"{var_label}" + (f" ({var_units})" if var_units else ""))
            ax.set_ylabel(level_label)
            if invert:
                ax.invert_yaxis()
            ax.grid(True)
            legend_title = "Statistic" if table.columns.name == "stat" else table.columns.name
            ax.legend(title=legend_title)

            _extend_xlim_to_zero(ax)
            _extend_ylim_to_zero(ax)
            return ax

        # --------------------------------------------------------------- #
        # Grouped by: one subplot per group, one line per stat -- mirrors #
        # plot_return_values exactly (one table per stat here, instead of #
        # one table per return period there).                             #
        # --------------------------------------------------------------- #
        tables = {
            s: self.statistics(
                var=var, by=by, stat=s, direction=direction,
                sectors=sectors, seasons=seasons, levels=levels, interp=interp,
            ).drop(columns="All", errors="ignore")
            for s in stats
        }

        first_table = next(iter(tables.values()))
        columns = list(first_table.columns)

        if by == "direction":
            # Direction labels sort alphabetically as plain strings
            # ("112-158°" < "22-68°") -- reorder by each sector's starting
            # compass angle so subplots progress sensibly, rather than
            # re-sorting arbitrarily. Applied consistently to every stat's
            # table via `columns`, since they all share the same columns.
            def _sector_start_angle(label: str) -> float:
                return float(label.split("-")[0])
            columns = sorted(columns, key=_sector_start_angle)

        owns_figure = ax is None
        if ax is None:
            fig, ax_arr = get_n_axes(len(columns), max_cols=ncols)
        else:
            fig = None
            ax_arr = np.array(ax).ravel()
            if len(ax_arr) < len(columns):
                raise ValueError(
                    f"Expected ax to contain {len(columns)} axes "
                    f"(one per {by} group), got {len(ax_arr)}."
                )

        # Shared x/y limits across all subplots, same reasoning as
        # plot_return_values: explicit xlim/ylim rather than sharex/sharey,
        # so every subplot keeps its own readable ticks.
        share_limits = owns_figure
        if share_limits:
            all_values = np.concatenate([t.values.ravel() for t in tables.values()])
            x_min, x_max = np.nanmin(all_values), np.nanmax(all_values)
            y_min, y_max = first_table.index.min(), first_table.index.max()
            x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
            y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0

        cmap = plt.get_cmap("viridis")
        n_stat = len(stats)
        colors = [cmap(i / max(n_stat - 1, 1)) for i in range(n_stat)]

        for col_idx, column in enumerate(columns):
            axi = ax_arr[col_idx]
            for s, color in zip(stats, colors):
                table = tables[s]
                style = dict(color=color, marker=".") if n_stat > 1 else dict(marker=".")
                style.update(plot_kwargs)
                axi.plot(
                    table[column].values, table.index.values,
                    label=str(s), **style,
                )
            # Mark the actual computed levels (this column's table index)
            # with light dotted guides, same treatment as plot_return_values.
            for level_value in first_table.index.values:
                axi.axhline(level_value, color="grey", linestyle=":",
                            linewidth=0.8, alpha=0.5, zorder=0)
            axi.set_title(str(column))
            axi.set_xlabel(
                f"{var_label}" + (f" ({var_units})" if var_units else "")
            )
            axi.set_ylabel(level_label)

            if share_limits:
                axi.set_xlim(x_min - x_pad, x_max + x_pad)
                axi.set_ylim(y_min - y_pad, y_max + y_pad)

            if invert:
                axi.invert_yaxis()
            axi.grid(True)

            if col_idx == 0:
                axi.legend(title="Statistic")

        for ax in ax_arr:
            _extend_xlim_to_zero(ax)
            _extend_ylim_to_zero(ax)
        return ax_arr

    def plot_rose(
        self,
        var: str | None = None,
        direction: str | None = None,
        levels: list[float] | None = None,
        interp: str = "nearest",
        going_to: bool = True,
        n_dir_bins: int = 16,
        mag_step: float = None,
        mag_max: float = None,
        cmap: str = "viridis",
        calm_threshold: float = 0.0,
        ncols: int = 3,
        rose_kwargs: dict | None = None,
        ax: "np.ndarray[plt.Axes] | None" = None,
    ) -> "np.ndarray[plt.Axes]":
        """
        Rose plot(s) of one profile variable against a paired direction
        variable, at one or more levels -- one polar subplot per requested
        level, with a single shared legend drawn below the whole figure.

        Levels are resolved exactly as in `statistics`/`table_return_values`
        (any numeric values, not necessarily ones present on the level
        coordinate, matched via `interp`), going through u/v (east/north)
        components rather than magnitude/direction independently, since
        direction is circular -- see `_resolve_profile_levels_uv`.

        All panels share identical magnitude bin edges, colormap, and radial
        scale. A single legend is drawn once below the whole figure, not per
        panel and not inside the axes.

        Parameters
        ----------
        var : str, optional
            Magnitude profile variable (e.g. current speed). Defaults to
            `self.primary.col`.
        direction : str, optional
            Direction profile variable. Falls back to `self.direction.col`
            if not given; raises if neither is available.
        levels : list[float], optional
            Levels to plot, as any numeric values. Defaults to every level
            on `var`'s level dimension.
        interp : {"nearest", "linear", "quadratic", "cubic", ...}
            How to resolve a requested level that doesn't fall exactly on
            the data's level coordinate. "nearest" (the default) uses the
            actual closest level's data unchanged; anything else
            interpolates the underlying u/v components and converts back.
            Ignored if `levels` is None.
        going_to : bool, default True
            Direction convention of `direction`'s stored values (True =
            oceanographic "going to"; False = meteorological "coming
            from"). Must match the data -- not inferred.
        n_dir_bins : int, default 16
            Number of compass sectors inside each rose panel.
        mag_step : float, optional
            Magnitude bin step. Auto-selected via `infer_step` if None.
        mag_max : float, optional
            Upper edge of the magnitude bins. Auto-selected as the 99th
            percentile of active (above `calm_threshold`) values pooled
            across every requested level if None. Set this explicitly
            together with `mag_step` to force identical bins across
            separate calls. An open-ended ">X" bin is still added if any
            level's data exceeds the resulting top edge.
        cmap : str, default "viridis"
            Colormap for magnitude classes, shared by every panel and the
            legend.
        calm_threshold : float, default 0.0
            Observations at or below this value are counted as calm.
        ncols : int, optional
            Max columns in the subplot grid. Defaults to 3, matching
            `plot_return_values`'s default.
        rose_kwargs : dict, optional
            Extra keyword arguments passed to every per-panel `plot_rose(...)`
            call. `mag_bins`, `r_max`, `cmap`, `calm_threshold`, `legend`,
            `mag_label`, `mag_unit`, and `title` are all set here and should
            not be overridden through this dict.
        ax : array of matplotlib.axes.Axes, optional
            Polar axes to draw on -- one per requested level. If omitted, a
            new constrained-layout figure with polar axes is created. When
            supplied, the caller owns the figure and no legend is drawn.

        Returns
        -------
        np.ndarray of matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if var is None:
            var = self.primary.col
        if var not in self.profile_vars:
            raise ValueError(
                f"'{var}' is not a recognised profile variable. "
                f"Available: {self.profile_vars}"
            )

        dir_var = self._require_direction("plot_rose", direction)

        level_dim = [d for d in self.ds[var].dims if d != self.time_dim][0]
        dir_level_dim = [d for d in self.ds[dir_var].dims if d != self.time_dim][0]
        if dir_level_dim != level_dim:
            raise ValueError(
                f"direction variable '{dir_var}' varies over level "
                f"dimension '{dir_level_dim}', but '{var}' varies over "
                f"'{level_dim}'. plot_rose requires both to share the same "
                "level dimension."
            )

        rose_kwargs = {} if rose_kwargs is None else dict(rose_kwargs)

        per_level = self._resolve_profile_levels_uv(
            var, dir_var, level_dim, levels=levels, interp=interp, going_to=going_to,
        )
        resolved_levels = list(per_level.keys())

        # ------------------------------------------------------------------ #
        # Shared magnitude bins, mirroring TimeSeries.plot_rose's own        #
        # pooled-data derivation, so every panel + the shared legend agree.  #
        # ------------------------------------------------------------------ #
        all_mag = np.concatenate([df["magnitude"].values for df in per_level.values()])
        active = all_mag[all_mag > calm_threshold]

        mag_top = float(mag_max) if mag_max is not None else (
            float(np.percentile(active, 99)) if len(active) else 1.0
        )
        _mag_step = mag_step or infer_step(pd.Series(active), target=7)
        mag_bins = np.arange(0, mag_top + _mag_step, _mag_step)

        legend_edges = mag_bins
        if len(active) and active.min() < legend_edges[0]:
            legend_edges = np.concatenate(([-np.inf], legend_edges))
        if len(active) and active.max() >= legend_edges[-1]:
            legend_edges = np.concatenate((legend_edges, [np.inf]))

        # ------------------------------------------------------------------ #
        # Shared radial scale across every level's panel.                    #
        # ------------------------------------------------------------------ #
        def _sector_max(mag_vals, dir_vals):
            sector_width = 360.0 / n_dir_bins
            dir_edges = np.linspace(-sector_width / 2, 360 - sector_width / 2, n_dir_bins + 1)
            n_total = len(mag_vals)
            active_m = mag_vals > calm_threshold
            dir_norm = dir_vals[active_m] % 360.0
            totals = np.zeros(n_dir_bins)
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
            _sector_max(df["magnitude"].values, df["direction"].values)
            for df in per_level.values()
        ) * 1.15

        # ------------------------------------------------------------------ #
        # Shared legend patches, built once and drawn below the whole figure #
        # ------------------------------------------------------------------ #
        n_mag = len(legend_edges) - 1
        cm_obj = plt.get_cmap(cmap, n_mag)
        colors = [cm_obj(i / n_mag) for i in range(n_mag)]

        var_meta = self._variables[var]
        mag_label = var_meta.name
        mag_unit = var_meta.unit
        unit_str = f" {mag_unit}" if mag_unit else ""

        def _legend_label(lo, hi):
            if np.isneginf(lo):
                return f"<{hi:g}{unit_str}"
            if np.isposinf(hi):
                return f">{lo:g}{unit_str}"
            return f"{lo:g}\u2013{hi:g}{unit_str}"

        patches = [
            mpatches.Patch(facecolor=colors[i], label=_legend_label(legend_edges[i], legend_edges[i + 1]))
            for i in range(n_mag)
        ]

        # ------------------------------------------------------------------ #
        # Polar axes grid. Created directly with plt.subplots (not by        #
        # converting rectangular axes) and with constrained layout, so the   #
        # layout engine can measure polar titles and reserve space for the   #
        # figure-level legend.                                               #
        # ------------------------------------------------------------------ #
        owns_figure = ax is None
        if owns_figure:
            n = len(resolved_levels)
            ncols = min(ncols, n)
            nrows = int(np.ceil(n / ncols))
            fig, ax_arr = plt.subplots(
                nrows, ncols,
                subplot_kw={"projection": "polar"},
                squeeze=False,
                layout="constrained",
            )
            ax_arr = ax_arr.ravel()
        else:
            fig = None
            ax_arr = np.array(ax).ravel()
            if len(ax_arr) < len(resolved_levels):
                raise ValueError(
                    f"Expected ax to contain {len(resolved_levels)} axes "
                    f"(one per requested level), got {len(ax_arr)}."
                )

        # ------------------------------------------------------------------ #
        # Level label for titles: "Depth 1.0 m" / "Height 98.6 m".           #
        # ------------------------------------------------------------------ #
        level_word = self.level_convention.capitalize()  # "Depth" or "Height"

        for idx, level in enumerate(resolved_levels):
            df = per_level[level]
            axi = ax_arr[idx]
            plot_rose(
                df["magnitude"].values, df["direction"].values,
                ax=axi,
                n_dir_bins=n_dir_bins,
                mag_bins=legend_edges,
                cmap=cmap,
                r_max=r_max,
                calm_threshold=calm_threshold,
                title=f"{level_word} {level:.1f} m  (n={len(df):,})",
                legend=False,
                mag_label=mag_label,
                mag_unit=mag_unit,
                **rose_kwargs,
            )

        for a in ax_arr[len(resolved_levels):]:
            a.set_visible(False)

        if owns_figure:
            fig.legend(
                handles=patches,
                title=mag_label,
                loc="outside lower center",
                ncol=min(n_mag, 6),
                frameon=False,
                handlelength=1.2,
            )

        return ax_arr

    def to_timeseries(
        self,
        level: float,
        interp: str = "nearest",
        primary: str | None = None,
        direction: str | None = None,
        secondary: list[str] | None = None,
        sectors: int = 12,
    ) -> "TimeSeries":
        """
        Export one level of this profile as a `TimeSeries`.

        Useful for anything that only makes sense at a single level and
        doesn't suit a per-level grid of subplots (e.g. a scatter diagram,
        or any other `TimeSeries` method) -- resolve the level once here,
        then use the full `TimeSeries` API on the result. Variable metadata
        (name/symbol/unit) carries over unchanged.

        Every profile variable sharing `primary`'s level dimension is
        sliced at `level` (resolved via `interp`, exactly as every other
        level-resolving method in this class -- see `_resolve_level_axis`).
        Scalar (time-only) variables are included by default. A
        profile variable on a *different* level dimension than `primary`
        cannot be sliced at the same `level` meaningfully, so it is only
        included if explicitly requested via `secondary` (which raises a
        clear error in that case, rather than silently misaligning two
        different grids) -- otherwise it's simply left out.

        Parameters
        ----------
        level : float
            Level to export, as any numeric value -- not necessarily one
            present on `primary`'s level coordinate.
        interp : {"nearest", "linear", "quadratic", "cubic", ...}
            How to resolve `level` against the level coordinate actually
            present. "nearest" (the default) selects the closest level;
            anything else interpolates numerically between the two
            bracketing levels. See `_resolve_level_axis` for details --
            applied independently to every exported column, including
            direction (circular interpolation is not special-cased here,
            consistent with every other level-resolving method in this
            class other than `plot_rose`).
        primary : str, optional
            Profile variable to use as the new TimeSeries's primary
            variable. Defaults to `self.primary`.
        direction : str, optional
            Profile variable to use as the new TimeSeries's direction
            variable. Defaults to `self.direction` if it shares the
            resolved `primary`'s level dimension, otherwise omitted.
        secondary : list[str], optional
            Additional profile/scalar variables to include. Defaults to
            every remaining profile variable sharing `primary`'s level
            dimension, plus every scalar (time-only) variable.
        sectors : int, default 12
            Passed straight through to the new `TimeSeries`.

        Returns
        -------
        TimeSeries
        """
        if primary is not None and primary not in self._variables:
            raise ValueError(
                f"primary='{primary}' is not a known variable. "
                f"Available: {list(self._variables)}"
            )
        primary_var = self._variables[primary] if primary else self.primary
        if primary_var.col not in self.profile_vars:
            raise ValueError(
                f"primary='{primary_var.col}' is not a profile variable. "
                f"Available: {self.profile_vars}"
            )
        level_dim = [d for d in self.ds[primary_var.col].dims if d != self.time_dim][0]

        def _at_level(col: str) -> pd.Series:
            df = self._profile_var_as_dataframe(col, level_dim).T  # level x time
            resolved = _resolve_level_axis(df, [level], interp)
            return resolved.loc[resolved.index[0]].rename(col)

        def _col_level_dim(col: str) -> str:
            return [d for d in self.ds[col].dims if d != self.time_dim][0]

        data = {primary_var.col: _at_level(primary_var.col)}

        # --- direction -------------------------------------------------- #
        if direction is not None and direction not in self._variables:
            raise ValueError(
                f"direction='{direction}' is not a known variable. "
                f"Available: {list(self._variables)}"
            )
        dir_var = self._variables[direction] if direction else self.direction
        if dir_var is not None and _col_level_dim(dir_var.col) != level_dim:
            if direction is not None:
                raise ValueError(
                    f"direction='{dir_var.col}' varies over level dimension "
                    f"'{_col_level_dim(dir_var.col)}', not '{level_dim}' "
                    f"(primary='{primary_var.col}'s level dimension) -- "
                    "cannot slice both at the same level."
                )
            dir_var = None  # default direction simply doesn't apply here
        if dir_var is not None:
            data[dir_var.col] = _at_level(dir_var.col)

        # --- secondary ---------------------------------------------------- #
        if secondary is None:
            secondary_cols = [
                v for v in self.profile_vars
                if v not in {primary_var.col, dir_var.col if dir_var else None}
                and _col_level_dim(v) == level_dim
            ] + list(self.scalar_vars)
        else:
            secondary_cols = list(secondary)

        secondary_vars = []
        for col in secondary_cols:
            if col in self.profile_vars:
                if _col_level_dim(col) != level_dim:
                    raise ValueError(
                        f"secondary variable '{col}' varies over level "
                        f"dimension '{_col_level_dim(col)}', not '{level_dim}' "
                        f"-- cannot slice at the same level as "
                        f"primary='{primary_var.col}'."
                    )
                data[col] = _at_level(col)
            elif col in self.scalar_vars:
                data[col] = self.ds[col].to_series().rename(col)
            else:
                raise ValueError(
                    f"secondary variable '{col}' is neither a profile "
                    "variable nor a scalar (time-only) variable on this "
                    "ProfileTimeSeries."
                )
            secondary_vars.append(self._variables[col])

        df = pd.DataFrame(data)

        return TimeSeries(
            df=df,
            primary=primary_var,
            direction=dir_var,
            secondary=secondary_vars,
            sectors=sectors,
        )

def _resolve_level_axis(
    table: pd.DataFrame,
    levels: list[float] | None,
    interp: str = "nearest",
) -> pd.DataFrame:
    """
    Reindex a level-indexed table (index = level values) onto a
    requested set of levels, resolving values not already present in
    the table's index according to `interp`.

    Shared by every table-producing method's `levels=`/`interp=`
    parameters (statistics, table_return_values). Deliberately a
    post-processing step applied to an already-computed table, never
    threaded into the computation itself: an EVA model cannot be
    interpolated (there is no meaningful "halfway between two fitted
    GPDs"), so an interpolated row here is always an interpolation of
    *results* (return values, statistics), computed at the levels that
    actually have data/models.

    Parameters
    ----------
    table : pd.DataFrame
        Any level-indexed table (numeric index, one row per level).
        Column structure is irrelevant and preserved as-is.
    levels : list[float], optional
        Requested level values. None means "leave the table's existing
        index untouched" -- the default for every caller.
    interp : {"nearest", "linear", "quadratic", "cubic", ...}
        How to resolve a requested level that isn't already an exact
        index value:
          - "nearest": snap to the closest existing level (no
            numerical interpolation of values -- the default, since it
            never fabricates a value the data didn't actually produce).
            The returned index holds the actual matched level, not the
            requested one.
          - anything else ("linear", "quadratic", "cubic", ...) is
            passed straight to pandas' ``DataFrame.interpolate(method=...)``,
            which wraps ``scipy.interpolate`` for the non-linear
            methods. The returned index holds the requested levels
            exactly.

    Returns
    -------
    pd.DataFrame
        Re-indexed per `levels`, if `levels` is not None (see Index
        labelling above for what the resulting index actually
        contains). Otherwise `table` unchanged.

    Raises
    ------
    ValueError
        If a requested level falls outside the table's existing index
        range and interp is not "nearest" -- extrapolation is not
        attempted; raising here makes the unsupported case explicit
        instead of a silent NaN in a results table.
    """
    if levels is None:
        return table

    table = table.sort_index()
    existing = table.index.to_numpy()

    if interp == "nearest":
        pos = table.index.get_indexer(levels, method="nearest")
        resolved = table.iloc[pos].copy()
        # Index intentionally kept as the ACTUAL matched levels (not
        # `levels`, the requested ones) -- see docstring.
        return resolved

    out_of_range = [
        lv for lv in levels if lv < existing.min() or lv > existing.max()
    ]
    if out_of_range:
        raise ValueError(
            f"Requested level(s) {out_of_range} fall outside the "
            f"available range [{existing.min():g}, {existing.max():g}] "
            f"and interp={interp!r} does not extrapolate. Use "
            "interp='nearest', or request levels within range."
        )

    union_index = table.index.union(levels)
    interpolated = table.reindex(union_index).interpolate(
        method=interp, limit_area="inside"
    )
    # Index intentionally kept as the REQUESTED levels -- see docstring.
    return interpolated.loc[levels]

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
    """

    def __init__(self, pts: "ProfileTimeSeries", var: str):
        self._pts = pts
        #: Name of the profile variable EVA is run on.
        self.var = var
        #: The level dimension `var` varies over.
        self.level_dim = [d for d in pts.ds[var].dims if d != pts.time_dim][0]
        #: Every level value on `level_dim`, in dataset order.
        self.levels = pts.ds.coords[self.level_dim].values
        #: Cache of per-level UnivariateEVA instances, keyed by level value.
        #: Empty until `get_extremes` has run.
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

    def get_extremes(
        self,
        th_omni: float | None = None,
        th_monthly: list[float] | None = None,
        th_sectors: list[float] | None = None,
        block_size: str = "365.2425D",
        min_last_block: float = 0.9,
        r: str = "48h",
        extremes_type: str = "high",
        errors: str = "raise",
        th_percentile: float = 0.98,
        poisson_correction: bool = True,
        sectors: int = 12,
    ) -> "ProfileEVA":
        """
        Build the UnivariateEVA for every level and run its `get_extremes`.

        Every level is extracted with the same settings. Thresholds left as
        None are derived per level from `th_percentile`, using that level's
        own data.

        Parameters
        ----------
        th_omni : float, optional
            Omni POT threshold. See `UnivariateEVA.get_extremes`.
        th_monthly : list of float, optional
            Monthly POT thresholds. See `UnivariateEVA.get_extremes`.
        th_sectors : list of float, optional
            Sector POT thresholds. See `UnivariateEVA.get_extremes`.
        block_size : str, default "365.2425D"
            AM block length. See `UnivariateEVA.get_extremes`.
        min_last_block : float, default 0.9
            Minimum fraction of a final block to keep.
        r : str, default "48h"
            POT declustering run length.
        extremes_type : {"high", "low"}, default "high"
            Whether to extract maxima or minima.
        errors : str, default "raise"
            Passed to `UnivariateEVA.get_extremes`.
        th_percentile : float, default 0.98
            Quantile used for the default thresholds.
        poisson_correction : bool, default True
            Passed to each level's `UnivariateEVA`.
        sectors : int, default 12
            Number of directional sectors, used if a direction variable is
            set. Ignored otherwise.

        Returns
        -------
        ProfileEVA
            self, so this chains directly into `.fit(...)`.
        """
        for level in tqdm_levels(self.levels, desc=f"EVA: {self.var}"):
            df = self._level_dataframe(level)
            model = UnivariateEVA(
                data=df,
                var=self.var,
                var_dir=self._dir_var,
                var_name=self._var_name,
                var_symbol=self._var_symbol,
                var_unit=self._var_unit,
                sectors=sectors,
                poisson_correction=poisson_correction,
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
        AM_dist: list[str] = [],
        POT_dist: list[str] = [],
        IDM_dist: list[str] = [],
        errors: str = "raise",
    ) -> "ProfileEVA":
        """
        Fit every already-extracted level's UnivariateEVA. All keyword
        arguments are passed straight through to
        `UnivariateEVA.fit`. Raises if `get_extremes` hasn't been run
        yet (nothing to fit), matching UnivariateEVA's own
        `_got_extremes` check.

        Parameters
        ----------
        AM_dist, POT_dist, IDM_dist, errors
            Passed straight through to `UnivariateEVA.fit`.

        Returns
        -------
        ProfileEVA
            self, so this chains directly into a plot/table method.
        """
        if not self.models:
            raise ValueError(
                "No extremes found -- call get_extremes() before fit()."
            )

        for level in tqdm_levels(list(self.models), desc=f"Fitting: {self.var}"):
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
        interp: str = "nearest",
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
            Levels to report, as any numeric values -- not necessarily
            ones that were fitted. Resolved against whichever levels
            ended up fitted and tabulated, according to `interp`. None
            (the default) returns one row per fitted level (every level
            with a fitted model for this `(method, dist)`, skipped with a
            warning otherwise, exactly as before).
        interp : {"nearest", "linear", "quadratic", "cubic", ...}
            How to resolve a requested level in `levels` that wasn't
            itself fitted. "nearest" (the default) reports whichever
            fitted level's row is closest; anything else interpolates
            numerically between fitted levels' return values via
            pandas/scipy. Ignored if `levels` is None. See
            `_resolve_level_axis` for details.

        Returns
        -------
        pd.DataFrame
            Index = level values, named by the level axis label (e.g.
            "Height (m)"), columns =
            month/sector labels (or a single "Omni" column if
            grouping="omni"). Values are return-value estimates for the
            requested return period, sourced by column from the
            corresponding *grouping subset's* actual model, exactly as
            UnivariateEVA.table_return_values_final's own columns are.
        """
        rp_label = f"{return_period}-year"
        target_levels = list(self.models)

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
                "get_extremes() and fit() first -- both always run "
                "every level, so a missing model usually means that "
                f"({method}, {dist}) failed to fit at this level.",
                UserWarning,
                stacklevel=2,
            )
        if not rows:
            raise ValueError(
                "No level had a fitted model to tabulate -- call "
                "get_extremes() and fit() first."
            )

        table = pd.DataFrame(rows).T
        table.index.name = self._pts._level_axis_label(self.level_dim)
        return _resolve_level_axis(table, levels, interp)

    def plot_return_values(
        self,
        return_periods: list[float],
        grouping: Literal["omni", "monthly", "sectors"],
        method: str,
        dist: str,
        return_period_size: str = "365.2425D",
        levels: list[float] | None = None,
        interp: str = "nearest",
        ncols: int = 3,
        plot_kwargs: dict | None = None,
        ax: "np.ndarray[plt.Axes] | None" = None,
    ) -> "np.ndarray[plt.Axes]":
        """
        One subplot per month/sector (laid out the same way
        `UnivariateEVA.plot_return_value_comparison` lays out its
        per-group subplots, via the same `utils.get_n_axes` grid logic), each
        showing the vertical profile -- level on the y-axis, return
        value on the x-axis, one line per requested return period.
        "omni" produces a single subplot (no month/sector split), same
        convention as `table_return_values`.

        Built on top of `table_return_values` -- one call per requested
        return period, then re-sliced by column (month/sector) to get
        each subplot's per-return-period lines -- so the same
        UnivariateEVA return-value math and the same skipped-level
        handling apply here too.

        For grouping="monthly" or "sectors", every subplot is drawn with
        the same x- and y-axis limits, computed as the overall min/max
        across all subplots/return periods, so values are directly
        comparable by eye across months/sectors. This is done by setting
        explicit xlim/ylim on each axis rather than matplotlib's
        sharex/sharey, since sharing hides tick labels on interior
        subplots -- here every subplot keeps its own readable ticks, just
        pinned to the same range. Not applied for "omni" -- there is only
        one subplot, so a shared range has no effect. Only applies when a
        new grid is created (ax=None); an existing `ax` array passed in is
        left with whatever limits it already has after plotting.

        Each actual computed level (the ones in `table.index`, i.e. the
        levels that had a fitted model, not necessarily every level in
        `self.levels`) is marked with a light dotted horizontal guide
        line, so the reader can see which depths the profile was actually
        evaluated at. The plotted return-value lines themselves also carry
        a marker at each computed level, for the same reason.

        A legend for the return periods is drawn on the first (top-left)
        subplot only, since every subplot uses the same return periods and
        colours.

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
        ncols : int, optional
            Max columns in the subplot grid. Defaults to 3, matching
            `UnivariateEVA.plot_return_value_comparison`'s default.
        plot_kwargs : dict, optional
            Passed to `ax.plot()` for every line.
        ax : array of matplotlib.axes.Axes, optional
            Axes to draw on -- one per month/sector, already flattened
            or in any array shape. A new grid is created via the same
            `utils.get_n_axes` layout `UnivariateEVA` uses if omitted.

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
                levels=levels, interp=interp,
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

        owns_figure = ax is None
        if ax is None:
            fig, ax_arr = get_n_axes(len(columns), max_cols=ncols)
        else:
            fig = None
            ax_arr = np.array(ax).ravel()
            if len(ax_arr) < len(columns):
                raise ValueError(
                    f"Expected ax to contain {len(columns)} axes "
                    f"(one per {grouping} group), got {len(ax_arr)}."
                )

        # Shared x/y limits across all subplots, computed manually rather
        # than via sharex/sharey so every subplot keeps its own tick
        # labels. Only meaningful (and only applied) when there's more
        # than one subplot to compare.
        share_limits = owns_figure and grouping in ("monthly", "sectors")
        if share_limits:
            all_values = np.concatenate([t.values.ravel() for t in tables.values()])
            x_min, x_max = np.nanmin(all_values), np.nanmax(all_values)
            y_min, y_max = first_table.index.min(), first_table.index.max()
            x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
            y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0

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
                style = dict(color=color, marker=".") if n_rp > 1 else dict(marker=".")
                style.update(plot_kwargs)
                axi.plot(
                    table[column].values, table.index.values,
                    label=f"{rp}-year", **style,
                )
            # Mark the actual computed levels (this column's table index)
            # with light dotted guides, so it's visible which depths were
            # evaluated versus interpolated between.
            for level_value in first_table.index.values:
                axi.axhline(level_value, color="grey", linestyle=":",
                            linewidth=0.8, alpha=0.5, zorder=0)
            axi.set_title(str(column))
            axi.set_xlabel(
                f"{var_label}" + (f" ({var_units})" if var_units else "")
            )
            axi.set_ylabel(self._pts._level_axis_label(self.level_dim))

            if share_limits:
                axi.set_xlim(x_min - x_pad, x_max + x_pad)
                axi.set_ylim(y_min - y_pad, y_max + y_pad)

            if self._pts.level_convention == "depth":
                axi.invert_yaxis()
            axi.grid(True)

            if col_idx == 0:
                axi.legend(title="Return period")

        for ax in ax_arr:
            _extend_xlim_to_zero(ax)
            _extend_ylim_to_zero(ax)
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

        Month filtering mirrors the package's own month-name resolution
        (full or abbreviated name, case-insensitive) rather than
        reimplementing a separate convention -- built directly here
        instead of going through `groupby_with_metadata`, since that
        groups into all 12 months and Turkstra only ever wants one.
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
        interp: str = "nearest",
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
            Levels to report, as any numeric values -- not necessarily
            ones actually computed. Applied identically to both the
            output-level (row) axis and the anchor-level (column) axis.
            Resolved according to `interp`. None (the default) returns
            every computed level on both axes, unchanged.
        interp : {"nearest", "linear", "quadratic", "cubic", ...}
            How to resolve a requested level in `levels` that wasn't
            itself computed. "nearest" (the default) reports whichever
            computed level is closest, on each axis independently;
            anything else interpolates numerically between computed
            levels via pandas/scipy -- see the note above on why this is
            better-justified for rows than for columns. Ignored if
            `levels` is None. See `_resolve_level_axis` for the row-axis
            mechanics (applied a second time, transposed, for columns).

        Returns
        -------
        pd.DataFrame
            Index = level values (name = "Depth (m)" or "Height (m)"),
            one column per anchor level (name = "Anchor level (m)") plus
            a final "Worst case" column. Entry (row=`dd`, col=`d`) is the
            conditionally-expected value at level `dd` given level `d` is
            at its own worst case; diagonal entries equal that level's
            own "Worst case" value exactly (when `dd == d` survives
            resolution on both axes unchanged -- an interpolated/relabelled
            row or column is no longer guaranteed to line up with a
            specific original anchor, so this exact-diagonal property is
            only guaranteed for `levels=None`).
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
            target_levels = list(self.levels)
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
                return_period_size=return_period_size,
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

        level_label = self._pts._level_axis_label(self.level_dim)
        matrix.index.name = level_label

        if levels is not None:
            worst_case_col = matrix[["Worst case"]]
            anchor_cols = matrix.drop(columns="Worst case")

            # Row axis (output levels): same mechanism as table_return_values.
            anchor_cols = _resolve_level_axis(anchor_cols, levels, interp)
            worst_case_col = _resolve_level_axis(worst_case_col, levels, interp)

            # Column axis (anchor levels): transpose so anchors become the
            # row index, resolve, transpose back.
            anchor_cols = _resolve_level_axis(anchor_cols.T, levels, interp).T

            matrix = anchor_cols
            matrix["Worst case"] = worst_case_col["Worst case"]
            matrix.index.name = level_label

        matrix.columns.name = "Anchor level (m)"
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
        interp: str = "nearest",
        cmap: str = "viridis",
        fmt_legend: str = ".1f",
        legend_ncols: int = 2,
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
        within one plot).

        Accepts exactly the same basis/percentile/month/return_period/
        method/dist/return_period_size/levels arguments as
        `table_turkstra` -- this simply plots that table.

        Each actual computed level (the table's index) is marked with a
        light dotted horizontal guide line and a marker on each plotted
        line, matching plot_profile/plot_statistics/plot_return_values's
        visual treatment elsewhere in this class.

        No title is drawn -- this plot is intended to be captioned
        externally (e.g. in a LaTeX figure environment) rather than
        carry its own title.

        Legend ordering: anchor-level entries are sorted so the legend's
        top-to-bottom order matches the plot's visual top-to-bottom order
        -- ascending level values for depth-convention data (the y-axis is
        inverted, so small depths plot near the top), descending for
        height-convention data (the y-axis is not inverted, so large
        heights plot near the top). The "Worst case" entry is always
        listed last, regardless of sort direction, since it is a distinct
        summary line rather than a level value.

        Parameters
        ----------
        cmap : str, default "viridis"
            Colormap used to colour the anchor-level lines.
        fmt_legend : str, default ".1f"
            Format spec (as used in an f-string, e.g. ".1f", ".2f", "g")
            applied to each anchor level's numeric value in the legend.
        plot_kwargs : dict, optional
            Passed to `ax.plot()` for every anchor-level line, overriding
            this method's own color/marker defaults where they overlap.
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
            return_period_size=return_period_size, levels=levels, interp=interp,
        )
        anchor_columns = [c for c in table.columns if c != "Worst case"]

        # Sort anchor columns so the legend's top-to-bottom order matches
        # the plot's visual top-to-bottom order -- see docstring.
        is_depth = self._pts.level_convention == "depth"
        anchor_columns = sorted(anchor_columns, reverse=not is_depth)

        if ax is None:
            _, ax = plt.subplots()

        n = len(anchor_columns)
        cm_obj = plt.get_cmap(cmap)
        colors = [cm_obj(i / max(n - 1, 1)) for i in range(n)]

        for anchor, color in zip(anchor_columns, colors):
            style = dict(color=color, marker=".")
            style.update(plot_kwargs)
            ax.plot(
                table[anchor].values, table.index.values,
                label=f"{anchor:{fmt_legend}}", **style,
            )

        for level_value in table.index.values:
            ax.axhline(level_value, color="grey", linestyle=":",
                    linewidth=0.8, alpha=0.5, zorder=0)

        wc_style = dict(color="black", linewidth=2.5, marker=".")
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
        if is_depth:
            ax.invert_yaxis()

        xlim = ax.get_xlim()
        if xlim[0]>0 and xlim[1]>0:
            ax.set_xlim([0,xlim[1]])

        ax.grid(True, color="lightgray", linestyle=":")
        ax.legend(title="Anchor level (m)", loc="center left", bbox_to_anchor=(1, 0.5), ncols=legend_ncols)

        _extend_xlim_to_zero(ax)
        _extend_ylim_to_zero(ax)
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
                "get_extremes() first (it always extracts every level at "
                f"once). Levels currently available: {list(self.models)}."
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