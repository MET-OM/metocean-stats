"""
SpectralTimeSeries
-------------------
Single-point metocean spectral time series, built from a single xarray
Dataset.

Design
~~~~~~
The user hands over one ``xr.Dataset`` (e.g. opened straight from a NetCDF /
Zarr / GRIB file). We never require them to pre-sort their data into named
arguments, and we never rename anything in their dataset. Instead:

  1. Dimension names for "time", "freq", "dir" are *detected* via aliases
     (see ``_ALIASES_*`` below) -- e.g. a dim literally called "theta" or
     "frequency" is recognised as playing the directional / frequency
     role. Nothing is renamed. The mapping from canonical role to the
     dataset's actual dim name is recorded in ``dim_map`` and used
     internally wherever we need to look up "the frequency dimension,
     whatever it's called here".
  2. Every data variable in the Dataset is inspected by its *dimension
     signature* (the set of canonical roles its dims play) and matched
     against the table below. Only time-varying spectral signatures are
     recognised -- this package is fundamentally about time series, so a
     bare frequency or directional spectrum with no time axis is left
     alone.

         {time, freq}        -> frequency spectral timeseries  ("E")
         {time, dir}         -> directional spectral timeseries ("D")
         {time, freq, dir}   -> 2D spectral timeseries          ("S")

  3. If more than one variable shares the same signature, that's an error:
     we ask the user to fix their dataset (drop/rename one) rather than
     silently guessing. There should be at most one of each.
  4. The dataset is stored as a *single* object, ``self.ds`` -- byte-for-
     byte what was passed in, with every original variable, coordinate,
     and dimension name intact. We never split it into a "classified"
     copy plus a separate "auxiliary" copy, and we never rename a
     dimension just because we recognised its role. Instead, ``var_map``
     records which original variable name (if any) plays each spectral
     role ("S", "E", "D"), and ``dim_map`` records which original
     dimension name (if any) plays each coordinate role ("time", "freq",
     "dir"). Everything else -- wind speed, wind direction, water level,
     whatever -- simply stays in ``self.ds`` under its original name,
     untouched, available later as e.g. a `group_by` key.
  5. Detection is heuristic (alias- and signature-based), so it can be
     wrong. For that reason, classification never silently repairs the
     data it finds. Instead, each detected spectral variable is *validated*
     (datetime check, monotonicity, non-negativity, dim order) and any
     problems are recorded as warnings/issues rather than fixed in place.
     Call ``.standardize()`` to actually fix issues (sort, transpose,
     clip) -- this mutates ``self.ds`` in place and is the only place
     data is ever changed without being explicitly asked for. In practice
     this always runs during ``__init__``, so a constructed instance is
     always standardized.
  6. Metadata lives *only* as xarray attributes on ``self.ds`` -- there is
     no parallel metadata object. Every variable, coordinate, and derived
     output gets ``name`` (short, human-readable, e.g. "Peak wave period")
     and ``long_name`` (precise, lower_snake_case, e.g.
     "spectral_peak_period") attrs, plus ``units`` where applicable.
     Anything not explicitly recognised gets a bare placeholder built from
     its own variable name. Existing attrs the user's dataset already had
     are preserved except where a specific key (``name``, ``long_name``,
     ``units``, ``history``, ...) is deliberately set by this class --
     individual keys are overwritten, never the whole attrs dict.

Direction convention
~~~~~~~~~~~~~~~~~~~~~
Declared via ``dir_convention``:
  "from"  - direction the wave/wind is coming from  (default)
  "to"    - direction the wave/wind is going toward (e.g. currents)
All directional outputs honour this declaration without silent conversion.
"""

from __future__ import annotations
import itertools
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.patches as mpatches
from tqdm import tqdm


from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Canonical roles - single source of truth for *detection*, never for renaming
# ---------------------------------------------------------------------------

_ROLE_TIME = "time"
_ROLE_FREQ = "freq"
_ROLE_DIR = "dir"

# Accepted aliases, in order of preference, for each canonical role.
_ALIASES_TIME = ["time"]
_ALIASES_FREQ = ["freq", "frequency", "f", "hz"]
_ALIASES_DIR = ["dir", "direction", "directions", "theta", "θ"]

# Accepted unit strings for direction coordinate attribute cross-checks.
# Checked case-insensitively, and as substrings for composite unit strings
# like "m2 rad-1 hz-1". First match wins.
_DIR_UNIT_DEG_HINTS = {"deg", "degrees", "degree", "°"}
_DIR_UNIT_RAD_HINTS = {"rad", "radians", "radian"}
_RAD_TO_DEG = 180.0 / np.pi

_ALIAS_MAP = {
    _ROLE_TIME: _ALIASES_TIME,
    _ROLE_FREQ: _ALIASES_FREQ,
    _ROLE_DIR: _ALIASES_DIR,
}

# Recognised time-varying spectral signatures, expressed as the set of
# canonical *roles* (not actual dim names) a data variable must carry.
# Value : (internal slot name, human-readable label)
_SPECTRAL_SIGNATURES: dict[frozenset, tuple[str, str]] = {
    frozenset({_ROLE_TIME, _ROLE_FREQ}): ("E", "frequency spectral timeseries"),
    frozenset({_ROLE_TIME, _ROLE_DIR}): ("D", "directional spectral timeseries"),
    frozenset({_ROLE_TIME, _ROLE_FREQ, _ROLE_DIR}): ("S", "2D spectral timeseries"),
}

_SPECTRAL_SIGNATURES_BY_SLOT = {
    "E": "frequency spectral timeseries",
    "D": "directional spectral timeseries",
    "S": "2D spectral timeseries",
}

_SLOT_ROLES = {
    "E": (_ROLE_TIME, _ROLE_FREQ),
    "D": (_ROLE_TIME, _ROLE_DIR),
    "S": (_ROLE_TIME, _ROLE_FREQ, _ROLE_DIR),
}


def _signature_roles_str(slot: str) -> str:
    return "(" + ", ".join(_SLOT_ROLES[slot]) + ")"


# ---------------------------------------------------------------------------
# Default attribute metadata
# ---------------------------------------------------------------------------
# Keyed by internal slot name ("S", "E", "D") and by canonical role
# ("freq", "dir"). Each entry supplies "name" (short/human), "long_name"
# (precise, lower_snake_case), and "units". Applied only to variables/dims
# nobody has already described via existing attrs of the same key.
_DEFAULT_ATTRS: dict[str, dict] = {
    "S":         dict(name="2D wave spectrum",            long_name="2d_variance_density_spectrum",          units="m^2/Hz/deg"),
    "E":         dict(name="Frequency spectrum",           long_name="frequency_variance_density_spectrum",   units="m^2/Hz"),
    "D":         dict(name="Directional spectrum",         long_name="directional_variance_density_spectrum", units="m^2/deg"),
    _ROLE_FREQ:  dict(name="Frequency",                     long_name="wave_frequency",                        units="Hz"),
    _ROLE_DIR:   dict(name="Direction",                     long_name="wave_direction",                        units="deg"),
}

# Names for the two 1D spectra derived from S (never a variable in var_map,
# but written into self.ds by compute_frequency_spectrum() /
# compute_directional_spectrum() the first time they're called).
_DERIVED_E_NAME = "freq_spectrum"
_DERIVED_D_NAME = "dir_spectrum"

# Metadata for every variable written by integrate() and the two
# compute_*_spectrum() helpers. Keyed by the fixed output name. These
# names/long_names are used both for the ds attrs and are the single
# source of truth -- nothing else duplicates them.
_INTEGRATED_VARS: dict[str, dict] = {
    # --- frequency-derived (need E or S) ---
    "hs":    dict(name="Significant wave height",       long_name="sea_surface_wave_significant_height",        units="m"),
    "tm01":  dict(name="Mean wave period",              long_name="sea_surface_wave_mean_period_from_moments_1", units="s"),
    "tm02":  dict(name="Mean zero-crossing period",     long_name="sea_surface_wave_zero_crossing_period",      units="s"),
    "tm_10": dict(name="Energy wave period",            long_name="sea_surface_wave_energy_period",             units="s"),
    "fp":    dict(name="Peak frequency",                long_name="sea_surface_wave_peak_frequency",            units="Hz"),
    "tp":    dict(name="Peak period",                   long_name="sea_surface_wave_peak_period",               units="s"),
    "fpI":   dict(name="Interpolated peak frequency",   long_name="sea_surface_wave_peak_frequency_interpolated", units="Hz"),
    "tpI":   dict(name="Interpolated peak period",      long_name="sea_surface_wave_peak_period_interpolated",  units="s"),
    # --- directional-derived (need D or S) ---
    "pdir":  dict(name="Peak wave direction",           long_name="sea_surface_wave_peak_direction",            units="deg"),
    "pdirI": dict(name="Interpolated peak direction",   long_name="sea_surface_wave_peak_direction_interpolated", units="deg"),
    "mdir":  dict(name="Mean wave direction",           long_name="sea_surface_wave_mean_direction",            units="deg"),
    "spr":   dict(name="Directional spread",            long_name="sea_surface_wave_directional_spread",       units="deg"),
    # --- 2D-only (need S) ---
    "pdir2": dict(name="Peak direction (2D)",           long_name="sea_surface_wave_peak_direction_from_2d_spectrum", units="deg"),
    "fp2":   dict(name="Peak frequency (2D)",           long_name="sea_surface_wave_peak_frequency_from_2d_spectrum", units="Hz"),
}


def _apply_attrs(da: xr.DataArray, meta: dict, extra: dict | None = None) -> None:
    """
    Set 'name', 'long_name', 'units' (and any extra key/value pairs) on
    da.attrs, overwriting only those specific keys. Everything else
    already present in da.attrs (e.g. a source file's own conventions)
    is left untouched.
    """
    da.attrs["name"] = meta["name"]
    da.attrs["long_name"] = meta["long_name"]
    if "units" in meta:
        da.attrs["units"] = meta["units"]
    if extra:
        da.attrs.update(extra)


def _default_attrs_for(col: str) -> dict:
    """Bare placeholder metadata for a variable nobody described."""
    return dict(name=col, long_name=col)


class SpectralTimeSeries:
    """
    Single-point metocean spectral time series.

    Construction is a strict three-phase pipeline, always run in full:

      1. **Identify** -- detect which dims play the time/freq/dir roles
         (via aliases, never renamed) and which data variables are
         time-varying spectra (S, E, and/or D), by dimension signature.
         Only these recognised variables are reported at this stage.
      2. **Standardize** -- immediately check the recognised variables for
         problems (direction unit, dimension order, sort order, sign,
         and exact dimensionality) and fix everything that is safely
         fixable, printing each fix as it is applied. Anything not
         safely fixable raises immediately, before any fix is applied,
         so you never end up with a partially patched dataset. There is
         no "unstandardized" object you can hold and forget to fix -- if
         construction succeeds, the data is clean.
      3. **Describe** -- now that units and structure are trustworthy,
         attach ``name``/``long_name``/``units`` attrs (only where not
         already meaningfully set) to every variable in the dataset --
         spectral, freq/dir, and auxiliary -- and print the full table.

    Every recognised spectral variable is required to have *exactly* the
    dims implied by its signature (e.g. an "S" variable must be exactly
    (time, freq, dir) -- no additional dims such as "station"). This is
    enforced in ``standardize()``; see there for details.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset (e.g. opened directly from NetCDF/Zarr/GRIB via
        xarray). Dimension names playing the time/freq/dir role are
        *detected* via aliases but never renamed. Data variables are
        classified by which roles their dims play:

            {time, freq}        -> frequency spectral timeseries
            {time, dir}         -> directional spectral timeseries
            {time, freq, dir}   -> 2D spectral timeseries

        At most one variable may match each signature; if more than one
        does, a ValueError is raised asking the user to resolve the
        ambiguity in their dataset. Variables that don't match any of
        these signatures are left completely untouched (original names,
        original dims) -- useful later as `group_by` keys (e.g.
        wind_speed, wind_direction).

        The dataset is stored as ``self.ds``, mutated in place during
        standardization (dimension order, sort order, sign, and
        direction unit) but never renamed. Classification is recorded in
        ``var_map`` (slot -> original variable name) and ``dim_map``
        (role -> original dim name).
    dir_convention : {"from", "to"}, default "from"
        Directional convention for all directional quantities.
        "from" - the direction the signal is coming from (wind, waves).
        "to"   - the direction the signal is going toward (currents, swell
                  tracking in some model conventions).
    depth : float, optional
        Water depth in metres.
    name : str, optional
        Human-readable label for this record (site name, model run, etc.).

    Attributes
    ----------
    ds : xr.Dataset
        The full source dataset. Always standardized by the time
        ``__init__`` returns: canonical dim order, ascending freq/dir,
        non-negative spectral values, direction in degrees. All metadata
        (``name``, ``long_name``, ``units``) lives in ``ds[var].attrs``.
    var_map : dict[str, str | None]
        Internal slot ("S", "E", "D") -> original variable name in the
        input dataset, or None if that slot wasn't found.
    dim_map : dict[str, str | None]
        Canonical role ("time", "freq", "dir") -> original dimension name
        in the input dataset, or None if that role wasn't found.
    has_2d, has_frequency, has_directional : bool
    dir_convention : {"from", "to"}

    Notes
    -----
    There is no ``standardized`` flag and no separate metadata object --
    by design, an instance of this class is always standardized, and all
    metadata is attached directly to ``self.ds`` as attrs. If the input
    cannot be safely standardized, construction raises instead of
    returning a partially-fixed object.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        dir_convention: Literal["from", "to"] = "from",
        depth: float | None = None,
        name: str | None = None,
    ):
        if not isinstance(ds, xr.Dataset):
            raise TypeError(
                f"ds must be an xr.Dataset, got {type(ds).__name__}. "
                "Load your source file with xarray first:\n"
                "    ds = xr.open_dataset('file.nc')"
            )

        if dir_convention not in ("from", "to"):
            raise ValueError(
                f"dir_convention must be 'from' or 'to', got {dir_convention!r}."
            )

        self.dir_convention = dir_convention
        self.depth          = depth
        self.name           = name

        title = f"SpectralTimeSeries: {name}" if name else "SpectralTimeSeries"
        print(title)
        print("=" * len(title))

        # ------------------------------------------------------------------ #
        # PHASE 1 -- Identify                                                  #
        # ------------------------------------------------------------------ #
        dim_map = self._detect_dims(ds)
        var_map = self._classify_variables(ds, dim_map)

        if not any(var_map.values()):
            raise ValueError(
                "No time-varying spectral variable found. Expected at least one "
                "data variable whose dims play the (time, freq), (time, dir), or "
                "(time, freq, dir) roles -- aliases accepted. "
                f"Variables present: {list(ds.data_vars)}."
            )

        for slot, var_name in var_map.items():
            if var_name is not None:
                self._check_time_is_datetime(ds[var_name], var_name, dim_map)

        dir_unit = (
            self._detect_dir_unit(ds, var_map, dim_map)
            if dim_map[_ROLE_DIR] is not None else None
        )

        print("\nRecognised variables:")
        for slot, var_name in var_map.items():
            if var_name is not None:
                label = _SPECTRAL_SIGNATURES_BY_SLOT[slot]
                print(f"  '{var_name}'  ->  {label} ({slot}), dims {ds[var_name].dims}")
        for role in (_ROLE_FREQ, _ROLE_DIR):
            dim_name = dim_map[role]
            if dim_name is not None:
                unit_str = f", unit = {dir_unit}" if role == _ROLE_DIR else ""
                print(f"  '{dim_name}'  ->  {role} dimension{unit_str}")

        # ------------------------------------------------------------------ #
        # PHASE 2 -- Standardize (always, immediately, no opt-out)           #
        # ------------------------------------------------------------------ #
        print("\nStandardizing:")
        ds = self._standardize(ds, var_map, dim_map, dir_unit)

        # ------------------------------------------------------------------ #
        # Store                                                               #
        # ------------------------------------------------------------------ #
        self.ds      = ds
        self.var_map = var_map
        self.dim_map = dim_map

        self.ds.attrs["dir_convention"] = dir_convention
        if depth is not None:
            self.ds.attrs["depth_m"] = depth
        if name is not None:
            self.ds.attrs["name"] = name

        # ------------------------------------------------------------------ #
        # PHASE 3 -- Describe (attrs only, applied after standardization so  #
        # unit/role detection above can still see the dataset's own hints)  #
        # ------------------------------------------------------------------ #
        self._apply_default_attrs(self.ds, var_map, dim_map)

        print()
        self._print_variable_table()
        print(f"\ndir_convention : {self.dir_convention}")
        print(f"depth          : {self.depth} m" if self.depth is not None else "depth          : not set")
        print(f"input          : {self._input_summary()}")

    # ---------------------------------------------------------------------- #
    # PHASE 1 helpers -- Identify (read-only, never renames, never mutates)  #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _detect_dims(ds: xr.Dataset) -> dict[str, str | None]:
        """
        For each canonical role (time/freq/dir), find which actual dim
        name in ``ds`` matches one of its aliases. Returns role -> actual
        dim name, or None if no alias is present. The dataset is never
        modified.
        """
        dim_map: dict[str, str | None] = {}
        for role, aliases in _ALIAS_MAP.items():
            found = next((a for a in aliases if a in ds.dims), None)
            dim_map[role] = found
        return dim_map

    @staticmethod
    def _classify_variables(
        ds: xr.Dataset, dim_map: dict[str, str | None]
    ) -> dict[str, str | None]:
        """
        Walk every data variable in ``ds`` and bucket it by dimension
        signature, expressed in terms of canonical *roles* rather than
        actual dim names (so a variable with dims ("time", "frequency")
        is still recognised as the "E" slot if "frequency" was detected
        as playing the freq role). Read-only: never modifies or copies
        data, only decides which original variable name (if any) plays
        each spectral role.

        A variable only matches a signature if *every one* of its dims
        plays a recognised role and the set of roles present exactly
        matches one of the known signatures -- a variable with an extra,
        unrecognised dim (e.g. "station") is not classified here and is
        left as an auxiliary variable. (``standardize()`` additionally
        guards against the case where a variable happens to carry only
        recognised-role dims but too many/few of them.)
        """
        actual_to_role = {v: k for k, v in dim_map.items() if v is not None}

        var_map: dict[str, str | None] = {"S": None, "E": None, "D": None}
        matches: dict[str, list[str]] = {"S": [], "E": [], "D": []}

        for var_name, da in ds.data_vars.items():
            roles_present = frozenset(
                actual_to_role[d] for d in da.dims if d in actual_to_role
            )
            all_dims_have_roles = len(roles_present) == len(da.dims)
            hit = _SPECTRAL_SIGNATURES.get(roles_present) if all_dims_have_roles else None

            if hit is None:
                continue
            slot, _label = hit
            matches[slot].append(var_name)

        for slot, names in matches.items():
            if len(names) > 1:
                label = _SPECTRAL_SIGNATURES_BY_SLOT[slot]
                raise ValueError(
                    f"Found {len(names)} variables matching the {label} signature "
                    f"({_signature_roles_str(slot)}): {names}. "
                    "Only one variable per signature is allowed. Drop or rename "
                    "the extras before constructing SpectralTimeSeries, e.g.:\n"
                    f"    ds = ds.drop_vars({names[1:]!r})"
                )
            if len(names) == 1:
                var_map[slot] = names[0]

        return var_map

    @staticmethod
    def _aux_names(ds: xr.Dataset, var_map: dict[str, str | None]) -> list[str]:
        classified_names = {n for n in var_map.values() if n is not None}
        return [v for v in ds.data_vars if v not in classified_names]

    @staticmethod
    def _check_time_is_datetime(
        da: xr.DataArray, var_name: str, dim_map: dict[str, str | None]
    ) -> None:
        """
        Hard structural check, not deferred to standardize(): a non-datetime
        time axis indicates a fundamentally mis-specified input, not a
        cosmetic ordering/unit issue, so it is raised immediately.
        """
        time_dim = dim_map[_ROLE_TIME]
        time_vals = da.coords[time_dim].values
        if not np.issubdtype(time_vals.dtype, np.datetime64):
            raise ValueError(
                f"'{var_name}': '{time_dim}' coordinate must be datetime64, "
                f"got dtype {time_vals.dtype}. "
                "Convert with pd.to_datetime() or xr.decode_cf()."
            )

    @staticmethod
    def _detect_dir_unit(
        ds: xr.Dataset,
        var_map: dict[str, str | None],
        dim_map: dict[str, str | None],
    ) -> str:
        """
        Infer whether the directional coordinate is in degrees or radians.

        The coordinate *values* are ground truth: if any |value| exceeds
        2π, the axis cannot possibly be radians, so it is degrees, and
        vice versa. This single range check is deliberately simple and
        covers every common convention -- [0, 360), (-180, 180],
        [0, 2π), (-π, π] -- because all of them either do or don't
        exceed 2π in absolute value.

        The ``units``/``unit`` attribute of the direction coordinate (and,
        as a fallback, of the classified spectral variables) is checked
        purely as a *cross-validation*: if an attribute is present and it
        contradicts the value-based verdict, that is treated as a likely
        metadata error in the source file and raises immediately, rather
        than silently trusting one signal over the other. An absent or
        empty attribute is fine and simply isn't used.

        Returns
        -------
        {"deg", "rad"}
        """
        dir_dim = dim_map[_ROLE_DIR]
        dirs = ds.coords[dir_dim].values.astype(float)

        # --- ground truth: value range ---
        value_unit = "deg" if np.nanmax(np.abs(dirs)) > 2.0 * np.pi else "rad"

        # --- cross-check: attribute hints, coordinate first, then spectra ---
        def _hint_from(attrs: dict) -> str | None:
            for key in ("units", "unit"):
                raw = str(attrs.get(key, "")).strip().lower()
                if not raw:
                    continue
                if any(h in raw for h in _DIR_UNIT_DEG_HINTS):
                    return "deg"
                if any(h in raw for h in _DIR_UNIT_RAD_HINTS):
                    return "rad"
            return None

        attr_unit = _hint_from(ds.coords[dir_dim].attrs)
        attr_source = f"coordinate '{dir_dim}'"
        if attr_unit is None:
            for slot in ("S", "D"):
                var_name = var_map.get(slot)
                if var_name is None:
                    continue
                attr_unit = _hint_from(ds[var_name].attrs)
                if attr_unit is not None:
                    attr_source = f"variable '{var_name}'"
                    break

        if attr_unit is not None and attr_unit != value_unit:
            raise ValueError(
                f"Conflicting directional unit signals for '{dir_dim}':\n"
                f"  coordinate values : range=[{dirs.min():.4g}, {dirs.max():.4g}] "
                f"-> {value_unit!r}\n"
                f"  units attribute   : {attr_source} declares {attr_unit!r}\n"
                "This likely indicates a metadata error in the source file. "
                "Fix the 'units' attribute, or the coordinate values "
                "themselves, before constructing SpectralTimeSeries."
            )

        return value_unit

    # ---------------------------------------------------------------------- #
    # PHASE 2 -- Standardize (always run; mutates and returns ds)            #
    # ---------------------------------------------------------------------- #

    @classmethod
    def _standardize(
        cls,
        ds: xr.Dataset,
        var_map: dict[str, str | None],
        dim_map: dict[str, str | None],
        dir_unit: str | None,
    ) -> xr.Dataset:
        """
        Check every recognised spectral variable for problems and fix
        everything that is safely fixable, printing each action as it is
        taken. Unfixable problems raise immediately, before any fix is
        applied to anything, so a ValueError never leaves ds half-patched.

        Fix order (later fixes may depend on earlier ones being applied):
          1. Unfixable checks first, across all variables -- freq must be
             strictly positive; direction, once in degrees, must fall in
             [0, 360); each recognised spectral variable must have
             *exactly* the dims implied by its signature (e.g. an "S"
             variable must be exactly (time, freq, dir), nothing more).
             Raising here happens before any mutation below.
          2. Direction unit: radians -> degrees, coordinate and density
             both rescaled to conserve total variance.
          3. Ascending sort of time / freq / dir coordinates.
          4. Per-variable dimension order (transpose to canonical order).
          5. Per-variable non-negativity (clip).
        """
        freq_dim = dim_map[_ROLE_FREQ]
        dir_dim  = dim_map[_ROLE_DIR]
        time_dim = dim_map[_ROLE_TIME]

        # -------------------------------------------------------------- #
        # 1. Unfixable checks -- raise before touching anything.          #
        # -------------------------------------------------------------- #
        unfixable: list[str] = []

        if freq_dim is not None:
            freqs = ds.coords[freq_dim].values.astype(float)
            if np.any(freqs <= 0):
                unfixable.append(
                    f"recognised '{freq_dim}' as the freq dimension, but found "
                    f"non-positive value(s) (min={freqs.min():.4g}). A frequency "
                    "axis cannot contain zero or negative values -- check the "
                    "source data."
                )

        if dir_dim is not None:
            dirs = ds.coords[dir_dim].values.astype(float)
            # Range is only meaningful once we know we're looking at degrees;
            # if currently radians, it will become degrees in step 2 below,
            # so we re-check range *after* conversion, not here. Here we only
            # catch degree-labelled inputs that are already out of range.
            if dir_unit == "deg" and (np.any(dirs < -180.0) or np.any(dirs > 360.0)):
                unfixable.append(
                    f"recognised '{dir_dim}' as the dir dimension (degrees), but "
                    f"found value(s) far outside any standard range "
                    f"(range=[{dirs.min():.4g}, {dirs.max():.4g}]°). This is not "
                    "a simple wrap-around or sign convention issue -- check the "
                    "source data."
                )

        # Exact-dimensionality check: a recognised spectral variable must
        # carry *only* the dims implied by its signature. This matters
        # because step 4 below transposes to `expected_dims`, which would
        # otherwise fail with a cryptic xarray error (or silently succeed
        # while ignoring an extra dim) if the variable carried more dims
        # than its slot allows.
        for slot, var_name in var_map.items():
            if var_name is None:
                continue
            actual_dims = ds[var_name].dims
            expected_roles = _SLOT_ROLES[slot]
            if len(actual_dims) != len(expected_roles):
                label = _SPECTRAL_SIGNATURES_BY_SLOT[slot]
                unfixable.append(
                    f"'{var_name}' was recognised as the {label} ({slot}) "
                    f"signature {_signature_roles_str(slot)}, but has "
                    f"{len(actual_dims)} dimension(s) {actual_dims} instead of "
                    f"the expected {len(expected_roles)}. Every spectral "
                    "variable must have exactly the dims (time, freq), "
                    "(time, dir), or (time, freq, dir) -- no additional "
                    "dimensions (e.g. 'station') are allowed. Select or "
                    "squeeze out the extra dimension before constructing "
                    "SpectralTimeSeries, e.g.:\n"
                    f"    ds = ds.isel({{'<extra_dim>': 0}})"
                )

        if unfixable:
            msg = "\n".join(f"  ERROR: {m}" for m in unfixable)
            print(msg)
            raise ValueError(
                "Cannot safely standardize -- the following issue(s) require "
                "manual inspection of the source dataset:\n" + msg
            )

        # -------------------------------------------------------------- #
        # 2. Direction unit -- radians -> degrees.                        #
        # -------------------------------------------------------------- #
        if dir_dim is not None and dir_unit == "rad":
            old_vals = ds.coords[dir_dim].values.astype(float)
            new_vals = old_vals * _RAD_TO_DEG
            ds = ds.assign_coords({dir_dim: new_vals})
            print(
                f"  '{dir_dim}' coordinate converted radians -> degrees "
                f"(×{_RAD_TO_DEG:.6g})"
            )
            for slot in ("S", "D"):
                var_name = var_map.get(slot)
                if var_name is not None and dir_dim in ds[var_name].dims:
                    ds[var_name] = ds[var_name] / _RAD_TO_DEG
                    print(
                        f"  '{var_name}' density rescaled ÷{_RAD_TO_DEG:.6g} "
                        "to conserve variance under the unit change"
                    )

            # Re-check range now that we're in degrees -- a radian axis with
            # a genuinely bad range (e.g. corrupted values) should still be
            # caught rather than silently sorted/wrapped below.
            dirs_deg = ds.coords[dir_dim].values.astype(float)
            if np.any(dirs_deg < -180.0) or np.any(dirs_deg > 360.0):
                raise ValueError(
                    f"'{dir_dim}': after converting radians -> degrees, values "
                    f"still fall far outside any standard range "
                    f"(range=[{dirs_deg.min():.4g}, {dirs_deg.max():.4g}]°). "
                    "Check the source data."
                )

        # -------------------------------------------------------------- #
        # 3. Ascending sort of freq / dir coordinates.                    #
        # -------------------------------------------------------------- #
        if time_dim is not None:
            diffs = np.diff(ds.coords[time_dim].values)
            non_ascending = np.any(diffs <= np.timedelta64(0)) if np.issubdtype(diffs.dtype, np.timedelta64) else np.any(diffs <= 0)
            if non_ascending:
                ds = ds.sortby(time_dim)
                print(f"  sorted '{time_dim}' ascending")

        if freq_dim is not None and np.any(np.diff(ds.coords[freq_dim].values) <= 0):
            ds = ds.sortby(freq_dim)
            print(f"  sorted '{freq_dim}' ascending")

        if dir_dim is not None and np.any(np.diff(ds.coords[dir_dim].values) <= 0):
            ds = ds.sortby(dir_dim)
            print(f"  sorted '{dir_dim}' ascending")

        # -------------------------------------------------------------- #
        # 4 & 5. Per-variable dimension order and non-negativity.         #
        # -------------------------------------------------------------- #
        for slot, var_name in var_map.items():
            if var_name is None:
                continue
            da = ds[var_name]
            expected_dims = tuple(dim_map[role] for role in _SLOT_ROLES[slot])

            if da.dims != expected_dims:
                da = da.transpose(*expected_dims)
                ds[var_name] = da
                print(f"  '{var_name}' transposed to {expected_dims}")

            n_neg = int((da.values < 0).sum())
            if n_neg:
                pct = 100 * n_neg / da.values.size
                ds[var_name] = da.clip(min=0)
                print(
                    f"  '{var_name}' clipped {n_neg} negative value(s) "
                    f"({pct:.2f}%) to zero"
                )

        return ds

    # ---------------------------------------------------------------------- #
    # PHASE 3 helpers -- Describe (attrs only)                                #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _apply_default_attrs(
        ds: xr.Dataset,
        var_map: dict[str, str | None],
        dim_map: dict[str, str | None],
    ) -> None:
        """
        Set 'name' and 'long_name' (and 'units' where applicable) on every
        variable/coordinate in ``ds`` -- spectral, freq/dir, and auxiliary
        -- by mutating ``da.attrs`` directly. Only these specific keys are
        touched; anything else already in a variable's attrs (e.g. the
        source file's own conventions) is left exactly as it was. Runs
        after standardize() so a "dir" role is guaranteed to already be
        in degrees.

        A variable/coordinate that already carries a non-empty 'name' is
        assumed to have been deliberately described (e.g. by the source
        file) and is left alone; otherwise it gets the default for its
        recognised slot/role, or a bare placeholder (its own var name).
        """
        def _needs_default(col: str, in_coords: bool) -> bool:
            attrs = ds.coords[col].attrs if in_coords else ds[col].attrs
            existing = attrs.get("name")
            return not existing

        for slot, col in var_map.items():
            if col is not None and _needs_default(col, in_coords=False):
                _apply_attrs(ds[col], _DEFAULT_ATTRS[slot])

        for role in (_ROLE_FREQ, _ROLE_DIR):
            dim_name = dim_map[role]
            if dim_name is not None and _needs_default(dim_name, in_coords=True):
                _apply_attrs(ds.coords[dim_name], _DEFAULT_ATTRS[role])

        for col in SpectralTimeSeries._aux_names(ds, var_map):
            if _needs_default(col, in_coords=False):
                _apply_attrs(ds[col], _default_attrs_for(col))

    def _recognized_as(self, col: str) -> str:
        for slot, var_name in self.var_map.items():
            if var_name == col:
                return _SPECTRAL_SIGNATURES_BY_SLOT[slot] + f" ({slot})"
        for role in (_ROLE_FREQ, _ROLE_DIR):
            if self.dim_map[role] == col:
                return f"{role} dimension"
        return "—"

    def _dims_str(self, col: str) -> str:
        if col in self.ds.data_vars:
            return "(" + ", ".join(self.ds[col].dims) + ")"
        if col in self.ds.coords:
            return "(" + ", ".join(self.ds.coords[col].dims) + ")"
        return "—"

    def _all_described_names(self) -> list[str]:
        names = list(self.ds.data_vars)
        for role in (_ROLE_FREQ, _ROLE_DIR):
            dim_name = self.dim_map[role]
            if dim_name is not None and dim_name not in names:
                names.append(dim_name)
        return names

    def _print_variable_table(self) -> None:
        headers = ["variable", "recognized as", "name", "units", "dims"]
        rows = []
        for col in self._all_described_names():
            attrs = self.ds.coords[col].attrs if col in self.ds.coords else self.ds[col].attrs
            rows.append([
                col,
                self._recognized_as(col),
                attrs.get("name", col),
                attrs.get("units", "—"),
                self._dims_str(col),
            ])

        widths = [
            max(len(headers[i]), max((len(r[i]) for r in rows), default=0)) + 2
            for i in range(len(headers))
        ]

        def fmt_row(cells: list[str]) -> str:
            return "".join(f"{c:<{widths[i]}}" for i, c in enumerate(cells))

        total_width = sum(widths)
        print(fmt_row(headers))
        print("-" * total_width)
        for row in rows:
            print(fmt_row(row))

    # ---------------------------------------------------------------------- #
    # Guards                                                                  #
    # ---------------------------------------------------------------------- #

    def _requires_2d_spectrum(self, method_name: str) -> None:
        if not self.has_2d:
            raise AttributeError(
                f"{method_name}() requires the full 2D spectrum S(f, θ, t), "
                f"but this instance holds only {self._input_summary()}. "
                "Provide a variable with dims (time, freq, dir)."
            )

    def _requires_frequency_spectrum(self, method_name: str) -> None:
        if not self.has_frequency:
            raise AttributeError(
                f"{method_name}() requires a frequency spectrum E(f, t), "
                f"but this instance holds only {self._input_summary()}. "
                "Provide a variable with dims (time, freq) or (time, freq, dir)."
            )

    def _requires_directional_spectrum(self, method_name: str) -> None:
        if not self.has_directional:
            raise AttributeError(
                f"{method_name}() requires a directional spectrum D(θ, t), "
                f"but this instance holds only {self._input_summary()}. "
                "Provide a variable with dims (time, dir) or (time, freq, dir)."
            )

    def _input_summary(self) -> str:
        parts = []
        if self.var_map.get("S"): parts.append("S(f,θ,t)")
        if self.var_map.get("E"): parts.append("E(f,t)")
        if self.var_map.get("D"): parts.append("D(θ,t)")
        return ", ".join(parts) if parts else "no spectral data"

    # ---------------------------------------------------------------------- #
    # Properties                                                              #
    # ---------------------------------------------------------------------- #

    @property
    def has_2d(self) -> bool:
        return self.var_map.get("S") is not None

    @property
    def has_frequency(self) -> bool:
        return self.var_map.get("E") is not None or self.var_map.get("S") is not None

    @property
    def has_directional(self) -> bool:
        return self.var_map.get("D") is not None or self.var_map.get("S") is not None

    def spectrum(self, slot: Literal["S", "E", "D"]) -> xr.DataArray | None:
        """Look up a classified spectral variable by canonical slot name."""
        var_name = self.var_map.get(slot)
        return self.ds[var_name] if var_name is not None else None

    @property
    def time(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.ds.coords[self.dim_map[_ROLE_TIME]].values)

    @property
    def freqs(self) -> np.ndarray | None:
        freq_dim = self.dim_map[_ROLE_FREQ]
        if freq_dim is not None:
            return self.ds.coords[freq_dim].values
        return None

    @property
    def dirs(self) -> np.ndarray | None:
        dir_dim = self.dim_map[_ROLE_DIR]
        if dir_dim is not None:
            return self.ds.coords[dir_dim].values
        return None

    @property
    def duration(self) -> pd.Timedelta:
        t = self.time
        return t[-1] - t[0]

    @property
    def timestep(self) -> pd.Timedelta:
        return pd.Series(self.time).diff().median()

    @property
    def n_years(self) -> float:
        return self.duration.total_seconds() / (365.2425 * 24 * 3600)

    # ---------------------------------------------------------------------- #
    # Dunder                                                                  #
    # ---------------------------------------------------------------------- #

    def __repr__(self) -> str:
        t = self.time
        aux_names = self._aux_names(self.ds, self.var_map)
        lines = [
            "SpectralTimeSeries(",
            f"  name          = {self.name}",
            f"  input         = {self._input_summary()}",
            f"  period        = {t[0].date()} → {t[-1].date()} ({self.n_years:.1f} years)",
            f"  timestep      = {self.timestep}",
        ]
        if self.freqs is not None:
            lines.append(
                f"  freq          = {len(self.freqs)} bins "
                f"[{self.freqs[0]:.4g}–{self.freqs[-1]:.4g} Hz]"
            )
        if self.dirs is not None:
            lines.append(
                f"  dir           = {len(self.dirs)} bins "
                f"[{self.dirs[0]:.4g}–{self.dirs[-1]:.4g}°]"
            )
        if aux_names:
            lines.append(f"  aux           = {aux_names}")
        lines += [
            f"  depth         = {self.depth} m",
            f"  dir_convention= {self.dir_convention}",
            ")",
        ]
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.time)

    # ---------------------------------------------------------------------- #
    # Integrated parameters                                                  #
    # ---------------------------------------------------------------------- #

    # ---------------------------------------------------------------------- #
    # Direction-integration helper: S(t,f,θ) -> E(t,f)                       #
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _batch_dims(da: xr.DataArray, *spectral_dims: str) -> tuple[str, ...]:
        """
        Every dimension of `da` except the named spectral dimension(s), in
        their original order. Used throughout the integration helpers so
        they work identically whether `da` has just (time,) as its batch
        dimension (the ordinary case) or several (e.g. a grouped/aggregated
        result with dims like (hs_bin, month) instead of time) -- see
        _bulk_from_E's docstring for the general contract this establishes.
        """
        return tuple(d for d in da.dims if d not in spectral_dims)
    
    @staticmethod
    def _dir_integrate_S(S: xr.DataArray, dir_dim: str) -> xr.DataArray:
        """
        Integrate S(t, f, θ) over direction using the trapezoidal rule with
        periodic closure, yielding E(t, f) in m²/Hz.

        Periodic closure: the first directional bin is appended at the end
        with its coordinate shifted by +360°.  Because the spectrum is
        periodic, both endpoints carry the same values -- no energy is
        invented.  The integral therefore covers exactly [θ₀, θ₀+360°).

        Integration is performed in radians (dθ in rad) to satisfy
            E(f) = ∫₀²π S(f,θ) dθ   [m²/Hz]
        """
        dirs_deg = S.coords[dir_dim].values.astype(float)

        # --- append wrap-around point ---
        wrap_coord = dirs_deg[0] + 360.0
        wrap_slice = S.isel({dir_dim: 0})
        wrap_slice = wrap_slice.assign_coords(
            {dir_dim: wrap_coord}
        ).expand_dims(dir_dim, axis=list(S.dims).index(dir_dim))
        S_closed = xr.concat([S, wrap_slice], dim=dir_dim)

        # --- switch coordinate to radians and integrate ---
        dirs_rad = np.deg2rad(S_closed.coords[dir_dim].values.astype(float))
        S_closed = S_closed.assign_coords({dir_dim: dirs_rad})
        E = S_closed.integrate(coord=dir_dim)
        return E

    # ---------------------------------------------------------------------- #
    # Frequency-integration helper: S(t,f,θ) -> D(t,θ)                       #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _freq_integrate_S(S: xr.DataArray, freq_dim: str) -> xr.DataArray:
        """
        Integrate S(t, f, θ) over frequency using the trapezoidal rule,
        yielding D(t, θ) in m²/deg.

        The directional coordinate is intentionally left in degrees; the
        unit m²/deg is the declared convention for the D slot.
        """
        return S.integrate(coord=freq_dim)

    # ---------------------------------------------------------------------- #
    # Spectral moment helper                                                  #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _spectral_moment(E: xr.DataArray, freq_dim: str, n: int) -> xr.DataArray:
        """
        Compute the n-th spectral moment  mₙ = ∫ fⁿ · E(f) df
        using the trapezoidal rule.

        n = -1 -> energy period numerator  (m₋₁)
        n =  0 -> variance                 (m₀)
        n =  1 -> mean period numerator    (m₁)
        n =  2 -> zero-crossing numerator  (m₂)
        """
        freqs = E.coords[freq_dim].astype(float)
        return (E * freqs ** n).integrate(coord=freq_dim)

    # ---------------------------------------------------------------------- #
    # Parabolic peak interpolation                                           #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _parabolic_peak_1d(values: np.ndarray, coords: np.ndarray) -> np.ndarray:
        orig_shape = values.shape[:-1]
        N = values.shape[-1]
        flat = values.reshape(-1, N)
        R = flat.shape[0]

        k = np.argmax(flat, axis=1)                 # (R,) vectorized, one call
        out = coords[k].astype(float)                # default: edge/flat fallback

        interior = (k > 0) & (k < N - 1)
        idx = np.nonzero(interior)[0]
        if idx.size:
            ki = k[idx]
            ym, y0, yp = flat[idx, ki - 1], flat[idx, ki], flat[idx, ki + 1]
            xm, x0, xp = coords[ki - 1], coords[ki], coords[ki + 1]

            denom_check = ym - 2.0 * y0 + yp
            num = xm**2 * (y0 - yp) + x0**2 * (yp - ym) + xp**2 * (ym - y0)
            den = xm    * (y0 - yp) + x0    * (yp - ym) + xp    * (ym - y0)

            valid = (denom_check < 0.0) & (den != 0.0)
            vidx = idx[valid]
            out[vidx] = 0.5 * num[valid] / den[valid]

        return out.reshape(orig_shape) if orig_shape else out.reshape(())


    @staticmethod
    def _parabolic_peak_dir(D_vals: np.ndarray, dirs_deg: np.ndarray) -> np.ndarray:
        orig_shape = D_vals.shape[:-1]
        N = D_vals.shape[-1]
        flat = D_vals.reshape(-1, N)
        R = flat.shape[0]

        # circular gap to the *next* bin, for every bin i (length N, no python loop)
        gaps = np.diff(dirs_deg, append=dirs_deg[0] + 360.0)   # gaps[i] = dist(i -> i+1)

        k = np.argmax(flat, axis=1)                 # (R,) one vectorized call
        km1 = (k - 1) % N
        kp1 = (k + 1) % N

        ym, y0, yp = flat[np.arange(R), km1], flat[np.arange(R), k], flat[np.arange(R), kp1]
        xm = -gaps[km1]     # local coord of previous bin, peak bin fixed at x0 = 0
        xp = gaps[k]        # local coord of next bin

        denom_check = ym - 2.0 * y0 + yp
        num = xm**2 * (y0 - yp) + (yp - ym) * 0.0 + xp**2 * (ym - y0)   # x0 = 0 term drops out
        den = xm * (y0 - yp) + xp * (ym - y0)                            # x0 term drops out

        offset = np.zeros(R, dtype=float)
        valid = (denom_check < 0.0) & (den != 0.0)
        offset[valid] = 0.5 * num[valid] / den[valid]

        out = (dirs_deg[k] + offset) % 360.0
        return out.reshape(orig_shape) if orig_shape else float(out[0])

    # ---------------------------------------------------------------------- #
    # Bulk parameters from E(f)                                               #
    # ---------------------------------------------------------------------- #
    
    @staticmethod
    def _bulk_from_E(
        E: xr.DataArray,
        freq_dim: str,
        source_name: str,
    ) -> dict[str, xr.DataArray]:
        """
        Compute all frequency-derived bulk parameters from E(f, ...).

        Parameters
        ----------
        E : xr.DataArray
            Frequency spectrum with the freq_dim dimension plus any number
            of other ("batch") dimensions -- typically just (time,), but
            equally (hs_bin, month), (station, time), or any grouping
            produced upstream (e.g. GroupedSpectralTimeSeries.aggregate()).
            Every dimension other than freq_dim is treated as a batch
            dimension and preserved as-is in every output.
        freq_dim : str
            Name of the frequency dimension in E.
        source_name : str
            Name of the variable E was taken/derived from, used only to
            populate each output's 'history' attribute.

        Returns
        -------
        dict with keys: "hs", "tm01", "tm02", "tm_10",
                        "fp", "tp", "fpI", "tpI".
        """
        m_n1 = SpectralTimeSeries._spectral_moment(E, freq_dim, -1)
        m0   = SpectralTimeSeries._spectral_moment(E, freq_dim,  0)
        m1   = SpectralTimeSeries._spectral_moment(E, freq_dim,  1)
        m2   = SpectralTimeSeries._spectral_moment(E, freq_dim,  2)

        eps = 1e-30
        hs    = 4.0 * np.sqrt(m0.clip(min=0.0))
        tm01  = m0 / (m1  + eps)
        tm02  = np.sqrt(m0 / (m2  + eps))
        tm_10 = m_n1 / (m0 + eps)

        freqs      = E.coords[freq_dim].values.astype(float)
        batch_dims = SpectralTimeSeries._batch_dims(E, freq_dim)
        batch_coords = {d: E.coords[d] for d in batch_dims if d in E.coords}

        # --- discrete peak ---
        peak_idx = E.argmax(dim=freq_dim)
        freqs_da = xr.DataArray(freqs, dims=[freq_dim], coords={freq_dim: freqs})
        fp = freqs_da.isel({freq_dim: peak_idx}).drop_vars(freq_dim).rename("fp")
        tp = (1.0 / fp.where(fp > 0)).rename("tp")

        # --- parabolic interpolated peak ---
        # transpose puts freq_dim last; _parabolic_peak_1d collapses every
        # dimension before it into one flat batch axis internally, so this
        # already works for any number of batch dims, not just (time,).
        E_vals   = E.transpose(*batch_dims, freq_dim).values
        fpI_vals = SpectralTimeSeries._parabolic_peak_1d(E_vals, freqs)
        fpI = xr.DataArray(
            fpI_vals, dims=batch_dims, coords=batch_coords, name="fpI"
        )
        tpI = (1.0 / fpI.where(fpI > 0)).rename("tpI")

        h = lambda method: f"integrated from '{source_name}' via {method}"
        return {
            "hs":    (hs.rename("hs"),     h("xarray trapezoidal (spectral moment m0)")),
            "tm01":  (tm01.rename("tm01"), h("xarray trapezoidal (spectral moments m0, m1)")),
            "tm02":  (tm02.rename("tm02"), h("xarray trapezoidal (spectral moments m0, m2)")),
            "tm_10": (tm_10.rename("tm_10"), h("xarray trapezoidal (spectral moments m0, m-1)")),
            "fp":    (fp,  h("discrete argmax")),
            "tp":    (tp,  h("discrete argmax, inverted")),
            "fpI":   (fpI, h("parabolic peak interpolation")),
            "tpI":   (tpI, h("parabolic peak interpolation, inverted")),
        }
    # ---------------------------------------------------------------------- #
    # Bulk parameters from D(θ)                                               #
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _bulk_from_D(
        D: xr.DataArray,
        dir_dim: str,
        dir_convention: str,
        source_name: str,
    ) -> dict[str, xr.DataArray]:
        """
        Compute all directional bulk parameters from D(θ, ...).

        Mean direction: circular (unit-vector) average via first circular
        moments a₁, b₁.  Maps result to [0°, 360°).

        Directional spread: Kuik et al. (1988),
            σ = √(2(1 − r))  where  r = √(a₁² + b₁²),  converted to degrees.

        Peak direction: discrete argmax + parabolic interpolation with
        circular wrap handling.

        Parameters
        ----------
        D : xr.DataArray
            Directional spectrum with the dir_dim dimension plus any
            number of other ("batch") dimensions -- see _bulk_from_E's
            docstring for the same note; applies identically here.
            Coordinate values in degrees, in [0°, 360°).
        dir_dim : str
            Name of the directional dimension in D.
        dir_convention : str
            Stored in output attrs for traceability.
        source_name : str
            Name of the variable D was taken/derived from, used only to
            populate each output's 'history' attribute.

        Returns
        -------
        dict with keys: "pdir", "pdirI", "mdir", "spr".
        """
        dirs_deg   = D.coords[dir_dim].values.astype(float)
        dirs_rad   = np.deg2rad(dirs_deg)
        batch_dims = SpectralTimeSeries._batch_dims(D, dir_dim)
        batch_coords = {d: D.coords[d] for d in batch_dims if d in D.coords}

        # --- discrete peak ---
        peak_idx = D.argmax(dim=dir_dim)
        dirs_da  = xr.DataArray(dirs_deg, dims=[dir_dim], coords={dir_dim: dirs_deg})
        pdir     = dirs_da.isel({dir_dim: peak_idx}).drop_vars(dir_dim).rename("pdir")

        # --- parabolic interpolated peak (circular) ---
        D_vals     = D.transpose(*batch_dims, dir_dim).values
        pdirI_vals = SpectralTimeSeries._parabolic_peak_dir(D_vals, dirs_deg)
        pdirI = xr.DataArray(
            pdirI_vals, dims=batch_dims, coords=batch_coords, name="pdirI"
        )

        # --- circular mean direction ---
        D_rad = D.assign_coords({dir_dim: dirs_rad})
        m0_d  = D_rad.integrate(coord=dir_dim)
        eps   = 1e-30
        a1    = (D_rad * np.cos(D_rad.coords[dir_dim])).integrate(coord=dir_dim) / (m0_d + eps)
        b1    = (D_rad * np.sin(D_rad.coords[dir_dim])).integrate(coord=dir_dim) / (m0_d + eps)
        mdir  = (np.rad2deg(np.arctan2(b1, a1)) % 360).rename("mdir")

        # --- directional spread (Kuik et al. 1988) ---
        r   = np.sqrt(a1**2 + b1**2).clip(0.0, 1.0)
        spr = np.rad2deg(np.sqrt(2.0 * (1.0 - r))).rename("spr")

        h = lambda method: f"integrated from '{source_name}' via {method}"
        return {
            "pdir":  (pdir,  h("discrete argmax")),
            "pdirI": (pdirI, h("parabolic peak interpolation (circular)")),
            "mdir":  (mdir,  h("circular mean (first trigonometric moments a1, b1)")),
            "spr":   (spr,   h("circular spread, Kuik et al. (1988)")),
        }
    # ---------------------------------------------------------------------- #
    # 2D-only: peak direction from argmax of S(f,θ)                          #
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _peak2_from_S(
            S: xr.DataArray,
            freq_dim: str,
            dir_dim: str,
            dir_convention: str,
            source_name: str,
        ) -> tuple[xr.DataArray, xr.DataArray, str]:
        """
        Peak frequency and peak direction from the joint 2D argmax of
        S(f, θ, ...): the (f, θ) cell holding the most energy, evaluated
        jointly, independently for every batch dimension combination.

        Differs from fp (argmax of E(f)) and pdir (argmax of D(θ)) in that
        those are computed from marginals, while fp2/pdir2 jointly
        optimise over both axes. For narrow-banded, unimodal spectra all
        agree; for broad-banded or bimodal spectra they can differ.

        S may carry any number of batch dimensions in addition to
        freq_dim/dir_dim -- see _bulk_from_E's docstring for the general
        note; the joint argmax is taken independently within each batch
        cell.
        """
        freqs_hz   = S.coords[freq_dim].values.astype(float)
        dirs_deg   = S.coords[dir_dim].values.astype(float)
        batch_dims = SpectralTimeSeries._batch_dims(S, freq_dim, dir_dim)
        batch_coords = {d: S.coords[d] for d in batch_dims if d in S.coords}

        S_vals = S.transpose(*batch_dims, freq_dim, dir_dim).values  # (..., F, D)
        batch_shape = S_vals.shape[:-2]
        F, D = S_vals.shape[-2:]

        # Collapse every batch dim into one flat axis for the argmax, then
        # reshape the result back to the true batch shape at the end --
        # the same "collapse, compute, reshape back" pattern already used
        # by _parabolic_peak_1d/_parabolic_peak_dir, generalized here from
        # a single leading T axis to an arbitrary number of batch dims.
        flat = S_vals.reshape(-1, F * D)
        idx  = np.argmax(flat, axis=1)          # (prod(batch_shape),)

        freq_idx = idx // D
        dir_idx  = idx % D

        fp2_flat   = freqs_hz[freq_idx]
        pdir2_flat = dirs_deg[dir_idx]

        fp2_vals   = fp2_flat.reshape(batch_shape)
        pdir2_vals = pdir2_flat.reshape(batch_shape)

        fp2_da = xr.DataArray(fp2_vals, dims=batch_dims, coords=batch_coords, name="fp2")
        pdir2_da = xr.DataArray(pdir2_vals, dims=batch_dims, coords=batch_coords, name="pdir2")

        history = f"integrated from '{source_name}' via 2D joint argmax over (freq, dir)"
        return fp2_da, pdir2_da, history
    # ---------------------------------------------------------------------- #
    # Public: compute_frequency_spectrum()                                    #
    # ---------------------------------------------------------------------- #

    def compute_frequency_spectrum(self) -> xr.DataArray:
        """
        Return the frequency spectrum E(t, f) in m²/Hz.

        If the E slot is already present in the dataset it is returned
        directly.  If only S is available, E is derived by integrating S
        over direction (0→360° with periodic closure) and cached in
        ``self.ds`` under the name ``'freq_spectrum'``.  Subsequent calls
        are free (the cached result is returned immediately).
        """
        self._requires_frequency_spectrum("compute_frequency_spectrum")

        if self.var_map["E"] is not None:
            return self.ds[self.var_map["E"]]

        if _DERIVED_E_NAME in self.ds.data_vars:
            return self.ds[_DERIVED_E_NAME]

        S_name  = self.var_map["S"]
        S       = self.ds[S_name]
        dir_dim = self.dim_map[_ROLE_DIR]
        E       = self._dir_integrate_S(S, dir_dim)
        E.name  = _DERIVED_E_NAME
        _apply_attrs(
            E, _DEFAULT_ATTRS["E"],
            extra={"history": f"integrated from '{S_name}' via xarray trapezoidal "
                               "rule over direction (periodic closure)"},
        )
        self.ds[_DERIVED_E_NAME] = E
        return E

    # ---------------------------------------------------------------------- #
    # Public: compute_directional_spectrum()                                  #
    # ---------------------------------------------------------------------- #

    def compute_directional_spectrum(self) -> xr.DataArray:
        """
        Return the directional spectrum D(t, θ) in m²/deg.

        If the D slot is already present in the dataset it is returned
        directly.  If only S is available, D is derived by integrating S
        over frequency and cached in ``self.ds`` under ``'dir_spectrum'``.
        """
        self._requires_directional_spectrum("compute_directional_spectrum")

        if self.var_map["D"] is not None:
            return self.ds[self.var_map["D"]]

        if _DERIVED_D_NAME in self.ds.data_vars:
            return self.ds[_DERIVED_D_NAME]

        S_name   = self.var_map["S"]
        S        = self.ds[S_name]
        freq_dim = self.dim_map[_ROLE_FREQ]
        D        = self._freq_integrate_S(S, freq_dim)
        D.name   = _DERIVED_D_NAME
        _apply_attrs(
            D, _DEFAULT_ATTRS["D"],
            extra={"history": f"integrated from '{S_name}' via xarray trapezoidal "
                               "rule over frequency"},
        )
        self.ds[_DERIVED_D_NAME] = D
        return D

    # ---------------------------------------------------------------------- #
    # Public: integrate()                                                     #
    # ---------------------------------------------------------------------- #

    def integrate(
        self,
        compute: list[str] | None = None,
    ) -> "SpectralTimeSeries":
        """
        Compute bulk spectral parameters and store them as
        time-series variables in ``self.ds``, with ``name``/``long_name``/
        ``units``/``history`` attrs set on each.

        Which parameters are computed depends on what spectral data are
        available:

        From E(f,t)  [or S collapsed to E via integrate over θ]:
            hs     Significant wave height      4√m₀              m
            tm01   Mean wave period              m₀/m₁             s
            tm02   Mean zero-crossing period     √(m₀/m₂)          s
            tm_10  Energy wave period            m₋₁/m₀            s
            fp     Peak frequency                argmax E(f)        Hz
            tp     Peak period                   1/fp               s
            fpI    Interpolated peak frequency   Lagrange vertex    Hz
            tpI    Interpolated peak period      1/fpI              s

        From D(θ,t)  [or S collapsed to D via integrate over f]:
            pdir   Peak wave direction           argmax D(θ)        °
            pdirI  Interpolated peak direction   Lagrange vertex    °
            mdir   Mean wave direction           circular mean      °
            spr    Directional spread            Kuik et al. 1988   °

        From S(f,θ,t) only:
            pdir2  Peak direction (2D)           argmax S(f,θ)      °
            fp2    Peak frequency (2D)           argmax S(f,θ)      Hz

        Parameters
        ----------
        compute : list[str] or None
            Subset of parameter names to compute, e.g. ``["hs", "fp", "fpI"]``.
            If None (default), all derivable parameters are computed.
            Valid names: "hs", "tm01", "tm02", "tm_10", "fp", "tp", "fpI",
            "tpI", "pdir", "pdirI", "mdir", "spr", "pdir2", "fp2".

        Returns
        -------
        self
        """

        _FREQ_PARAMS = {"hs", "tm01", "tm02", "tm_10", "fp", "tp", "fpI", "tpI"}
        _DIR_PARAMS  = {"pdir", "pdirI", "mdir", "spr"}
        _2D_PARAMS   = {"pdir2", "fp2"}    
        wanted = set(compute) if compute is not None else (_FREQ_PARAMS | _DIR_PARAMS | _2D_PARAMS)

        # Validate requested names
        all_valid = _FREQ_PARAMS | _DIR_PARAMS | _2D_PARAMS
        unknown = wanted - all_valid
        if unknown:
            raise ValueError(
                f"Unknown parameter(s): {sorted(unknown)}. "
                f"Valid names: {sorted(all_valid)}"
            )

        freq_dim = self.dim_map[_ROLE_FREQ]
        dir_dim  = self.dim_map[_ROLE_DIR]
        results: dict[str, tuple[xr.DataArray, str]] = {}

        if self.has_frequency and wanted & _FREQ_PARAMS:
            E      = self.compute_frequency_spectrum()
            E_name = self.var_map["E"] or _DERIVED_E_NAME
            all_freq = self._bulk_from_E(E, freq_dim, E_name)
            results.update({k: v for k, v in all_freq.items() if k in wanted})

        if self.has_directional and wanted & _DIR_PARAMS:
            D      = self.compute_directional_spectrum()
            D_name = self.var_map["D"] or _DERIVED_D_NAME
            all_dir = self._bulk_from_D(D, dir_dim, self.dir_convention, D_name)
            results.update({k: v for k, v in all_dir.items() if k in wanted})

        if self.has_2d and wanted & _2D_PARAMS:
            S_name = self.var_map["S"]
            fp2_da, pdir2_da, history = self._peak2_from_S(
                self.ds[S_name], freq_dim, dir_dim, self.dir_convention, S_name
            )
            if "fp2"   in wanted: results["fp2"]   = (fp2_da,   history)
            if "pdir2" in wanted: results["pdir2"] = (pdir2_da, history)

        for out_name, (da, history) in results.items():
            meta = _INTEGRATED_VARS[out_name]
            _apply_attrs(da, meta, extra={
                "history": history,
                **({"dir_convention": self.dir_convention}
                if out_name in ("pdir", "pdirI", "mdir", "spr", "pdir2") else {}),
            })
            self.ds[out_name] = da

        print(f"integrate(): computed {len(results)} parameter(s): {sorted(results)}")
        return self


    @staticmethod
    def _compute_dpm(
        S: xr.DataArray,
        E: xr.DataArray,
        freq_dim: str,
        dir_dim: str,
    ) -> xr.DataArray:
        """
        Mean direction at the peak frequency, Dpm.

        Evaluated at the discrete peak of E(f) — the frequency bin carrying
        the most energy — using the first circular moments of S(f,θ) at that
        frequency slice. Not stored in self.ds; computed on demand for
        tracking.

        Parameters
        ----------
        S : xr.DataArray
            2D spectrum (time, freq, dir).
        E : xr.DataArray
            Frequency spectrum (time, freq), used only to locate the peak bin.
        freq_dim : str
            Name of the frequency dimension.
        dir_dim : str
            Name of the directional dimension.

        Returns
        -------
        xr.DataArray
            Mean direction at peak frequency, shape (time,), degrees in [0°, 360°).
        """
        time_dim = next(d for d in E.dims if d != freq_dim)

        # Peak frequency index per timestep
        ipeak = E.argmax(dim=freq_dim)  # (time,)

        # S at peak frequency: (time, dir)
        S_at_peak = S.isel({freq_dim: ipeak})

        dirs_deg = S.coords[dir_dim].values.astype(float)
        dirs_rad = np.deg2rad(dirs_deg)
        dirs_da  = xr.DataArray(dirs_rad, dims=[dir_dim], coords={dir_dim: S.coords[dir_dim]})

        eps  = 1e-30
        m0   = S_at_peak.integrate(coord=dir_dim)
        sin_ = (S_at_peak * np.sin(dirs_da)).integrate(coord=dir_dim) / (m0 + eps)
        cos_ = (S_at_peak * np.cos(dirs_da)).integrate(coord=dir_dim) / (m0 + eps)

        dpm = (270.0 - np.rad2deg(np.arctan2(sin_, cos_))) % 360.0
        dpm.name = "dpm"
        return dpm


    # ---------------------------------------------------------------------- #
    # Public: aggregate()                                                     #
    # ---------------------------------------------------------------------- #

    _DEFAULT_SEASONS = {
        "DJF": [12, 1, 2],
        "MAM": [3, 4, 5],
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11],
    }

    @staticmethod
    def _infer_step(vals: np.ndarray, target: int = 10) -> float:
        """
        Infer a round step size from the data range, snapped to a nice
        value. Direct port of the TimeSeries module's ``infer_step``,
        operating on a plain array instead of a pd.Series.
        """
        NICE_STEPS = [0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10]
        raw       = (np.nanmax(vals) - np.nanmin(vals)) / target
        magnitude = 10 ** np.floor(np.log10(raw))
        scaled    = raw / magnitude
        nice      = min(NICE_STEPS, key=lambda x: abs(x - scaled))
        return nice * magnitude

    @staticmethod
    def _bin_edges(vals: np.ndarray, step: float) -> np.ndarray:
        """
        Compute bin edges snapped to a given step size. Starts from zero
        if all values are non-negative, otherwise fits outer edges to the
        data range. Direct port of the TimeSeries module's ``bin_edges``.
        """
        lo = 0.0 if np.nanmin(vals) >= 0 else np.floor(np.nanmin(vals) / step) * step
        hi = np.ceil(np.nanmax(vals) / step) * step
        n  = int(round((hi - lo) / step)) + 1
        return np.linspace(lo, hi, n)

    def _get_time_groups(
        self,
        by: str,
        sectors: int | list[float] | None = None,
        seasons: dict | None = None,
        step: float | None = None,
        circular_by: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Resolve ``by`` into a dict of group_label -> boolean mask over the
        time axis (length == len(self.time)). Mirrors the ``by`` vocabulary
        and labelling conventions of the plain ``TimeSeries`` module's
        ``groupby_month`` / ``groupby_season`` / ``groupby_sector``,
        adapted to produce index masks against ``self.time`` / ``self.ds``
        rather than grouping a DataFrame directly.

        Supported ``by`` values:
          "month", "season", "year", "week", "day", "hour"  -- calendar groups
          <variable name>                                    -- numeric bin
                                                                  groups of any
                                                                  existing 1D
                                                                  (time,) variable.
                                                                  Pass
                                                                  ``circular_by=True``
                                                                  for a variable
                                                                  that wraps at
                                                                  360° (e.g. a
                                                                  direction),
                                                                  using the same
                                                                  north-centred
                                                                  sector
                                                                  convention as
                                                                  ``groupby_sector``.
        """
        t = self.time

        if by == "month":
            months = t.month
            return {
                pd.to_datetime(str(m), format="%m").month_name()[:3]: (months == m)
                for m in range(1, 13) if (months == m).any()
            }

        if by == "season":
            seasons = seasons or self._DEFAULT_SEASONS
            months = t.month
            return {
                label: np.isin(months, mlist)
                for label, mlist in seasons.items()
                if np.isin(months, mlist).any()
            }

        if by == "year":
            years = t.year
            return {str(y): (years == y) for y in sorted(set(years))}

        if by == "week":
            weeks = t.isocalendar().week.values
            return {str(int(w)): (weeks == w) for w in sorted(set(weeks))}

        if by == "day":
            days = t.dayofyear
            return {str(int(d)): (days == d) for d in sorted(set(days))}

        if by == "hour":
            hours = t.hour
            return {str(h): (hours == h) for h in range(24) if (hours == h).any()}

        # --- fall back: bin by an arbitrary existing numeric variable ---
        col = self._resolve_scalar_var(by, by)
        vals = self.ds[col].values.astype(float)
        finite = np.isfinite(vals)

        if circular_by:
            # Same north-centred sector convention as groupby_sector: an
            # int `sectors` gives that many equally-sized sectors with the
            # first centred on 0°; a list gives explicit sector edges.
            edges = sectors if hasattr(sectors, "__len__") else np.linspace(
                0, 360, (sectors or 12) + 1
            )
            offset = (edges[1] - edges[0]) / 2.0
            labels = [
                f"{(edges[i] - offset) % 360:.0f}-{(edges[i + 1] - offset) % 360:.0f}°"
                for i in range(len(edges) - 1)
            ]
            shifted = (vals + offset) % 360.0
            bin_idx = np.digitize(shifted, edges, right=False) - 1
            bin_idx = np.clip(bin_idx, 0, len(labels) - 1)
            groups = {}
            for i, label in enumerate(labels):
                mask = finite & (bin_idx == i)
                if mask.any():
                    groups[label] = mask
            return groups

        _step = step or self._infer_step(vals[finite])
        edges = self._bin_edges(vals[finite], _step)
        labels = edges[:-1].astype(float)
        bin_idx = np.digitize(vals, edges[1:-1], right=False)
        groups = {}
        for i, label in enumerate(labels):
            mask = finite & (bin_idx == i)
            if mask.any():
                groups[label] = mask
        return groups

    def _resolve_scalar_var(self, requested: str, by_label: str) -> str:
        """
        Resolve a requested scalar variable name (e.g. "hs") to an actual
        1D (time,)-only variable name in ``self.ds``. Raises with a hint to
        run ``integrate()`` first if the name is a known integrated output
        that simply hasn't been computed yet.

        ``integrate()`` is never run implicitly here: the user may already
        have supplied their own variable under that name, and silently
        overwriting it would be surprising.
        """
        time_dim = self.dim_map[_ROLE_TIME]

        if requested in self.ds.data_vars and self.ds[requested].dims == (time_dim,):
            return requested

        if requested in _INTEGRATED_VARS:
            raise ValueError(
                f"'{by_label}={requested!r}' requires a scalar time series "
                f"variable ('{requested}') that isn't present in this "
                f"dataset yet. Run `.integrate()` first to compute it, "
                "then call aggregate() again."
            )
        raise ValueError(
            f"'{requested}' is not a variable in this dataset with dims "
            f"('{time_dim}',). Available scalar variables: "
            f"{[v for v in self.ds.data_vars if self.ds[v].dims == (time_dim,)]}."
        )

    def aggregate(
        self,
        by: str,
        stat: str = "mean",
        select_by: str | None = None,
        sectors: int | list[float] | None = None,
        seasons: dict | None = None,
        step: float | None = None,
        circular_by: bool = False,
    ) -> xr.Dataset:
        """
        Aggregate the spectral time series into groups defined by ``by``,
        collapsing the time dimension into a new group dimension named
        after ``by``. Operates on the 2D spectrum if present, otherwise on
        whichever of the 1D frequency/directional spectra are present
        (both, if both exist).

        Two aggregation modes, selected by whether ``select_by`` is given:

        **Element-wise mode** (``select_by=None``, default): for each
        (freq[, dir]) bin independently, compute ``stat`` ("mean", "max",
        "min", or "p<NN>" e.g. "p95") across all spectra in the group. The
        result is a synthetic spectrum -- no single time step necessarily
        looked like it -- useful for e.g. "the average winter directional
        spread" but not guaranteed physically self-consistent (a bulk
        parameter computed from the aggregated spectrum need not equal the
        same stat computed from the per-timestep bulk parameter).

        **Selection mode** (``select_by=<variable name>``): computes no new
        spectral shape. Instead, ranks the existing scalar variable
        ``select_by`` (e.g. "hs") within each group, finds the time step
        nearest the requested ``stat`` percentile (or nearest the group
        mean, if ``stat="mean"``), and returns *that actual spectrum* --
        always a real, physically consistent sea state. ``select_by`` must
        already exist in ``self.ds`` (run ``.integrate()`` first if it's a
        derivable quantity like "hs" that hasn't been computed yet -- this
        method never runs ``integrate()`` implicitly, since you may already
        have supplied your own variable under that name).

        Single-level convenience wrapper -- see `groupby()` for the
        chained, multi-level version this now delegates to.

        Parameters
        ----------
        by : str
            One of "month", "season", "year", "week", "day", "hour", or
            the name of any existing scalar (time,)-only variable to bin
            numerically (e.g. "hs", or a directional variable such as
            "mdir" -- pair with ``circular_by=True`` for the latter).
        stat : str, default "mean"
            "mean", "max", "min", or "p<NN>" (e.g. "p50", "p95"). In
            selection mode, "mean" selects the time step nearest the
            group's mean of ``select_by`` (not an element-wise mean of
            spectra) -- see notes above.
        select_by : str, optional
            Name of an existing scalar variable to select representative
            spectra by. If omitted, aggregation is element-wise (see above).
        sectors : int or list of float, optional
            Only used when ``circular_by=True``. If int, that many
            equally-sized sectors with the first centred on 0°/360° (same
            convention as the TimeSeries module's ``groupby_sector``); if
            a list, explicit sector edges. Default 12 (30° sectors).
        seasons : dict, optional
            Custom season->months mapping, only used when ``by="season"``.
        step : float, optional
            Bin width for non-circular numeric ``by`` variables. Inferred
            via the same "nice number" logic as the TimeSeries module's
            ``infer_step`` if omitted. Ignored when ``circular_by=True``.
        circular_by : bool, default False
            If True, ``by`` is treated as a variable that wraps at 360°
            (e.g. a wave/wind direction) and binned into sectors using
            ``sectors`` instead of ``step``, with labels like "345-15°"
            (north-centred), matching the TimeSeries module's
            ``groupby_sector`` convention.

        Returns
        -------
        xr.Dataset
            A new dataset (not ``self.ds``, and ``self.ds`` is never
            mutated) with the time dimension replaced by a new dimension
            named ``by``, holding whichever spectra were aggregated (S,
            and/or E/D), each with its usual attrs plus a "history" attr
            describing the aggregation performed. Also includes a
            ``_group_counts`` variable along the same group dimension,
            giving the number of time steps aggregated into each group.
        """
        return self.groupby(
            by, sectors=sectors, seasons=seasons, step=step, circular_by=circular_by,
        ).aggregate(stat=stat, select_by=select_by)

    @staticmethod
    def _coord_array(labels: list) -> np.ndarray:
        """
        Build a 1D coordinate array from a list of labels, robust to labels
        being tuples (as produced by a chained `.groupby()` -- even a
        single-level chain produces length-1 tuples). Passing a list of
        tuples straight to `assign_coords` lets numpy infer an extra
        dimension from the tuple length, which xarray then rejects as
        ambiguous ("cannot set variable with 2-dimensional data"). Building
        the object array by hand keeps each label -- string or tuple -- as
        a single opaque coordinate value.
        """
        arr = np.empty(len(labels), dtype=object)
        for i, label in enumerate(labels):
            arr[i] = label
        return arr

    def _aggregate_elementwise(
        self,
        source_vars: list[str],
        groups: dict[str, np.ndarray],
        group_labels: list[str],
        group_dim: str,
        stat: str,
        time_dim: str,
    ) -> tuple[dict[str, xr.DataArray], str]:
        stat_fn, stat_desc = self._resolve_stat_fn(stat)
        out: dict[str, xr.DataArray] = {}

        for var_name in source_vars:
            da = self.ds[var_name]
            slices = []
            for label in group_labels:
                mask = groups[label]
                sub = da.isel({time_dim: mask})
                slices.append(stat_fn(sub, dim=time_dim))
            stacked = xr.concat(slices, dim=group_dim)
            stacked = stacked.assign_coords({group_dim: self._coord_array(group_labels)})
            stacked.attrs.update(da.attrs)
            stacked.attrs["history"] = (
                f"aggregated from '{var_name}' via element-wise {stat_desc} "
                f"within each '{group_dim}' group"
            )
            out[var_name] = stacked

        history = (
            f"element-wise {stat_desc} of {source_vars} grouped by '{group_dim}'"
        )
        return out, history

    def _aggregate_by_selection(
        self,
        source_vars: list[str],
        groups: dict[str, np.ndarray],
        group_labels: list[str],
        group_dim: str,
        scalar_vals: np.ndarray,
        sel_col: str,
        stat: str,
        time_dim: str,
    ) -> tuple[dict[str, xr.DataArray], str]:
        _, stat_desc = self._resolve_stat_fn(stat, for_selection=True)
        selected_idx = []

        for label in group_labels:
            mask = groups[label]
            idx_in_group = np.nonzero(mask)[0]
            vals = scalar_vals[idx_in_group]
            finite = np.isfinite(vals)
            idx_in_group = idx_in_group[finite]
            vals = vals[finite]

            if stat == "mean":
                target = np.mean(vals)
            elif stat == "max":
                target = np.max(vals)
            elif stat == "min":
                target = np.min(vals)
            elif stat.startswith("p"):
                q = float(stat[1:])
                target = np.percentile(vals, q)
            else:
                raise ValueError(f"Unrecognised stat {stat!r}.")

            nearest = idx_in_group[np.argmin(np.abs(vals - target))]
            selected_idx.append(nearest)

        out: dict[str, xr.DataArray] = {}
        for var_name in source_vars:
            da = self.ds[var_name]
            picked = da.isel({time_dim: selected_idx})
            if time_dim in picked.coords:
                picked = picked.drop_vars(time_dim)
            if time_dim in picked.dims:
                picked = picked.rename({time_dim: group_dim})
            picked = picked.assign_coords({group_dim: self._coord_array(group_labels)})
            picked.attrs.update(da.attrs)
            picked.attrs["history"] = (
                f"selected from '{var_name}' -- the real spectrum whose "
                f"'{sel_col}' is nearest the {stat_desc} of '{sel_col}' "
                f"within each '{group_dim}' group"
            )
            out[var_name] = picked

        history = (
            f"selection aggregation of {source_vars}: for each '{group_dim}' "
            f"group, the actual spectrum nearest the {stat_desc} of '{sel_col}'"
        )
        return out, history

    @staticmethod
    def _resolve_stat_fn(stat: str, for_selection: bool = False):
        """
        Returns (fn, description). ``fn`` takes (dataarray, dim=...) and is
        only used in element-wise mode; in selection mode only the
        description is used (the actual selection logic lives in
        ``_aggregate_by_selection``).
        """
        if stat == "mean":
            return (lambda da, dim: da.mean(dim=dim)), "mean"
        if stat == "max":
            return (lambda da, dim: da.max(dim=dim)), "max"
        if stat == "min":
            return (lambda da, dim: da.min(dim=dim)), "min"
        if stat.startswith("p"):
            try:
                q = float(stat[1:])
            except ValueError:
                raise ValueError(
                    f"stat={stat!r} not recognised. Use 'mean', 'max', 'min', "
                    "or 'p<NN>' e.g. 'p95'."
                )
            return (lambda da, dim, q=q: da.quantile(q / 100.0, dim=dim, keep_attrs=True)), f"p{q:g}"
        raise ValueError(
            f"stat={stat!r} not recognised. Use 'mean', 'max', 'min', or "
            "'p<NN>' e.g. 'p95'."
        )
    
    def groupby(
        self,
        by: str,
        sectors: int | list[float] | None = None,
        seasons: dict | None = None,
        step: float | None = None,
        circular_by: bool = False,
    ) -> "_GroupedSpectralTimeSeries":
        """
        Start a (possibly chained) groupby, terminated by
        `.aggregate(stat=, select_by=)`. E.g.:

            sts.groupby("season") \\
               .groupby("hs", step=2) \\
               .groupby("tp", step=1) \\
               .aggregate(select_by="hs", stat="max")

        nests by season, then by hs-bin, then by tp-bin, then picks the
        representative spectrum in each leaf cell. A single
        `.groupby(by, ...).aggregate(...)` call is equivalent to the old
        one-level `aggregate(by, ...)`, which is now just a thin wrapper
        around this.
        """
        groups = self._get_time_groups(
            by, sectors=sectors, seasons=seasons, step=step, circular_by=circular_by,
        )
        return _GroupedSpectralTimeSeries(self, [(by, groups)])

    def _spectral_source_vars(self) -> list[str]:
        """
        Which spectral variable(s) aggregation operates on: the 2D
        spectrum if present, else whichever of the 1D frequency/
        directional spectra exist (both, if both exist).
        """
        source_vars: list[str] = []
        if self.has_2d:
            source_vars.append(self.var_map["S"])
        else:
            if self.var_map["E"] is not None:
                source_vars.append(self.var_map["E"])
            if self.var_map["D"] is not None:
                source_vars.append(self.var_map["D"])
        if not source_vars:
            raise ValueError(
                "aggregate() found no spectral variable to aggregate. This "
                "should not happen on a validly constructed SpectralTimeSeries."
            )
        return source_vars

    # -------------------------------------------------------------------------- #
    # Partitioning                                                                #
    # -------------------------------------------------------------------------- #

    def _resolve_var(self, var_name: str, label: str) -> xr.DataArray:
        """Fetch a variable from self.ds by name, with a clear error."""
        if var_name not in self.ds:
            raise ValueError(
                f"{label}={var_name!r} not found in self.ds. "
                f"Available variables: {list(self.ds.data_vars)}"
            )
        return self.ds[var_name]

    def _wind_inputs(
        self,
        wspd_var: str,
        wdir_var: str,
        dpt_var: str | None,
        wdir_from: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Resolve and return wind/depth as plain numpy arrays (one value per
        timestep), with the wdir convention flip applied if needed.

        Returns (wspd, wdir, dpt) where each is a 1D numpy array of length
        n_times, and dpt is None if dpt_var is None.
        """
        wspd_da = self._resolve_var(wspd_var, "wspd_var")
        wdir_da = self._resolve_var(wdir_var, "wdir_var")
        dpt_da  = self._resolve_var(dpt_var, "dpt_var") if dpt_var is not None else None

        time_dim = self.dim_map[_ROLE_TIME]
        wspd = wspd_da.values.astype(float)
        wdir = wdir_da.values.astype(float)
        if wdir_from:
            wdir = (wdir + 180.0) % 360.0
        dpt  = dpt_da.values.astype(float) if dpt_da is not None else None

        return wspd, wdir, dpt

    def _smooth_spectrum(
        self,
        S: xr.DataArray,
        freq_window: int,
        dir_window: int,
    ) -> xr.DataArray:
        """Rolling-average smoothing along freq and dir."""
        if freq_window % 2 == 0 or dir_window % 2 == 0:
            raise ValueError(
                "freq_window and dir_window must be odd integers "
                f"(got {freq_window}, {dir_window})."
            )
        freq_dim = self.dim_map[_ROLE_FREQ]
        dir_dim  = self.dim_map[_ROLE_DIR]
        return (
            S
            .rolling({freq_dim: freq_window, dir_dim: dir_window}, center=True)
            .mean()
            .fillna(0.0)
        )

    def _build_collection(
        self,
        out: np.ndarray,
        labels: list[str],
    ) -> "SpectralTimeSeriesCollection":
        spec_var = self.var_map["S"]
        freq_dim = self.dim_map[_ROLE_FREQ]
        dir_dim  = self.dim_map[_ROLE_DIR]
        time_dim = self.dim_map[_ROLE_TIME]

        S      = self.ds[spec_var]
        times  = self.ds.coords[time_dim].values
        freqs  = self.ds.coords[freq_dim].values
        dirs   = self.ds.coords[dir_dim].values

        stale_1d_names = [
            n for n in (
                self.var_map.get("E"),
                self.var_map.get("D"),
                _DERIVED_E_NAME,
                _DERIVED_D_NAME,
            )
            if n is not None and n in self.ds.data_vars
        ]

        vars_to_drop = [spec_var] + stale_1d_names + [
            v for v in _INTEGRATED_VARS if v in self.ds.data_vars
        ]

        members: dict[str, SpectralTimeSeries] = {}
        for pi, label in enumerate(labels):
            da_part = xr.DataArray(
                out[pi],
                coords={time_dim: times, freq_dim: freqs, dir_dim: dirs},
                dims=(time_dim, freq_dim, dir_dim),
                attrs={**S.attrs, "partition": label},
            )
            ds_part = self.ds.drop_vars(vars_to_drop).assign({spec_var: da_part})

            # var_map for each partition member must reflect that E/D are no
            # longer present as named variables -- otherwise has_frequency /
            # has_directional would report True from a stale var_map even
            # though the actual data was just dropped above.
            part_var_map = self.var_map.copy()
            part_var_map["E"] = None
            part_var_map["D"] = None

            members[label] = SpectralTimeSeries.__new__(SpectralTimeSeries)._init_from_ds(
                ds=ds_part,
                var_map=part_var_map,
                dim_map=self.dim_map.copy(),
                dir_convention=self.dir_convention,
                depth=self.depth,
                name=f"{self.name} / {label}" if self.name else label,
            )
        return SpectralTimeSeriesCollection(members)

    def _run_partition(
        self,
        np_func,
        labels: list[str],
        per_timestep_args,   # callable(ti) -> list of positional args for np_func
        smooth: bool,
        freq_window: int,
        dir_window: int,
    ) -> "SpectralTimeSeriesCollection":
        """
        Core dispatcher: loop over timesteps, call np_func, pack results.

        Parameters
        ----------
        np_func : callable
            One of the wavespectra np_* functions. Must accept
            (spectrum, spectrum_smooth, ...) and return an array of shape
            (n_parts, nf, nd).
        labels : list[str]
            Partition labels in slot order, e.g. ["wind_sea", "swell_0", ...].
        per_timestep_args : callable(ti: int) -> list
            Returns the extra positional arguments for np_func at timestep ti
            (everything after spectrum and spectrum_smooth).
        smooth : bool
            Whether to smooth the spectrum before partitioning.
        freq_window, dir_window : int
            Smoothing window sizes (must be odd).
        """

        spec_var = self.var_map["S"]
        freq_dim = self.dim_map[_ROLE_FREQ]
        dir_dim  = self.dim_map[_ROLE_DIR]
        time_dim = self.dim_map[_ROLE_TIME]

        S       = self.ds[spec_var]
        n_times = S.sizes[time_dim]
        n_freq  = S.sizes[freq_dim]
        n_dir   = S.sizes[dir_dim]
        n_parts = len(labels)

        S_smooth = self._smooth_spectrum(S, freq_window, dir_window) if smooth else S

        out = np.zeros((n_parts, n_times, n_freq, n_dir), dtype=float)

        for ti in tqdm(range(n_times)):
            spec    = S.isel({time_dim: ti}).values.astype(float)
            spec_sm = S_smooth.isel({time_dim: ti}).values.astype(float)
            args    = per_timestep_args(ti)
            out[:, ti, :, :] = np_func(spec, spec_sm, *args)

        return self._build_collection(out, labels)

    # ---------------------------------------------------------------------- #
    # Public partitioning methods                                             #
    # ---------------------------------------------------------------------- #

    def partition_ptm1(
        self,
        wspd_var: str,
        wdir_var: str,
        dpt_var: str | None = None,
        wdir_from: bool = True,
        agefac: float = 1.7,
        wscut: float = 0.3333,
        swells: int = 3,
        ihmax: int = 100,
        smooth: bool = False,
        freq_window: int = 3,
        dir_window: int = 3,
    ) -> "SpectralTimeSeriesCollection":
        """
        PTM1 watershed partitioning.

        Topographic partitions for which the percentage of wind-sea energy
        exceeds ``wscut`` are aggregated into the wind-sea component. The
        remaining partitions are assigned as swell components in order of
        decreasing Hs.

        Parameters
        ----------
        wspd_var : str
            Wind speed variable in ``self.ds`` (m/s).
        wdir_var : str
            Wind direction variable in ``self.ds`` (degrees).
        dpt_var : str or None
            Water depth variable in ``self.ds`` (m). If None, deep-water
            celerity is used (c = 1.56 / f).
        wdir_from : bool
            If True (default), ``wdir_var`` is in the meteorological
            "coming from" convention and is flipped by 180° internally.
            Set to False if the variable is already in the "going to"
            convention expected by the wave-age criterion.
        agefac : float
            Wave-age factor (default 1.7).
        wscut : float
            Wind-sea energy fraction cutoff (default 0.3333).
        swells : int
            Number of swell partitions to return (default 3). Overflow
            partitions are discarded (wavespectra default behaviour).
        ihmax : int
            Number of immersion levels for the watershed (default 100).
        smooth : bool
            Smooth the spectrum before computing watershed boundaries
            (Portilla et al. 2009). Reduces over-segmentation on noisy
            spectra.
        freq_window : int
            Smoothing window along freq (odd integer, default 3).
        dir_window : int
            Smoothing window along dir (odd integer, default 3).

        Returns
        -------
        SpectralTimeSeriesCollection
            Keys: ``"wind_sea"``, ``"swell_0"``, …, ``"swell_{swells-1}"``.
        """
        from wavespectra.partition.partition import np_ptm1

        self._requires_2d_spectrum("partition_ptm1")
        wspd, wdir, dpt = self._wind_inputs(wspd_var, wdir_var, dpt_var, wdir_from)
        freqs = self.ds.coords[self.dim_map[_ROLE_FREQ]].values.astype(float)
        dirs  = self.ds.coords[self.dim_map[_ROLE_DIR]].values.astype(float)

        labels = ["wind_sea"] + [f"swell_{i}" for i in range(swells)]

        def args(ti):
            return [freqs, dirs, wspd[ti], wdir[ti],
                    dpt[ti] if dpt is not None else None,
                    agefac, wscut, swells, ihmax]

        return self._run_partition(np_ptm1, labels, args, smooth, freq_window, dir_window)

    def partition_ptm2(
        self,
        wspd_var: str,
        wdir_var: str,
        dpt_var: str | None = None,
        wdir_from: bool = True,
        agefac: float = 1.7,
        wscut: float = 0.3333,
        swells: int = 3,
        ihmax: int = 100,
        smooth: bool = False,
        freq_window: int = 3,
        dir_window: int = 3,
    ) -> "SpectralTimeSeriesCollection":
        """
        PTM2 watershed partitioning with secondary wind-sea.

        Like PTM1 but wind-sea energy found within swell partitions is
        stripped out and collected into a secondary wind-sea slot (index 1)
        rather than being left in the swell. Swell slots start at index 2.

        Parameters
        ----------
        wspd_var, wdir_var, dpt_var, wdir_from, agefac, wscut, ihmax,
        smooth, freq_window, dir_window :
            Same as ``partition_ptm1``.
        swells : int
            Number of swell partitions (default 3). Output has
            ``2 + swells`` total slots.

        Returns
        -------
        SpectralTimeSeriesCollection
            Keys: ``"wind_sea_primary"``, ``"wind_sea_secondary"``,
            ``"swell_0"``, …, ``"swell_{swells-1}"``.
        """
        from wavespectra.partition.partition import np_ptm2

        self._requires_2d_spectrum("partition_ptm2")
        wspd, wdir, dpt = self._wind_inputs(wspd_var, wdir_var, dpt_var, wdir_from)
        freqs = self.ds.coords[self.dim_map[_ROLE_FREQ]].values.astype(float)
        dirs  = self.ds.coords[self.dim_map[_ROLE_DIR]].values.astype(float)

        labels = ["wind_sea_primary", "wind_sea_secondary"] + [f"swell_{i}" for i in range(swells)]

        def args(ti):
            return [freqs, dirs, wspd[ti], wdir[ti],
                    dpt[ti] if dpt is not None else None,
                    agefac, wscut, swells, ihmax]

        return self._run_partition(np_ptm2, labels, args, smooth, freq_window, dir_window)

    def partition_ptm3(
        self,
        parts: int = 3,
        ihmax: int = 100,
        smooth: bool = False,
        freq_window: int = 3,
        dir_window: int = 3,
    ) -> "SpectralTimeSeriesCollection":
        """
        PTM3 watershed partitioning — no wind-sea/swell classification.

        Partitions are ordered by descending Hs only. Useful when wind
        information is unavailable or when classification is not needed
        (e.g. spectral reconstruction).

        Parameters
        ----------
        parts : int
            Number of partitions to return (default 3).
        ihmax : int
            Number of immersion levels for the watershed (default 100).
        smooth : bool
            Smooth before computing watershed boundaries.
        freq_window, dir_window : int
            Smoothing window sizes (odd integers, default 3).

        Returns
        -------
        SpectralTimeSeriesCollection
            Keys: ``"part_0"``, ``"part_1"``, …, ``"part_{parts-1}"``.
        """
        from wavespectra.partition.partition import np_ptm3

        self._requires_2d_spectrum("partition_ptm3")
        freqs = self.ds.coords[self.dim_map[_ROLE_FREQ]].values.astype(float)
        dirs  = self.ds.coords[self.dim_map[_ROLE_DIR]].values.astype(float)

        labels = [f"part_{i}" for i in range(parts)]

        def args(ti):
            return [freqs, dirs, parts, ihmax]

        return self._run_partition(np_ptm3, labels, args, smooth, freq_window, dir_window)

    def partition_ptm4(
        self,
        wspd_var: str,
        wdir_var: str,
        dpt_var: str | None = None,
        wdir_from: bool = True,
        agefac: float = 1.7,
    ) -> "SpectralTimeSeriesCollection":
        """
        PTM4 WAM-style partitioning by wave-age criterion.

        Splits the spectrum into wind sea and a single swell partition by
        masking: bins where celerity ≤ agefac × wspd × cos(dir − wdir) are
        wind sea, everything else is swell. No watershed is computed.

        Parameters
        ----------
        wspd_var, wdir_var, dpt_var, wdir_from, agefac :
            Same as ``partition_ptm1``.

        Returns
        -------
        SpectralTimeSeriesCollection
            Keys: ``"wind_sea"``, ``"swell_0"``.
        """
        from wavespectra.core.utils import celerity as ws_celerity

        self._requires_2d_spectrum("partition_ptm4")
        wspd, wdir, dpt = self._wind_inputs(wspd_var, wdir_var, dpt_var, wdir_from)

        spec_var = self.var_map["S"]
        freq_dim = self.dim_map[_ROLE_FREQ]
        dir_dim  = self.dim_map[_ROLE_DIR]
        time_dim = self.dim_map[_ROLE_TIME]

        S     = self.ds[spec_var]
        freqs = self.ds.coords[freq_dim].values.astype(float)
        dirs  = self.ds.coords[dir_dim].values.astype(float)
        times = self.ds.coords[time_dim].values
        n_times, n_freq, n_dir = S.sizes[time_dim], S.sizes[freq_dim], S.sizes[dir_dim]

        out = np.zeros((2, n_times, n_freq, n_dir), dtype=float)

        D2R = np.pi / 180.0
        for ti in tqdm(range(n_times)):
            spec     = S.isel({time_dim: ti}).values.astype(float)
            dpt_val  = dpt[ti] if dpt is not None else None
            cel      = ws_celerity(freqs, dpt_val)                          # (nf,)
            wind_c   = agefac * wspd[ti] * np.cos(D2R * (dirs - wdir[ti])) # (nd,)
            wsmask   = cel[:, None] <= wind_c[None, :]                      # (nf, nd)
            out[0, ti] = np.where(wsmask, spec, 0.0)   # wind sea
            out[1, ti] = np.where(~wsmask, spec, 0.0)  # swell

        return self._build_collection(out, ["wind_sea", "swell_0"])

    def partition_ptm5(
        self,
        fcut: float,
    ) -> "SpectralTimeSeriesCollection":
        """
        PTM5 static frequency-cutoff partitioning.

        Splits the spectrum at ``fcut`` Hz: bins at or above the cutoff are
        wind sea, bins at or below are swell. No wind or depth information
        is required.

        Parameters
        ----------
        fcut : float
            Cutoff frequency in Hz.

        Returns
        -------
        SpectralTimeSeriesCollection
            Keys: ``"wind_sea"``, ``"swell_0"``.

        Notes
        -----
        Unlike wavespectra's PTM5, this implementation does not interpolate
        the spectrum at ``fcut`` — the split is applied to the existing
        frequency grid as-is. Bins exactly at ``fcut`` are included in both
        partitions (the boundary bin carries the full energy in each).
        """
        self._requires_2d_spectrum("partition_ptm5")

        spec_var = self.var_map["S"]
        freq_dim = self.dim_map[_ROLE_FREQ]
        dir_dim  = self.dim_map[_ROLE_DIR]
        time_dim = self.dim_map[_ROLE_TIME]

        S     = self.ds[spec_var]
        freqs = self.ds.coords[freq_dim].values.astype(float)
        n_times, n_freq, n_dir = S.sizes[time_dim], S.sizes[freq_dim], S.sizes[dir_dim]

        hf_mask = freqs >= fcut   # (nf,)  wind sea
        lf_mask = freqs <= fcut   # (nf,)  swell

        spec_vals = S.values.astype(float)  # (n_times, nf, nd)
        out = np.zeros((2, n_times, n_freq, n_dir), dtype=float)
        out[0] = spec_vals * hf_mask[None, :, None]
        out[1] = spec_vals * lf_mask[None, :, None]

        return self._build_collection(out, ["wind_sea", "swell_0"])

    def partition_hp01(
        self,
        wspd_var: str,
        wdir_var: str,
        dpt_var: str | None = None,
        wdir_from: bool = True,
        agefac: float = 1.7,
        wscut: float = 0.3333,
        swells: int = 3,
        ihmax: int = 100,
        smooth: bool = False,
        freq_window: int = 3,
        dir_window: int = 3,
        kappa: float = 0.4,
        zeta: float = 0.65,
        angle_max: float = 30.0,
        hs_min: float = 0.2,
        noise_a: float | None = None,
        noise_b: float = 0.0,
        combine_extra_swells: bool = True,
    ) -> "SpectralTimeSeriesCollection":
        """
        Hanson and Phillips (2001) partitioning with swell merging.

        Runs PTM1-style watershed + wind-sea classification, then merges
        adjacent swell partitions belonging to the same wave system
        following the criteria of Hanson and Phillips (2001) and Hanson et
        al. (2009). Useful for noisy measured spectra that the watershed
        algorithm tends to over-segment.

        Parameters
        ----------
        wspd_var, wdir_var, dpt_var, wdir_from, agefac, wscut, swells,
        ihmax, smooth, freq_window, dir_window :
            Same as ``partition_ptm1``.
        kappa : float
            Spread factor in the peak-separation criterion (HP01 eq. 9).
            Larger values combine more partitions. Default 0.4.
        zeta : float
            Peak-minimum factor: fraction of the smaller peak density that
            the saddle point between two partitions must exceed for them to
            be combined. Smaller values combine more. Default 0.65.
        angle_max : float
            Maximum angle (degrees) between partition mean directions for
            combining (Hanson et al. 2009). Default 30°.
        hs_min : float
            Minimum Hs (m) of swell partitions; smaller ones are always
            merged with their most connected neighbour. Default 0.2 m.
        noise_a : float or None
            Factor A in HP01 eq. 10 noise threshold
            (e ≤ A / (fp⁴ + B)). Disabled if None.
        noise_b : float
            Factor B in HP01 eq. 10. Default 0.0.
        combine_extra_swells : bool
            If True (default), extra swells beyond ``swells`` are merged
            into their nearest neighbour. If False, the smallest extras are
            discarded (energy not conserved).

        Returns
        -------
        SpectralTimeSeriesCollection
            Keys: ``"wind_sea"``, ``"swell_0"``, …, ``"swell_{swells-1}"``.
        """
        from wavespectra.partition.partition import np_hp01
        from wavespectra.core.utils import celerity as ws_celerity

        self._requires_2d_spectrum("partition_hp01")
        wspd, wdir, dpt = self._wind_inputs(wspd_var, wdir_var, dpt_var, wdir_from)
        freqs = self.ds.coords[self.dim_map[_ROLE_FREQ]].values.astype(float)
        dirs  = self.ds.coords[self.dim_map[_ROLE_DIR]].values.astype(float)

        labels = ["wind_sea"] + [f"swell_{i}" for i in range(swells)]

        D2R = np.pi / 180.0

        def args(ti):
            dpt_val = dpt[ti] if dpt is not None else None
            cel     = ws_celerity(freqs, dpt_val)
            wind_c  = agefac * wspd[ti] * np.cos(D2R * (dirs - wdir[ti]))
            wsmask  = cel[:, None] <= wind_c[None, :]
            return [wsmask, freqs, dirs,
                    wscut, swells,
                    kappa, zeta, angle_max, hs_min,
                    noise_a, noise_b, ihmax, combine_extra_swells]

        return self._run_partition(np_hp01, labels, args, smooth, freq_window, dir_window)

    # ---------------------------------------------------------------------- #
    # denoising()                                                            #
    # ---------------------------------------------------------------------- #

    def denoise_gaussian(
        self,
        sigma: float = 0.8,
        inplace: bool = False,
    ) -> "SpectralTimeSeries":
        """
        Gaussian-smooth whichever spectra are present (S, and/or the 1D E/D
        spectra) to suppress single-bin noise spikes, e.g. before calling
        partition_watershed() on noisy observed data.

        This is a general-purpose denoising step, entirely separate from
        partitioning: partition_watershed() always operates on the spectrum
        exactly as it finds it in self.ds and never smooths internally,
        specifically so that its partition-count control (see its docstring)
        stays deterministic and inspectable. If you want smoothed input,
        call this method first and partition the result.

        The directional axis, if present, is treated as circular: smoothing
        is done on a copy of the array padded by mirroring a few bins across
        the 0°/360° wrap, so a wave system straddling that boundary is not
        artificially blurred against a false edge. The frequency axis (and
        time) are smoothed as ordinary open boundaries (no wrap).

        Parameters
        ----------
        sigma : float, default 0.8
            Gaussian smoothing sigma, in bin units, applied independently to
            each spectral variable found (S get sigma applied on both its
            freq and dir axes; E only on freq; D only on dir, circularly).
            Set to 0 to make this a no-op (returns an unmodified copy, or
            self, if inplace=True).
        inplace : bool, default False
            If False (default), returns a new SpectralTimeSeries wrapping a
            deep copy of self.ds with the denoised spectra substituted in --
            self is left completely untouched. If True, mutates self.ds
            directly and returns self, for chaining
            (e.g. sts.denoise_spectra(inplace=True).partition_watershed()).

        Returns
        -------
        SpectralTimeSeries
            A new instance (inplace=False) or self (inplace=True), with
            each present spectral variable's values replaced by their
            smoothed version. All attrs (name, long_name, units, ...) are
            preserved as-is; only the numeric values change. A "history"
            attr is appended noting the denoising applied.
        """
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}.")

        target_ds = self.ds if inplace else self.ds.copy(deep=True)

        freq_dim = self.dim_map[_ROLE_FREQ]
        dir_dim  = self.dim_map[_ROLE_DIR]

        for slot, var_name in self.var_map.items():
            if var_name is None:
                continue
            da = target_ds[var_name]

            if slot == "S":
                smoothed_vals = self._denoise_2d(da.values, sigma, sigma)
            elif slot == "E":
                smoothed_vals = self._denoise_1d_open(da.values, sigma, freq_dim, da.dims)
            elif slot == "D":
                smoothed_vals = self._denoise_1d_circular(da.values, sigma, dir_dim, da.dims)
            else:
                continue

            target_ds[var_name].values[...] = smoothed_vals
            existing_history = target_ds[var_name].attrs.get("history", "")
            note = f"denoised via Gaussian smoothing (sigma={sigma:g} bins)"
            target_ds[var_name].attrs["history"] = (
                f"{existing_history}; {note}" if existing_history else note
            )

        if inplace:
            return self

        return SpectralTimeSeries.__new__(SpectralTimeSeries)._init_from_ds(
            target_ds, self.var_map, self.dim_map,
            self.dir_convention, self.depth, self.name,
        )

    def _init_from_ds(
        self,
        ds: xr.Dataset,
        var_map: dict[str, str | None],
        dim_map: dict[str, str | None],
        dir_convention: str,
        depth: float | None,
        name: str | None,
    ) -> "SpectralTimeSeries":
        """
        Internal constructor bypass: build a SpectralTimeSeries instance
        directly from an already-standardized, already-classified dataset,
        skipping __init__'s identify/standardize/describe pipeline entirely.
        Used by denoise_spectra(inplace=False) to hand back a new instance
        without re-running detection on a dataset we already fully
        understand (and, more importantly, without re-printing the
        constructor's banner/table for what is really just a light
        transformation of an existing, already-valid instance).
        """
        self.ds             = ds
        self.var_map        = var_map
        self.dim_map        = dim_map
        self.dir_convention  = dir_convention
        self.depth           = depth
        self.name             = name
        return self

    @staticmethod
    def _denoise_2d(vals: np.ndarray, sigma_freq: float, sigma_dir: float) -> np.ndarray:
        """
        Smooth a (time, freq, dir) array along its last two axes only
        (never across time), with the dir axis treated as circular via
        mirror-padding before filtering and un-padding after.
        """
        if sigma_freq == 0 and sigma_dir == 0:
            return vals

        n_dir = vals.shape[-1]
        wrap = min(4, max(1, n_dir // 2))
        padded = np.concatenate(
            [vals[..., -wrap:], vals, vals[..., :wrap]], axis=-1
        )
        smoothed = gaussian_filter(
            padded, sigma=(0, sigma_freq, sigma_dir), mode="nearest"
        )
        return smoothed[..., wrap:wrap + n_dir]

    @staticmethod
    def _denoise_1d_open(
        vals: np.ndarray, sigma: float, axis_dim: str, dims: tuple[str, ...]
    ) -> np.ndarray:
        """Smooth a (time, freq) array along freq only; freq is not circular."""
        if sigma == 0:
            return vals
        axis = dims.index(axis_dim)
        sigmas = [0] * vals.ndim
        sigmas[axis] = sigma
        return gaussian_filter(vals, sigma=sigmas, mode="nearest")

    @staticmethod
    def _denoise_1d_circular(
        vals: np.ndarray, sigma: float, axis_dim: str, dims: tuple[str, ...]
    ) -> np.ndarray:
        """Smooth a (time, dir) array along dir only, treating dir as circular."""
        if sigma == 0:
            return vals
        axis = dims.index(axis_dim)
        n_dir = vals.shape[axis]
        wrap = min(4, max(1, n_dir // 2))
        padded = np.concatenate(
            [np.take(vals, range(n_dir - wrap, n_dir), axis=axis),
             vals,
             np.take(vals, range(0, wrap), axis=axis)],
            axis=axis,
        )
        sigmas = [0] * vals.ndim
        sigmas[axis] = sigma
        smoothed = gaussian_filter(padded, sigma=sigmas, mode="nearest")
        return np.take(smoothed, range(wrap, wrap + n_dir), axis=axis)

    # ---------------------------------------------------------------------- #
    # Public: plot()                                                         #
    # ---------------------------------------------------------------------- #

    def plot(
        self,
        var: str | None = None,
        time=None,
        row: str | None = None,
        col: str | None = None,
        col_wrap: int | None = None,
        radius: str = "frequency",
        plot_type: str = "pcolormesh",
        cmap: str = "viridis",
        vmax: float | None = None,
        dir_letters: bool = False,
        panel_size: float = 3.0,
    ):
        """
        Plot this instance's spectrum -- one timestep, a time slice faceted
        via row=/col=, etc. Thin wrapper around plotting.plot_spectra(); see
        that function's docstring for full parameter documentation.

        Examples
        --------
        >>> sts.plot(time="2020-01-15T00")                     # doctest: +SKIP
        >>> sts.plot(time=slice("2020-01-01", "2020-01-03"),
        ...          col="time", col_wrap=4)                    # doctest: +SKIP
        """
        from .plots import plot_spectra
        return plot_spectra(
            self, var=var, time=time, row=row, col=col, col_wrap=col_wrap,
            radius=radius, plot_type=plot_type, cmap=cmap, vmax=vmax,
            dir_letters=dir_letters, panel_size=panel_size,
        )

    # ------------------------------------------------------------------ #
    # Public: plot_overview()                                            #
    # ------------------------------------------------------------------ #

    def plot_overview(
        self,
        time_reduce: str = "mean",
        wspd_var: str | None = None,
        wdir_var: str | None = None,
        dpt_var: str | None = None,
        wdir_from: bool = True,
        partition_method: str | None = None,
        partition_kwargs: dict | None = None,
        plot_type: str = "pcolormesh",
        radius: str = "frequency",
        dir_letters: bool = False,
        max_spec_value: float | None = None,
        title: str | None = None,
        output_file: str | None = None,
    ) -> "plt.Figure":
        """
        Summary plot of the (time-reduced) spectrum: 2D polar spectrum if
        present, else whichever 1D frequency/directional spectra exist;
        a 1D frequency-spectrum inset; an Hs bar; and, if wind data is
        available, a wind-direction arrow (plus windsea/swell direction
        arrows if `partition_method` is given).

        Parameters
        ----------
        time_reduce : {"mean", "max_hs"}, default "mean"
            How to collapse the time axis to one representative spectrum
            before plotting.
            "mean"    - element-wise mean spectrum across all timesteps.
            "max_hs"  - the single actual timestep with the highest hs
                        (a real physical spectrum, not a synthetic
                        average) -- requires `integrate()` to have been
                        run already, or is run here on demand if `hs`
                        isn't yet in `self.ds`.
        wspd_var, wdir_var, dpt_var : str, optional
            Names of wind speed / wind direction / depth variables in
            `self.ds`, forwarded to `partition_ptm1`/`ptm2`/`ptm3` if
            `partition_method` is given, and used to draw the wind
            direction arrow if `wdir_var` alone is given. `dpt_var` is
            only used for partitioning, not for the plain wind arrow.
        wdir_from : bool, default True
            Passed through to the partition method -- see
            `partition_ptm1`'s docstring. Also governs how the plain wind
            arrow (drawn when no `partition_method` is given) is
            oriented: True draws the arrow pointing *from* the compass
            direction in `wdir_var` (meteorological convention); False
            draws it pointing *toward* that direction.
        partition_method : {"ptm1", "ptm2", "ptm3"}, optional
            If given, also draws mean-direction arrows for each resulting
            partition (via `self.partition_{method}(...)`), colour-coded.
            Requires `wspd_var`/`wdir_var` (ptm1/ptm2) or nothing extra
            (ptm3, though ptm3's partitions have no wind_sea/swell
            semantics -- arrows are still drawn per partition, just
            unlabelled as such).
        partition_kwargs : dict, optional
            Extra keyword arguments forwarded to the chosen partition
            method (e.g. `{"swells": 2, "smooth": True}`).
        plot_type : {"pcolormesh", "contour"}, default "pcolormesh"
        radius : {"frequency", "period"}, default "frequency"
        dir_letters : bool, default False
        max_spec_value : float, optional
        title : str, optional
        output_file : str, optional
            If given, saves the figure to this path (dpi=200).

        Returns
        -------
        matplotlib.figure.Figure
        """
        if radius not in ("frequency", "period"):
            raise ValueError(f"radius must be 'frequency' or 'period', got {radius!r}.")
        if plot_type not in ("pcolormesh", "contour"):
            raise ValueError(f"plot_type must be 'pcolormesh' or 'contour', got {plot_type!r}.")
        if time_reduce not in ("mean", "max_hs"):
            raise ValueError(f"time_reduce must be 'mean' or 'max_hs', got {time_reduce!r}.")

        agg, agg_desc = self._reduce_time(time_reduce)

        has_2d = self.var_map.get("S") is not None
        has_e  = self.var_map.get("E") is not None
        has_d  = self.var_map.get("D") is not None
        if not (has_2d or has_e or has_d):
            raise ValueError("plot_overview() found no spectral variable (S, E, or D) to plot.")

        hs  = float(agg["hs"].values) if "hs" in agg else None
        pdir = float(agg["pdir"].values) if "pdir" in agg else None

        fig = plt.figure(figsize=(7.5, 6))
        cmap = mcm.ocean_r
        accent = "#0463d7"
        alpha = 0.22

        if has_2d:
            ax_main = fig.add_axes([0.25, 0.25, 0.6, 0.6], projection="polar")
            self._plot_2d_panel(ax_main, agg, plot_type, radius, dir_letters, cmap, max_spec_value)
            ax_inset = fig.add_axes([0.06, 0.15, 0.3, 0.3])
            self._plot_1d_freq_panel(ax_inset, agg, has_e, has_d, radius, accent, alpha)
        else:
            ax_main = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            self._plot_1d_freq_panel(ax_main, agg, has_e, has_d, radius, accent, alpha)

        if hs is not None:
            self._plot_hs_bar(fig, hs, accent, alpha)

        if wdir_var is not None and wdir_var in self.ds.data_vars:
            self._plot_direction_arrows(
                fig, agg, pdir,
                wspd_var, wdir_var, dpt_var, wdir_from,
                partition_method, partition_kwargs or {},
                accent, alpha,
            )

        self._plot_summary_text(fig, hs, agg_desc)
        fig.suptitle(title or "Spectrum overview", fontsize=16, x=0.02, ha="left", y=0.99)

        if output_file is not None:
            fig.savefig(output_file, dpi=200, bbox_inches=None, pad_inches=0)

        return fig

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _reduce_time(self, time_reduce: str) -> tuple[xr.Dataset, str]:
        """
        Collapse the whole time axis to one representative spectrum,
        returning (reduced dataset, human-readable description). Ensures
        hs/pdir are present on the result by running integrate() first if
        needed -- on `self` (cheap, cached in self.ds for later reuse),
        not on the already-reduced result, since integrate()'s moment
        calculations are only meaningful evaluated per-timestep.
        """
        time_dim = self.dim_map[_ROLE_TIME]

        if "hs" not in self.ds.data_vars:
            self.integrate(compute=["hs", "fp", "pdir"] if (self.has_frequency or self.has_directional) else None)

        if time_reduce == "mean":
            return self.ds.mean(dim=time_dim, keep_attrs=True), "mean over full record"

        # "max_hs": actual timestep with highest hs -- a real spectrum
        idx = int(self.ds["hs"].argmax(dim=time_dim).values)
        picked = self.ds.isel({time_dim: idx})
        when = str(picked[time_dim].values)[:19]
        return picked, f"timestep with max hs ({when})"

    def _direction_to_math_radians(self, deg) -> np.ndarray:
        """
        Compass degrees (0 = N, clockwise -- the standardized on-disk
        convention per __init__) -> standard math radians (0 = East,
        counter-clockwise) for arrow-drawing arithmetic. This conversion
        itself doesn't depend on dir_convention ("from"/"to" is a
        semantic label for what the angle *means*, not a different
        numeral system) -- dir_convention is used only for the arrow's
        text label, in _plot_direction_arrows.
        """
        deg = np.asarray(deg, dtype=float)
        return np.deg2rad((450.0 - deg) % 360.0)

    def _plot_2d_panel(self, ax, agg, plot_type, radius, dir_letters, cmap, vmax):
        freq_dim = self.dim_map[_ROLE_FREQ]
        dir_dim  = self.dim_map[_ROLE_DIR]
        s = agg[self.var_map["S"]]

        dirs = agg[dir_dim].values
        freqs = agg[freq_dim].values
        rad = 1.0 / freqs if radius == "period" else freqs
        theta = self._direction_to_math_radians(dirs)

        order = np.argsort(theta)
        theta = theta[order]
        s_vals = s.transpose(freq_dim, dir_dim).values[:, order]

        vmax = vmax if vmax is not None else float(np.nanmax(s_vals))
        if plot_type == "pcolormesh":
            ax.pcolormesh(theta, rad, s_vals, cmap=cmap, vmax=vmax, shading="auto")
        else:
            step = max(np.round(vmax / 10, 1), 0.05)
            levels = np.round(np.arange(0, vmax + step, step), 2)
            ax.contourf(theta, rad, s_vals, levels=levels, cmap=cmap)

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(315)
        ax.grid(True)
        if dir_letters:
            ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
            ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

    def _plot_1d_freq_panel(self, ax, agg, has_e, has_d, radius, color, alpha):
        freq_dim = self.dim_map[_ROLE_FREQ]
        dir_dim  = self.dim_map[_ROLE_DIR]
        has_2d = self.var_map.get("S") is not None

        if has_2d:
            e_f = agg[self.var_map["S"]].sum(dim=dir_dim)
        elif has_e:
            e_f = agg[self.var_map["E"]]
        else:
            e_f = None

        if e_f is not None:
            freqs = agg[freq_dim].values
            x = 1.0 / freqs if radius == "period" else freqs
            ax.plot(x, e_f.values, color=color)
            ax.fill_between(x, e_f.values, color=color, alpha=alpha)
            ax.set_xlabel("Period [s]" if radius == "period" else "Frequency [Hz]")
            ax.set_ylabel("Energy density")
        elif has_d:
            d = agg[self.var_map["D"]]
            dirs = agg[dir_dim].values
            ax.plot(dirs, d.values, color=color)
            ax.fill_between(dirs, d.values, color=color, alpha=alpha)
            ax.set_xlabel("Direction [deg]")
            ax.set_ylabel("Directional density")
        else:
            ax.text(0.5, 0.5, "No 1D spectrum available", ha="center", va="center")
            ax.set_axis_off()

    def _plot_hs_bar(self, fig, hs, color, alpha):
        cbar_ax = fig.add_axes([0.87, 0.3, 0.03, 0.5])
        cbar_ax.bar(0, hs, color=color, alpha=min(alpha + 0.5, 1))
        cbar_ax.set_ylim(0, 25 if hs > 20 else 20)
        cbar_ax.set_xlim(-0.5, 0.5)
        cbar_ax.set_xticks([])
        cbar_ax.yaxis.tick_right()
        cbar_ax.tick_params(axis="y", labelsize=12)
        cbar_ax.text(0, -1, r"H$_{m_0}$ [m]", ha="center", va="top", fontsize=12)

    def _plot_direction_arrows(
        self, fig, agg, pdir,
        wspd_var, wdir_var, dpt_var, wdir_from,
        partition_method, partition_kwargs,
        color, alpha,
    ):
        ax = fig.add_axes([0.02, 0.55, 0.2, 0.3], frameon=False)
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zorder(20)
        x0, y0, length = 0.5, 0.4, 0.35

        def draw(direction_deg, arrow_color, style="line"):
            rad = self._direction_to_math_radians(direction_deg)
            dx, dy = np.cos(rad) * length, np.sin(rad) * length
            if style == "line":
                ax.plot([x0, x0 + dx], [y0, y0 + dy], color=arrow_color, linewidth=2)
            else:
                ax.annotate(
                    "", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                    arrowprops=dict(facecolor=arrow_color, edgecolor="none", width=2, headwidth=8),
                )

        wind_dir = float(agg[wdir_var].values)
        # wdir_from=True: variable holds "coming from" -- the arrow should
        # point where the wind is going, i.e. wind_dir + 180.
        arrow_dir = (wind_dir + 180.0) % 360.0 if wdir_from else wind_dir
        draw(arrow_dir, color, style="line")
        ax.text(0.02, 0.02, f"wind ({'from' if wdir_from else 'to'} {wind_dir:.0f}\u00b0)",
                fontsize=7, color=color, transform=ax.transAxes)

        if partition_method is not None:
            method_fn = getattr(self, f"partition_{partition_method}", None)
            if method_fn is None:
                ax.text(0.02, 0.10, f"(unknown partition_method '{partition_method}')",
                        fontsize=7, color="gray", transform=ax.transAxes)
                return
            try:
                kwargs = dict(partition_kwargs)
                if partition_method in ("ptm1", "ptm2"):
                    kwargs.setdefault("wspd_var", wspd_var)
                    kwargs.setdefault("wdir_var", wdir_var)
                    kwargs.setdefault("dpt_var", dpt_var)
                    kwargs.setdefault("wdir_from", wdir_from)
                collection = method_fn(**kwargs)

                palette = {"wind_sea": "#E69F00", "wind_sea_primary": "#E69F00",
                           "wind_sea_secondary": "#E69F00"}
                for label in collection.labels():
                    member = collection[label]
                    member.integrate(compute=["pdir"])
                    if "pdir" not in member.ds:
                        continue
                    member_pdir = float(member.ds["pdir"].mean().values)
                    arrow_color = palette.get(label, "green")
                    draw(member_pdir, arrow_color, style="arrow")
            except Exception as exc:
                # Partitioning is a nice-to-have on this plot, not the
                # point of it -- don't let a partition failure take down
                # the whole overview plot.
                ax.text(0.02, 0.10, f"(partition arrows unavailable: {exc})",
                        fontsize=7, color="gray", transform=ax.transAxes)
        elif pdir is not None:
            draw(pdir, "black", style="arrow")

        rect = mpatches.Rectangle(
            (0, 0), 1.1, 1.1, transform=ax.transAxes,
            facecolor=color, alpha=alpha, edgecolor="none",
        )
        ax.add_patch(rect)
        rect.set_clip_on(False)

    def _plot_summary_text(self, fig, hs, agg_desc):
        label = rf"$\mathbf{{Aggregation:}}$ {agg_desc}"
        if hs is not None:
            label += "\n" + rf"$\mathbf{{H_{{m_0}}:}}$ {hs:.1f} m"
        fig.text(0.02, 0.02, label, fontsize=11, transform=fig.transFigure)



# ---------------------------------------------------------------------- #
# Public: groupby() / aggregate()                                        #
# ---------------------------------------------------------------------- #

class _GroupedSpectralTimeSeries:
    """
    Intermediate object representing a chain of one or more `.groupby()`
    calls on a `SpectralTimeSeries`, not yet aggregated. Call `.aggregate()`
    to collapse the chain into an N-D `xr.Dataset`, with one output
    dimension per level in the chain (e.g. `season x hs x tp`).

    Each level's groups are resolved once, globally, against the *full*
    time axis -- exactly as the single-level `SpectralTimeSeries.aggregate()`
    resolves them -- not independently re-inferred within each parent leaf.
    This keeps bin edges/labels for a given `by` consistent across every
    branch of the chain, which is what lets the leaves be unstacked into a
    clean rectangular grid rather than a ragged one where, say, the "hs"
    bins in winter don't line up with the "hs" bins in summer.
    """

    def __init__(
        self,
        parent: "SpectralTimeSeries",
        levels: list[tuple[str, dict[str, np.ndarray]]],
    ):
        self._parent = parent
        self._levels = levels  # [(by_name, {label: full-length bool mask}), ...]

    def groupby(
        self,
        by: str,
        sectors: int | list[float] | None = None,
        seasons: dict | None = None,
        step: float | None = None,
        circular_by: bool = False,
    ) -> "_GroupedSpectralTimeSeries":
        """
        Add another grouping level to the chain. Resolves ``by`` with its
        own independent ``sectors``/``seasons``/``step``/``circular_by`` --
        no relation to whatever grouping variables came before it.
        """
        existing = [name for name, _ in self._levels]
        if by in existing:
            raise ValueError(
                f"'{by}' is already a level in this groupby chain "
                f"({existing}); grouping by the same variable twice isn't "
                "supported."
            )
        new_groups = self._parent._get_time_groups(
            by, sectors=sectors, seasons=seasons, step=step, circular_by=circular_by,
        )
        return _GroupedSpectralTimeSeries(self._parent, self._levels + [(by, new_groups)])

    def aggregate(self, stat: str = "mean", select_by: str | None = None) -> xr.Dataset:
        """
        Terminate the groupby chain, collapsing time into one output
        dimension per level. Same two modes (element-wise vs. selection)
        as ``SpectralTimeSeries.aggregate()`` -- see its docstring for the
        full explanation of ``stat``/``select_by`` semantics.

        If any bulk parameters (hs, tp, fp, ...) were already present in
        the parent's dataset (i.e. ``.integrate()`` had been run before
        ``.groupby()``), they are re-computed from the aggregated spectra
        rather than aggregated directly -- aggregation does not commute
        with the nonlinear operations integrate() performs (e.g.
        mean(hs) != 4*sqrt(mean(m0)) in general), so re-deriving from the
        highest-order spectrum available (S if present, else E/D) is the
        only way to keep the output internally consistent. Only the
        parameter names that were present before are recomputed -- this
        never adds parameters nobody had asked for via
        ``integrate(compute=[...])``.

        Leaf combinations with no time steps in them (e.g. no JJA
        observations with hs in the 8-10 bin) are simply absent from the
        result, which surfaces as ``NaN``/missing at that grid cell once
        unstacked -- no explicit sparsity handling needed.
        """
        parent = self._parent
        level_names = [name for name, _ in self._levels]
        level_masks = [masks for _, masks in self._levels]
        time_dim = parent.dim_map[_ROLE_TIME]

        # --- cartesian product of every level's labels, ANDing masks,
        #     dropping any combination with zero time steps in it ---
        combo_labels: list[tuple] = []
        combo_masks: dict[tuple, np.ndarray] = {}
        for label_tuple in itertools.product(*(d.keys() for d in level_masks)):
            mask = np.ones(len(parent.time), dtype=bool)
            for lvl_masks, label in zip(level_masks, label_tuple):
                mask = mask & lvl_masks[label]
            if mask.any():
                combo_labels.append(label_tuple)
                combo_masks[label_tuple] = mask

        if not combo_labels:
            raise ValueError(
                "No non-empty combinations found across the groupby chain "
                f"({level_names}) -- every leaf cell would be empty."
            )

        combo_dim = "_combo"
        source_vars = parent._spectral_source_vars()

        if select_by is not None:
            sel_col = parent._resolve_scalar_var(select_by, "select_by")
            scalar_vals = parent.ds[sel_col].values.astype(float)
            out_arrays, history = parent._aggregate_by_selection(
                source_vars, combo_masks, combo_labels, combo_dim,
                scalar_vals, sel_col, stat, time_dim,
            )
        else:
            out_arrays, history = parent._aggregate_elementwise(
                source_vars, combo_masks, combo_labels, combo_dim, stat, time_dim,
            )

        # --- re-integrate, if the parent had any bulk parameters computed ---
        present_integrated = [k for k in _INTEGRATED_VARS if k in parent.ds.data_vars]
        if present_integrated:
            integrated_arrays = self._reintegrate(
                out_arrays, present_integrated, combo_dim, parent,
            )
            out_arrays = {**out_arrays, **integrated_arrays}

        combo_counts = np.array([int(combo_masks[c].sum()) for c in combo_labels])
        multi_index = pd.MultiIndex.from_tuples(combo_labels, names=level_names)
        mindex_coords = xr.Coordinates.from_pandas_multiindex(multi_index, combo_dim)

        out_ds = xr.Dataset(out_arrays)
        out_ds = out_ds.drop_vars(combo_dim)
        out_ds = out_ds.assign_coords(mindex_coords)

        out_ds["_group_counts"] = (combo_dim, combo_counts)
        out_ds["_group_counts"].attrs.update({
            "name": "Group counts",
            "long_name": "number_of_time_steps_aggregated_per_group",
        })

        out_ds = out_ds.unstack(combo_dim)  # "_combo" -> season, hs, tp, ...

        out_ds.attrs["history"] = (
            f"{history}, chained over levels {level_names} and unstacked "
            "into an N-D grid" + (
                f"; re-integrated {sorted(present_integrated)} from the "
                "aggregated spectra" if present_integrated else ""
            )
        )
        out_ds.attrs["dir_convention"] = parent.dir_convention
        return out_ds

    @staticmethod
    def _reintegrate(
        out_arrays: dict[str, xr.DataArray],
        present_integrated: list[str],
        combo_dim: str,
        parent: "SpectralTimeSeries",
    ) -> dict[str, xr.DataArray]:
        """
        Re-derive whichever integrated parameters were present in the
        parent before aggregation, from the just-aggregated spectra in
        ``out_arrays`` (still flat along ``combo_dim`` -- called before
        the final unstack, so the integration helpers only ever see one
        batch dimension, exactly like the ordinary time-indexed case).

        Uses S if present in the aggregated output, else E and/or D,
        exactly mirroring _spectral_source_vars()'s own S-supremacy rule
        -- the aggregated spectra are the only source of truth here, the
        parent's own pre-aggregation E/D/S is not touched or reused, since
        it has the wrong (unaggregated) shape.
        """
        _FREQ_PARAMS = {"hs", "tm01", "tm02", "tm_10", "fp", "tp", "fpI", "tpI"}
        _DIR_PARAMS  = {"pdir", "pdirI", "mdir", "spr"}
        _2D_PARAMS   = {"pdir2", "fp2"}

        wanted = set(present_integrated)
        freq_dim = parent.dim_map[_ROLE_FREQ]
        dir_dim  = parent.dim_map[_ROLE_DIR]

        S_name = parent.var_map.get("S")
        E_name = parent.var_map.get("E")
        D_name = parent.var_map.get("D")

        results: dict[str, xr.DataArray] = {}

        S_agg = out_arrays.get(S_name) if S_name else None
        E_agg = out_arrays.get(E_name) if E_name else None
        D_agg = out_arrays.get(D_name) if D_name else None

        # --- frequency-derived: use E if aggregated directly, else
        #     derive it from the aggregated S (never the parent's
        #     pre-aggregation E/S, which has the wrong shape) ---
        if wanted & _FREQ_PARAMS:
            if E_agg is not None:
                E_for_integration, E_source = E_agg, E_name
            elif S_agg is not None:
                E_for_integration = SpectralTimeSeries._dir_integrate_S(S_agg, dir_dim)
                E_source = S_name
            else:
                E_for_integration = None

            if E_for_integration is not None:
                all_freq = SpectralTimeSeries._bulk_from_E(
                    E_for_integration, freq_dim, E_source,
                )
                for k, (da, history) in all_freq.items():
                    if k in wanted:
                        results[k] = (da, history + " (re-integrated post-aggregation)")

        # --- direction-derived: same pattern, D or S-derived-D ---
        if wanted & _DIR_PARAMS:
            if D_agg is not None:
                D_for_integration, D_source = D_agg, D_name
            elif S_agg is not None:
                D_for_integration = SpectralTimeSeries._freq_integrate_S(S_agg, freq_dim)
                D_source = S_name
            else:
                D_for_integration = None

            if D_for_integration is not None:
                all_dir = SpectralTimeSeries._bulk_from_D(
                    D_for_integration, dir_dim, parent.dir_convention, D_source,
                )
                for k, (da, history) in all_dir.items():
                    if k in wanted:
                        results[k] = (da, history + " (re-integrated post-aggregation)")

        # --- 2D-only: requires the aggregated S itself ---
        if wanted & _2D_PARAMS and S_agg is not None:
            fp2_da, pdir2_da, history = SpectralTimeSeries._peak2_from_S(
                S_agg, freq_dim, dir_dim, parent.dir_convention, S_name,
            )
            history += " (re-integrated post-aggregation)"
            if "fp2" in wanted:
                results["fp2"] = (fp2_da, history)
            if "pdir2" in wanted:
                results["pdir2"] = (pdir2_da, history)

        out: dict[str, xr.DataArray] = {}
        for out_name, (da, history) in results.items():
            meta = _INTEGRATED_VARS[out_name]
            _apply_attrs(da, meta, extra={
                "history": history,
                **({"dir_convention": parent.dir_convention}
                   if out_name in ("pdir", "pdirI", "mdir", "spr", "pdir2") else {}),
            })
            out[out_name] = da

        return out

# ---------------------------------------------------------------------------
# SpectralTimeSeriesCollection
# ---------------------------------------------------------------------------

class SpectralTimeSeriesCollection:
    """
    A named/labelled collection of related SpectralTimeSeries instances,
    for comparison, validation, or plotting between them.

    Typical sources: several wave model outputs for the same point and
    period; or the partitions (wind sea / swell components) derived from
    one spectrum via partition_watershed(). Members need not share
    variable names or dim names -- each member's own dim_map is consulted
    whenever "the freq dimension" or "the dir dimension" of that specific
    member is needed, exactly as SpectralTimeSeries itself never assumes a
    fixed name for these.

    This class holds no computational logic of its own beyond alignment
    bookkeeping -- it is a container plus a compatibility check. Methods
    that need directly-comparable arrays (e.g. a future RMSE/diff method)
    should guard themselves with `_requires_aligned()`; methods that only
    need per-member scalars (e.g. comparing integrated bulk parameters)
    do not, since those can be compared across members regardless of
    whether the underlying grids match.

    Parameters
    ----------
    members : dict[str, SpectralTimeSeries] or list[SpectralTimeSeries]
        The collection's members. If a dict, its keys are used directly as
        labels. If a list, each member's own `.name` is used as its label
        if set and unique among the list; otherwise a default label
        ("member_0", "member_1", ...) is assigned.

    Attributes
    ----------
    members : dict[str, SpectralTimeSeries]
        Label -> instance.
    aligned : bool (property)
        Whether every member's freq/dir coordinates (for whichever roles
        they have) and dir_convention match exactly across the whole
        collection. Does not consider time -- see class docstring and
        `align()`. Cheap: compares coordinate arrays only, no
        interpolation or copying.

    Notes
    -----
    There is no "standardized" state here distinct from `aligned`: every
    individual SpectralTimeSeries member is already always standardized
    internally by its own constructor (see that class's docstring). What
    this class adds is purely cross-member *compatibility* -- do the
    members' own grids agree with each other -- which is a different,
    genuinely collection-level property, hence `aligned` rather than
    reusing the word "standardized".
    """

    def __init__(
        self,
        members: dict[str, "SpectralTimeSeries"] | list["SpectralTimeSeries"],
    ):
        if isinstance(members, dict):
            if not members:
                raise ValueError("members must be a non-empty dict or list.")
            for label, m in members.items():
                if not isinstance(m, SpectralTimeSeries):
                    raise TypeError(
                        f"members[{label!r}] must be a SpectralTimeSeries, "
                        f"got {type(m).__name__}."
                    )
            self.members: dict[str, SpectralTimeSeries] = dict(members)

        elif isinstance(members, list):
            if not members:
                raise ValueError("members must be a non-empty dict or list.")
            for i, m in enumerate(members):
                if not isinstance(m, SpectralTimeSeries):
                    raise TypeError(
                        f"members[{i}] must be a SpectralTimeSeries, "
                        f"got {type(m).__name__}."
                    )
            names = [m.name for m in members]
            unique_named = (
                all(n is not None for n in names) and len(set(names)) == len(names)
            )
            if unique_named:
                self.members = {m.name: m for m in members}
            else:
                self.members = {f"member_{i}": m for i, m in enumerate(members)}

        else:
            raise TypeError(
                f"members must be a dict[str, SpectralTimeSeries] or "
                f"list[SpectralTimeSeries], got {type(members).__name__}."
            )

    # ---------------------------------------------------------------------- #
    # Container dunders                                                      #
    # ---------------------------------------------------------------------- #

    def __getitem__(self, label: str) -> "SpectralTimeSeries":
        return self.members[label]

    def __iter__(self):
        return iter(self.members.values())

    def __len__(self) -> int:
        return len(self.members)

    def __contains__(self, label: str) -> bool:
        return label in self.members

    def labels(self) -> list[str]:
        return list(self.members.keys())

    def items(self):
        return self.members.items()

    def __repr__(self) -> str:
        lines = [f"SpectralTimeSeriesCollection ({len(self.members)} members)"]
        lines.append(f"  aligned = {self.aligned}")
        for label, m in self.members.items():
            lines.append(f"  '{label}': {m._input_summary()}, {len(m)} steps")
        return "\n".join(lines)

    # ---------------------------------------------------------------------- #
    # Alignment check (cheap, read-only, no interpolation)                   #
    # ---------------------------------------------------------------------- #

    @property
    def aligned(self) -> bool:
        """
        Whether every member's freq/dir coordinates and dir_convention
        match exactly across the collection, for whichever roles are
        present. Time is deliberately not considered here -- see class
        docstring and `align()`'s `time` parameter.

        Roles absent from *every* member are ignored (e.g. an all-1D-E
        collection is judged only on freq). A role present on *some* but
        not *all* members makes the collection unaligned on that basis
        alone -- comparability requires the same roles everywhere.

        Returns
        -------
        bool
        """
        members = list(self.members.values())
        if len(members) <= 1:
            return True

        first = members[0]
        if any(m.dir_convention != first.dir_convention for m in members[1:]):
            return False

        for role in (_ROLE_FREQ, _ROLE_DIR):
            has_role = [m.dim_map[role] is not None for m in members]
            if not any(has_role):
                continue
            if not all(has_role):
                return False

            getter = (lambda m: m.freqs) if role == _ROLE_FREQ else (lambda m: m.dirs)
            ref = getter(first)
            for m in members[1:]:
                vals = getter(m)
                if vals.shape != ref.shape or not np.allclose(vals, ref):
                    return False

        return True

    def _requires_aligned(self, method_name: str) -> None:
        if not self.aligned:
            raise AttributeError(
                f"{method_name}() requires an aligned collection (matching "
                "freq/dir coordinates and dir_convention across all "
                f"members), but this collection is not aligned. Call "
                ".align(target=<label>) first to produce an aligned copy, "
                "then call this method on the result."
            )

    # ---------------------------------------------------------------------- #
    # Public: align()                                                        #
    # ---------------------------------------------------------------------- #

    def align(
        self,
        target: str,
        method: Literal["interp", "nearest"] = "interp",
        time: Literal["inner", "outer", "interp", "none"] = "inner",
    ) -> "SpectralTimeSeriesCollection":
        """
        Return a new SpectralTimeSeriesCollection where every member has
        been interpolated (or subsampled) onto `target`'s freq/dir grid,
        making the result's `.aligned` True. The original collection and
        its members are never mutated.

        Parameters
        ----------
        target : str
            Label of the member whose freq/dir coordinates every other
            member is aligned onto. Required -- there is no default,
            since which member is the reference is a modelling choice,
            not something to infer silently (e.g. "first in the dict").
        method : {"interp", "nearest"}, default "interp"
            How to resample freq/dir onto the target's grid.
            "interp" - linear interpolation (xr.DataArray.interp).
            "nearest" - subsample to the nearest existing bin instead of
                        interpolating -- avoids inventing values between
                        bins, at the cost of not landing exactly on the
                        target grid's spacing if the source grid is coarser.
        time : {"inner", "outer", "interp", "none"}, default "inner"
            How to reconcile the time axis across members.
            "inner"  - keep only timestamps common to all members
                        (intersection). Never invents data.
            "outer"  - keep every timestamp appearing in any member
                        (union); members are NaN where they had no data at
                        a given step.
            "interp" - reindex every member onto the target's own time
                        axis, linearly interpolating the others onto it.
            "none"   - leave each member's time axis untouched entirely.
                        `.aligned` only ever reflects freq/dir/dir_convention
                        (see its docstring), so this is a legitimate choice
                        for e.g. `align(time="none")` before a method that
                        only needs matching freq/dir (like plotting spectra
                        side by side) and handles time overlap itself, or
                        not at all.

        Returns
        -------
        SpectralTimeSeriesCollection
            A new collection, same labels, each member rebuilt on the
            common grid. `.aligned` is True on the result (assuming at
            least one freq/dir role was actually present to align; an
            all-scalar edge case trivially returns aligned=True already).
        """
        if target not in self.members:
            raise ValueError(
                f"target={target!r} is not a member of this collection. "
                f"Available labels: {self.labels()}."
            )
        if method not in ("interp", "nearest"):
            raise ValueError(f"method must be 'interp' or 'nearest', got {method!r}.")
        if time not in ("inner", "outer", "interp", "none"):
            raise ValueError(
                f"time must be one of 'inner', 'outer', 'interp', 'none', got {time!r}."
            )

        target_member = self.members[target]
        target_freq = target_member.freqs
        target_dir  = target_member.dirs

        # --- resolve the common time index, if requested ---
        common_time = None
        if time == "inner":
            common_time = target_member.time
            for label, m in self.members.items():
                if label == target:
                    continue
                common_time = common_time.intersection(m.time)
        elif time == "outer":
            common_time = target_member.time
            for label, m in self.members.items():
                if label == target:
                    continue
                common_time = common_time.union(m.time)
        elif time == "interp":
            common_time = target_member.time
        # "none": leave each member's time axis as-is, common_time stays None

        new_members: dict[str, SpectralTimeSeries] = {}
        for label, m in self.members.items():
            new_ds = self._align_member_ds(
                m, target_freq, target_dir, method, time, common_time,
            )
            new_members[label] = SpectralTimeSeries.__new__(
                SpectralTimeSeries
            )._init_from_ds(
                new_ds, m.var_map, m.dim_map, m.dir_convention, m.depth, m.name,
            )

        return SpectralTimeSeriesCollection(new_members)

    @staticmethod
    def _align_member_ds(
        m: "SpectralTimeSeries",
        target_freq: np.ndarray | None,
        target_dir: np.ndarray | None,
        method: str,
        time_policy: str,
        common_time: pd.DatetimeIndex | None,
    ) -> xr.Dataset:
        """
        Build one member's realigned dataset: interpolate/subsample its
        freq/dir coordinates (whichever it has) onto the target's, then
        reconcile its time axis per `time_policy`. Uses this member's own
        dim_map throughout, so it works regardless of what this member's
        dims are actually called.
        """
        ds = m.ds
        freq_dim = m.dim_map[_ROLE_FREQ]
        dir_dim  = m.dim_map[_ROLE_DIR]
        time_dim = m.dim_map[_ROLE_TIME]

        interp_kwargs = {}
        if freq_dim is not None and target_freq is not None:
            interp_kwargs[freq_dim] = target_freq
        if dir_dim is not None and target_dir is not None:
            interp_kwargs[dir_dim] = target_dir

        if interp_kwargs:
            if method == "interp":
                ds = ds.interp(**interp_kwargs, kwargs={"fill_value": "extrapolate"})
            else:  # "nearest"
                ds = ds.sel(**interp_kwargs, method="nearest")

        if time_policy in ("inner", "outer") and common_time is not None:
            ds = ds.reindex({time_dim: common_time})
        elif time_policy == "interp" and common_time is not None:
            ds = ds.interp({time_dim: common_time}, kwargs={"fill_value": "extrapolate"})
        # "none": leave ds's time axis untouched

        return ds
    
    def track(
        self,
        wspd_var: str | None = None,
        ddpm_sea_max: float = 30.0,
        ddpm_swell_max: float = 20.0,
        dfp_sea_scaling: float = 1.0,
        dfp_swell_source_distance: float = 1e6,
        min_duration: int = 1,
    ) -> "SpectralTimeSeriesCollection":
        """
        Track wave systems across time from an already-partitioned collection.

        Matches partitions between consecutive timesteps based on the evolution
        of peak frequency and mean direction at peak frequency, following
        Ewans & Kibblewhite (1986) for wind sea and Snodgrass et al. (1966)
        for swell. Each continuously tracked wave system is returned as a
        separate SpectralTimeSeries member, covering only the timesteps where
        that system is active.

        Parameters
        ----------
        wspd_var : str or None
            Name of the wind speed variable used for wind-sea tracking
            thresholds. Searched across all members; the first member
            containing it is used. Required if any member label starts with
            "wind_sea". If None and no wind-sea partitions are present
            (e.g. PTM3), wind thresholds are not applied.
        ddpm_sea_max : float
            Maximum direction difference (degrees) for matching wind-sea
            partitions. Default 30°.
        ddpm_swell_max : float
            Maximum direction difference (degrees) for matching swell
            partitions. Default 20°.
        dfp_sea_scaling : float
            Scaling factor for the wind-sea peak frequency rate of change.
            Default 1.0.
        dfp_swell_source_distance : float
            Assumed swell source distance (m) for swell fp rate of change.
            Default 1e6 m.
        min_duration : int
            Minimum number of timesteps a system must be active to be
            included in the output. Default 1 (keep all).

        Returns
        -------
        SpectralTimeSeriesCollection
            One member per tracked wave system, keyed ``"system_0"``,
            ``"system_1"``, …, ordered by first appearance. Each member
            covers only the timesteps where that system is active.
        """
        from wavespectra.partition.tracking import np_track_partitions

        members     = self.members
        labels      = list(members.keys())
        first       = next(iter(members.values()))
        time_dim    = first.dim_map[_ROLE_TIME]
        freq_dim    = first.dim_map[_ROLE_FREQ]
        dir_dim     = first.dim_map[_ROLE_DIR]
        times       = first.ds.coords[time_dim].values  # shared time axis
        n_times     = len(times)
        n_parts     = len(labels)

        # ------------------------------------------------------------------ #
        # Infer nsea from labels                                               #
        # ------------------------------------------------------------------ #
        nsea = sum(1 for lbl in labels if lbl.startswith("wind_sea"))

        # ------------------------------------------------------------------ #
        # Resolve wspd                                                         #
        # ------------------------------------------------------------------ #
        wspd_vals: np.ndarray | None = None
        if nsea > 0:
            if wspd_var is None:
                raise ValueError(
                    "wspd_var is required for tracking wind-sea partitions "
                    f"(found {nsea} wind-sea slot(s): "
                    f"{[l for l in labels if l.startswith('wind_sea')]}). "
                    "Pass wspd_var=<variable name> or use a PTM3/PTM5 collection."
                )
            for member in members.values():
                if wspd_var in member.ds:
                    wspd_vals = member.ds[wspd_var].values.astype(float)
                    break
            if wspd_vals is None:
                raise ValueError(
                    f"wspd_var={wspd_var!r} not found in any member's dataset. "
                    f"Available variables: {list(first.ds.data_vars)}"
                )

        # ------------------------------------------------------------------ #
        # Compute fp and dpm per partition                                     #
        # ------------------------------------------------------------------ #
        fp_arr  = np.full((n_parts, n_times), np.nan)
        dpm_arr = np.full((n_parts, n_times), np.nan)

        for pi, (label, member) in enumerate(members.items()):
            if member.var_map["S"] is None:
                continue

            # fp: compute and cache if not already present
            if "fp" not in member.ds:
                member.integrate(compute=["fp"])
            fp_arr[pi] = member.ds["fp"].values

            # dpm: always computed fresh (not stored)
            E = member.compute_frequency_spectrum()
            S = member.ds[member.var_map["S"]]
            dpm_da = SpectralTimeSeries._compute_dpm(S, E, freq_dim, dir_dim)
            dpm_arr[pi] = dpm_da.values

        # ------------------------------------------------------------------ #
        # Run tracker                                                          #
        # ------------------------------------------------------------------ #
        track_ids, n_tracks = np_track_partitions(
            times=times,
            fp=fp_arr,
            dpm=dpm_arr,
            wspd=wspd_vals,
            ddpm_sea_max=ddpm_sea_max,
            ddpm_swell_max=ddpm_swell_max,
            dfp_sea_scaling=dfp_sea_scaling,
            dfp_swell_source_distance=dfp_swell_source_distance,
            nsea=nsea,
        )
        # track_ids shape: (n_parts, n_times)
        # value = global track id (>= 0), -888 = unmatched energetic, -999 = null

        # ------------------------------------------------------------------ #
        # Build one SpectralTimeSeries per tracked system                      #
        # ------------------------------------------------------------------ #
        spec_var = first.var_map["S"]

        # For each system id, find which (partition, timestep) pairs belong to it
        system_members: dict[str, SpectralTimeSeries] = {}

        for sys_id in range(n_tracks):
            pi_arr, ti_arr = np.nonzero(track_ids == sys_id)

            if len(ti_arr) < min_duration:
                continue

            # Active timesteps for this system (sorted, unique -- tracker
            # guarantees at most one partition holds a given system at each t)
            active_ti = np.sort(ti_arr)
            active_times = times[active_ti]

            # Pull the spectral slice for each active timestep from whichever
            # partition holds the system at that time
            spec_slices = []
            for ti, pi in zip(ti_arr[np.argsort(ti_arr)], pi_arr[np.argsort(ti_arr)]):
                member  = members[labels[pi]]
                sl      = member.ds[spec_var].isel({time_dim: ti})
                spec_slices.append(sl)

            spec_da = xr.concat(spec_slices, dim=time_dim)
            spec_da.attrs = first.ds[spec_var].attrs.copy()

            # Take all non-spectral variables from the first member, sliced to
            # the active timesteps
            ds_base = first.ds.drop_vars(spec_var).isel({time_dim: active_ti})
            ds_sys  = ds_base.assign({spec_var: spec_da})

            system_label = f"system_{sys_id}"
            system_members[system_label] = SpectralTimeSeries.__new__(SpectralTimeSeries)._init_from_ds(
                ds=ds_sys,
                var_map=first.var_map.copy(),
                dim_map=first.dim_map.copy(),
                dir_convention=first.dir_convention,
                depth=first.depth,
                name=system_label,
            )

        return SpectralTimeSeriesCollection(system_members) 
    
    def plot_timeseries(
        self,
        var: str,
        t_start=None,
        t_end=None,
        ax=None,
        figsize=(12, 4),
        **plot_kwargs,
    ) -> "plt.Axes":
        """
        Plot a scalar time-series variable from each member on a shared axis.

        Parameters
        ----------
        var : str
            Variable name to plot (e.g. ``"hs"``, ``"fp"``). Must exist in
            at least one member's dataset.
        t_start, t_end : str, datetime-like, or None
            Optional time window. Any format accepted by xarray's ``.sel``
            (e.g. ``"2024-01-01"``, a ``datetime`` object, or a
            ``numpy.datetime64``). Members with no data in the window are
            silently skipped.
        ax : matplotlib.axes.Axes or None
            Axes to plot onto. A new figure is created if None.
        figsize : tuple
            Figure size when a new figure is created. Default (12, 4).
        **plot_kwargs
            Passed directly to ``ax.plot()`` for every member (e.g.
            ``linewidth=1.5``, ``alpha=0.8``).

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        ylabel = var  # fallback
        plotted = 0

        for label, member in self.members.items():
            if var not in member.ds:
                continue

            da = member.ds[var]

            # Infer time dim -- the one dim remaining after dropping freq/dir
            time_dim = member.dim_map[_ROLE_TIME]

            # Slice to requested window
            if t_start is not None or t_end is not None:
                slc = {time_dim: slice(t_start, t_end)}
                da = da.sel(slc)

            # Skip members with no data in the window
            if da.sizes[time_dim] == 0 or da.isnull().all():
                continue

            # Use "name" attr for y-label if present (take from first found)
            if plotted == 0:
                ylabel = da.attrs.get("name", var)
                units  = da.attrs.get("units", "")
                if units:
                    ylabel = f"{ylabel} ({units})"

            times = da.coords[time_dim].values
            vals  = da.values

            ax.plot(times, vals, label=label, **plot_kwargs)
            plotted += 1

        if plotted == 0:
            raise ValueError(
                f"Variable {var!r} not found or no data in the requested time "
                f"window in any member. Available variables: "
                f"{sorted({v for m in self.members.values() for v in m.ds.data_vars})}"
            )

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return ax

    def integrate(self, **kwargs) -> "SpectralTimeSeriesCollection":
        """
        Run .integrate() on every member.
        """
        for member in self.members.values():
            member.integrate(**kwargs)
        return self

    # ---------------------------------------------------------------------- #
    # Public: plot()                                                         #
    # ---------------------------------------------------------------------- #

    def plot(
        self,
        time,
        radius: str = "frequency",
        plot_type: str = "pcolormesh",
        cmap: str = "viridis",
        vmax: float | None = None,
        dir_letters: bool = False,
        panel_size: float = 3.0,
        ncols: int | None = None,
    ) -> "tuple[plt.Figure, np.ndarray]":
        """
        Plot the spectrum at `time` from every member, side by side, one
        panel per member (labelled by its key). `time` is required -- unlike
        SpectralTimeSeries.plot(), there is no single sensible default across
        a whole collection whose members may cover different periods.

        Members with no data at `time` are skipped, with a note printed --
        not silently dropped without explanation. Does not require
        `.aligned` -- comparing panels visually side by side is fine even on
        different grids; alignment only matters for numeric operations
        (see `_requires_aligned`'s docstring).

        Parameters
        ----------
        time : required
            Selector applied via `.sel()` against each member's own time
            dimension.
        ncols : int, optional
            Panels per row. Defaults to one row of all members.
        (other parameters as in plotting.plot_spectra())

        Returns
        -------
        (matplotlib.figure.Figure, np.ndarray of Axes)
        """
        from .plots import _draw_one_panel

        panels = []
        for label, member in self.members.items():
            time_dim = member.dim_map[_ROLE_TIME]
            spec_var = member.var_map["S"] or member.var_map["E"] or member.var_map["D"]
            if spec_var is None or time_dim not in member.ds[spec_var].dims:
                print(f"  skipped '{label}': no time-varying spectrum present")
                continue
            try:
                da = member.ds[spec_var].sel({time_dim: time})
            except (KeyError, ValueError):
                print(f"  skipped '{label}': no data at time={time!r}")
                continue
            panels.append((label, member, da))

        if not panels:
            raise ValueError(f"No member had data at time={time!r}.")

        ncols = ncols or len(panels)
        nrows = int(np.ceil(len(panels) / ncols))
        has_dir = any(m.dim_map[_ROLE_DIR] in da.dims for _, m, da in panels)
        fig_kw = dict(subplot_kw=dict(projection="polar")) if has_dir else {}
        fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows), **fig_kw)
        axes_flat = np.atleast_1d(axes).ravel()

        vmax_shared = vmax if vmax is not None else max(float(np.nanmax(da.values)) for _, _, da in panels)

        last_mesh = None
        for ax, (label, member, da) in zip(axes_flat, panels):
            freq_dim = member.dim_map[_ROLE_FREQ] if member.dim_map[_ROLE_FREQ] in da.dims else None
            dir_dim  = member.dim_map[_ROLE_DIR]  if member.dim_map[_ROLE_DIR]  in da.dims else None
            mesh = _draw_one_panel(ax, da, freq_dim, dir_dim, radius, plot_type, cmap, vmax_shared, dir_letters)
            last_mesh = mesh if mesh is not None else last_mesh
            ax.set_title(label, fontsize=9)

        for ax in axes_flat[len(panels):]:
            ax.set_visible(False)
        if last_mesh is not None:
            fig.colorbar(last_mesh, ax=axes_flat.tolist(), pad=0.02, shrink=0.6)
        fig.tight_layout()
        return fig, axes