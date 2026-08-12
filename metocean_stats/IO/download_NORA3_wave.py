"""
download_nora3.py
-----------------
Download NORA3 wave hindcast data for the nearest grid point to a target
location from the MET Norway THREDDS server.

Usage
-----
    df = download_nora3(
        lon=2.28,
        lat=58.97,
        start="2020-01-01",
        end="2021-01-01",
    )
"""

import numpy as np
import pandas as pd
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.metrics.pairwise import haversine_distances
from typing import Literal

# --------------------------------------------------------------------------- #
# Haversine helper                                                             #
# --------------------------------------------------------------------------- #

def _get_lonlat_distance(lon0, lat0, lon1, lat1):
    """
    Get distance from each point in (lon0, lat0) to each point in (lon1, lat1).
    Multiply by earth radius 6371 to get distance in km.
    2D arrays will be flattened.
    """
    if not hasattr(lon0, "__len__"): lon0 = [lon0]
    if not hasattr(lat0, "__len__"): lat0 = [lat0]
    if not hasattr(lon1, "__len__"): lon1 = [lon1]
    if not hasattr(lat1, "__len__"): lat1 = [lat1]

    lon0 = np.array(lon0)
    lat0 = np.array(lat0)
    lon1 = np.array(lon1)
    lat1 = np.array(lat1)

    target_coordinates = np.array([lat0.flatten(), lon0.flatten()]).swapaxes(0, 1)
    origin_coordinates = np.array([lat1.flatten(), lon1.flatten()]).swapaxes(0, 1)

    return haversine_distances(
        np.radians(target_coordinates),
        np.radians(origin_coordinates),
    )


# --------------------------------------------------------------------------- #
# Public function                                                              #
# --------------------------------------------------------------------------- #

_PATH_FMT = (
    "https://thredds.met.no/thredds/dodsC/nora3_subset_wave/wave/"
    "%Y%m_NORA3wave_sub.nc"
)

_EARTH_RADIUS_KM = 6371.0

_COMMON_VARS = ["ff", "dd", "hs", "tp", "thq", "fpI", "Pdir"]


def _fetch_month(
    path: str,
    position: dict,
    variables: list[str],
) -> "xr.Dataset | None":
    """
    Fetch a single monthly file. Module-level so it is picklable by
    ProcessPoolExecutor. Each worker process has its own HDF5 state,
    avoiding the thread-safety issues of the netCDF4/OPeNDAP backend.
    """
    try:
        return xr.open_dataset(path, cache=False).isel(position)[variables].load()
    except Exception as exc:
        print(f"  WARNING — skipped {path}: {exc}")
        return None


def download_nora3_wave(
    lon: float,
    lat: float,
    start: str = "1960",
    end: str = "2026",
    variables: list[str] | Literal["common","all"] = "common",
) -> pd.DataFrame:
    """
    Download NORA3 wave hindcast data for the nearest grid point to a
    target location.

    Monthly NetCDF files are fetched via OPeNDAP from the MET Norway
    THREDDS server. Only the single nearest grid point is extracted, so
    no spatial subsetting or radius trimming is needed.

    Parameters
    ----------
    lon : float
        Target longitude in decimal degrees.
    lat : float
        Target latitude in decimal degrees.
    start : str
        Start of the temporal window, as a datetime-compatible string.
        Defaults to ``"1960"``, the beginning of the NORA3 record.
    end : str
        End of the temporal window, same format as ``start``.
        Defaults to ``"2026"``, the end of the current NORA3 record.
    variables : list[str] or str, optional
        Variables to extract. Three forms accepted:

        - ``"common"`` *(default)* — the most commonly used wave/wind variables:
          ``["ff", "dd", "hs", "tp", "thq", "fpI", "Pdir"]``.
        - ``"all"`` — every time-varying variable found in the dataset.
        - ``list[str]`` — an explicit list of variable names.

        Static fields (``latitude``, ``longitude``, ``model_depth``) are
        always appended as scalar columns from the reference file,
        regardless of this parameter.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame with one row per NORA3 time step.
        Includes all requested time-varying variables plus the static
        fields ``latitude``, ``longitude``, and ``model_depth`` so the
        caller can verify the snap distance and depth.

    Raises
    ------
    RuntimeError
        If no monthly files could be loaded for the requested period.
    """

    # ------------------------------------------------------------------ #
    # Build list of monthly file paths within the requested window        #
    # ------------------------------------------------------------------ #
    t_start = pd.Timestamp(start)
    t_end   = pd.Timestamp(end)
    dates   = pd.date_range(t_start, t_end, freq="MS")
    paths   = sorted(d.strftime(_PATH_FMT) for d in dates)

    print(
        f"Downloading NORA3 wave data — "
        f"({lat:.4f}°N, {lon:.4f}°E), "
        f"{t_start.strftime('%Y-%m-%d')} → {t_end.strftime('%Y-%m-%d')}, "
        f"{len(paths)} monthly file(s) ..."
    )

    # ------------------------------------------------------------------ #
    # Find nearest grid point and resolve variables from the first file   #
    # ------------------------------------------------------------------ #
    ref        = xr.open_dataset(paths[0], cache=False)
    latitudes  = ref["latitude"].load()
    longitudes = ref["longitude"].load()

    distances = _get_lonlat_distance(lon, lat, longitudes, latitudes)
    ix, iy    = np.unravel_index(distances.argmin(), longitudes.shape)
    position  = {latitudes.dims[0]: ix, latitudes.dims[1]: iy}

    grid_lat = float(ref.isel(position).latitude.values)
    grid_lon = float(ref.isel(position).longitude.values)
    snap_km  = float(distances.min()) * _EARTH_RADIUS_KM

    print(
        f"  Nearest grid point: ({grid_lat:.4f}°N, {grid_lon:.4f}°E) "
        f"— {snap_km:.2f} km from target."
    )

    # Time-varying variables: resolve from the variables argument
    _static = {"latitude", "longitude", "model_depth"}

    if variables == "common":
        time_vars = _COMMON_VARS
    elif variables == "all":
        time_vars = [
            v for v in ref.data_vars
            if "time" in ref[v].dims and v not in _static
        ]
    elif isinstance(variables, list):
        time_vars = variables
    else:
        raise ValueError(
            f"'variables' must be a list of strings, \"common\", or \"all\", "
            f"got {variables!r}."
        )

    # Static fields: read once from the reference file
    ref_point   = ref.isel(position)
    static_vals = {
        col: float(ref_point[col].values)
        for col in _static
        if col in ref.data_vars
    }

    # ------------------------------------------------------------------ #
    # Load each monthly file (time-varying variables only)                #
    # ------------------------------------------------------------------ #
    chunks: list[xr.Dataset] = []

    args = [(p, position, time_vars) for p in paths]
    with ProcessPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_fetch_month, *a): a[0] for a in args}
        for future in tqdm(as_completed(futures), total=len(futures), smoothing=0):
            result = future.result()
            if result is not None:
                chunks.append(result)

    if not chunks:
        raise RuntimeError(
            "No monthly files could be loaded for the requested period. "
            "Check your start/end dates and network connection."
        )

    # ------------------------------------------------------------------ #
    # Concatenate, convert to DataFrame, append static fields             #
    # ------------------------------------------------------------------ #
    ds_full = xr.concat(chunks, dim="time", data_vars="minimal")
    df      = ds_full.to_dataframe()

    # Add static fields as scalar columns
    for col, val in static_vals.items():
        df[col] = val

    print(f"  Done. {len(df):,} time steps retained.")

    return df


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download NORA3 wave hindcast data for a target point."
    )
    parser.add_argument("--lon",   type=float, required=True, help="Target longitude")
    parser.add_argument("--lat",   type=float, required=True, help="Target latitude")
    parser.add_argument("--start", type=str,   default="1960", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",   type=str,   default="2026", help="End date (YYYY-MM-DD)")
    parser.add_argument("--out",   type=str,   default=None,   help="Output CSV path (optional)")
    args = parser.parse_args()

    df = download_nora3_wave(
        lon=args.lon,
        lat=args.lat,
        start=args.start,
        end=args.end,
    )

    print(df)

    if args.out:
        df.to_csv(args.out)
        print(f"Saved to {args.out}")