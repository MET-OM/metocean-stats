"""
download_satellite.py
---------------------
Download CCI sea-state altimeter data around a target point and trim to a
given radius.

Usage
-----
    df = download_satellite(
        lon=-3.6,
        lat=48.2,
        radius_km=50,
        start="2020-01-01",
        end="2021-01-01",
    )
"""

import io
import json
import pickle

import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import haversine_distances


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

_VARIABLES = [
    "bathymetry",
    "cycle",
    "distance_to_coast",
    "lat",
    "lon",
    "relative_pass",
    "satellite",
    "swh",
    "swh_adjusted",
    "swh_denoised",
    "swh_uncertainty",
    "time",
]

_URL = "https://cci-seastate.ifremer.fr/download_data"
_EARTH_RADIUS_KM = 6371.0


def download_satellite(
    lon: float,
    lat: float,
    radius_km: float = 25,
    start: str = "1990",
    end: str = "2025",
    timeout: int = 600,
) -> pd.DataFrame:
    """
    Download CCI sea-state altimeter data around a target point.

    A bounding box large enough to fully contain the circle of ``radius_km``
    is sent to the API. The returned data are then trimmed to the exact radius
    using the Haversine distance.

    Parameters
    ----------
    lon : float
        Target longitude in decimal degrees.
    lat : float
        Target latitude in decimal degrees.
    radius_km : float, optional
        Radius around the target point to keep, in kilometres. Default 25.
    start : str
        Start of the temporal window, as a datetime-compatible string.
        Default is 1990, which is the start of the dataset.
    end : str
        End of the temporal window, same format as ``start``. 
        Default is 2025, which is the end of the dataset.
    timeout : int, default 600
        Request timeout in seconds.

    Returns
    -------
    pd.DataFrame
        All satellite observations within ``radius_km`` of (lon, lat) for the
        requested period. An extra column ``"distance_km"`` is added with the
        Haversine distance from the target point.

    Raises
    ------
    requests.HTTPError
        If the API returns a non-200 status code.
    ValueError
        If the response cannot be parsed as a DataFrame.
    """
    # ------------------------------------------------------------------ #
    # Bounding box                                                         #
    # A degree of latitude is always ~111 km. A degree of longitude       #
    # shrinks with cos(lat), so we inflate accordingly to be safe.        #
    # ------------------------------------------------------------------ #
    delta_lat = 1.1*radius_km / _EARTH_RADIUS_KM * (180.0 / np.pi)
    delta_lon = delta_lat / max(np.cos(np.radians(lat)), 1e-6)

    area = {
        "west":  lon - delta_lon,
        "east":  lon + delta_lon,
        "south": lat - delta_lat,
        "north": lat + delta_lat,
    }

    # ------------------------------------------------------------------ #
    # Temporal extent                                                      #
    # ------------------------------------------------------------------ #
    def _fmt(t: str) -> str:
        ts = pd.Timestamp(t)
        return ts.strftime("%Y-%m-%d %H:%M")

    temporal_extent = [_fmt(start), _fmt(end)]

    # ------------------------------------------------------------------ #
    # Request                                                              #
    # ------------------------------------------------------------------ #
    payload = {
        "area":             json.dumps(area),
        "temporal_extent":  json.dumps(temporal_extent),
        "variable_extent":  json.dumps(_VARIABLES),
        "program":          "l3_altimeter",
        "format":           "pandas",
    }

    print(
        f"Requesting satellite data — "
        f"box [{area['west']:.3f}°E, {area['south']:.3f}°N] to "
        f"[{area['east']:.3f}°E, {area['north']:.3f}°N], "
        f"{temporal_extent[0]} → {temporal_extent[1]} ..."
    )

    response = requests.post(_URL, data=payload, timeout=timeout)
    response.raise_for_status()

    try:
        df = pickle.load(io.BytesIO(response.content))
    except Exception as exc:
        raise ValueError(
            f"Could not parse API response as a DataFrame: {exc}"
        ) from exc

    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            f"Expected a DataFrame from the API, got {type(df).__name__}."
        )

    print(f"  Downloaded {len(df):,} observations. Trimming to {radius_km} km radius ...")

    # ------------------------------------------------------------------ #
    # Trim to radius                                                       #
    # ------------------------------------------------------------------ #
    dist_matrix = _get_lonlat_distance(
        lon, lat,
        df["lon"].values, df["lat"].values,
    )
    dist_km = dist_matrix.flatten() * _EARTH_RADIUS_KM

    df = df.copy()
    df["distance_km"] = dist_km
    df = df.loc[dist_km <= radius_km].reset_index(drop=True)

    print(f"  Retained {len(df):,} observations within radius.")

    return df.set_index("time")

# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download CCI altimeter satellite data around a point."
    )
    parser.add_argument("--lon",       type=float, required=True, help="Target longitude")
    parser.add_argument("--lat",       type=float, required=True, help="Target latitude")
    parser.add_argument("--radius",    type=float, required=True, help="Radius in km")
    parser.add_argument("--start",     type=str,   required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",       type=str,   required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--out",       type=str,   default=None,  help="Output CSV path (optional)")
    args = parser.parse_args()

    df = download_satellite(
        lon=args.lon,
        lat=args.lat,
        radius_km=args.radius,
        start=args.start,
        end=args.end,
    )

    print(df)

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Saved to {args.out}")