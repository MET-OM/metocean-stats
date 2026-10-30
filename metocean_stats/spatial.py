"""Spatial: a class for plotting oceanographic/meteorological gridded data on a map.

Design: add_* methods only record what to draw and expand the tracked data
bounds - nothing is plotted and no figure exists until `.plot()` is called.
At that point the extent (if not given explicitly), projection (if not given
explicitly) and feature resolution are all derived from the data added.

Usage
-----
>>> m = Spatial()
>>> m.add_points_of_interest([4.5, 5.0], [60.4, 60.6], ["Station A", "Station B"])
>>> fig, ax = m.plot()

Add data layers with the ``add_*`` methods (for example `add_bathymetry`,
`add_quiver` or `add_coastline`) before calling `plot`.

To draw into an existing axes instead (e.g. one panel of a subplot grid),
pass it to `plot`: ``m.plot(ax=ax)``. The extent is then taken from the axes,
or from an explicit ``extent`` given to `Spatial`, rather than computed from
the data.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

_PROJECTIONS = {
    "PlateCarree": ccrs.PlateCarree,
    "Mercator": ccrs.Mercator,
    "NorthPolarStereo": ccrs.NorthPolarStereo,
    "SouthPolarStereo": ccrs.SouthPolarStereo,
    "LambertConformal": ccrs.LambertConformal,
    "LambertAzimuthalEqualArea": ccrs.LambertAzimuthalEqualArea,
    "Orthographic": ccrs.Orthographic,
}

# Span above which data is considered "global" -> PlateCarree.
_GLOBAL_LON_SPAN_THRESHOLD = 300.0
_GLOBAL_LAT_SPAN_THRESHOLD = 100.0

# Latitude beyond which data is considered a polar cap regardless of longitude
_POLAR_LAT_THRESHOLD = 45.0

# (max_span_degrees, resolution) tiers, checked in order, coarsest first.
_RESOLUTION_TIERS = [
    (60.0, "110m"),
    (10.0, "50m"),
    (0.0, "10m"),
]


def _to_ndarray(data):
    """Coerce numpy / pandas / xarray inputs to a plain numpy array of values."""
    if data is None:
        return None
    if hasattr(data, "values"):
        return np.asarray(data.values)
    return np.asarray(data)


def _check_shapes(lon, lat, **fields):
    """Coerce lon/lat (+ named fields) to ndarray and verify shapes are compatible."""
    lon = _to_ndarray(lon)
    lat = _to_ndarray(lat)
    if lon is None or lat is None:
        raise ValueError("lon and lat must be provided")

    if lon.ndim == 1 and lat.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    elif lon.ndim == 2 and lat.ndim == 2:
        if lon.shape != lat.shape:
            raise ValueError(f"lon and lat must have matching shapes, got {lon.shape} and {lat.shape}")
        lon2d, lat2d = lon, lat
    else:
        raise ValueError("lon and lat must both be 1D (will be meshgridded) or both 2D")

    coerced = {}
    for name, value in fields.items():
        if value is None:
            coerced[name] = None
            continue
        arr = _to_ndarray(value)
        if arr.shape != lon2d.shape:
            raise ValueError(f"'{name}' has shape {arr.shape}, expected {lon2d.shape} (derived from lon/lat)")
        coerced[name] = arr

    return lon2d, lat2d, coerced


def _auto_levels(data, n=10):
    """Return ~n evenly spaced contour levels spanning the finite range of data."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("data contains no finite values to derive levels from")
    vmin, vmax = np.nanmin(finite), np.nanmax(finite)
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    return np.linspace(vmin, vmax, n)


def _wrap_lon(lon):
    """Wrap a longitude (or array) into the conventional [-180, 180) range."""
    return ((lon + 180.0) % 360.0) - 180.0


def _circular_lon_extent(lons):
    """Compute the smallest longitude span containing all points, handling
    wraparound so that e.g. [-5, 0, 5], [355, 0, 5] and [355, 360, 5] are all
    recognized as the same ~10 degree span rather than ~360 degrees.

    Method: place all points on the circle (mod 360), find the largest empty
    gap between consecutive points, and take the extent as the complement of
    that gap (i.e. everything *outside* the biggest gap is the data span).

    Returns (lon_min, lon_max) with lon_min possibly < -180 or lon_max possibly
    > 180 if the span crosses the antimeridian (e.g. -5 to 5 stays as -5/5, but
    350 to 10 becomes -10/10) - both remain valid, ordinary min/max in a shared
    unwrapped frame suitable for downstream extent/projection calculations.
    """
    lons_mod = np.mod(np.asarray(lons, dtype=float), 360.0)  # now in [0, 360)
    uniq = np.unique(lons_mod)

    if uniq.size == 1:
        lo = _wrap_lon(uniq[0])
        return lo, lo

    gaps = np.diff(uniq, append=uniq[0] + 360.0)
    max_gap_idx = int(np.argmax(gaps))
    max_gap = gaps[max_gap_idx]

    # Data spans everything *except* the largest gap. The span starts right
    # after the gap and runs forward (increasing, unwrapped) to close it.
    lon_min = uniq[(max_gap_idx + 1) % uniq.size]
    span = 360.0 - max_gap
    lon_max = lon_min + span

    # Re-center into a conventional frame close to [-180, 180] so downstream
    # numbers look sane (e.g. -5/5 rather than 355/365).
    lon_min_wrapped = _wrap_lon(lon_min)
    shift = lon_min_wrapped - lon_min
    return lon_min + shift, lon_max + shift

def _resolve_projection(projection, extent):
    """Resolve a projection spec into a cartopy CRS instance.

    projection may be: None (auto-select based on extent), a known name (str),
    or an already-instantiated CRS.

    Auto-selection logic (in order):
      1. Polar cap (data confined to lat >= 45 or lat <= -45): Lambert
         Azimuthal Equal-Area centered on the relevant pole. Full longitude
         coverage is normal here and does NOT imply a global map.
      2. Genuinely global (large longitude span AND large latitude span):
         PlateCarree.
      3. Otherwise (regional, non-polar): Lambert Azimuthal Equal-Area
         centered on the data/extent centroid.
    """
    if projection is not None:
        if isinstance(projection, str):
            if projection not in _PROJECTIONS:
                raise ValueError(
                    f"Unknown projection '{projection}'. Available: {list(_PROJECTIONS)}. "
                    "You can also pass a cartopy CRS instance directly."
                )
            return _PROJECTIONS[projection]()
        return projection  # already a CRS instance

    lon_min, lon_max, lat_min, lat_max = extent
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min

    # 1. Polar cap - checked first, since a full lon ring at high latitude
    #    must not be mistaken for global coverage.
    if lat_min >= _POLAR_LAT_THRESHOLD:
        return ccrs.LambertAzimuthalEqualArea(central_longitude=0.0, central_latitude=90.0)
    if lat_max <= -_POLAR_LAT_THRESHOLD:
        return ccrs.LambertAzimuthalEqualArea(central_longitude=0.0, central_latitude=-90.0)

    # 2. Genuinely global - both dimensions must be large.
    if lon_span >= _GLOBAL_LON_SPAN_THRESHOLD and lat_span >= _GLOBAL_LAT_SPAN_THRESHOLD:
        return ccrs.PlateCarree()

    # 3. Regional.
    lon_center = _wrap_lon((lon_min + lon_max) / 2.0)
    lat_center = (lat_min + lat_max) / 2.0
    return ccrs.LambertAzimuthalEqualArea(central_longitude=lon_center, central_latitude=lat_center)

def _auto_resolution(extent):
    """Pick a coastline/feature resolution tier based on the extent's spatial scale."""
    lon_min, lon_max, lat_min, lat_max = extent
    span = max(lon_max - lon_min, lat_max - lat_min)
    for threshold, resolution in _RESOLUTION_TIERS:
        if span >= threshold:
            return resolution
    return _RESOLUTION_TIERS[-1][1]


def _pad_extent(lon_min, lon_max, lat_min, lat_max, frac=0.05, min_pad=0.5):
    """Add proportional padding around a data bounding box, with a floor for
    degenerate (point-like or zero-span) inputs. Latitude is clamped to
    [-90, 90]; longitude is intentionally left unclamped here so a span
    crossing the antimeridian is preserved for the projection/extent logic
    downstream (cartopy handles lon_min/lon_max outside [-180, 180] fine)."""
    lon_pad = max((lon_max - lon_min) * frac, min_pad)
    lat_pad = max((lat_max - lat_min) * frac, min_pad)
    return (
        lon_min - lon_pad,
        lon_max + lon_pad,
        max(lat_min - lat_pad, -90.0),
        min(lat_max + lat_pad, 90.0),
    )

def _edge_sample(lon_min, lon_max, lat_min, lat_max, n=100):
    """Densely sample the border of a lon/lat box (not just its 4 corners),
    since projected box edges are generally curved, not straight."""
    lons = np.linspace(lon_min, lon_max, n)
    lats = np.linspace(lat_min, lat_max, n)
    top = np.column_stack([lons, np.full(n, lat_max)])
    bottom = np.column_stack([lons, np.full(n, lat_min)])
    left = np.column_stack([np.full(n, lon_min), lats])
    right = np.column_stack([np.full(n, lon_max), lats])
    pts = np.concatenate([top, bottom, left, right])
    return pts[:, 0], pts[:, 1]


def _tight_projected_extent(proj, lons, lats, pad_frac=0.05):
    """Project lon/lat points into the map's own coordinate system and return
    a tight (x_min, x_max, y_min, y_max) bounding box in those native
    coordinates, with proportional padding. Returns None if no finite points
    result (e.g. all points are outside the projection's valid domain)."""
    pts = proj.transform_points(ccrs.PlateCarree(), np.asarray(lons), np.asarray(lats))
    x, y = pts[:, 0], pts[:, 1]
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size == 0:
        return None

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_pad = max((x_max - x_min) * pad_frac, 1.0)
    y_pad = max((y_max - y_min) * pad_frac, 1.0)
    return (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)

def _batch_shape_of(da_or_shape, freq_dim: str, dir_dim: str, dims=None):
    """
    Given a DataArray (or an explicit `dims` tuple + shape), return
    (batch_dims, batch_shape): every dim except freq_dim/dir_dim, in
    their existing order, and the corresponding sizes. Possibly empty
    (a bare (freq, dir) spectrum with no batch dims at all) -- this is
    a valid, supported case, not an error.
    """
    if dims is None:
        dims = da_or_shape.dims
        sizes = da_or_shape.sizes
    else:
        sizes = dict(zip(dims, da_or_shape))  # da_or_shape is a shape tuple here
 
    batch_dims = tuple(d for d in dims if d not in (freq_dim, dir_dim))
    batch_shape = tuple(sizes[d] for d in batch_dims)
    return batch_dims, batch_shape


def _broadcast_point_params(n, **params):
    """Broadcast per-point-of-interest parameters against n points.

    Each keyword value may be:
      - a single "scalar" value (including a plain string, which is treated
        as one atomic value, NOT as a sequence of characters, even though
        strings have __len__) -> repeated for all n points.
      - a sequence of length n -> used as-is, one entry per point.
      - a sequence of length 1 -> that single value repeated for all n points.

    Any sequence-like value whose length is neither 1 nor n raises ValueError.
    Returns a dict of the same keys, each now a list of length n.

    "Sequence-like" here means: has __len__ and __getitem__, but is not a
    str/bytes (those are always atomic) and is not None.
    """
    def is_scalar_like(v):
        if v is None:
            return True
        if isinstance(v, (str, bytes)):
            return True
        if not hasattr(v, "__len__"):
            return True
        return False

    out = {}
    for name, value in params.items():
        if is_scalar_like(value):
            out[name] = [value] * n
            continue

        seq = list(value)
        if len(seq) == n:
            out[name] = seq
        elif len(seq) == 1:
            out[name] = seq * n
        else:
            raise ValueError(
                f"'{name}' has length {len(seq)}, but must have length 1 or "
                f"{n} (the number of points) to match lons/lats."
            )
    return out

# ---------------------------------------------------------------------------
# Drawing implementations - plain functions taking an already-created ax and
# transform. Always called from within plot(), in call order.
# ---------------------------------------------------------------------------
def _draw_quiver(ax, transform, *, lon2d, lat2d, u2d, v2d, color, cmap, magnitude_color,
                  scale, width, subsample, label, reference_length, reference_units, **kwargs):
    sl = (slice(None, None, subsample), slice(None, None, subsample))
    lon_s, lat_s, u_s, v_s = lon2d[sl], lat2d[sl], u2d[sl], v2d[sl]

    kw = dict(scale=scale, width=width, transform=transform)
    kw.update(kwargs)

    if magnitude_color:
        magnitude = np.hypot(u_s, v_s)
        kw["cmap"] = cmap
        q = ax.quiver(lon_s, lat_s, u_s, v_s, magnitude, **{k: v for k, v in kw.items() if v is not None})
    else:
        kw["color"] = color
        q = ax.quiver(lon_s, lat_s, u_s, v_s, **{k: v for k, v in kw.items() if v is not None})

    if reference_length is not None:
        ax.quiverkey(
            q, X=0.9, Y=1.03, U=reference_length,
            label=f"{reference_length:g} {reference_units}".strip(),
            labelpos="E", coordinates="axes",
        )
    if label is not None:
        proxy_color = color if not magnitude_color else "k"
        ax.plot([], [], color=proxy_color, marker=">", linestyle="none", label=label)
    return q

def _grid_index_velocities(lon2d, lat2d, u2d, v2d):
    """Convert physical (east, north) vector components defined on a
    curvilinear lon/lat grid into grid-index velocities (di/dt, dj/dt), by
    inverting the local coordinate Jacobian at each grid point.

    i indexes columns (array axis=1), j indexes rows (array axis=0),
    matching standard array indexing / np.meshgrid convention.

    Longitude is unwrapped along both array axes before differentiating, so
    that the antimeridian seam (e.g. neighboring cells going 179 -> -179) or
    a pole-crossing grid does not register as a huge artificial jump in the
    finite-difference gradient. Without this, the local Jacobian blows up at
    the seam/pole row, corrupting the computed velocities there and
    producing spurious closed-loop streamlines - not an integration bug, but
    a bad input velocity field right at the discontinuity.
    """
    R = 6371000.0  # meters; only the *ratio* between i/j scaling matters here
    lat_rad = np.deg2rad(lat2d)

    lon_rad = np.deg2rad(lon2d)
    lon_unwrapped_rad = np.unwrap(np.unwrap(lon_rad, axis=1), axis=0)

    dlon_di = np.gradient(lon_unwrapped_rad, axis=1)
    dlon_dj = np.gradient(lon_unwrapped_rad, axis=0)
    dlat_di = np.gradient(lat_rad, axis=1)
    dlat_dj = np.gradient(lat_rad, axis=0)

    # local physical displacement (meters) per unit step in i / j
    dx_di = R * np.cos(lat_rad) * dlon_di
    dx_dj = R * np.cos(lat_rad) * dlon_dj
    dy_di = R * dlat_di
    dy_dj = R * dlat_dj

    det = dx_di * dy_dj - dx_dj * dy_di
    det_safe = np.where(np.abs(det) < 1e-30, np.nan, det)

    # invert [[dx_di, dx_dj], [dy_di, dy_dj]] @ [di, dj] = [u, v]
    inv_a = dy_dj / det_safe
    inv_b = -dx_dj / det_safe
    inv_c = -dy_di / det_safe
    inv_d = dx_di / det_safe

    ui = inv_a * u2d + inv_b * v2d
    uj = inv_c * u2d + inv_d * v2d
    return ui, uj

def _draw_streamlines(ax, transform, *, lon2d, lat2d, u2d, v2d, color, cmap,
                       color_by_magnitude, density, linewidth, label,
                       grid_atol=1e-6, grid_rtol=1e-5,
                       arrow_spacing=None, arrow_size=3, **kwargs):

    """Plot streamlines of a 2D vector field.

    Works on any grid (plain rectilinear lon/lat, or curvilinear grids where
    lon/lat both vary with both array dimensions), and correctly handles
    poles and the antimeridian seam.

    Streamlines are always integrated in array index space (i, j) - which is
    regular by construction and has no pole singularity or dateline seam -
    with physical u/v first converted to grid-index velocities via the local
    coordinate Jacobian. The resulting path vertices are then remapped to
    lon/lat for display. This deliberately avoids matplotlib's native
    streamplot(transform=...) route, which integrates in lon/lat space and
    produces spurious circular artifacts near the pole or across the
    antimeridian.

    Known limitation: this route omits streamplot's directional arrowheads
    (matplotlib's arrow placement isn't easily remapped point-by-point).
    """
    from matplotlib.figure import Figure
    from matplotlib.collections import LineCollection
    from scipy.ndimage import map_coordinates

    ny, nx = lon2d.shape
    i = np.arange(nx, dtype=float)
    j = np.arange(ny, dtype=float)

    ui, uj = _grid_index_velocities(lon2d, lat2d, u2d, v2d)
    ui = np.nan_to_num(ui, nan=0.0)
    uj = np.nan_to_num(uj, nan=0.0)

    tmp_fig = Figure()
    tmp_ax = tmp_fig.add_subplot(111)
    strm_tmp = tmp_ax.streamplot(i, j, ui, uj, density=density, linewidth=1.0)
    segments = strm_tmp.lines.get_segments()  # list of (k_n, 2) arrays, k_n varies per streamline

    if not segments:
        return None

    seg_lengths = [len(seg) for seg in segments]
    seg_arr = np.concatenate(segments, axis=0)  # (total_points, 2) -> columns are (i, j)
    i_coords, j_coords = seg_arr[:, 0], seg_arr[:, 1]

    lon_pts = map_coordinates(lon2d, [j_coords, i_coords], order=1, mode="nearest")
    lat_pts = map_coordinates(lat2d, [j_coords, i_coords], order=1, mode="nearest")

    splits = np.cumsum(seg_lengths)[:-1]
    lon_polylines = np.split(lon_pts, splits)
    lat_polylines = np.split(lat_pts, splits)

    lon_polylines = [np.unwrap(poly, period=360.0) for poly in lon_polylines]

    lon_lat_segments = []
    for lon_poly, lat_poly in zip(lon_polylines, lat_polylines):
        if len(lon_poly) < 2:
            continue
        pts = np.stack([lon_poly, lat_poly], axis=-1)  # (k, 2)
        pairs = np.stack([pts[:-1], pts[1:]], axis=1)  # (k-1, 2, 2)
        lon_lat_segments.append(pairs)

    if not lon_lat_segments:
        return None
    lon_lat_segments = np.concatenate(lon_lat_segments, axis=0)

    if color_by_magnitude:
        mag2d = np.hypot(u2d, v2d)
        mag_pts = map_coordinates(mag2d, [j_coords, i_coords], order=1, mode="nearest")
        mag_polylines = np.split(mag_pts, splits)
        seg_mag = []
        for mag_poly in mag_polylines:
            if len(mag_poly) < 2:
                continue
            seg_mag.append((mag_poly[:-1] + mag_poly[1:]) / 2.0)
        seg_mag = np.concatenate(seg_mag, axis=0)

        lc = LineCollection(lon_lat_segments, cmap=cmap, transform=transform,
                            linewidths=linewidth or 1.0, **kwargs)
        lc.set_array(seg_mag)
    else:
        lc = LineCollection(lon_lat_segments, colors=color, transform=transform,
                            linewidths=linewidth or 1.0, **kwargs)

    def _add_streamline_arrows(ax, transform, lon_polylines, lat_polylines, color,
                                seg_colors_per_polyline=None, cmap=None,
                                arrow_spacing=None, arrow_size=3, zorder=3):
        """Place directional arrowheads at intervals along each streamline
        polyline, since matplotlib's native streamplot arrows can't be reused
        once we bypass streamplot's own drawing (necessary to avoid antimeridian/
        pole artifacts - see _draw_streamlines).

        arrow_spacing: draw one arrow every N points along each polyline.
                    Set to None to disable arrows entirely.
        arrow_size: scales the arrowhead size (matplotlib mutation_scale).
        """
        if arrow_spacing is None:
            return

        from matplotlib.patches import FancyArrowPatch

        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

        for idx, (lon_poly, lat_poly) in enumerate(zip(lon_polylines, lat_polylines)):
            n = len(lon_poly)
            if n < 2:
                continue

            if seg_colors_per_polyline is not None:
                arrow_color = cmap_obj(plt.Normalize()(seg_colors_per_polyline[idx].mean())) \
                    if cmap_obj is not None else color
            else:
                arrow_color = color

            for k in range(arrow_spacing // 2, n - 1, arrow_spacing):
                start = (lon_poly[k], lat_poly[k])
                end = (lon_poly[k + 1], lat_poly[k + 1])
                arrow = FancyArrowPatch(
                    start, end, transform=transform, arrowstyle="-|>",
                    mutation_scale=12 * arrow_size, color=arrow_color,
                    linewidth=0, zorder=zorder,
                )
                ax.add_patch(arrow)

    ax.add_collection(lc)

    _add_streamline_arrows(
        ax, transform, lon_polylines, lat_polylines,
        color=color, cmap=cmap if color_by_magnitude else None,
        seg_colors_per_polyline=(
            [np.split(mag_pts, splits)[i] for i in range(len(lon_polylines))]
            if color_by_magnitude else None
        ),
        arrow_spacing=arrow_spacing, arrow_size=arrow_size,
    )

    if label is not None:
        proxy_color = color if not color_by_magnitude else "k"
        ax.plot([], [], color=proxy_color, label=label)
    return lc

def _levels_norm(levels, cmap):
    """Build a BoundaryNorm so that each interval BETWEEN consecutive levels
    maps to an equal-sized slice of the colormap, regardless of how unevenly
    the level *values* are spaced.

    Without this, contour/contourf's default Normalize scales colors linearly
    by level value, so e.g. levels [50,100,200,500,1000,2000,5000] puts almost
    the whole colormap in the 2000-5000 band and makes 50/100/200/500 look
    nearly identical. BoundaryNorm instead assigns colormap slice k to the
    k-th interval regardless of its width, so every band is visually distinct.
    """
    levels = np.asarray(levels)
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) or cmap is None else cmap
    n_bins = max(len(levels) - 1, 1)
    return mcolors.BoundaryNorm(levels, ncolors=cmap_obj.N, extend="both") if n_bins > 0 else None


def _draw_contour(ax, transform, *, lon2d, lat2d, z, levels, cmap, colors, linewidths,
                   clabel, clabel_fmt, label, even_color_spacing=True, **kwargs):
    norm = None
    if even_color_spacing and colors is None and levels is not None and cmap is not None:
        norm = _levels_norm(levels, cmap)
    kw = dict(levels=levels, cmap=cmap, colors=colors, linewidths=linewidths,
              norm=norm, transform=transform, transform_first=True)
    kw.update(kwargs)
    cs = ax.contour(lon2d, lat2d, z, **{k: v for k, v in kw.items() if v is not None})
    if clabel:
        ax.clabel(cs, fmt=clabel_fmt)
    if label is not None:
        proxy_color = colors if isinstance(colors, str) else "k"
        ax.plot([], [], color=proxy_color, label=label)
    return cs


def _draw_contourf(ax, transform, *, lon2d, lat2d, z, levels, cmap, extend, label,
                    colorbar, colorbar_label, even_color_spacing=True, **kwargs):
    norm = None
    if even_color_spacing and levels is not None:
        norm = _levels_norm(levels, cmap)
    kw = dict(levels=levels, cmap=cmap, extend=extend, norm=norm,
              transform=transform, transform_first=True)
    kw.update(kwargs)
    cf = ax.contourf(lon2d, lat2d, z, **{k: v for k, v in kw.items() if v is not None})
    if colorbar:
        cbar = ax.figure.colorbar(cf, ax=ax, orientation="vertical")
        cbar.set_label(colorbar_label or label or "")
    return cf


def _draw_pcolormesh(ax, transform, *, lon2d, lat2d, z, cmap, vmin, vmax, label,
                      colorbar, colorbar_label, shading, **kwargs):
    kw = dict(cmap=cmap, vmin=vmin, vmax=vmax, shading=shading, transform=transform)
    kw.update(kwargs)
    pc = ax.pcolormesh(lon2d, lat2d, z, **{k: v for k, v in kw.items() if v is not None})
    if colorbar:
        cbar = ax.figure.colorbar(pc, ax=ax, orientation="vertical")
        cbar.set_label(colorbar_label or label or "")
    return pc


def _draw_points(ax, transform, *, lons, lats, texts, markers, colors, markersizes,
                  text_offsets, text_kwargs_list, labels, **kwargs):
    """Draw one or more points of interest. All the per-point arguments here
    are already-broadcast lists of equal length (see _broadcast_point_params);
    this function just loops over them and draws each one, using **kwargs
    (shared, non-broadcast extras) for every point."""
    artists = []
    n = len(lons)
    for idx in range(n):
        lon, lat = lons[idx], lats[idx]
        text = texts[idx]
        marker = markers[idx]
        color = colors[idx]
        markersize = markersizes[idx]
        text_offset = text_offsets[idx]
        text_kwargs = text_kwargs_list[idx]
        label = labels[idx]

        kw = dict(marker=marker, color=color, markersize=markersize, linestyle="none", transform=transform)
        kw.update(kwargs)
        if label is not None:
            kw["label"] = label
        (point,) = ax.plot([lon], [lat], **{k: v for k, v in kw.items() if v is not None})

        txt = None
        if text:
            dx, dy = text_offset
            txt = ax.text(
                lon + dx, lat + dy, text, transform=transform,
                **{"fontsize": 9, "ha": "left", "va": "bottom", **(text_kwargs or {})},
            )
        artists.append((point, txt))
    return artists if n > 1 else artists[0]


def _draw_coastline(ax, transform, *, resolution, color, linewidth, **kwargs):
    return ax.coastlines(resolution=resolution, color=color, linewidth=linewidth, **kwargs)


def _draw_land(ax, transform, *, resolution, facecolor, edgecolor, zorder, **kwargs):
    land = cfeature.NaturalEarthFeature("physical", "land", resolution, facecolor=facecolor, edgecolor=edgecolor)
    return ax.add_feature(land, zorder=zorder, **kwargs)


def _draw_borders(ax, transform, *, resolution, color, linewidth, **kwargs):
    return ax.add_feature(cfeature.BORDERS.with_scale(resolution), edgecolor=color, linewidth=linewidth, **kwargs)


def _draw_image_tiles(ax, transform, *, tiler, zoom, interpolation, zorder, **kwargs):
    # add_image fetches tiles for the axes' current extent, set by plot()
    # before any queued layer is drawn -- same precondition as coastline/land.
    return ax.add_image(tiler, zoom, interpolation=interpolation, zorder=zorder, **kwargs)


def _draw_gridlines(ax, transform, *, draw_labels, linewidth, color, alpha, linestyle, **kwargs):
    return ax.gridlines(draw_labels=draw_labels, linewidth=linewidth, color=color, alpha=alpha,
                         linestyle=linestyle, **kwargs)


class Spatial:
    """
    Map of gridded oceanographic or meteorological data.

    The ``add_*`` methods queue layers and track data bounds. Nothing is
    drawn until `plot` is called. At that point the extent, projection and
    feature resolution are derived from the data, unless given explicitly.
    """

    def __init__(self, projection=None, extent=None, figsize=None, title=None, layout="constrained"):
        """
        Parameters
        ----------
        projection : str or cartopy.crs.Projection, optional
            Projection name: "PlateCarree", "Mercator", "NorthPolarStereo",
            "SouthPolarStereo", "LambertConformal",
            "LambertAzimuthalEqualArea" or "Orthographic". A cartopy
            projection instance is also accepted. If None, it is chosen from
            the extent in `plot`.
        extent : sequence of 4 float, optional
            Map extent as ``(lon_min, lon_max, lat_min, lat_max)``. If None,
            it is derived from the data.
        figsize : tuple of float, optional
            Figure size passed to `matplotlib.pyplot.subplots`.
        title : str, optional
            Axes title.
        layout : str, default "constrained"
            Layout engine passed to `matplotlib.pyplot.subplots`.

        Raises
        ------
        ValueError
            If `extent` does not contain exactly four values.
        """
        self.projection_spec = projection


        if extent is not None:
            extent = tuple(extent)
            if len(extent) != 4:
                raise ValueError(
                    f"extent must have exactly 4 values (lon_min, lon_max, lat_min, lat_max), "
                    f"got {len(extent)}: {extent}. Omit extent entirely (or pass extent=None) "
                    "to have it inferred automatically from the data added."
                )
        self.extent = extent
        self.figsize = figsize
        self.title = title
        self.layout = layout

        self._layers = []
        self.artists = []

        self._lon_samples = []
        self._lat_samples = []
        self._lat_min = None
        self._lat_max = None

        self.fig = None
        self.ax = None
        self.transform = ccrs.PlateCarree()

    # -- bounds tracking -------------------------------------------------

    def _update_bounds(self, lon, lat):
        lon = np.asarray(lon, dtype=float).ravel()
        lat = np.asarray(lat, dtype=float).ravel()
        self._lon_samples.append(lon)
        self._lat_samples.append(lat)
        lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
        self._lat_min = lat_min if self._lat_min is None else min(self._lat_min, lat_min)
        self._lat_max = lat_max if self._lat_max is None else max(self._lat_max, lat_max)

    def _has_bounds(self):
        return bool(self._lon_samples) and self._lat_min is not None

    def _compute_data_extent(self):
        """Compute a padded (lon_min, lon_max, lat_min, lat_max) from all data
        added so far, using circular longitude logic to avoid inflating the
        span for data that merely straddles 0 deg or the antimeridian."""
        all_lon = np.concatenate(self._lon_samples)
        lon_min, lon_max = _circular_lon_extent(all_lon)
        return _pad_extent(lon_min, lon_max, self._lat_min, self._lat_max)

    # -- layer queueing ----------------------------------------------------

    def _add_layer(self, draw_func, **kwargs):
        self._layers.append((draw_func, kwargs))
        return self
    
    def plot(self, ax=None):
        """
        Draw all queued layers.

        Parameters
        ----------
        ax : cartopy.mpl.geoaxes.GeoAxes, optional
            Existing axes to draw on. If None, a new figure is created. An
            existing axes keeps its current extent, unless `extent` was given
            to the constructor.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : cartopy.mpl.geoaxes.GeoAxes
        """
        if ax is not None:
            self.ax = ax
            self.fig = ax.figure

            if self.extent is not None:
                extent = self.extent
            else:
                try:
                    extent = ax.get_extent(crs=self.transform)
                except Exception:
                    extent = self._compute_data_extent() if self._has_bounds() else None

            if extent is not None:
                self.extent = extent
                ax.set_extent(extent, crs=self.transform)

        else:
            if self.extent is not None:
                lonlat_extent = self.extent
                # explicit extent given as a lon/lat box - densely sample its
                # border so the projected footprint is still tight, not just
                # the 4 corners.
                sample_lons, sample_lats = _edge_sample(*lonlat_extent)
            elif self._has_bounds():
                lonlat_extent = self._compute_data_extent()
                # use the actual data points for a tight fit, not the box edges
                sample_lons = np.concatenate(self._lon_samples)
                sample_lats = np.concatenate(self._lat_samples)
            else:
                raise ValueError(
                    "No data has been added and no extent was given, so a map extent "
                    "cannot be determined. Add data via add_quiver/add_contourf/etc., "
                    "or pass extent=[lon_min, lon_max, lat_min, lat_max] explicitly."
                )
            self.extent = lonlat_extent

            proj = _resolve_projection(self.projection_spec, lonlat_extent)
            # constrained layout keeps colorbars, gridline tick labels, and
            # axes from overlapping (e.g. a right-hand colorbar colliding
            # with gridline labels drawn on that same side) by re-solving
            # the layout whenever new artists (colorbars, tick labels, etc.)
            # are added, instead of using fixed subplot margins.
            self.fig, self.ax = plt.subplots(
                figsize=self.figsize, subplot_kw={"projection": proj},
                layout=self.layout,
            )

            projected_extent = _tight_projected_extent(proj, sample_lons, sample_lats)
            if projected_extent is not None:
                self.ax.set_extent(projected_extent, crs=proj)
            else:
                # fall back if projection produced no finite points
                self.ax.set_extent(lonlat_extent, crs=self.transform)

        if self.title:
            self.ax.set_title(self.title)

        auto_res = _auto_resolution(self.extent) if self.extent is not None else "50m"
        for draw_func, kwargs in self._layers:
            if kwargs.get("resolution", "unset") is None:
                kwargs["resolution"] = auto_res
            self.artists.append(draw_func(self.ax, self.transform, **kwargs))

        return self.fig, self.ax
    
    def legend(self, **kwargs):
        """
        Add a legend to the map. Call after `plot`.

        Parameters
        ----------
        **kwargs
            Passed to `matplotlib.axes.Axes.legend`.

        Returns
        -------
        Spatial
            This instance, for chaining.
        """
        self.ax.legend(**kwargs)
        return self

    def savefig(self, path, **kwargs):
        """
        Save the figure. Call after `plot`.

        Parameters
        ----------
        path : str or path-like
            Output file name. The format is inferred from the extension.
        **kwargs
            Passed to `matplotlib.figure.Figure.savefig`.

        Returns
        -------
        Spatial
            This instance, for chaining.
        """
        self.fig.savefig(path, **kwargs)
        return self

    # -- plotting methods (always queued; executed by plot()) --------------

    def add_quiver(self, u, v, lon, lat, *, color="k", cmap=None, magnitude_color=False,
                    scale=None, width=None, subsample=1, label=None,
                    reference_length=None, reference_units="", **kwargs):
        """
        Queue a vector field as arrows.

        Parameters
        ----------
        u, v : array-like
            Eastward and northward components. The shape must match the grid
            defined by `lon` and `lat`.
        lon, lat : array-like
            1D coordinates (meshed onto a grid) or 2D coordinate arrays.
        color : str, default "k"
            Arrow colour. Ignored if `magnitude_color` is True.
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap used if `magnitude_color` is True.
        magnitude_color : bool, default False
            Colour the arrows by vector magnitude.
        scale : float, optional
            Passed to `matplotlib.axes.Axes.quiver`. Smaller values give longer
            arrows. If None, matplotlib chooses.
        width : float, optional
            Shaft width, passed to `matplotlib.axes.Axes.quiver`.
        subsample : int, default 1
            Plot every Nth vector in each dimension.
        label : str, optional
            Legend label. Adds an invisible proxy artist.
        reference_length : float, optional
            Length of a reference arrow (quiver key). If None, no key is drawn.
        reference_units : str, default ""
            Units shown with the reference arrow.
        **kwargs
            Passed to `matplotlib.axes.Axes.quiver`.

        Returns
        -------
        Spatial
            This instance, for chaining.
        """
        lon2d, lat2d, fields = _check_shapes(lon, lat, u=u, v=v)
        self._update_bounds(lon2d, lat2d)
        return self._add_layer(
            _draw_quiver, lon2d=lon2d, lat2d=lat2d, u2d=fields["u"], v2d=fields["v"],
            color=color, cmap=cmap, magnitude_color=magnitude_color, scale=scale, width=width,
            subsample=subsample, label=label, reference_length=reference_length,
            reference_units=reference_units, **kwargs,
        )

    def add_streamlines(self, u, v, lon, lat, *, color="k", cmap=None, color_by_magnitude=False,
                          density=1.0, linewidth=None, label=None,
                          grid_atol=1e-6, grid_rtol=1e-5,
                          arrow_spacing=None, arrow_size=3, **kwargs):
        """
        Queue streamlines of a 2D vector field.

        Parameters
        ----------
        u, v, lon, lat : array-like
            As for `add_quiver`.
        color : str, default "k"
            Line colour. Ignored if `color_by_magnitude` is True.
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap used if `color_by_magnitude` is True.
        color_by_magnitude : bool, default False
            Colour the streamlines by vector magnitude.
        density : float, default 1.0
            Streamline density, as in `matplotlib.axes.Axes.streamplot`.
        linewidth : float, optional
            Line width. If None, 1.0 is used.
        label : str, optional
            Legend label. Adds an invisible proxy artist.
        grid_atol, grid_rtol : float
            Unused. Kept for backward compatibility.
        arrow_spacing : int, optional
            Place one arrowhead every N points along each streamline. If None,
            no arrowheads are drawn.
        arrow_size : float, default 3
            Scales the arrowhead size.
        **kwargs
            Passed to `matplotlib.collections.LineCollection`.

        Returns
        -------
        Spatial
            This instance, for chaining.

        Notes
        -----
        Streamlines are integrated in grid index space and then mapped back to
        lon/lat. This avoids the artefacts that `streamplot` with a cartopy
        transform produces near the poles and the antimeridian.
        """
        lon2d, lat2d, fields = _check_shapes(lon, lat, u=u, v=v)
        self._update_bounds(lon2d, lat2d)
        return self._add_layer(
            _draw_streamlines, lon2d=lon2d, lat2d=lat2d, u2d=fields["u"], v2d=fields["v"],
            color=color, cmap=cmap, color_by_magnitude=color_by_magnitude,
            density=density, linewidth=linewidth, label=label,
            grid_atol=grid_atol, grid_rtol=grid_rtol,
            arrow_spacing=arrow_spacing, arrow_size=arrow_size, **kwargs,
        )

    def add_contour(self, data, lon, lat, *, levels=None, n_levels=10, cmap=None, colors=None,
                      linewidths=None, clabel=False, clabel_fmt="%.1f", label=None,
                      even_color_spacing=True, **kwargs):
        """
        Queue line contours of a scalar field.

        Parameters
        ----------
        data, lon, lat : array-like
            Scalar field and its coordinates. The shape of `data` must match the
            grid defined by `lon` and `lat`.
        levels : array-like, optional
            Contour levels. If None, `n_levels` evenly spaced levels spanning the
            finite data range are used.
        n_levels : int, default 10
            Number of automatic levels. Ignored if `levels` is given.
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap for the lines. Ignored if `colors` is given.
        colors : str or list of str, optional
            Single colour or one colour per level.
        linewidths : float, optional
            Line width.
        clabel : bool, default False
            Label the contour lines with their values.
        clabel_fmt : str, default "%.1f"
            Format of the contour labels.
        label : str, optional
            Legend label. Adds an invisible proxy artist.
        even_color_spacing : bool, default True
            If True and `cmap` is used, colours are assigned evenly across the
            levels by index, not by value. This keeps unevenly spaced levels
            visually distinct. Set to False for value-based colour scaling.
        **kwargs
            Passed to `matplotlib.axes.Axes.contour`.

        Returns
        -------
        Spatial
            This instance, for chaining.
        """
        lon2d, lat2d, fields = _check_shapes(lon, lat, data=data)
        self._update_bounds(lon2d, lat2d)
        z = fields["data"]
        if levels is None:
            levels = _auto_levels(z, n_levels)
        return self._add_layer(
            _draw_contour, lon2d=lon2d, lat2d=lat2d, z=z, levels=levels, cmap=cmap,
            colors=colors, linewidths=linewidths, clabel=clabel, clabel_fmt=clabel_fmt,
            label=label, even_color_spacing=even_color_spacing, **kwargs,
        )

    def add_contourf(self, data, lon, lat, *, levels=None, n_levels=10, cmap="viridis",
                       extend="both", label=None, colorbar=False, colorbar_label=None,
                       even_color_spacing=True, **kwargs):
        """
        Queue filled contours of a scalar field.

        Parameters
        ----------
        data, lon, lat : array-like
            Scalar field and its coordinates. The shape of `data` must match the
            grid defined by `lon` and `lat`.
        levels : array-like, optional
            Contour levels. If None, `n_levels` evenly spaced levels spanning the
            finite data range are used.
        n_levels : int, default 10
            Number of automatic levels. Ignored if `levels` is given.
        cmap : str or matplotlib.colors.Colormap, default "viridis"
            Colormap.
        extend : {"both", "min", "max", "neither"}, default "both"
            Passed to `matplotlib.axes.Axes.contourf`.
        label : str, optional
            Used as the colorbar label if `colorbar_label` is not given.
        colorbar : bool, default False
            Attach a colorbar.
        colorbar_label : str, optional
            Colorbar label.
        even_color_spacing : bool, default True
            If True, colours are assigned evenly across the levels by index, not
            by value. This keeps unevenly spaced levels visually distinct. Set
            to False for value-based colour scaling.
        **kwargs
            Passed to `matplotlib.axes.Axes.contourf`.

        Returns
        -------
        Spatial
            This instance, for chaining.
        """
        lon2d, lat2d, fields = _check_shapes(lon, lat, data=data)
        self._update_bounds(lon2d, lat2d)
        z = fields["data"]
        if levels is None:
            levels = _auto_levels(z, n_levels)
        return self._add_layer(
            _draw_contourf, lon2d=lon2d, lat2d=lat2d, z=z, levels=levels, cmap=cmap,
            extend=extend, label=label, colorbar=colorbar, colorbar_label=colorbar_label,
            even_color_spacing=even_color_spacing, **kwargs,
        )

    def add_pcolormesh(self, data, lon, lat, *, cmap="viridis", vmin=None, vmax=None, label=None,
                         colorbar=False, colorbar_label=None, shading="auto", **kwargs):
        """
        Queue a pseudocolour plot of a scalar field.

        Parameters
        ----------
        data, lon, lat : array-like
            Scalar field and its coordinates. The shape of `data` must match the
            grid defined by `lon` and `lat`.
        cmap : str or matplotlib.colors.Colormap, default "viridis"
            Colormap.
        vmin, vmax : float, optional
            Colour scale limits.
        label : str, optional
            Used as the colorbar label if `colorbar_label` is not given.
        colorbar : bool, default False
            Attach a colorbar.
        colorbar_label : str, optional
            Colorbar label.
        shading : str, default "auto"
            Passed to `matplotlib.axes.Axes.pcolormesh`.
        **kwargs
            Passed to `matplotlib.axes.Axes.pcolormesh`.

        Returns
        -------
        Spatial
            This instance, for chaining.
        """
        lon2d, lat2d, fields = _check_shapes(lon, lat, data=data)
        self._update_bounds(lon2d, lat2d)
        z = fields["data"]
        return self._add_layer(
            _draw_pcolormesh, lon2d=lon2d, lat2d=lat2d, z=z, cmap=cmap, vmin=vmin, vmax=vmax,
            label=label, colorbar=colorbar, colorbar_label=colorbar_label, shading=shading, **kwargs,
        )

    def add_bathymetry(self, depth, lon, lat, *, cmap="Blues", levels=None, n_levels=10,
                         colorbar=True, colorbar_label="Depth (m)", even_color_spacing=True,
                         **kwargs):
        """
        Queue a filled depth or elevation field, with defaults for bathymetry.

        A wrapper around `add_contourf`.

        Parameters
        ----------
        depth, lon, lat : array-like
            Depth field and its coordinates. The shape of `depth` must match the
            grid defined by `lon` and `lat`.
        cmap : str or matplotlib.colors.Colormap, default "Blues"
            Colormap.
        levels : array-like, optional
            Contour levels. See `add_contourf`.
        n_levels : int, default 10
            Number of automatic levels. See `add_contourf`.
        colorbar : bool, default True
            Attach a colorbar.
        colorbar_label : str, default "Depth (m)"
            Colorbar label.
        even_color_spacing : bool, default True
            Passed to `add_contourf`.
        **kwargs
            Passed to `add_contourf`.

        Returns
        -------
        Spatial
            This instance, for chaining.
        """
        return self.add_contourf(
            depth, lon, lat, cmap=cmap, levels=levels, n_levels=n_levels,
            colorbar=colorbar, colorbar_label=colorbar_label,
            even_color_spacing=even_color_spacing, **kwargs,
        )

    def add_points_of_interest(self, lons, lats, texts=None, *, markers="o", colors="k",
                                 markersizes=6, text_offsets=(0.02, 0.02), text_kwargs=None,
                                 labels=None, **kwargs):
        """
        Queue one or more points of interest, such as stations or platforms.

        Parameters
        ----------
        lons, lats : float or array-like
            Point coordinates in degrees. Must have the same length.
        texts : str or list of str, optional
            Label text drawn next to each point.
        markers : str or list of str, default "o"
            Marker style, passed to `matplotlib.axes.Axes.plot`.
        colors : str or list of str, default "k"
            Marker colour.
        markersizes : float or list of float, default 6
            Marker size.
        text_offsets : tuple of float or list of tuple, default (0.02, 0.02)
            Offset of each label from its point, as ``(dx, dy)`` in degrees.
        text_kwargs : dict or list of dict, optional
            Keyword arguments for `matplotlib.axes.Axes.text`.
        labels : str or list of str, optional
            Legend labels.
        **kwargs
            Passed to `matplotlib.axes.Axes.plot` for every point.

        Returns
        -------
        Spatial
            This instance, for chaining.

        Raises
        ------
        ValueError
            If `lons` and `lats` differ in length, or a per-point argument
            does not have length 1 or the number of points.

        Notes
        -----
        Each per-point argument is either a single value, applied to every
        point, or a sequence with one entry per point. A sequence of length 1
        is also applied to every point. A string is always a single value.

        Examples
        --------
        >>> m = Spatial()
        >>> m.add_points_of_interest([4.5, 5.0], [60.4, 60.6], ["A", "B"], colors=["red", "blue"])
        """
        lons_arr = np.atleast_1d(np.asarray(lons, dtype=float))
        lats_arr = np.atleast_1d(np.asarray(lats, dtype=float))
        if lons_arr.shape != lats_arr.shape:
            raise ValueError(f"lons and lats must have the same length, got {lons_arr.shape} and {lats_arr.shape}")
        n = lons_arr.size
        self._update_bounds(lons_arr, lats_arr)

        # text_offsets is special: a single offset is itself a 2-tuple, which
        # looks like "a length-2 sequence" - we must not mistake that for "one
        # offset per point" when n==2. Detect "a single (dx, dy) pair" up
        # front and wrap it before generic broadcasting sees it.
        def is_single_offset(v):
            return (
                len(v) == 2
                and not hasattr(v[0], "__len__")
                and not hasattr(v[1], "__len__")
            )

        if is_single_offset(text_offsets):
            text_offsets_in = [text_offsets] * n if n != 1 else [text_offsets]
        else:
            text_offsets_in = text_offsets

        broadcast = _broadcast_point_params(
            n,
            texts=texts, markers=markers, colors=colors, markersizes=markersizes,
            text_offsets=text_offsets_in, text_kwargs=text_kwargs, labels=labels,
        )

        return self._add_layer(
            _draw_points, lons=list(lons_arr), lats=list(lats_arr),
            texts=broadcast["texts"], markers=broadcast["markers"], colors=broadcast["colors"],
            markersizes=broadcast["markersizes"], text_offsets=broadcast["text_offsets"],
            text_kwargs_list=broadcast["text_kwargs"], labels=broadcast["labels"], **kwargs,
        )

    def add_coastline(self, *, resolution=None, color="black", linewidth=1.0, **kwargs):
        """
        Queue a coastline overlay.

        Parameters
        ----------
        resolution : {"110m", "50m", "10m"}, optional
            Natural Earth resolution. If None, it is chosen in `plot` from the
            map scale.
        color : str, default "black"
            Line colour.
        linewidth : float, default 1.0
            Line width.
        **kwargs
            Passed to `cartopy.mpl.geoaxes.GeoAxes.coastlines`.

        Returns
        -------
        Spatial
            This instance, for chaining.
        """
        return self._add_layer(_draw_coastline, resolution=resolution, color=color, linewidth=linewidth, **kwargs)

    def add_land(self, *, resolution=None, facecolor="lightgray", edgecolor="none", zorder=0, **kwargs):
        """
        Queue a filled land polygon, for use as a background.

        Parameters
        ----------
        resolution : {"110m", "50m", "10m"}, optional
            Natural Earth resolution. If None, it is chosen in `plot` from the
            map scale.
        facecolor : str, default "lightgray"
            Fill colour.
        edgecolor : str, default "none"
            Outline colour.
        zorder : int, default 0
            Drawing order.
        **kwargs
            Passed to `cartopy.mpl.geoaxes.GeoAxes.add_feature`.

        Returns
        -------
        Spatial
            This instance, for chaining.
        """
        return self._add_layer(
            _draw_land, resolution=resolution, facecolor=facecolor, edgecolor=edgecolor, zorder=zorder, **kwargs
        )

    def add_borders(self, *, resolution=None, color="gray", linewidth=0.5, **kwargs):
        """
        Queue national and administrative borders.

        Parameters
        ----------
        resolution : {"110m", "50m", "10m"}, optional
            Natural Earth resolution. If None, it is chosen in `plot` from the
            map scale.
        color : str, default "gray"
            Line colour.
        linewidth : float, default 0.5
            Line width.
        **kwargs
            Passed to `cartopy.mpl.geoaxes.GeoAxes.add_feature`.

        Returns
        -------
        Spatial
            This instance, for chaining.
        """
        return self._add_layer(_draw_borders, resolution=resolution, color=color, linewidth=linewidth, **kwargs)

    def add_image_tiles(self, tiler, *, zoom=3, interpolation="spline36", zorder=0, **kwargs):
        """
        Queue raster tiles from any cartopy tile source, as a background.

        Parameters
        ----------
        tiler : cartopy.io.img_tiles.GoogleWTS
            Tile source, such as ``cartopy.io.img_tiles.OSM()``.
        zoom : int, default 3
            Tile zoom level. Higher values give more detail but fetch many more
            tiles, so the default is low. Raise it when the extent is known.
        interpolation : str, default "spline36"
            Passed to `matplotlib.axes.Axes.imshow`.
        zorder : int, default 0
            Drawing order.
        **kwargs
            Passed to `cartopy.mpl.geoaxes.GeoAxes.add_image`.

        Returns
        -------
        Spatial
            This instance, for chaining.

        See Also
        --------
        add_land_osm : OpenStreetMap tiles with the source fixed.
        """
        return self._add_layer(
            _draw_image_tiles, tiler=tiler, zoom=zoom, interpolation=interpolation,
            zorder=zorder, **kwargs
        )

    def add_land_osm(self, *, zoom=3, interpolation="spline36", **kwargs):
        """
        Queue OpenStreetMap raster tiles, as a background.

        Calls `add_image_tiles` with ``cartopy.io.img_tiles.OSM()``.

        Parameters
        ----------
        zoom : int, default 3
            Tile zoom level. See `add_image_tiles`.
        interpolation : str, default "spline36"
            Passed to `matplotlib.axes.Axes.imshow`.
        **kwargs
            Passed to `add_image_tiles`.

        Returns
        -------
        Spatial
            This instance, for chaining.
        """
        import cartopy.io.img_tiles as cimgt
        return self.add_image_tiles(cimgt.OSM(), zoom=zoom, interpolation=interpolation, **kwargs)

    def add_gridlines(self, *, draw_labels=True, linewidth=0.5, color="gray", alpha=0.5,
                        linestyle="--", **kwargs):
        """
        Queue latitude and longitude gridlines.

        Parameters
        ----------
        draw_labels : bool, default True
            Draw tick labels on the map edges.
        linewidth : float, default 0.5
            Line width.
        color : str, default "gray"
            Line colour.
        alpha : float, default 0.5
            Line transparency.
        linestyle : str, default "--"
            Line style.
        **kwargs
            Passed to `cartopy.mpl.geoaxes.GeoAxes.gridlines`.

        Returns
        -------
        Spatial
            This instance, for chaining.
        """
        return self._add_layer(
            _draw_gridlines, draw_labels=draw_labels, linewidth=linewidth, color=color,
            alpha=alpha, linestyle=linestyle, **kwargs,
        )