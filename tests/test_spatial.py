import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM

from metocean_stats.spatial import Spatial

LON, LAT = np.meshgrid(np.linspace(0.0, 10.0, 12), np.linspace(58.0, 62.0, 8))
DATA = np.sin(np.radians(LON)) * np.cos(np.radians(LAT))
U = np.cos(np.radians(LAT))
V = np.sin(np.radians(LON))


def test_plot():
    Spatial().add_contourf(DATA, LON, LAT).plot()


def test_plot_on_axes():
    ax = plt.axes(projection=ccrs.PlateCarree())
    Spatial(extent=[0, 10, 58, 62]).add_contourf(DATA, LON, LAT).plot(ax=ax)


def test_legend_and_savefig(tmp_path):
    m = Spatial().add_points_of_interest(4.5, 60.4, "A").add_contourf(DATA, LON, LAT)
    m.plot()
    m.legend()
    m.savefig(str(tmp_path / "map.png"))


def test_add_quiver():
    Spatial().add_quiver(U, V, LON, LAT, subsample=2, reference_length=1.0, reference_units="m/s").plot()


def test_add_streamlines():
    Spatial().add_streamlines(U, V, LON, LAT).plot()


def test_add_contour():
    Spatial().add_contour(DATA, LON, LAT).plot()


def test_add_contourf():
    Spatial().add_contourf(DATA, LON, LAT, colorbar=True).plot()


def test_add_pcolormesh():
    Spatial().add_pcolormesh(DATA, LON, LAT).plot()


def test_add_bathymetry():
    Spatial().add_bathymetry(DATA, LON, LAT).plot()


def test_add_points_of_interest():
    Spatial().add_points_of_interest([4.5, 5.0], [60.4, 60.6], ["A", "B"]).plot()


def test_add_coastline():
    Spatial().add_contourf(DATA, LON, LAT).add_coastline().plot()


def test_add_land():
    Spatial().add_contourf(DATA, LON, LAT).add_land().plot()


def test_add_borders():
    Spatial().add_contourf(DATA, LON, LAT).add_borders().plot()


def test_add_image_tiles():
    assert isinstance(Spatial().add_image_tiles(OSM()), Spatial)


def test_add_land_osm():
    assert isinstance(Spatial().add_land_osm(), Spatial)


def test_add_gridlines():
    Spatial().add_contourf(DATA, LON, LAT).add_gridlines().plot()
