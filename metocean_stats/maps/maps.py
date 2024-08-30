import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# Helper function to handle projection and coordinate transformation
def get_transformed_coordinates(ds, lon_var, lat_var, projection_type='rotated_pole'):
    """
    Get transformed coordinates and the appropriate Cartopy projection based on the dataset.

    Parameters:
    - ds: xarray.Dataset, the dataset to extract projection info from
    - lon_var: str, the longitude variable name in the dataset
    - lat_var: str, the latitude variable name in the dataset
    - projection_type: str, type of projection to use ('rotated_pole' or 'lambert_conformal')

    Returns:
    - transformed_lon: np.array, transformed longitudes
    - transformed_lat: np.array, transformed latitudes
    - proj: cartopy.crs.Projection, the projection object
    """
    lon = ds[lon_var].values
    lat = ds[lat_var].values
    lon, lat = np.meshgrid(lon, lat)

    if projection_type == 'rotated_pole':
        rotated_pole = ccrs.RotatedPole(
            pole_latitude=ds.projection_ob_tran.grid_north_pole_latitude,
            pole_longitude=ds.projection_ob_tran.grid_north_pole_longitude
        )
        proj = ccrs.PlateCarree()
        transformed_coords = proj.transform_points(rotated_pole, lon, lat)
    elif projection_type == 'lambert_conformal':
        lambert_params = ds['projection_lambert'].attrs
        lambert_conformal = ccrs.LambertConformal(
            central_longitude=lambert_params['longitude_of_central_meridian'],
            central_latitude=lambert_params['latitude_of_projection_origin'],
            standard_parallels=lambert_params.get('standard_parallel', None),
            globe=ccrs.Globe(ellipse='sphere', semimajor_axis=lambert_params.get('earth_radius', 6371000.0))
        )
        proj = ccrs.PlateCarree()
        transformed_coords = proj.transform_points(lambert_conformal, lon, lat)
    else:
        raise ValueError(f"Unknown projection type: {projection_type}")

    transformed_lon = transformed_coords[..., 0]
    transformed_lat = transformed_coords[..., 1]
    return transformed_lon, transformed_lat, proj

# Function to plot points on the map
def plot_points_on_map(lon, lat, label, bathymetry='NORA3', output_file='map.png'):
    lon_list = lon if isinstance(lon, (list, tuple)) else [lon]
    lat_list = lat if isinstance(lat, (list, tuple)) else [lat]
    label_list = label if isinstance(label, (list, tuple)) else [label]

    if bathymetry == 'NORA3':
        ds = xr.open_dataset('https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_files/2022/12/20221231_MyWam3km_hindcast.nc')
        standard_lon, standard_lat, _ = get_transformed_coordinates(ds, 'rlon', 'rlat', projection_type='rotated_pole')
        depth = ds['model_depth'].values
    else:
        pass

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    for lon, lat, label in zip(lon_list, lat_list, label_list):
        ax.plot(lon, lat, marker='o', markersize=8, linewidth=0, label=f'{label}', transform=ccrs.PlateCarree())

    lat_min, lat_max = max(min(lat_list) - 3, -90), min(max(lat_list) + 3, 90)
    lon_min, lon_max = max(min(lon_list) - 5, -180), min(max(lon_list) + 5, 180)

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    coast = cfeature.NaturalEarthFeature(category='physical', name='coastline', scale='10m', edgecolor='lightgrey', facecolor='darkkhaki')
    ax.add_feature(coast)

    if bathymetry == 'NORA3':
        mask_extent = (
            (standard_lon >= lon_min) & (standard_lon <= lon_max) &
            (standard_lat >= lat_min) & (standard_lat <= lat_max)
        )
        depth_extent = np.ma.masked_where(~mask_extent, depth)
        cont = ax.contourf(standard_lon, standard_lat, depth_extent, levels=30,
                           cmap='binary', transform=ccrs.PlateCarree())
        cbar = plt.colorbar(cont, orientation='vertical', pad=0.02, aspect=16, shrink=0.6)
        cbar.set_label('Depth [m]')
    else:
        pass

    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    return fig

# Function to plot extreme wave map
def plot_extreme_wave_map(return_period=50, product='NORA3', title='empty title', set_extent=[0, 30, 52, 73], output_file='extreme_wave_map.png'):
    if product == 'NORA3':
        ds = xr.open_dataset(f'https://thredds.met.no/thredds/dodsC/nora3_subset_stats/wave/CF_hs_{return_period}y_extreme_gumbel_NORA3.nc')
        standard_lon, standard_lat, _ = get_transformed_coordinates(ds, 'rlon', 'rlat', projection_type='rotated_pole')
        hs_flat = ds['hs'].values.flatten()
    else:
        print(product, 'is not available')
        return

    mask = ~np.isnan(hs_flat)
    hs_flat = hs_flat[mask]
    lon_flat = standard_lon.flatten()[mask]
    lat_flat = standard_lat.flatten()[mask]

    cmap = ListedColormap(plt.cm.get_cmap('coolwarm', 25)(np.linspace(0, 1, 25)))

    fig = plt.figure(figsize=(9, 10))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=15))
    ax.coastlines(resolution='10m', zorder=3)
    ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)

    hb = ax.hexbin(lon_flat, lat_flat, C=hs_flat, gridsize=180, cmap=cmap, vmin=0, vmax=25, edgecolors='white', transform=ccrs.PlateCarree(), reduce_C_function=np.mean, zorder=1)
    ax.add_feature(cfeature.LAND, color='darkkhaki', zorder=2)

    ax.set_extent(set_extent, crs=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.5, color='black', alpha=1, linestyle='--')
    gl.right_labels = False
    gl.top_labels = False
    gl.rotate_labels=False

    cb = plt.colorbar(hb, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
    cb.set_label(ds['hs'].attrs.get('standard_name', 'hs') + ' [' + ds['hs'].attrs.get('units', 'm') + ']', fontsize=14)
    plt.tight_layout()

    if title:
        plt.title(title, fontsize=16)
    plt.savefig(output_file, dpi=300)
    return fig

# Function to plot extreme wind map
def plot_extreme_wind_map(return_period=50, product='NORA3', z=0, title='empty title', set_extent=[0, 30, 52, 73], output_file='extreme_wind_map.png'):
    """
    Plots an extreme wind speed map based on the specified return level and dataset.

    Parameters:
    - return_period: int, optional, default=50
        The return period in years (e.g., 50 years) for the extreme wind speed.
    - product: str, optional, default='NORA3'
        The dataset to use for the wind speed data. Currently only 'NORA3' is supported.
    - title: str, optional, default='empty title'
        The title to display on the map.
    - set_extent: list, optional, default=[0, 30, 52, 73]
        The map extent specified as [lon_min, lon_max, lat_min, lat_max].
    - output_file: str, optional, default='extreme_wind_map.png'
        The filename to save the plot.

    Returns:
    - fig: matplotlib.figure.Figure
        The figure object containing the plot.
    """    
    if product == 'NORA3':
        ds = xr.open_dataset(f'https://thredds.met.no/thredds/dodsC/nora3_subset_stats/atm/CF_Return_Levels_3hourly_WindSpeed_6heights_1991_2020_zlev_period{return_period}yr.nc')
        standard_lon, standard_lat, _ = get_transformed_coordinates(ds, 'x', 'y', projection_type='lambert_conformal')
        hs_flat = ds['wind_speed'].sel(z=z).values.flatten()
    else:
        print(product, 'is not available')
        return

    mask = ~np.isnan(hs_flat)
    hs_flat = hs_flat[mask]
    lon_flat = standard_lon.flatten()[mask]
    lat_flat = standard_lat.flatten()[mask]

    cmap = ListedColormap(plt.cm.get_cmap('terrain', int(70/2.5))(np.linspace(0, 1, int(70/2.5))))

    fig = plt.figure(figsize=(9, 10))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=15))
    ax.coastlines(resolution='10m', zorder=3)
    ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)

    hb = ax.hexbin(lon_flat, lat_flat, C=hs_flat, gridsize=180, cmap=cmap, vmin=0, vmax=70, edgecolors='white', transform=ccrs.PlateCarree(), reduce_C_function=np.mean, zorder=1)
    ax.add_feature(cfeature.LAND, color='darkkhaki', zorder=2)

    ax.set_extent(set_extent, crs=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.5, color='black', alpha=1, linestyle='--')
    gl.right_labels = False
    gl.top_labels = False
    gl.rotate_labels=False

    cb = plt.colorbar(hb, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
    cb.set_label(ds['wind_speed'].attrs.get('standard_name', 'wind_speed') + ' [' + ds['wind_speed'].attrs.get('units', 'm/s') + ']', fontsize=14)
    plt.tight_layout()

    if title:
        plt.title(title, fontsize=16)
    plt.savefig(output_file, dpi=300)
    return fig

# Function to plot mean air temperature map
def plot_mean_air_temperature_map(product='NORA3', title='empty title', set_extent=[0, 30, 52, 73], unit='degC', mask_land=True, output_file='mean_air_temperature_map.png'):
    """
    Plots an extreme wind speed map based on the specified return level and dataset.

    Parameters:
    - product: str, optional, default='NORA3'
        The dataset to use for the wind speed data. Currently only 'NORA3' is supported.
    - title: str, optional, default='empty title'
        The title to display on the map.
    - set_extent: list, optional, default=[0, 30, 52, 73]
        The map extent specified as [lon_min, lon_max, lat_min, lat_max].
    - unit: str, optional, default='degC' (Celsius degrees), option 'K' for Kelvin
    - mask_land. bool, optional, default=True for masked land
    - output_file: str, optional, default='extreme_wind_map.png'
        The filename to save the plot.

    Returns:
    - fig: matplotlib.figure.Figure
        The figure object containing the plot.
    """    
    if product == 'NORA3':
        ds = xr.open_dataset(f'https://thredds.met.no/thredds/dodsC/nora3_subset_stats/atm/CF_Overall_Mean_AirTemperature2m_1991_2020.nc')
        #standard_lon, standard_lat, _ = get_transformed_coordinates(ds, 'x', 'y', projection_type='lambert_conformal')
        hs_flat = ds['air_temperature_2m'].isel(time=0).values.flatten()
        lon_flat = ds['longitude'].values.flatten()
        lat_flat = ds['latitude'].values.flatten()
    else:
        print(product, 'is not available')
        return

    #mask = ~np.isnan(hs_flat)
    #hs_flat = hs_flat[mask]
    #lon_flat = standard_lon.flatten()[mask]
    #lat_flat = standard_lat.flatten()[mask]
    if unit=='degC':
        vmin=-14
        vmax=14
    elif unit=='K':
        hs_flat=hs_flat+273.15
        vmin=260
        vmax=290
    else:
        print('Chosen unit not correct, should be K or degC')
        return

    cmap = ListedColormap(plt.cm.get_cmap('RdYlBu_r', int((vmax-vmin)/1))(np.linspace(0, 1, int((vmax-vmin)/1))))

    fig = plt.figure(figsize=(9, 10))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=set_extent[0]+(set_extent[1]-set_extent[0])/2))
    ax.coastlines(resolution='10m', zorder=3)
    if mask_land:
        ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)

    hb = ax.hexbin(lon_flat, lat_flat, C=hs_flat, gridsize=180, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors='white', transform=ccrs.PlateCarree(), reduce_C_function=np.mean, zorder=1)
    ax.add_feature(cfeature.LAND, color='darkkhaki', zorder=2)

    ax.set_extent(set_extent, crs=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.5, color='black', alpha=1, linestyle='--')
    gl.right_labels = False
    gl.top_labels = False
    gl.rotate_labels=False

    cb = plt.colorbar(hb, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
    cb.set_label(ds['air_temperature_2m'].attrs.get('standard_name', 'air_temperature') + ' [' + unit + ']', fontsize=14)
    plt.tight_layout()

    if title:
        plt.title(title, fontsize=16)
    plt.savefig(output_file, dpi=300)
    return fig
