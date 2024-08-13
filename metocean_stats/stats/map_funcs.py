import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_points_on_map(lon, lat, label, bathymetry='NORA3',output_file='map.png'):
    lon_list = lon
    lat_list = lat
    label_list = label

    # Ensure lon_list and lat_list are iterable (lists or arrays)
    if not isinstance(lon_list, (list, tuple)):
        lon_list = [lon_list]
    if not isinstance(lat_list, (list, tuple)):
        lat_list = [lat_list]
    if not isinstance(label_list, (list, tuple)):
        label_list = [label_list]

    # Extract the data variables
    if bathymetry == 'NORA3':
        var = 'model_depth'
        file = 'https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_files/2022/12/20221231_MyWam3km_hindcast.nc'
        ds = xr.open_dataset(file)
        depth = ds[var].values
        rlat = ds['rlat'].values
        rlon = ds['rlon'].values

        # Get the rotated pole parameters from the dataset attributes
        rotated_pole = ccrs.RotatedPole(
            pole_latitude=ds.projection_ob_tran.grid_north_pole_latitude,
            pole_longitude=ds.projection_ob_tran.grid_north_pole_longitude
        )

        # Create meshgrid for lat/lon
        lon, lat = np.meshgrid(rlon, rlat)

        # Transform rotated coordinates to standard lat/lon coordinates
        proj = ccrs.PlateCarree()
        transformed_coords = proj.transform_points(rotated_pole, lon, lat)
        standard_lon = transformed_coords[..., 0]
        standard_lat = transformed_coords[..., 1]
    else:
        pass

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    # Plot the points with automatic labels
    for i, (lon, lat, label) in enumerate(zip(lon_list, lat_list, label_list)):
        ax.plot(lon, lat, marker='o', markersize=8, linewidth=0, label=f'{label}', transform=ccrs.PlateCarree())

    # Calculate zoom-in extent
    lat_min = max(min(lat_list) - 3, -90)  # Ensure latitude does not go below -90
    lat_max = min(max(lat_list) + 3, 90)   # Ensure latitude does not go above 90
    lon_min = max(min(lon_list) - 5, -180) # Ensure longitude does not go below -180
    lon_max = min(max(lon_list) + 5, 180)  # Ensure longitude does not go above 180

    # Set extent of the map
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())    
    # Add coastlines and other features
    # Add high-resolution coastlines with grey facecolor for land
    coast = cfeature.NaturalEarthFeature(
        category='physical', name='coastline', scale='10m', edgecolor='lightgrey', facecolor='darkkhaki')
    ax.add_feature(coast)

    # Plot depth
    if bathymetry == 'NORA3':
        # Mask depth data to the zoomed extent
        mask_extent = (
            (standard_lon >= lon_min) & (standard_lon <= lon_max) &
            (standard_lat >= lat_min) & (standard_lat <= lat_max)
        )
        depth_extent = np.ma.masked_where(~mask_extent, depth)

        # Plot depth with masked data
        cont = ax.contourf(standard_lon, standard_lat, depth_extent, levels=30,
                           cmap='binary', transform=ccrs.PlateCarree())

        # Add colorbar with limits based on the depth in the zoomed extent
        cbar = plt.colorbar(cont, orientation='vertical', pad=0.02, aspect=16, shrink=0.6)
        cbar.set_label('Depth [m]')
    else:
        pass
        
    # Add gridlines
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    # Add legend
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    return fig

def plot_extreme_wave_map(return_level=50, product='NORA3', title='empty title', set_extent = [0,30,52,73], output_file='extreme_wave_map.png'):
    if product == 'NORA3':
        lon = 'rlon'
        lat = 'rlat'
        var = 'hs'
        num_colors= 21
        file = f'https://thredds.met.no/thredds/dodsC/nora3_subset_stats/wave/CF_hs_{return_level}y_extreme_gumbel_NORA3.nc'    
    else:
        print(product,'is not available')
    
    ds = xr.open_dataset(file)
    # Extract the data variables
    hs = ds[var].values
    rlat = ds[lat].values
    rlon = ds[lon].values

    # Get the rotated pole parameters from the dataset attributes
    rotated_pole = ccrs.RotatedPole(
        pole_latitude=ds.projection_ob_tran.grid_north_pole_latitude,
        pole_longitude=ds.projection_ob_tran.grid_north_pole_longitude
    )

    # Create meshgrid for lat/lon
    lon, lat = np.meshgrid(rlon, rlat)

    # Transform rotated coordinates to standard lat/lon coordinates
    proj = ccrs.PlateCarree()
    transformed_coords = proj.transform_points(rotated_pole, lon, lat)
    standard_lon = transformed_coords[..., 0]
    standard_lat = transformed_coords[..., 1]

    # Flatten the arrays
    hs_flat = hs.flatten()
    lat_flat = standard_lat.flatten()
    lon_flat = standard_lon.flatten()

    # Create a mask for NaN values
    mask = ~np.isnan(hs_flat)
    hs_flat = hs_flat[mask]
    lat_flat = lat_flat[mask]
    lon_flat = lon_flat[mask]

    # Create a colormap with discrete colors
    num_colors = num_colors   # Number of discrete colors
    cmap = plt.cm.get_cmap('coolwarm', num_colors)
    cmap = ListedColormap(cmap(np.linspace(0, 1, num_colors)))

    # Create the plot
    fig = plt.figure(figsize=(9, 10))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=15))
    ax.coastlines(resolution='10m', zorder=3)
    ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)

    # Plot the significant wave height using hexbin with a higher gridsize and discrete colors
    hb = ax.hexbin(lon_flat, lat_flat, C=hs_flat, gridsize=180, cmap=cmap, vmin=0, vmax=21, edgecolors='white', transform=ccrs.PlateCarree(), reduce_C_function=np.mean, zorder=1)

    # Add the land feature on top of the hexbin plot
    ax.add_feature(cfeature.LAND, color='darkkhaki', zorder=2)

    # Add gridlines for latitude and longitude
    coast=cfeature.NaturalEarthFeature(category='physical', name='coastline', scale='10m', edgecolor='lightgrey', facecolor='darkkhaki')
    ax.add_feature(coast)

    ax.set_extent(set_extent, crs=ccrs.PlateCarree())
    gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,x_inline=False,y_inline=False,linewidth=0.5,color='black',alpha=1,linestyle='--')
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    gl.right_labels=False
    gl.left_labels=True
    gl.top_labels=False
    gl.bottom_labels=True
    gl.rotate_labels=False

    # Add colorbar with discrete colors
    cb = plt.colorbar(hb, ax=ax, orientation='vertical', shrink=0.7, pad=0.05, ticks=np.arange(0, 21, 2))
    cb.set_label(ds[var].attrs.get('standard_name', var)+' ['+ds[var].attrs.get('units', var)+']', fontsize=14)
    plt.tight_layout()

    if title is None:
        pass
    else:
        plt.title(title, fontsize=16)
    plt.savefig(output_file,dpi=300)
    return fig
     
def plot_extreme_wind_map(return_level=50, product='NORA3', level=0, title='empty title', set_extent=[0, 30, 52, 73],output_file='extreme_wind_map.png'):
    if product == 'NORA3':
        lon = 'x'
        lat = 'y'
        var = 'wind_speed'
        num_colors= 21
        file= f'https://thredds.met.no/thredds/dodsC/nora3_subset_stats/atm/CF_Return_Levels_3hourly_WindSpeed_6heights_1991_2020_zlev_period{return_level}yr.nc'
    else:
        print(product,'is not available')

    ds = xr.open_dataset(file)
    # Extract the data variables
    if isinstance(level, int):
         hs = ds[var][:,:].values
    else:
         hs = ds[var][level,:,:].values

    rlat = ds[lat].values
    rlon = ds[lon].values


    # Get the Lambert conformal parameters from the dataset attributes
    lambert_params = ds['projection_lambert'].attrs
    rotated_pole = ccrs.LambertConformal(
        central_longitude=lambert_params['longitude_of_central_meridian'],
        central_latitude=lambert_params['latitude_of_projection_origin'],
        standard_parallels=lambert_params.get('standard_parallel', None),
        globe=ccrs.Globe(ellipse='sphere', semimajor_axis=lambert_params.get('earth_radius', 6371000.0))
    )

    # Create meshgrid for lat/lon
    lon, lat = np.meshgrid(rlon, rlat)

    # Transform rotated coordinates to standard lat/lon coordinates
    proj = ccrs.PlateCarree()
    transformed_coords = proj.transform_points(rotated_pole, lon, lat)
    standard_lon = transformed_coords[..., 0]
    standard_lat = transformed_coords[..., 1]

    # Flatten the arrays
    hs_flat = hs.flatten()
    lat_flat = standard_lat.flatten()
    lon_flat = standard_lon.flatten()

    # Create a mask for NaN values
    mask = ~np.isnan(hs_flat)
    hs_flat = hs_flat[mask]
    # Remove values greater than 60
    hs_flat = hs_flat[hs_flat <= 60]
    #lat_flat = lat_flat[mask]
    #lon_flat = lon_flat[mask]

    # Create a colormap with discrete colors
    cmap = plt.cm.get_cmap('terrain', num_colors)
    cmap = ListedColormap(cmap(np.linspace(0, 1, num_colors)))

    # Create the plot
    fig = plt.figure(figsize=(9, 10))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=15))
    ax.coastlines(resolution='10m', zorder=3)
    ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)
    # Plot the significant wave height using hexbin with a higher gridsize and discrete colors
    hb = ax.hexbin(lon_flat, lat_flat, C=hs_flat, gridsize=100, cmap=cmap, vmin=hs_flat.min(), vmax=hs_flat.max(), edgecolors='white', transform=ccrs.PlateCarree(), reduce_C_function=np.mean, zorder=1)

    # Add the land feature on top of the hexbin plot
    ax.add_feature(cfeature.LAND, color='darkkhaki', zorder=2)

    # Add gridlines for latitude and longitude
    coast = cfeature.NaturalEarthFeature(category='physical', name='coastline', scale='10m', edgecolor='lightgrey', facecolor='darkkhaki')
    ax.add_feature(coast)

    ax.set_extent(set_extent, crs=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.5, color='black', alpha=1, linestyle='--')
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    gl.right_labels = False
    gl.left_labels = True
    gl.top_labels = False
    gl.bottom_labels = True
    gl.rotate_labels = False

    # Add colorbar with discrete colors
    cb = plt.colorbar(hb, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
    cb.set_label(ds[var].attrs.get('standard_name', var)+' ['+ds[var].attrs.get('units', var)+']', fontsize=14)
    plt.tight_layout()
    if title is not None:
        plt.title(title, fontsize=16)
    plt.savefig(output_file, dpi=300)
    return fig