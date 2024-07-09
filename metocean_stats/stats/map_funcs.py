import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

def plot_points_on_map(lon, lat, label, bathymetry='NORA3'):
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
        ax.plot(lon, lat, marker='o', markersize=8, label=f'{label}', transform=ccrs.PlateCarree())

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
        cont = ax.contourf(standard_lon, standard_lat, depth_extent, levels=100,
                           cmap='binary', transform=ccrs.PlateCarree())

        # Add colorbar with limits based on the depth in the zoomed extent
        cbar = plt.colorbar(cont, orientation='vertical', pad=0.02, aspect=16, shrink=0.8)
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
    plt.savefig('map_'+str(lon)+'_'+str(lat)+'.png')
    plt.close()
    return fig
