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
    # lon, lat = np.meshgrid(lon, lat)
    if lon.ndim == 1 and lat.ndim == 1:
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
def plot_points_on_map(lon:list[float]|float, 
                       lat:list[float]|float, 
                       label:list[str]|str, 
                       bathymetry:str='NORA3', 
                       output_file:str='map.png',
                       lon_lim:tuple|list=(),
                       lat_lim:tuple|list=()):
    '''
    Plot a set of longitude and latitude on a map, with bathymetry and land features. 
    
    Arguments
    ---------
    lon : float or list
        longitude(s) to plot
    lat : float or list
        latitude(s) to plot
    label : str or list of str
        names corresponding to points
    bathymetry : str
        Bathymetry source data. Currently, only NORA3 is supported.
    output_file : str
        Output image file name.
    lon_lim : list
        Longitude boundaries. Default: input data longitude +- 5.
    lat_lim : list
        Latitude boundaries. Default: input data latitude +- 3.
    
    Returns
    -------
    fig : matplotlib figure object
    '''
    
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

    # Set limits based on plotted points only if no lon/lat limits specified
    if lat_lim == ():
        lat_min, lat_max = max(min(lat_list) - 3, -90), min(max(lat_list) + 3, 90)
    else:
        lat_min, lat_max = lat_lim
    if lon_lim == ():
        lon_min, lon_max = max(min(lon_list) - 5, -180), min(max(lon_list) + 5, 180)
    else:
        lon_min, lon_max = lon_lim

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
    if output_file != "":
        plt.savefig(output_file)
    plt.close()
    return fig


# Function to plot several points and one main on a map wth LambertConformal projection
def plot_points_on_map_lc(lon,lat,label,bathymetry='NORA3',output_file='map.png',lon_lim=(),lat_lim=()):
    '''
    Plot a set of longitude and latitude on a map, with bathymetry and land features. 
    
    Arguments
    ---------
    lon : float or list
        longitude(s) to plot
    lat : float or list
        latitude(s) to plot
    label : str or list of str
        names corresponding to points
    bathymetry : str
        Bathymetry source data. Currently, only NORA3 is supported.
    output_file : str
        Output image file name.
    lon_lim : list
        Longitude boundaries. Default: input data longitude +- 5.
    lat_lim : list
        Latitude boundaries. Default: input data latitude +- 3.
    
    Returns
    -------
    fig : matplotlib figure object

    Modified version of function plot_points_on_map by clio-met
    '''
    
    lon_list = lon if isinstance(lon, (list, tuple)) else [lon]
    lat_list = lat if isinstance(lat, (list, tuple)) else [lat]
    label_list = label if isinstance(label, (list, tuple)) else [label]

    if bathymetry == 'NORA3':
        ds = xr.open_dataset('https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_files/2022/12/20221231_MyWam3km_hindcast.nc')
        standard_lon, standard_lat, _ = maps.get_transformed_coordinates(ds, 'rlon', 'rlat', projection_type='rotated_pole')
        depth = ds['model_depth'].values
    else:
        pass

    # Set limits based on plotted points only if no lon/lat limits specified
    if lat_lim == ():
        lat_min, lat_max = max(min(lat_list) - 3, -90), min(max(lat_list) + 3, 90)
    else:
        lat_min, lat_max = lat_lim
    if lon_lim == ():
        lon_min, lon_max = max(min(lon_list) - 5, -180), min(max(lon_list) + 5, 180)
    else:
        lon_min, lon_max = lon_lim

    # Central longitude
    c_l=lon_min+(lon_max-lon_min)/2

    # Markers customization
    nl=len(label)-1
    if nl>14:
        print('The number of points to plot should be < 14')
        return
    colors=['k']*nl+['r']
    markers_symb=['+','D','x','s','o','^','*','v','d','<','p','>','h']
    markers=markers_symb[0:nl]+['o']
    markerfacecolors=["None"]*nl+['r']
    markersize=[10]*nl+[8]

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=c_l))
    i=0
    for lon, lat, lab in zip(lon_list, lat_list, label_list):
        ax.plot(lon, lat, marker=markers[i], markersize=markersize[i], markerfacecolor=markerfacecolors[i], linewidth=0, markeredgecolor=colors[i], label=lab, transform=ccrs.PlateCarree())
        i=i+1

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

    gl = ax.gridlines(draw_labels=True,x_inline=False,y_inline=False)
    gl.right_labels=False
    gl.left_labels=True
    gl.top_labels=False
    gl.bottom_labels=True
    gl.rotate_labels=False
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}
    ax.legend(loc='upper left',fontsize='x-large')

    plt.tight_layout()
    if output_file != "":
        plt.savefig(output_file)
    plt.close()
    return fig


# Function to plot extreme wave map
def plot_extreme_wave_map(file ='CF_Return_Levels_3hourly_WindSpeed_6heights_1991_2020_zlev_period100yr.nc', product='NORA3', title='empty title', set_extent=[0, 30, 52, 73], output_file='extreme_wind_map.png', method='hexbin', cutoff_contour = 50):
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
    - method : str
        Visualization method for the data. Choose one of the following:
            - 'hexbin' : Displays the data using a hexagonal grid with color shading.
            - 'contour' : Displays the data using contour lines.
    - cutoff_contour : int or str  
            Percentile cutoff for drawing contours:
                - 50 plots the top 50% of values.
                - 66.6 plots the top 33.4%.
                - 'min' includes all values from the minimum.

    Returns:
    - fig: matplotlib.figure.Figure
        The figure object containing the plot.
    """    
    if product == 'NORA3':

        # ds = xr.open_dataset(f'https://thredds.met.no/thredds/dodsC/nora3_subset_stats/atm/CF_hs_{return_period}y_extreme_gumbel_NORA3.nc)
        # ds = xr.open_dataset(f'https://thredds.met.no/thredds/dodsC/nora3_subset_stats/atm/{file}')
        ds = xr.open_dataset(file)
        lon, lat, _ = get_transformed_coordinates(ds, 'rlon', 'rlat', projection_type= 'rotated_pole') 
        standard_lon, standard_lat = ds['rlon'].values, ds['rlat'].values 
        hs_flat = ds['hs'].values.flatten()
        
    else:
        print(product, 'is not available')
        return
    
    cmap = 'afmhot_r'

    mask = ~np.isnan(hs_flat)
    hs_flat = hs_flat[mask]
    lon_flat = lon.flatten()[mask]
    lat_flat = lat.flatten()[mask]

    fig = plt.figure(figsize=(9, 10))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=15))
    ax.coastlines(resolution='10m', zorder=3)
    ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)

    if method == 'hexbin':
        hb = ax.hexbin(lon_flat, lat_flat, C=hs_flat, gridsize=180, cmap='coolwarm', vmin=0, vmax=25, edgecolors='white', transform=ccrs.PlateCarree(), reduce_C_function=np.mean, zorder=1)
        cb = plt.colorbar(hb, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
        cb.set_label(ds['hs'].attrs.get('standard_name', 'hs') + ' [' + ds['hs'].attrs.get('units', 'm') + ']', fontsize=14)

    elif method == 'contour':
   
        hs = xr.DataArray(ds['hs'], dims=ds['hs'].dims, coords=ds['hs'].coords) # Convert back to xarray
        max_val = np.max(hs)
        min_val = np.min(hs)

        # Calculating dynamic step size for contour and contourf
        value_range = int(max_val - min_val)
        step_contour = int(value_range / 10)       # integer step for contours
        step_contourf = int(value_range / 20)       # integer step for contoursf (colormap)
           
        # Define the cutoff percentile for contour visualization
        # cutoff_contour = 50     # Display contours for the top 50% of values; use 'min' to start from the lowest values
    
        # Sets the starting threshold for contour lines and labels — controls whether all levels or only the highest levels are plotted
        start_contour = int(np.nanmin(hs) if cutoff_contour == 'min' else np.nanpercentile(hs, cutoff_contour)) # Selects min value or percentile based on choosen parameter

        # Ensure starting threshold is an even number by subtracting/adding 1 if it's odd
        start_contour = start_contour - 1 if start_contour % 2 != 0 else start_contour 
        step_contour = step_contour + 1 if step_contour % 2 != 0 else step_contour 
        start_contourf = int(np.nanmin(hs)) - 1 if int(np.nanmin(hs)) % 2 != 0 else int(np.nanmin(hs))

        # Generate contour and label levels starting from their respective start values,
        # spaced by the step sizes, and extended to ensure the maximum data value is included.
        levels_contour = np.arange(start_contour, int(np.max(hs)+step_contour*2), step_contour)
        levels_contourf = np.arange(start_contourf,int(np.max(hs)+step_contourf*2),step_contourf)

        print('Levels contour:', levels_contour)
        print('Levels contourf:', levels_contourf)

        rotated_pole = ccrs.RotatedPole(
            pole_latitude=ds.projection_ob_tran.grid_north_pole_latitude,
            pole_longitude=ds.projection_ob_tran.grid_north_pole_longitude
        )   

        cm = plt.contourf(standard_lon, standard_lat, hs,levels=levels_contourf, transform=rotated_pole,cmap=cmap, alpha = 0.6, zorder=1, vmin =0, vmax =int(max_val))
        cb = plt.colorbar(cm, label=ds['hs'].attrs.get('standard_name', 'hs') + ' [' + ds['hs'].attrs.get('units', 'm') + ']', ax=ax, orientation='vertical', shrink=0.7, pad=0.05)

        cs = ax.contour(standard_lon, standard_lat, hs, levels=levels_contour, transform=rotated_pole,cmap=cmap, zorder=1, vmin = 0, vmax = int(max_val))
        ax.clabel(cs, levels = levels_contour, inline=False, fontsize=16, fmt='%d', colors='black')
    ax.set_extent(set_extent, crs=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.5, color='black', alpha=1, linestyle='--')
    gl.right_labels = False
    gl.top_labels = False
    gl.rotate_labels=False

    plt.tight_layout()

    if title:
        plt.title(title, fontsize=16)
    plt.savefig(output_file, dpi=300)
    return fig


# Function to plot extreme wind map
def plot_extreme_wind_map(return_period=50, file ='CF_Return_Levels_3hourly_WindSpeed_6heights_1991_2020_zlev_period100yr.nc', product='NORA3', z=0, title='empty title', set_extent=[0, 30, 52, 73], output_file='extreme_wind_map.png', method='hexbin', cutoff_contour=50):
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
    - method : str
        Visualization method for the data. Choose one of the following:
            - 'hexbin' : Displays the data using a hexagonal grid with color shading.
            - 'contour' : Displays the data using contour lines.
    - cmap : str
        The colormap indicating wind speeds.


    Returns:
    - fig: matplotlib.figure.Figure
        The figure object containing the plot.
    """    

    # Added package
    from scipy.ndimage import gaussian_filter

    if product == 'NORA3':
        x = 'x'
        y = 'y'
        # ds = xr.open_dataset(f'https://thredds.met.no/thredds/dodsC/nora3_subset_stats/atm/CF_hs_{return_period}y_extreme_gumbel_NORA3.nc)
        # ds = xr.open_dataset(f'https://thredds.met.no/thredds/dodsC/nora3_subset_stats/atm/{file}')
        ds = xr.open_dataset(file)
        # ds = xr.open_dataset('CF_Return_Levels_3hourly_WindSpeed_6heights_1991_2020_zlev_period100yr.nc')
        standard_lon, standard_lat, _ = get_transformed_coordinates(ds, x, y, projection_type='lambert_conformal')
        hs_flat = ds['wind_speed'].sel(z=z).values.flatten()
        # hs_flat = ds['wind_speed'][0].values.flatten()
    else:
        print(product, 'is not available')
        return

    mask = ~np.isnan(hs_flat)
    hs_flat = hs_flat[mask]
    lon_flat = standard_lon.flatten()[mask]
    lat_flat = standard_lat.flatten()[mask]

    hs = ds['wind_speed'].sel(z=z)

    # Filtering the data with NaN handling
    nan_mask = np.isnan(hs)
    hs_filled = np.copy(hs)
    hs_filled[nan_mask] = 0

    weights = (~nan_mask).astype(float)  # Create weights ( 1 where data is valid, 0 where NaN)
    
    smoothed_hs = gaussian_filter(hs_filled, sigma = 1.5)  # Gaussian filter to data and weights
    smoothed_weights = gaussian_filter(weights, sigma=1.5)
    
    result = smoothed_hs/smoothed_weights  # Normalize to ignore NaNs in the average
    result[smoothed_weights==0] = np.nan   # Reassign NaNs where no weight

    hs = xr.DataArray(result, dims=hs.dims, coords=hs.coords) # Convert back to xarray

    # Finding max value in specified area
    lon_min, lon_max, lat_min, lat_max = set_extent

    mask = (
            (standard_lon >= lon_min) & (standard_lon <= lon_max) &
            (standard_lat >= lat_min) & (standard_lat <= lat_max)
        )
    
    hs_area = hs.where(mask)

    max_val = np.round(hs_area.max().item())
    min_val = np.round(hs_area.min().item())

    # Set up map with coastlines and country borders
    fig = plt.figure(figsize=(9, 10))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=15))
    ax.coastlines(resolution='10m', zorder=3)
    ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)
    cmap='afmhot_r'

    if method == 'hexbin':
        hb = ax.hexbin(lon_flat, lat_flat, C=hs_flat, gridsize=180, cmap=cmap, vmin=min_val, vmax=max_val, edgecolors='white', transform=ccrs.PlateCarree(), reduce_C_function=np.mean, zorder=1)
        cb = plt.colorbar(hb, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
        cb.set_label(ds['wind_speed'].attrs.get('standard_name', 'wind speed') + ' [' + ds['wind_speed'].attrs.get('units', 'm/s') + ']', fontsize=14)

    elif method == 'contour':
       
        # Calculating dynamic step size for contour and contourf
        value_range = int(max_val - min_val)
        step_contour = np.max([int(value_range / 10), 4])       # integer step for contours
        step_contourf = np.max([int(value_range / 20), 2])       # integer step for contoursf (colormap)
           
        # Define the cutoff percentile for contour visualization
        # cutoff_contour = 50     # Display contours for the top 50% of values; use 'min' to start from the lowest values
    
        # Sets the starting threshold for contour lines and labels — controls whether all levels or only the highest levels are plotted
        start_contour = int(np.nanmin(hs) if cutoff_contour == 'min' else np.nanpercentile(hs, cutoff_contour)) # Selects min value or percentile based on choosen parameter

        # Ensure starting threshold is an even number by subtracting/adding 1 if it's odd
        start_contour = start_contour - 1 if start_contour % 2 != 0 else start_contour 
        step_contour = step_contour + 1 if step_contour % 2 != 0 else step_contour 
        start_contourf = int(np.nanmin(hs)) - 1 if int(np.nanmin(hs)) % 2 != 0 else int(np.nanmin(hs))

        # Generate contour and label levels starting from their respective start values,
        # spaced by the step sizes, and extended to ensure the maximum data value is included.
        levels_contour = np.arange(start_contour, int(np.max(hs)+step_contour*2), step_contour)
        levels_contourf = np.arange(start_contourf,int(np.max(hs)+step_contourf*2),step_contourf)

        # Removes values that are over the maximum
        max_level_contour = max_val if max_val in levels_contour else levels_contour[np.searchsorted(levels_contour, max_val, side='right')]
        max_level_contourf = max_val if max_val in levels_contourf else levels_contourf[np.searchsorted(levels_contourf, max_val, side='right')]
        levels_contour = levels_contour[levels_contour <= max_level_contour]
        levels_contourf = levels_contourf[levels_contourf <= max_level_contourf]
        print('Levels contour:', levels_contour)
        print('Levels contourf:', levels_contourf)

        cm = plt.contourf(standard_lon, standard_lat, hs,levels=levels_contourf, transform=ccrs.PlateCarree(),cmap=cmap, alpha = 0.6, zorder=1, vmin =0, vmax =int(max_val))
        cb = plt.colorbar(cm, label=ds['wind_speed'].attrs.get('standard_name', 'wind speed') + ' [' + ds['wind_speed'].attrs.get('units', 'm') + ']', ax=ax, orientation='vertical', shrink=0.7, pad=0.05)

        cs = ax.contour(standard_lon, standard_lat, hs, levels=levels_contour, transform=ccrs.PlateCarree(),cmap=cmap, zorder=1, vmin = 0, vmax = int(max_val))
        ax.clabel(cs, levels = levels_contour, inline=False, fontsize=16, fmt='%d', colors='white')

    ax.set_extent(set_extent, crs=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.5, color='black', alpha=1, linestyle='--')
    gl.right_labels = False
    gl.top_labels = False
    gl.rotate_labels=False

    plt.tight_layout()

    if title:
        plt.title(title, fontsize=16)
    plt.savefig(output_file, dpi=300)
    return fig

# Function to plot extreme current map
def plot_extreme_current_map(file ='CF_Return_Levels_3hourly_WindSpeed_6heights_1991_2020_zlev_period100yr.nc', product='NORA3',z='surface', title='empty title', set_extent=[0, 30, 52, 73], output_file='extreme_wind_map.png', method='hexbin', cmap='Reds'):
    '''
    Plots an extreme current speed map based on the specified return level and dataset.

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
    - method : str
        Visualization method for the data. Choose one of the following:
            - 'hexbin' : Displays the data using a hexagonal grid with color shading.
            - 'contour' : Displays the data using contour lines.
    - cmap : str
        The colormap indicating wind speeds.


    Returns:
    - fig: matplotlib.figure.Figure
        The figure object containing the plot.
    '''

    # Added package
    from scipy.ndimage import gaussian_filter

    if product == 'NORA3':
        # ds = xr.open_dataset(f'https://thredds.met.no/thredds/dodsC/nora3_subset_stats/atm/CF_hs_{return_period}y_extreme_gumbel_NORA3.nc)
        # ds = xr.open_dataset(f'https://thredds.met.no/thredds/dodsC/nora3_subset_stats/atm/{file}')
        ds = xr.open_dataset(file)
        if z=='surface':
            parameter = 'genextreme_surface_100year'
        elif z=='seafloor':
             parameter = 'genextreme_seafloor_100year'           
        hs_flat = ds[parameter].values.flatten()
        
    else:
        print(product, 'is not available')
        return

    mask = ~np.isnan(hs_flat)
    hs_flat = hs_flat[mask]
    lon_flat = ds['lon'].values.flatten()[mask]
    lat_flat = ds['lat'].values.flatten()[mask]

    fig = plt.figure(figsize=(9, 10))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=15))
    ax.coastlines(resolution='10m', zorder=3)
    ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)

    if method == 'hexbin':
        hb = ax.hexbin(lon_flat, lat_flat, C=hs_flat, gridsize=180, cmap=cmap, vmin=int(np.nanmin(hs_flat)), vmax=int(np.nanmax(hs_flat)), edgecolors='white', transform=ccrs.PlateCarree(), reduce_C_function=np.mean, zorder=1)
        ax.add_feature(cfeature.LAND, color='darkkhaki', zorder=2)
        cb = plt.colorbar(hb, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
        cb.set_label(ds[parameter].attrs.get('standard_name', 'wind speed') + ' [' + ds[parameter].attrs.get('units', 'm/s') + ']', fontsize=14)

    elif method == 'contour':
        
        hs = xr.DataArray(ds[parameter])
        standard_lon = xr.DataArray(ds['lon'])
        standard_lat = xr.DataArray(ds['lat'])

        # Filtering the data with NaN handling
        nan_mask = np.isnan(hs)
        hs_filled = np.copy(hs)
        hs_filled[nan_mask] = 0

        weights = (~nan_mask).astype(float)  # Create weights ( 1 where data is valid, 0 where NaN)
        
        smoothed_hs = gaussian_filter(hs_filled, sigma = 1.5)  # Gaussian filter to data and weights
        smoothed_weights = gaussian_filter(weights, sigma=1.5)
        
        result = smoothed_hs/smoothed_weights  # Normalize to ignore NaNs in the average
        result[smoothed_weights==0] = np.nan   # Reassign NaNs where no weight

        hs = xr.DataArray(result, dims=hs.dims, coords=hs.coords) # Convert back to xarray

        print('max:', round(np.nanmax(hs),2))
        print('min', round(np.nanmin(hs)),2)

        # Calculating dynamic step size for contour and contourf
        value_range = round(np.nanmax(hs) - np.nanmin(hs),2)
        step_contour = round(value_range / 10,1) or 0.2               # integer step for contours
        step_contourf = 0.05
           
        # Define the cutoff percentile for contour visualization
        cutoff_contour = 50     # Display contours for the top 50% of values; use 'min' to start from the lowest values
    
        # Sets the starting threshold for contour lines and labels — controls whether all levels or only the highest levels are plotted
        start_contour = round(np.nanmin(hs) if cutoff_contour == 'min' else np.nanpercentile(hs, cutoff_contour),1) # Selects min value or percentile based on choosen parameter
        start_contourf = np.min(hs)
        
        # Generate contour and label levels starting from their respective start values,
        # spaced by the step sizes, and extended to ensure the maximum data value is included.
        max_val = np.nanmax(hs)
        levels_contour = np.round(np.arange(start_contour, max_val + step_contour, step_contour),1)
        levels_contourf = np.round(np.arange(start_contourf, max_val + step_contourf, step_contourf),2)

        # Removes values that are over the maximum
        max_level_contour = max_val if max_val in levels_contour else levels_contour[np.searchsorted(levels_contour, max_val, side='right')]
        max_level_contourf = max_val if max_val in levels_contourf else levels_contourf[np.searchsorted(levels_contourf, max_val, side='right')]
        levels_contour = levels_contour[levels_contour <= max_level_contour]
        levels_contourf = levels_contourf[levels_contourf <= max_level_contourf]
        print('Levels contour:', levels_contour)
        print('Levels contourf:', levels_contourf)
        print()

        cm = plt.contourf(standard_lon, standard_lat, hs,levels=levels_contourf, transform=ccrs.PlateCarree(),cmap=cmap, alpha = 0.6, zorder=1, vmin =0, vmax =max_val)
        cb = plt.colorbar(cm, label=ds[parameter].attrs.get('standard_name', 'Current speed') + ' [' + ds[parameter].attrs.get('units', 'm/s') + ']', ax=ax, orientation='vertical', shrink=0.7, pad=0.05)

        cs = ax.contour(standard_lon, standard_lat, hs, levels=levels_contour, transform=ccrs.PlateCarree(),cmap=cmap, zorder=1, vmin = int(np.nanmin(hs)), vmax = max_val)
        ax.clabel(cs, levels = levels_contour, inline=False, fontsize=16, fmt='%.1f', colors='black')

    ax.set_extent(set_extent, crs=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.5, color='black', alpha=1, linestyle='--')
    gl.right_labels = False
    gl.top_labels = False
    gl.rotate_labels=False

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
        ds = xr.open_dataset('https://thredds.met.no/thredds/dodsC/nora3_subset_stats/atm/CF_Overall_Mean_AirTemperature2m_1991_2020.nc')
        ## The code `standard_lon` appears to be a variable name in Python. It is not assigned any
        # value or operation in the provided snippet, so it is not doing anything specific in this
        # context.
        standard_lon, standard_lat, _ = get_transformed_coordinates(ds, 'x', 'y', projection_type='lambert_conformal')
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


#####

import re
from matplotlib import colors as mcolors
#############--------------------Extract which depths we have data from in the NORKYST800-----------------#########################
def extract_unique_depths_from_NORKYST800(df):
    # Get column names from the df
    column_names = df.columns
    depths = set()
    
    # Extract the different depths 
    for col in column_names:
        # Check if the column contain a depth in the name (for example "temp_0m")
        match = re.search(r'(\d+\.?\d*)m$', col)  # Find the number after the underscore and before "m"
        if match:
            depth = match.group(0)  # Find the depth (example: '0m', '2.5m', osv.)
            depths.add(depth)
    depths_sorted = sorted(depths, key=lambda x: float(x.replace('m', '')))
    # Return a list of unique values of depths in the given NORKYST800.
    return depths_sorted
###########--------------------------Get the wanted varible out of the NORKYST800 dataset--------------------#######################
def get_a_chosen_variable_from_NORKYST800(df,salt,temp,u,v,magnitude):
    """
    - Variable will return in the shape (depth x time) 
    """
    df = pd.read_csv(df,comment="#",index_col=0,parse_dates=True)
    depths = extract_unique_depths_from_NORKYST800(df)
    num_depths = len(depths)
    num_timesteps = len(df)
    
    if salt == True:
        salinity = np.zeros((num_depths,num_timesteps))

        for i, d in enumerate(depths):
            salt = df[f'salinity_{d}'].fillna(0).values
            salinity[i,:] = salt
        unit="ppt"
        return salinity, [float(d.replace('m','')) for d in depths], unit

    if temp == True:
        temperature = np.zeros((num_depths,num_timesteps))

        for i, d in enumerate(depths):
            temp = df[f'temperature_{d}'].fillna(0).values
            temperature[i,:] = temp
        unit="°c/celsius"
        return temperature, [float(d.replace('m','')) for d in depths], unit

    if u == True:
        u_speeds = np.zeros((num_depths, num_timesteps))
    
        for i, d in enumerate(depths):
            u = df[f'v_{d}'].fillna(0).values
            u_speeds[i, :] = v
        unit="m/s"
        return u_speeds, [float(d.replace('m', '')) for d in depths], unit

    if v == True:
        v_speeds = np.zeros((num_depths, num_timesteps))
    
        for i, d in enumerate(depths):
            v = df[f'v_{d}'].fillna(0).values
            v_speeds[i, :] = v
        unit="m/s"
        return v_speeds, [float(d.replace('m', '')) for d in depths], unit
        
    if magnitude == True:
        current_speeds = np.zeros((num_depths, num_timesteps))
        
        for i, d in enumerate(depths):
            # Beregn strømstyrke √(u² + v²)
            u = df[f'u_{d}'].fillna(0).values
            v = df[f'v_{d}'].fillna(0).values
            current_speeds[i, :] = np.sqrt(u**2 + v**2)
        unit="m/s"
        return current_speeds, [float(d.replace('m', '')) for d in depths], unit

    else:
        print('none variables were chosen')

##############------------------------Choose what crossection you want----------------##############################
def find_rounded_lat_lon_index_from_NORKYST800_url(lat_array, lon_array, lat, lon, decimals=2, tolerance=0.005):
    lat_array_rounded = np.round(lat_array, decimals)
    lon_array_rounded = np.round(lon_array, decimals)

    # Find the index where both lat and lon are within the tolerance of the desired value
    match = np.where((np.abs(lat_array_rounded - lat) <= tolerance) & (np.abs(lon_array_rounded - lon) <= tolerance))
    
    # If a match return that index
    if match[0].size > 0:
        y1, x1 = match[0][0], match[1][0]  # Take the first match of the given values
        return [int(y1), int(x1)]
    else:
        return print(f"Try new value for the {lat} latitude and {lon} longitude point") 

    

def bathymetry_cross_section_in_point_longitude_with_NORKYST800_url(url, lat, lon_start,lon_end):
    dataset = xr.open_dataset(url)
    bathymetry_array = dataset['h'].values
    lat_array = dataset['lat'].values  # Extract 2D array from dataset['lat']
    lon_array = dataset['lon'].values  # Extract 2D array from dataset['lon']
    
    start = find_rounded_lat_lon_index_from_NORKYST800_url(lat_array, lon_array, lat, lon_start, decimals=1, tolerance=0.005)
    end = find_rounded_lat_lon_index_from_NORKYST800_url(lat_array, lon_array, lat, lon_end, decimals=1, tolerance=0.005)
    start = np.array(start)
    end = np.array(end)
    diff = end - start
    step=np.array([int(diff[0]/diff[1]),1]) #steps may vary between -2,-3
    coordinates = []
    current = start.copy()
        
    # add the initalpoint
    coordinates.append(current.tolist())

    for i in range (1,end[1]-start[1]+1):
        current += step
        coordinates.append(current.tolist())
        
    # Konverter listen til en numpy array (hvis ønskelig)
    coordinates_array = np.array(coordinates)
    bath_data = []
    for i in range(len(coordinates_array)):
        y, x = coordinates_array[i]
        bath_data.append(bathymetry_array[y, x])

    
    return bath_data, coordinates_array

def bathymetry_cross_section_in_point_latitude_with_NORKYST800_url(url,lat_start, lat_end ,lon):
    dataset = xr.open_dataset(url)
    bathymetry_array = dataset['h'].values
    lat_array = dataset['lat'].values  # Extract 2D array from dataset['lat']
    lon_array = dataset['lon'].values  # Extract 2D array from dataset['lon']
    start = find_rounded_lat_lon_index_from_NORKYST800_url(lat_array, lon_array, lat_start, lon, decimals=1, tolerance=0.005)
    end = find_rounded_lat_lon_index_from_NORKYST800_url(lat_array, lon_array, lat_end, lon, decimals=1, tolerance=0.005)
    start = np.array(start)
    end = np.array(end)
    diff = end-start
    step = np.array([1,int(diff[1]/diff[0])])
    coordinates = []
    current = start.copy()

    #add the initialpoint
    coordinates.append(current.tolist())

    for i in range (1,end[0]-start[0]+1):
        current += step
        coordinates.append(current.tolist())
        
    # Konverter listen til en numpy array (hvis ønskelig)
    coordinates_array = np.array(coordinates)
    bath_data = []
    for i in range(len(coordinates_array)):
        y, x = coordinates_array[i]
        bath_data.append(bathymetry_array[y, x])

    return bath_data, coordinates_array

################---------------------PLOTS---------------------##########################
def plot_bathymetry_cross_section(product: str = "NORKYST800",lon: float= 4,lat:float= 60.5 ,delta_lon: float = 1.0 ,delta_lat: float = 0.5):
    """
    - product = string, optional, default = "NORKYST800"
    - lon = longitude to the point, default set to 4 degrees longitude
    - lat = latitude to the point default set to 60.5 degrees latitude
    - delta_lon = +- value, standard set to 1.0
    - delta_lat = +- value, standard set to 0.5
    """
    if product == "NORKYST800":
        url = 'https://thredds.met.no/thredds/dodsC/fou-hi/norkyst800m-1h/NorKyst-800m_ZDEPTHS_his.an.2025020700.nc'
    else:
        pass
    lat_start = lat - delta_lat
    lat_end = lat + delta_lat

    lon_start = lon - delta_lon
    lon_end = lon + delta_lon
    
    bath_lon,_ = bathymetry_cross_section_in_point_longitude_with_NORKYST800_url(url,lat,lon_start,lon_end)
    bath_lat,_ = bathymetry_cross_section_in_point_latitude_with_NORKYST800_url(url,lat_start,lat_end,lon)

    fig, ax1 = plt.subplots(2,1,figsize=(12,6))
    # Distance between the plots
    plt.subplots_adjust(hspace=0.5)
        
    # Calculate values on the x-axis and finding the maximum depth
    lon_bath = np.linspace(lon_start, lon_end, len(bath_lon))
    max_bath_depth_1 = np.max(bath_lon)

    lat_bath = np.linspace(lat_start, lat_end, len(bath_lat))
    max_bath_depth_2 = np.max(bath_lat)

    #First subplot for longitude:

    ax2_0 = ax1[0].twinx()
    ax1[0].plot(lon_bath, bath_lon, color='black', linewidth=3, label="Bathymetry")
    ax1[0].axvline(x=lon, color='red', linestyle='--', linewidth=2, label=f'Linje ved lon={lon}') #Line indicating the point you choose

    ax1[0].set_xlabel("Longitude (°)")
    ax1[0].set_ylabel("Depth (m)", color='black')
    ax1[0].set_title("Bathymetry along the longitude")
    ax1[0].set_ylim(max_bath_depth_1, 0)
    ax2_0.set_ylim(max_bath_depth_1, 0)
    ax1[0].set_xlim(lon_start,lon_end)
    ax1[0].legend(loc='upper right')
    ax1[0].grid(True)

    #Second subplot for latitude:

    ax2_1 = ax1[1].twinx()
    ax1[1].plot(lat_bath, bath_lat, color='black', linewidth=3, label="Bathymetry")
    ax1[1].axvline(x=lat, color='red', linestyle='--', linewidth=2, label=f'Line at lat={lat}') #Line indicating the point you choose
    
    ax1[1].set_xlabel("Latitude (°)")
    ax1[1].set_ylabel("Depth (m)", color='black')
    ax1[1].set_title("Bathymetry along the latitude")
    ax1[1].set_ylim(max_bath_depth_2, 0)
    ax2_1.set_ylim(max_bath_depth_2, 0)
    ax1[1].set_xlim(lat_start,lat_end)
    ax1[1].legend(loc='upper right')
    ax1[1].grid(True)

    ## Save and show the plot
    plt.savefig("Bathymetry_cross_section", dpi=100, facecolor='white', bbox_inches="tight")
    return fig 

def plot_bathymetry_cross_section_with_variable_from_NORKYST800_overall(product: str = "NORKYST800", df: str = "NORKYST800", lon: float=4, lat: float = 60.5, var: str = "temperature", delta_lon:float =1, delta_lat:float=0.5):
    """
    product = string, optional, default = "NORKYST800"
    df = string, csv file from where the variable is extracted, default = "NORKYST800"
    lon= longitude to the point, default set to 4 degrees longitude
    lat= latitude to the point default set to 60.5 degrees latitude
    delta_lon= +- value, standard set to 1.0
    delta_lat= +- value, standard set to 0.5
    var = a chosen variable between 'magnitude'.'v','u','temperature' and 'salinity'
    """
    if product == "NORKYST800":
        url ='https://thredds.met.no/thredds/dodsC/fou-hi/norkyst800m-1h/NorKyst-800m_ZDEPTHS_his.an.2025020700.nc'
    else:
        pass

    if df == "NORKYST800":
        df = "NORKYST800_test.csv"
    else:
        pass

    if var == "magnitude":
        variable, depths,unit= get_a_chosen_variable_from_NORKYST800(df,salt=False,temp=False,u=False,v=False,magnitude=True)
    if var == "v":
        variable, depths, unit= get_a_chosen_variable_from_NORKYST800(df,salt=False,temp=False,u=False,v=True,magnitude=False)
    if var == "u":
        variable, depths, unit =get_a_chosen_variable_from_NORKYST800(df,salt=False,temp=False,u=True,v=False,magnitude=False)
    if var == "temperature":
        variable, depths, unit =get_a_chosen_variable_from_NORKYST800(df,salt=False,temp=True,u=False,v=False,magnitude=False)
    if var == "salinity":
        variable, depths, unit =get_a_chosen_variable_from_NORKYST800(df,salt=True,temp=False,u=False,v=False,magnitude=False)
    else:
        return "no valid variable, try either: 'magnitude'.'v','u','temperature' or 'salinity'."        
    
    lat_start = lat - delta_lat
    lat_end = lat + delta_lat

    lon_start = lon - delta_lon
    lon_end = lon + delta_lon
    
    bath_lon,_ = bathymetry_cross_section_in_point_longitude_with_NORKYST800_url(url,lat,lon_start,lon_end)
    bath_lat,_ = bathymetry_cross_section_in_point_latitude_with_NORKYST800_url(url,lat_start,lat_end,lon)
    
    variable = np.mean(variable.copy(), axis=1, keepdims=True) 
    fig, ax1 = plt.subplots(2,1,figsize=(12,6))
    
    # Distance between the plots
    plt.subplots_adjust(hspace=0.5)
        
    # Calculate values on the x-axis and finding the maximum depth
    lon_bath = np.linspace(lon_start, lon_end, len(bath_lon))
    max_bath_depth_1 = max(bath_lon)

    lat_bath = np.linspace(lat_start, lat_end, len(bath_lat))
    max_bath_depth_2 = max(bath_lat)

    ax2_0 = ax1[0].twinx()
    
    #First subplot for longitude:
    Lon, Depth = np.meshgrid(lon, depths)
    ### Variable plot ###
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=np.min(variable), vmax=np.max(variable))
    cf = ax2_0.imshow(variable, cmap=cmap, norm=norm, aspect='auto', extent=[lon_start, lon_end, max_bath_depth_1, 0], alpha=0.5)
    cf.set_zorder(0)                # Make sure that it will be made before the bathymetry line
    cbar = plt.colorbar(cf,ax=ax2_0, label=f"{var} ({unit})")
    
    ### Bathymetry plot ### 
    ax1[0].plot(lon_bath, bath_lon, color='black', linewidth=3, label="Bathymetry",zorder=3)
    ax1[0].axvline(x=lon, color='red', linestyle='--', linewidth=2, label=f'Line at lon={lon}') #Line indicating the point you choose

    ### Settings ### 
    ax1[0].set_xlabel("Longitude (°)")
    ax1[0].set_ylabel("Depth (m)", color='black')
    ax1[0].set_title("Bathymetry along the longitude")
    ax1[0].set_ylim(max_bath_depth_1, 0)
    ax2_0.set_ylim(max_bath_depth_1, 0)
    ax1[0].set_xlim(lon_start,lon_end)
    ax1[0].legend(loc='upper right')
    ax1[0].grid(True)

    ax2_1 = ax1[1].twinx()
    #Second subplot for latitude:
    Lat, Depth = np.meshgrid(lat, depths)
    
    ### Variable plot ###
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=np.min(variable), vmax=np.max(variable))
    cf = ax2_1.imshow(variable, cmap=cmap, norm=norm, aspect='auto', extent=[lat_start, lat_end, max_bath_depth_1, 0], alpha=0.5)
    cf.set_zorder(1)                # Make sure that it will be made before the bathymetry line
    cbar = plt.colorbar(cf,ax=ax2_1, label=f"{var} ({unit})")

    ### Bathymetry plot ### 
    ax1[1].plot(lat_bath, bath_lat, color='black', linewidth=3, label="Bathymetry",zorder=4)
    ax1[1].axvline(x=lat, color='red', linestyle='--', linewidth=2, label=f'Line at lat={lat}') #Line indicating the point you choose

    ### Settings ### 
    ax1[1].set_xlabel("Latitude (°)")
    ax1[1].set_ylabel("Depth (m)", color='black')
    ax1[1].set_title("Bathymetry along the latitude")
    ax1[1].set_ylim(max_bath_depth_2, 0)
    ax2_1.set_ylim(max_bath_depth_2, 0)
    ax1[1].set_xlim(lat_start,lat_end)
    ax1[1].legend(loc='upper right')
    ax1[1].grid(True)

    ## Save and show the plot
    plt.savefig(f"Bathymetry_cross_section_with_variable={var}.png", dpi=100, facecolor='white', bbox_inches="tight")
    return fig
