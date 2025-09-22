import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

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
        depth = ds['model_depth'].values
        lonarr = ds['longitude'].values
        latarr = ds['latitude'].values
        ds.close()
        del ds
        print(np.shape(lonarr))
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
    print(lon_min,lon_max)
    print(lat_min,lat_max)

    # Central longitude
    c_l=lon_min+(lon_max-lon_min)/2

    # Markers customization
    if len(lon)>1:
        nl=len(label)-1
        if nl>14:
            print('The number of points to plot should be < 14')
            return
        colors=['k']*nl+['r']
        markers_symb=['+','D','x','s','o','^','*','v','d','<','p','>','h']
        markers=markers_symb[0:nl]+['o']
        markerfacecolors=["None"]*nl+['r']
        markersize=[10]*nl+[8]
    else:
        colors=['r']
        markers=['o']
        markerfacecolors=colors
        markersize=[10]

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
            (lonarr <= lon_min-2) | (lonarr >= lon_max+2) &
            (latarr <= lat_min-2) | (latarr >= lat_max+2)
        )
        depth_extent = np.ma.masked_where(mask_extent, depth)
        cont = ax.contourf(lonarr, latarr, depth_extent, cmap='binary', levels=30, transform=ccrs.PlateCarree())
        cbar = plt.colorbar(cont, orientation='vertical', pad=0.02, aspect=16, shrink=0.8)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Depth [m]',fontsize=15)
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
def plot_extreme_wave_map(return_period=100, product='NORA3', title='empty title', set_extent=[0, 30, 52, 73], output_file='extreme_wave_map.png', method='hexbin', percentile_contour = 50):
    '''
    Plots a map of extreme significant wave height for a given return period using the specified dataset and visualization method.

    Parameters:
    - return_period : int, optional, default=100
        The return period in years for calculating the extreme current speeds. Choose between 50 or 100 years.
    - product : str, optional, default='NORA3'
        The dataset to use for current speed data. Currently, only 'NORA3' is supported.
    - title : str, optional, default='empty title'
        The title to display on the map plot.
    - set_extent : list of float, optional, default=[0, 30, 52, 73]
        The geographical extent of the map in the format [lon_min, lon_max, lat_min, lat_max].
    - output_file : str, optional, default='extreme_wave_map.png'
        The path and filename for saving the generated plot.
    - method : str, optional, default='hexbin'
        Visualization method to use. Options:
            - 'hexbin'  : Plot data using a hexagonal binning with color shading.
            - 'contour' : Plot data using filled contours and highlighted contour lines.
    - percentile_contour : int or str, optional, default=50
        Threshold for drawing highlighted contour lines and labels:
            - Integer : Plots the top percentage of values (e.g., 50 plots top 50%).
            - 'min'   : Includes all values from the minimum.

    Returns:
    - fig : matplotlib.figure.Figure
        The figure object containing the generated map plot.
    '''    
    if product == 'NORA3':
        ds = xr.open_dataset(f'https://thredds.met.no/thredds/dodsC/nora3_subset_stats/wave/CF_hs_{return_period}y_extreme_gumbel_NORA3.nc')

    else:
        print(product, 'is not available')
        return
      
    standard_lon, standard_lat = ds['rlon'].values, ds['rlat'].values 
    standard_lon, standard_lat = np.meshgrid(standard_lon, standard_lat)

    data = xr.DataArray(ds['hs'], dims=ds['hs'].dims, coords=ds['hs'].coords)
    data_flat = ds['hs'].values.flatten()

    # Remove NaN values
    mask = ~np.isnan(data_flat)
    data_flat = data_flat[mask]
    lon_flat = standard_lon.flatten()[mask]
    lat_flat = standard_lat.flatten()[mask]

    # Figure and map
    fig = plt.figure(figsize=(9, 10))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=15))
    ax.coastlines(resolution='10m', zorder=3)
    ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)
    ax.add_feature(cfeature.LAND, facecolor='white', zorder=2)
    cmap = 'afmhot_r'

    # Rotated projection
    rotated_pole = ccrs.RotatedPole(
        pole_latitude=ds.projection_ob_tran.grid_north_pole_latitude,
        pole_longitude=ds.projection_ob_tran.grid_north_pole_longitude
    )   

    if method == 'hexbin':
        hb = ax.hexbin(lon_flat, lat_flat, C=data_flat, gridsize=180, cmap=cmap, vmin=np.min(data_flat), vmax=np.max(data_flat), edgecolors='white', transform=rotated_pole, reduce_C_function=np.mean, zorder=1)
        cb = plt.colorbar(hb, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
        cb.set_label(ds['hs'].attrs.get('standard_name', 'hs') + ' [' + ds['hs'].attrs.get('units', 'm') + ']', fontsize=14)

    elif method == 'contour': 
        max_val = np.max(data)
        min_val = np.min(data)

        # Dynamic level spacing
        value_range = int(max_val - min_val)
        step_contour = int(value_range / 10)       
        step_contourf = int(value_range / 20)     
    
        # Starting contour level
        start_contour = int(np.nanmin(data) if percentile_contour == 'min' else np.nanpercentile(data, percentile_contour)) 
        start_contour = start_contour - 1 if start_contour % 2 != 0 else start_contour  # ensure even
        start_contourf = int(np.nanmin(data)) - 1 if int(np.nanmin(data)) % 2 != 0 else int(np.nanmin(data))

        levels_contour = np.arange(start_contour, int(np.max(data)+step_contour*2), step_contour)
        levels_contourf = np.arange(start_contourf,int(np.max(data)+step_contourf*2),step_contourf)

        print('Levels contour:', levels_contour)
        print('Levels contourf:', levels_contourf)

        # Filled contour
        cm = plt.contourf(standard_lon, standard_lat, data,levels=levels_contourf, transform=rotated_pole,cmap=cmap, alpha = 0.6, zorder=1, vmin =0, vmax =int(max_val))
        cb = plt.colorbar(cm, label=ds['hs'].attrs.get('standard_name', 'hs') + ' [' + ds['hs'].attrs.get('units', 'm') + ']', ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
        # Line contour
        cs = ax.contour(standard_lon, standard_lat, data, levels=levels_contour, transform=rotated_pole,cmap=cmap, zorder=1, vmin = 0, vmax = int(max_val))
        ax.clabel(cs, levels = levels_contour, inline=False, fontsize=16, fmt='%d', colors='black')
    
    # Map extent and gridlines
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
def plot_extreme_wind_map(return_period=100, product='NORA3', z=0, title='empty title', set_extent=[0, 30, 52, 73], output_file='extreme_wind_map.png', land_mask=False, method='hexbin', percentile_contour=50):
    '''
    Plots a map of extreme wind speeds for a given return period using the specified dataset and visualization method.

    Parameters:
    - return_period: int, optional, default=100
        The return period in years for the extreme wind speed. Choose between 50 or 100 years.
    - product: str, optional, default='NORA3'
        The dataset to use for the wind speed data. Currently only 'NORA3' is supported.
    - z : int, optional, default=100
        Height level (in meters above ground) at which the wind speed is evaluated. Choose between 10, 25, 50, 100, 150, 200 or 300.
    - title: str, optional, default='empty title'
        The title to display on the map.
    - set_extent: list, optional, default=[0, 30, 52, 73]
        The map extent specified as [lon_min, lon_max, lat_min, lat_max].
    - output_file: str, optional, default='extreme_wind_map.png'
        The filename to save the plot.
    - land_mask : bool, optional, default=False  
        If True, apply a mask to exclude land areas.
    - method : str, optional, default='hexbin'
        Visualization method to use. Options:
            - 'hexbin'  : Plot data using a hexagonal binning with color shading.
            - 'contour' : Plot data using filled contours and highlighted contour lines.
    - percentile_contour : int or str, optional, default=50
        Threshold for drawing highlighted contour lines and labels:
            - Integer : Plots the top percentage of values (e.g., 50 plots top 50%).
            - 'min'   : Includes all values from the minimum.


    Returns:
    - fig: matplotlib.figure.Figure
        The figure object containing the plot.
    '''    

    if product == 'NORA3':
        ds = xr.open_dataset(f'https://thredds.met.no/thredds/dodsC/nora3_subset_stats/atm/CF_Return_Levels_3hourly_WindSpeed_6heights_1991_2020_zlev_period{return_period}yr.nc')

    else:
        print(product, 'is not available')
        return
    
    standard_lon = xr.DataArray(ds['longitude'])
    standard_lat = xr.DataArray(ds['latitude'])

    data = ds['wind_speed'].sel(z=z)
    data = Gaussian_filter(data, 1.5)

    # Remove NaN values
    data_flat = ds['wind_speed'].sel(z=z).values.flatten()
    mask = ~np.isnan(data_flat)
    data_flat = data_flat[mask]
    lon_flat = standard_lon.values.flatten()[mask]
    lat_flat = standard_lat.values.flatten()[mask]

    # Finding max value in specified area
    lon_min, lon_max, lat_min, lat_max = set_extent

    mask = (
            (standard_lon >= lon_min-5) & (standard_lon <= lon_max+5) &    # Add ±5° to bounds to include all values shown on Lambert map.
            (standard_lat >= lat_min-5) & (standard_lat <= lat_max+5)
        )
    
    wind_area = data.where(mask)

    max_val = int(wind_area.max().item())
    min_val = int(wind_area.min().item())

    # Figure and map
    fig = plt.figure(figsize=(9, 10))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=15))
    ax.coastlines(resolution='10m', zorder=3)
    ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)
    cmap='afmhot_r'
    if land_mask == True:
        ax.add_feature(cfeature.LAND, facecolor='white', zorder=2)
    

    if method == 'hexbin':
        hb = ax.hexbin(lon_flat, lat_flat, C=data_flat, gridsize=180, cmap=cmap, vmin=min_val, vmax=max_val, edgecolors='white', transform=ccrs.PlateCarree(), reduce_C_function=np.mean, zorder=1)
        cb = plt.colorbar(hb, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
        cb.set_label(ds['wind_speed'].attrs.get('standard_name', 'wind speed') + ' [' + ds['wind_speed'].attrs.get('units', 'm/s') + ']', fontsize=14)

    elif method == 'contour':
        # Dynamic level spacing
        value_range = int(max_val - min_val)
        step_contour = np.max([int(value_range / 10), 4])                           # Step size for contour lines; minimum of 4 to avoid overly fine intervals
        step_contour = step_contour + 1 if step_contour % 2 != 0 else step_contour  # ensure even
        step_contourf = np.max([int(value_range / 20), 2])                          # Step size for filled contours (colormap); minimum of 2 for clarity
    
        # Starting contour level
        start_contour = int(np.min(data) if percentile_contour == 'min' else np.percentile(data, percentile_contour)) 
        start_contour = start_contour - 1 if start_contour % 2 != 0 else start_contour  #ensure even
        start_contourf = int(np.min(data)) - 1 if int(np.min(data)) % 2 != 0 else int(np.min(data))

        levels_contour = np.arange(start_contour, int(max_val+step_contour), step_contour)
        levels_contourf = np.arange(start_contourf,int(max_val+step_contourf),step_contourf)

        print('Levels contour:', levels_contour)
        print('Levels contourf:', levels_contourf)

        # Filled contour
        cm = plt.contourf(standard_lon, standard_lat, data,levels=levels_contourf, transform=ccrs.PlateCarree(),cmap=cmap, alpha = 0.6, zorder=1, vmin =0, vmax =int(max_val))
        cb = plt.colorbar(cm, label=ds['wind_speed'].attrs.get('standard_name', 'wind speed') + ' [' + ds['wind_speed'].attrs.get('units', 'm') + ']', ax=ax, orientation='vertical', shrink=0.7, pad=0.05)

        # Line contour
        cs = ax.contour(standard_lon, standard_lat, data, levels=levels_contour, transform=ccrs.PlateCarree(),cmap=cmap, zorder=1, vmin = 0, vmax = int(max_val))
        ax.clabel(cs, levels = levels_contour, inline=False, fontsize=16, fmt='%d', colors='white')

    # Map extent and grid lines
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
def plot_extreme_current_map(return_period=100, z='surface', distribution='gumbel', product='NORA3', title='empty title', set_extent=[0, 30, 52, 73], output_file='extreme_current_map.png', method='hexbin', percentile_contour=50):
    '''
    Plots a map of extreme current speeds for a given return period using the specified dataset and visualization method.

    Parameters:
    - return_period : int, optional, default=100
        The return period in years (e.g., 100 years) for calculating the extreme current speeds. Choose between 25, 50 or 100 years.
    - z : str, optional, default='surface'
        The depth level at which the current speed is analyzed. Choose between 'surface' or 'seafloor'.
    - distribution : str, optional, default='gumbel'
        Statistical distribution used to model the extreme values. Options:
            - 'gumbel'       : Gumbel distribution.
            - 'genextreme'   : Generalized Extreme Value (GEV) distribution.
    - product : str, optional, default='NORA3'
        The dataset to use for current speed data. Currently, only 'NORA3' is supported.
    - title : str, optional, default='empty title'
        The title to display on the map plot.
    - set_extent : list of float, optional, default=[0, 30, 52, 73]
        The geographical extent of the map in the format [lon_min, lon_max, lat_min, lat_max].
    - output_file : str, optional, default='extreme_current_map.png'
        The path and filename for saving the generated plot.
    - method : str, optional, default='hexbin'
        Visualization method to use. Options:
            - 'hexbin'  : Plot data using a hexagonal binning with color shading.
            - 'contour' : Plot data using filled contours and highlighted contour lines.
    - percentile_contour : int or str, optional, default=50
        Threshold for drawing highlighted contour lines and labels:
            - Integer : Plots the top percentage of values (e.g., 50 plots top 50%).
            - 'min'   : Includes all values from the minimum.

    Returns:
    - fig : matplotlib.figure.Figure
        The figure object containing the generated map plot.
    '''

    if product == 'NORA3':
        ds = xr.open_dataset(f'https://thredds.met.no/thredds/dodsC/nora3_subset_stats/ocean/norkyst2400_annual_maxima_rve.nc')
        
    else:
        print(product, 'is not available')
        return

    parameter = f'{distribution}_{z}_{return_period}year' 
    data = xr.DataArray(ds[parameter])
    standard_lon = xr.DataArray(ds['lon'])
    standard_lat = xr.DataArray(ds['lat'])

    # Flatten values and remove NaN
    data_flat = ds[parameter].values.flatten()
    mask = ~np.isnan(data_flat)
    data_flat = data_flat[mask]
    lon_flat = ds['lon'].values.flatten()[mask]
    lat_flat = ds['lat'].values.flatten()[mask]

    # Figure and map
    fig = plt.figure(figsize=(9, 10))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=15))
    ax.coastlines(resolution='10m', zorder=3)
    ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)
    ax.add_feature(cfeature.LAND, facecolor='white', zorder=2)
    cmap='afmhot_r'

    if method == 'hexbin':
        hb = ax.hexbin(lon_flat, lat_flat, C=data_flat, gridsize=80, cmap=cmap, vmin=int(np.min(data_flat)), vmax=int(np.max(data_flat)), edgecolors='white', transform=ccrs.PlateCarree(), reduce_C_function=np.mean, zorder=1)
        cb = plt.colorbar(hb, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
        cb.set_label(ds[parameter].attrs.get('standard_name', 'Current speed') + ' [' + ds[parameter].attrs.get('units', 'm/s') + ']', fontsize=14)

    elif method == 'contour':
        # Filter the data and handle the NaN values
        data = Gaussian_filter(data, 1.5)

        # Step spacing
        step_contour = np.round(np.max(data) / 7,1)      
        step_contourf = 0.05                                
    
        # Starting contour level
        start_contour = np.round(np.min(data) if percentile_contour == 'min' else np.nanpercentile(data, percentile_contour),1) 
        start_contourf = np.min(data)
        
        max_val = np.max(data)
        levels_contour = np.round(np.arange(start_contour, max_val + step_contour, step_contour),1)
        levels_contourf = np.round(np.arange(start_contourf, max_val + step_contourf, step_contourf),2)

        print('Levels contour:', levels_contour)
        print('Levels contourf:', levels_contourf)

        # Filled contour
        cm = plt.contourf(standard_lon, standard_lat, data,levels=levels_contourf, transform=ccrs.PlateCarree(),cmap=cmap, alpha = 0.6, zorder=1, vmin =0, vmax =max_val)
        cb = plt.colorbar(cm, label=ds[parameter].attrs.get('standard_name', 'Current speed') + ' [' + ds[parameter].attrs.get('units', 'm/s') + ']', ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
        # Line contour
        cs = ax.contour(standard_lon, standard_lat, data, levels=levels_contour, transform=ccrs.PlateCarree(),cmap=cmap, zorder=1, vmin = int(np.min(data)), vmax = max_val)
        ax.clabel(cs, levels = levels_contour, inline=False, fontsize=14, fmt='%.1f', colors='black')

    # Map extent and grid lines
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

# Function to smooth the data with a Gaussian filter
def Gaussian_filter(data, sigma=1.5):
        '''
         Smooths an xarray.DataArray using a Gaussian filter, ignoring NaNs.
         NaNs are excluded by applying the filter to both the data and a validity mask.

        Parameters
        - data : xarray.DataArray
            Input array to smooth.
        - sigma : float, default=1.5
            Standard deviation of the Gaussian filter.

        Returns
        - xarray.DataArray
            Smoothed data with NaNs preserved.
        '''
        # Filtering the data with NaN handling
        nan_mask = np.isnan(data)
        data_filled = np.copy(data)
        data_filled[nan_mask] = 0

        weights = (~nan_mask).astype(float)  # Create weights ( 1 where data is valid, 0 where NaN)
        
        smoothed_data = gaussian_filter(data_filled, sigma = sigma)  # Gaussian filter to data and weights
        smoothed_weights = gaussian_filter(weights, sigma=sigma)
        
        result = smoothed_data/smoothed_weights  # Normalize to ignore NaNs in the average
        result[smoothed_weights==0] = np.nan   # Reassign NaNs where no weight

        data = xr.DataArray(result, dims=data.dims, coords=data.coords) # Convert back to xarray
        
        return data


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
