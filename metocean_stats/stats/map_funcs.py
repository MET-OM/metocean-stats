import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

def plot_point_on_map(lon_list, lat_list):
    # Ensure lon_list and lat_list are iterable (lists or arrays)
    if not isinstance(lon_list, (list, tuple)):
        lon_list = [lon_list]
    if not isinstance(lat_list, (list, tuple)):
        lat_list = [lat_list]

    # Create the plot with Mollweide projection
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.Mollweide())
    
    # Add stock image as background
    ax.stock_img()
    
    # Add high-resolution coastlines
    coast = cfeature.NaturalEarthFeature(
        category='physical', name='coastline', scale='10m', facecolor='none')
    ax.add_feature(coast, edgecolor='black')
    
    # Plot the points with automatic labels
    for i, (lon, lat) in enumerate(zip(lon_list, lat_list)):
        ax.plot(lon, lat, marker='o', markersize=8, label=f'{lon,lat}', transform=ccrs.PlateCarree())
    
    # Calculate zoom-in extent
    lat_min = max(min(lat_list) - 10, -90)  # Ensure latitude does not go below -90
    lat_max = min(max(lat_list) + 10, 90)   # Ensure latitude does not go above 90
    lon_min = max(min(lon_list) - 15, -180) # Ensure longitude does not go below -180
    lon_max = min(max(lon_list) + 15, 180)  # Ensure longitude does not go above 180
    
    # Set extent of the map
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    # Add legend
    ax.legend(loc='upper left')
    
    #plt.title('...')
    plt.tight_layout()
    plt.savefig('map_'+str(lon)+'_'+str(lat)+'.png')


