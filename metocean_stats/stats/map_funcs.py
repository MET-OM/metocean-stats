import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

def plot_point_on_map(lon, lat, label):
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

    # Create the plot with Mollweide projection
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.Mollweide())

    # Add high-resolution coastlines with grey facecolor for land
    coast = cfeature.NaturalEarthFeature(
        category='physical', name='coastline', scale='10m', edgecolor='lightgrey', facecolor='darkkhaki')
    ax.add_feature(coast)

    # Plot the points with automatic labels
    for i, (lon, lat,label) in enumerate(zip(lon_list, lat_list,label_list)):
        ax.plot(lon, lat, marker='o', markersize=8, label=f'{label}', transform=ccrs.PlateCarree())

    # Calculate zoom-in extent
    lat_min = max(min(lat_list) - 5, -90)  # Ensure latitude does not go below -90
    lat_max = min(max(lat_list) + 5, 90)   # Ensure latitude does not go above 90
    lon_min = max(min(lon_list) - 10, -180) # Ensure longitude does not go below -180
    lon_max = min(max(lon_list) + 10, 180)  # Ensure longitude does not go above 180

    # Set extent of the map
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

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


