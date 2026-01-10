import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import matplotlib.patches as patches
import re
from ..stats import spec_funcs
from ..tables import spectra
from matplotlib.colors import ListedColormap
from matplotlib.ticker import ScalarFormatter

def plot_spectra_1d(data, var='SPEC', period=None, month=None, method='mean', output_file='monthly_spectra_1d.png'):
    '''
    Plots 1D spectra aggregated by month, for a specific month across multiple years, or for a single timestamp.

    - If no month is specified:
      Plots 12 monthly spectra (one subplot per month) plus an overall average spectrum
      computed from the monthly aggregates.
    - If a month is specified:
      Plots one spectrum per year for that month, with an additional average spectrum
      across all selected years.
    - If `period` contains a single timestamp:
      Plots only the spectrum for that specific time.

    Parameters
    - data : xarray.Dataset
        Dataset containing 2D wave spectra over time.
    - var : str, optional, default='SPEC'
        Name of the spectral variable to analyze.
    - period : tuple of two str, optional, default=None
        A tuple specifying the desired time range.
        - (start_time, end_time): Filters between start_time and end_time, e.g., ('2021-01-01T00', '2021-12-31T23').
        - (start_time): Filters to a single timestamp.
        - None: Uses the full time range available in data.
        Both start_time and end_time may be strings or datetime-like objects.
    - month : int or None, optional, default=None
        If set (1 = January, ..., 12 = December), plots only that month.
        If None, plots all monthly spectra.
    - method : str, optional, default='mean'
        Aggregation method:  
        - 'mean'                 : Average over time.  
        - 'top_1_percent_mean'   : Average over times where Hm0 ≥ 99th percentile.  
        - 'hm0_max'              : Use time step with maximum Hm0.  
        - 'hm0_top3_mean'        : Average of Hm0 over the three time steps with the highest values.
    output_file : str, optional, default='monthly_spectra_1d.png'
        Filename for the saved plot.

    Returns
    -------
    - fig : matplotlib.figure.Figure
        The figure object containing the plot.
    '''

    data = spec_funcs.standardize_wave_dataset(data)

    filtered_data, period_label = spec_funcs.filter_period(data=data[var], period=period)                                                    # Filters the dataset to the specified time period.

    df = spectra.table_monthly_freq_1dspectrum(filtered_data, var=var, month=month, method=method, average_over='month', output_file=None)   # DataFrame containing one aggregated 1D spectrum per month and a final row for the overall mean spectrum based on the selected method.

    freq = [col for col in df.columns if col not in ['Year', 'Month', 'Month_no', 'Hm0']]                                                    # Select all columns except 'Month', 'Month_no', and 'Hm0'
    spec_values = df[freq].to_numpy()
    hm0 = df['Hm0'].tolist()

    fig, ax = plt.subplots()

    color_list = [
        "#051BE2", 
        "#079FF7",  
        "#77C4F1",  
        "#117733", 
        "#A3C26B",  
        "#00EBAC",  
        "#EED039",  
        "#E69F00",  
        "#D55E00",  
        "#CC79A7",  
        "#999999",  
        "#990303",  
    ]

    dashed_lines = {"#E69F00", "#079FF7", "#A3C26B"}
    dotted_lines = {"#00EBAC", "#77C4F1", "#990303", "#051BE2"}

    # Build (color, linestyle) pairs
    line_styles_12 = []
    for color in color_list:
        linestyle = '--' if color in dashed_lines else (':' if color in dotted_lines else '-')
        line_styles_12.append((color, linestyle))

    # Title
    parts = period_label.split(" to ")
    year_label = parts[0][:4] if len(parts) == 1 or parts[0][:4] == parts[1][:4] else f"{parts[0][:4]}-{parts[1][:4]}"

   
    ######## Plot of monthly aggregation across all years ########
    if month is None and (period is None or isinstance(period, list)):
        months = df['Month'].tolist()

        for i in range(len(months)):
            color, ls = ('black', '-') if i == len(months) -1 else line_styles_12[i]
            if period is None or isinstance(period, tuple):
                ax.plot(freq, spec_values[i, :], label=rf"{months[i]} ($H_{{m_0}}$={round(hm0[i], 1)} m)", color=color, linestyle=ls, linewidth=1.5, markevery=5)
        ax.set_title(f'Period: {year_label}, method = {method}', fontsize=14)

    ######## Single timestamp plot ########
    elif isinstance(period, str):
        ax.plot(freq, spec_values[0, :], label=rf"{period_label} ($H_{{m_0}}$={round(hm0[0], 1)} m)", color='blue', linewidth=1.5)    

    ######## Plot of yearly aggregation for a specific month ########
    else:
        for i, label_val in enumerate(df['Year']):
            spec = spec_values[i]
            label = f"{label_val} ($H_{{m_0}}$={hm0[i]:.1f} m)"
            color, ls = ('black', '-') if i == len(df['Year']) - 1 else line_styles_12[i]
            ax.plot(freq, spec, linewidth=1.5, label=label,  color=color, linestyle=ls)
        ax.set_title(f'Period: {year_label}, month = {df.index.name}, method = {method}', fontsize=14)


    ax.legend(fontsize=8)
    ax.set_xlabel('Frequency [Hz]', fontsize=12)
    ax.set_ylabel(r'Variance Density [m$^2$/Hz]', fontsize=12)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    return fig

def plot_spectrum_2d(data,var='SPEC', period=None, month = None, method='mean', plot_type='pcolormesh', 
                          freq_mask = False, radius='frequency', dir_letters=False, output_file='spectrum_2d.png'):
    '''
    Plots a 2D directional spectrum for a specific month or time period.

    Parameters:
    - data : xarray.Dataset
        Dataset containing 2D wave spectra over time.
    - var : str, optional, default='SPEC'
        Name of the spectral variable to plot.
    - period : tuple of str, optional, default=None
        A tuple specifying the desired time range.
        - (start_time, end_time): Filters between start_time and end_time, e.g., ('2021-01-01T00', '2021-12-31T23').
        - (start_time): Filters to a single timestamp.
        - None: Uses the full time range available in data.
        Both start_time and end_time may be strings or datetime-like objects.
    - month : int or None, optional, default=None
        Filter data by calendar month (1 = January, ..., 12 = December).
        If None, no filtering by month is applied.
    - method : str, optional, default='mean'
        Aggregation method:  
        - 'mean'                : Average over time.  
        - 'top_1_percent_mean'  : Average over times when Hm0 ≥ 99th percentile.  
        - 'hm0_max'             : Use time step with maximum Hm0.  
        - 'hm0_top3_mean'       : Average of Hm0 over the three time steps with the highest values.
    - plot_type : str, optional, default='pcolormesh'
        2D spectrum plotting method. Choose 'pcolormesh' or 'contour'.
    - freq_mask : bool, optional, default=False
        If True, mask low-energy frequencies.
    - radius : str, optional, default='frequency'
        Units for radial axis, period or frequency.
    - dir_letters : bool, optional, default=False
        Use compass labels instead of degrees.
    - output_file : str, optional, default='spectrum_2d.png'
        Filename for the saved plot.

    Returns
    - fig: matplotlib.figure.Figure
        The figure object containing the plot.
    '''

    data = spec_funcs.standardize_wave_dataset(data)

    # Normalize and wrap directional spectral data to 0-360°.
    spec, dirc = spec_funcs.wrap_directional_spectrum(data, var=var)                                

    # Filters the dataset to the specified time period.
    filtered_data, period_label = spec_funcs.filter_period(data=spec, period=period)   

    # Add hm0 to dataset if not present
    if 'hm0' not in data:
        hm0 = spec_funcs.integrated_parameters_dict(data[var], data.freq, data.direction)['Hs']     
        filtered_data['hm0'] = ('time', hm0)
    
    # Aggregate the data using the specified method 
    data_aggregated = spec_funcs.aggregate_spectrum(data=filtered_data, hm0=filtered_data['hm0'], method=method, month=month)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
    cmap = plt.cm.hot_r

    frequencies=spec.coords['freq']

    max_data_aggregated = np.nanmax(data_aggregated)
    min_data_aggregated = np.nanmin(data_aggregated)
    # If masking enabled, keep frequencies where data exceeds threshold (focusing on the most significant parts of the spectrum)
    if freq_mask:
        threshold = max(0.0001, min(0.03, 0.01 / max_data_aggregated))* max_data_aggregated                                             # Set threshold dynamically: smaller (0.0001 * max) for large max values, larger (up to 0.03 * max) for small max values
        threshold_val = min_data_aggregated + (threshold * (max_data_aggregated - min_data_aggregated))                                 # Convert to data value
        frequencies = data_aggregated.freq[(data_aggregated >= threshold_val).any(dim='direction')]                                     # Keep frequencies where data exceeds threshold in any direction
        data_aggregated = data_aggregated.sel(freq=frequencies)           
    
    # Set radial coordinate data based on chosen radius type (period or frequency)
    if radius=='period':
        rad_data=1/frequencies.values
    elif radius=='frequency':
        rad_data=frequencies.values

    # Plots 2D spectrum based on plot_type
    if plot_type == 'contour':
                step = np.round(max_data_aggregated / 10, 1)
                levels = np.round(np.arange(0, max_data_aggregated + step, step), 1)
                cp = ax.contourf(np.radians(dirc.values), rad_data, data_aggregated, cmap=cmap, levels=levels)

    elif plot_type == 'pcolormesh':
        cp = ax.pcolormesh(np.radians(dirc.values), rad_data, data_aggregated, cmap=cmap, shading='auto')

    # Replace x-axis tick labels with compass directions (N, NE, E, ..., NW) if enabled
    if dir_letters:
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

    cbar = plt.colorbar(cp, pad=0.1, shrink=0.7) 
    cbar.set_label(r'Variance Density [m$^2$/Hz]', fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    method_label = method.replace("_", r"\_")
    label = f'$\mathbf{{Method:}}\ {method_label}$' + '\n' if isinstance(period, tuple) else None

    # Labels
    if month is None:
        label2 = (rf"$\mathbf{{Period:}}$ {period_label}" + " " * 15 )
    else:
        months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        label2 = (
            rf"$\mathbf{{Year:}}$ {filtered_data.time.dt.year.min().item()} - {filtered_data.time.dt.year.max().item()}"+ "\n" 
            + rf"$\mathbf{{Month:}}$ {months[month-1]}")

    # Text based on selected method
    if method == 'hm0_max':
        period_label = pd.to_datetime(data_aggregated.time.values).to_pydatetime().strftime('%Y-%m-%dT%H') + 'Z'
        label2 = (rf"$\mathbf{{Timestamp:}}$ {period_label}" + rf"  (Hm0 = {np.round(filtered_data['hm0'].sel(time=data_aggregated.time).values, 1)})" + "\n")

        times = pd.to_datetime(filtered_data.time.values)
        label2 += rf"$\mathbf{{Range:}}$ " + (
            f"{times.min().strftime('%Y-%m-%dT%H')}Z"
            if times.min() == times.max()
            else f"{times.min().strftime('%Y-%m-%dT%H')}Z to {times.max().strftime('%Y-%m-%dT%H')}Z"
        )

    elif method == 'hm0_top3_mean':
        hm0_top3_timesteps = [pd.to_datetime(t).strftime('%Y-%m-%dT%H') + 'Z' for t in data_aggregated.attrs['hm0_top3_timesteps']]
        period_label = ', '.join(hm0_top3_timesteps[:3])

        label2 = (rf"$\mathbf{{Timestamps:}}$ {period_label}" + " " * 15 + "\n")
        times = pd.to_datetime(filtered_data.time.values)
        label2 += (
            rf"$\mathbf{{Range:}}$ "
            f"{times.min().strftime('%Y-%m-%dT%H')}Z"
            if times.min() == times.max()
            else rf"$\mathbf{{Range:}}$ "
            f"{times.min().strftime('%Y-%m-%dT%H')}Z to {times.max().strftime('%Y-%m-%dT%H')}Z"
        )
        
    label = label + label2 if label != None else label2

    ax.text(-0.07,-0.15,label,
            transform=ax.transAxes,fontsize=16 if method == 'hm0_top3_mean' else 18, ha='left')
    
    lat_str = ", ".join(
        f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}"                                # Format latitudes with N/S
        for lat in np.unique(data['latitude'].round(3).values))

    lon_str = ", ".join(
        f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"                                # Format longitudes with E/W
        for lon in np.unique(data['longitude'].round(3).values))

    label_position = (
        rf"$\mathbf{{Position:}}$" +f" {lon_str}, {lat_str}"                   # Position label
        + "\u2007" * max(0, 27 - len(f"Lon: {lon_str}, Lat: {lat_str}")) + "\n"
    )

    ax.text(-0.07,-0.23,label_position,
        transform=ax.transAxes,fontsize=16 if method == 'hm0_top3_mean' else 18, ha='left')

    # Set up polar plot style and tick appearance
    ax.grid(True)
    ax.set_rlabel_position(315)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.tick_params(axis='y', labelsize=12) 
    ax.tick_params(axis='x', labelsize=12, pad=5)
    plt.tight_layout(pad=0)

    if output_file is not None:
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    return fig


def plot_diana_spectrum(data, var='SPEC', period=None, month = None, method='mean', partition=False, plot_type='pcolormesh', freq_mask=False,
                           radius='frequency', dir_letters=False, bar='hm0', arrow_dir='mean_dir', max_spec_value=None, output_file='diana_spectrum.png'):
    '''
    Generates a Diana plot showing the 2D wave spectrum with mean wave direction, and mean wind direction if available.
    Includes 1D spectrum inset, position, period, method and Hm0. Mean swell and windsea directions are added if 
    partitioning is enabled (requires wind data).

    Parameters
    - data : xarray.Dataset
        Dataset with 2D directional-frequency wave spectra.
    - var : str, optional, default='SPEC'
        Spectral variable name.
    - period : list of two str, optional, default=None
        A list specifying the desired time range.
        - [start_time, end_time]: Filters between start_time and end_time, e.g., ('2021-01-01T00', '2021-12-31T23').
        - [start_time]: Filters to a single timestamp.
        - None: Uses the full time range available in data.
        Both start_time and end_time may be strings or datetime-like objects.
    - month : int, optional, default=None
        Filter by month (1 = January, ..., 12 = December). If set, selects data from that month across all years.
        If None, no filtering is applied.
    - method : str, optional, default='mean'
        Aggregation method:  
        - 'mean'                : Average over time.  
        - 'top_1_percent_mean'  : Average over times where Hm0 ≥ 99th percentile.  
        - 'hm0_max'             : Use time step with maximum Hm0.  
        - 'hm0_top3_mean'       : Average of Hm0 over the three time steps with the highest values.
    - partition : bool, optional, default=False
        If True, partitions the wave spectrum into wind sea and swell components, and computes the mean direction for each.  
        This is only applicable when wind direction data is available.
    - plot_type : str, optional, default='pcolormesh'
        2D spectrum plotting method. Choose 'pcolormesh' or 'contour'.
    - freq_mask : bool, optional, default=False
        If True, mask low-energy frequencies.
    - radius : str, optional, default='frequency'
        Units for radial axis, period or frequency.
    - dir_letters : bool, optional, default=False
        Use compass labels instead of degrees.
    - bar : str, optional, default='hm0'
        Show colorbar for 'density' or significant wave height 'hm0'.
    - arrow_dir: str, optional, default='mean_dir'
        - 'mean_dir': Computes the mean wave direction.
        - 'pdir': Computes the mean peak wave direction. 
    - max_spec_value : float, optional, default=None
        Sets the 2D-spectrum `vmax`. If None, uses the aggregated dataset maximum. 
    - output_file : str, optional, default='diana_spectrum.png'
        Output filename for saving the figure. 

    Returns
    fig_main : matplotlib.figure.Figure
        The generated plot figure.
    '''

    if partition and 'wind_direction' not in data:
        raise ValueError("Wind data is required for partitioning. Either change partition to False or add wind data using: spec_funcs.merge_spec_wind(NORA3_wave_spec, NORA3_wind_sub).")

    valid_methods = ['mean', 'top_1_percent_mean', 'hm0_max', 'hm0_top3_mean']
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Available methods are: {', '.join(valid_methods)}.")
    
    ############ Data handling ############
    # Standardize the 2D wave spectrum dataset to match the WINDSURFER/NORA3 format.
    data = spec_funcs.standardize_wave_dataset(data)

    # Filters the dataset to the specified time period.
    filtered_data, period_label = spec_funcs.filter_period(data=data, period=period)

    if month is not None and month not in pd.to_datetime(filtered_data['time'].values).month:
        raise ValueError(f"Month {month} is not present in the filtered data period ({period_label}).")

    # Add hm0 to dataset if not present
    if 'hm0' not in data:
        int_params = spec_funcs.integrated_parameters_dict(filtered_data['SPEC'], filtered_data.freq, filtered_data.direction)
        hm0 = int_params['Hs']
        filtered_data['hm0'] = xr.DataArray(hm0, dims=['time'], coords={'time': filtered_data.time}, attrs={'units': 'm', 'standard_name': f'significant_wave_height'})
        filtered_data['fp'] = xr.DataArray(int_params['peak_freq'], dims=['time'], coords={'time': filtered_data.time}, attrs={'units': 'Hz', 'standard_name': f'peak_freq'})
        filtered_data['pdir'] = xr.DataArray(int_params['peak_dir'], dims=['time'], coords={'time': filtered_data.time}, attrs={'units': 'deg', 'standard_name': f'peak_dir'})
    
    # Compute u/v wind components if wind_direction is available
    if 'wind_direction' in filtered_data:
        filtered_data['u'] =  filtered_data['wind_speed'] * xr.ufuncs.sin(np.deg2rad((450 - filtered_data['wind_direction'])%360))
        filtered_data['v'] =  filtered_data['wind_speed'] * xr.ufuncs.cos(np.deg2rad((450 - filtered_data['wind_direction'])%360))

    # Normalize and wrap directional spectral data to 0-360°.
    spec, dirc = spec_funcs.wrap_directional_spectrum(filtered_data, var=var)                

    # Aggregate spectra using the specified method 
    data_aggregated_spec = spec_funcs.aggregate_spectrum(data=spec, hm0=filtered_data['hm0'], var=var, method=method, month=month)

    # Aggregate the whole data set using the specified method
    data_aggregated = spec_funcs.aggregate_spectrum(data=filtered_data, hm0=filtered_data['hm0'], var=None, method=method, month=month)

    hm0_aggregated = np.round(data_aggregated['hm0'].values,1)


    ############ 2D Spectrum ############
    fig_main = plt.figure(figsize=(7.5, 6))
    ax_2D_spectra = fig_main.add_axes([0.25, 0.25, 0.6, 0.6], projection='polar')        # [left, bottom, width, height]
    
    # Style preferences — change these to update all colors in the figure consistently
    cmap = cm.ocean_r                                                                    
    color="#0463d7ff"                                                                 
    alpha = 0.22
    textcolor = 'black'
    
    frequencies=spec.coords['freq']

    max_data_aggregated = np.nanmax(data_aggregated_spec)
    min_data_aggregated = np.nanmin(data_aggregated_spec)
    # If masking enabled, keep frequencies where data exceeds threshold (focusing on the most significant parts of the spectrum)
    if freq_mask:
        threshold = max(0.0001, min(0.03, 0.01 / max_data_aggregated))* max_data_aggregated                            # Set threshold dynamically: smaller (0.0001 * max) for large max values, larger (up to 0.03 * max) for small max values
        threshold_val = min_data_aggregated + (threshold * (max_data_aggregated - min_data_aggregated))                # Convert to data value
        frequencies = data_aggregated_spec.freq[(data_aggregated_spec >= threshold_val).any(dim='direction')]          # Keep frequencies where data exceeds threshold in any direction
        data_aggregated_spec = data_aggregated_spec.sel(freq=frequencies)                                              # Filter data to those frequencies

    # Set radial coordinate data based on chosen radius type (period or frequency)
    if radius=='period':
        rad_data=1/frequencies.values
    elif radius=='frequency':
        rad_data=frequencies.values

    # Plots 2D spectrum based on plot_type
    if plot_type == 'contour':
        step = np.maximum(np.round(max_data_aggregated / 10, 1), 0.05)
        levels = np.round(np.arange(0, max_data_aggregated + step, step), 1 if max_data_aggregated > 1 else 2)
        colors = cmap(np.linspace(0, 1, len(levels)-1))
        colors[0] = [1, 1, 1, 1]   # white RGBA
        cmap_contour = ListedColormap(colors)
        cp = ax_2D_spectra.contourf(np.radians(dirc.values), rad_data, data_aggregated_spec, cmap=cmap_contour, levels=levels)

    elif plot_type == 'pcolormesh':
        cp = ax_2D_spectra.pcolormesh(np.radians(dirc.values), rad_data, data_aggregated_spec, vmax=max_spec_value, cmap=cmap, shading='auto')

    # Show labels only on every second radial tick when radius is frequency
    if radius == 'frequency':
        ticks = ax_2D_spectra.get_yticks()
        reduced_ticks = ticks[::2]                              # keep every 2nd tick
        ax_2D_spectra.set_yticks(reduced_ticks)                 # Set fewer yticks
        labels = [str(round(t, 2)) for t in reduced_ticks]      # Set labels only on those ticks (rounded nicely)
        ax_2D_spectra.set_yticklabels(labels)

    # Replace x-axis tick labels with compass directions (N, NE, E, ..., NW) if enabled
    if dir_letters:
        ticks_loc = ax_2D_spectra.get_xticks().tolist()
        ax_2D_spectra.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax_2D_spectra.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

    # If variance selected, add variance density colorbar with label and ticks (dynamically adjusted based on plot type and data range)
    if bar == 'density':
        cbar_ax = fig_main.add_axes([0.87, 0.3, 0.03, 0.5])
        cbar = plt.colorbar(cp, cax=cbar_ax, pad=0.1, shrink=0.6)
        cbar_ax.text(0.5, 1.05, r'Density [m$^2$/Hz]', ha='center', va='bottom', fontsize=12, transform=cbar_ax.transAxes)
        cbar.ax.tick_params(labelsize=12)

        if plot_type == 'pcolormesh':
            step = (max_data_aggregated - min_data_aggregated) / 5
            prec = max(0, -int(np.floor(np.log10(step))) + 1)
            step = round(step, prec)
            
            fmt = ScalarFormatter()
            fmt.set_scientific(True)

            cbar.formatter = fmt
            cbar.update_ticks()
        else:
            cbar.set_ticks(levels[::2])
    
    # Set up polar plot style and tick appearance
    ax_2D_spectra.grid(True)
    ax_2D_spectra.set_rlabel_position(315)
    ax_2D_spectra.set_theta_zero_location('N')
    ax_2D_spectra.set_theta_direction(-1)
    ax_2D_spectra.tick_params(axis='y', labelsize=12) 
    ax_2D_spectra.tick_params(axis='x', labelsize=12, pad=5)


    ############ 1D wave spectrum ############
    ax_1D_spectra =  fig_main.add_axes([0.06, 0.15, 0.3, 0.3])          # [left, bottom, width, height]

    # Plot 1D wave spectrum and get the spectra dataframe
    plot_1d_spectrum_on_ax(data_aggregated_spec, ax=ax_1D_spectra, var=var, color=color, alpha=alpha)

    # If 'hm0' bar plot is requested, create a vertical bar representing Hm0 beside the 2D spectrum
    if bar == 'hm0':
        cbar_ax = fig_main.add_axes([0.87, 0.3, 0.03, 0.5])
        cbar_ax.bar(0, hm0_aggregated, color=color, alpha=min(alpha+0.5,1))
        cbar_ax.set_ylim(0, 20)
        if hm0_aggregated >20:
            cbar_ax.set_ylim(0, 25)
        cbar_ax.set_xlim(-0.5, 0.5)
        cbar_ax.set_xticks([])
        cbar_ax.yaxis.tick_right() 
        cbar_ax.set_yticks(np.linspace(0, 20, 5))
        cbar_ax.tick_params(axis='y', labelsize=12)
        cbar_ax.text(0, -1, r'H$_{m_0}$ [m]', ha='center', va='top', fontsize=12)

    ############ Direction arrows ############
    mean_dir_rad = spec_funcs.compute_mean_wave_direction(mean_pdir=True if arrow_dir=='pdir' else False, data=data_aggregated, var=var)

    arrow_length = 0.35                                                 # Total length of arrow

    dx = np.cos(mean_dir_rad) * arrow_length                            # Direction vector (unit length)
    dy = np.sin(mean_dir_rad) * arrow_length

    x0, y0 = 0.5,0.4
    end = (x0 + dx, y0 + dy)

    ax_arrow = fig_main.add_axes([0.02, 0.55, 0.2, 0.3], frameon=False)  # [left, bottom, width, height]
    ax_arrow.set_axis_off()
    ax_arrow.set_xlim(0, 1)
    ax_arrow.set_ylim(0, 1)
    ax_arrow.set_zorder(20)

    ax_arrow.annotate(
        '',
        xy=end,
        xytext=(x0,y0),
        xycoords='axes fraction',
        textcoords='axes fraction',
        arrowprops=dict(facecolor=textcolor, edgecolor='none', width=2, headwidth=8)
    )

    # If 'wind_direction' in data plot mean wind, swell wave and wind sea wave direction
    if 'wind_direction' in data:
        mean_wind_dir_rad = np.arctan2(data_aggregated['u'], data_aggregated['v'])
        data_aggregated['wind_direction'] = ((450 - np.rad2deg(mean_wind_dir_rad))%360)-180

        plot_direction_arrow_on_ax(ax_arrow,
                            direction=(data_aggregated['wind_direction'].sel(height=10) if 'height' in data else data_aggregated['wind_direction']),
                            speed_data=data_aggregated['wind_speed'].sel(height=10) if 'height' in data else data_aggregated['wind_speed'],
                            x0=x0, y0=y0,
                            color=color,
                            draw_ticks=True,
                            style='line')
        
        if partition == True:
            # Partition the aggregated dataset
            sp =  spec_funcs.Spectral_Partition_wind(data_aggregated, beta=1.3)

            # Plot swell
            plot_direction_arrow_on_ax(ax_arrow,
                                direction=sp['mean_pdir_swell_rad'] if arrow_dir == 'pdir' else sp['mean_dir_swell_rad'],
                                x0=x0, y0=y0,
                                color='green',
                                style='arrow')

            # Plot windsea
            plot_direction_arrow_on_ax(ax_arrow,
                                direction=sp['mean_pdir_windsea_rad'] if arrow_dir == 'pdir' else sp['mean_dir_windsea_rad'],
                                x0=x0, y0=y0,
                                color='#E69F00',
                                style='arrow')


    # Draw cross lines at the arrow base to show orientation relative to coordinate axes
    line_len = 0.15                                                             # Length of cross arms (in axes fraction)
    angles = [np.radians(0), np.radians(90), np.radians(45), np.radians(135)]   # Axes to display

    for angle in angles:
        dx = np.cos(angle) * line_len
        dy = np.sin(angle) * line_len
        ax_arrow.plot([x0 - dx, x0 + dx], [y0 - dy, y0 + dy], color='gray', linestyle='-', linewidth=1)

    # Title of the arrow
    if partition == False:
        ax_arrow.text(0.05 if'wind_direction' in data else 0.2, 0.85, "Mean wave" + ('/' if'wind_direction' in data else ''), color='black', transform=ax_arrow.transAxes, fontsize=12, va='bottom')
        if 'wind_direction' in data:
            ax_arrow.text(0.72, 0.85, "wind", color=color, transform=ax_arrow.transAxes, fontsize=12, va='bottom')
        ax_arrow.text(0.25, 0.75, "direction", color='black', transform=ax_arrow.transAxes, fontsize=12, va='bottom')

    elif partition == True:
        x = 0.05
        y = 0.99

        ax_arrow.text(x, y, 'Mean peak wave/' if arrow_dir=='pdir' else "Mean wave/", color='black', transform=ax_arrow.transAxes, fontsize=12, va='bottom')    
        ax_arrow.text(x + 0.08, y-0.1, "swell", color='green', transform=ax_arrow.transAxes, fontsize=12, va='bottom') if arrow_dir=='pdir' else ax_arrow.text(x + 0.67, y, "swell", color='green', transform=ax_arrow.transAxes, fontsize=12, va='bottom')
        ax_arrow.text(x + 0.36, y-0.1, "/", color='black', transform=ax_arrow.transAxes, fontsize=12, va='bottom')  if arrow_dir=='pdir' else ax_arrow.text(x + 0.95, y, "/", color='black', transform=ax_arrow.transAxes, fontsize=12, va='bottom') 
        ax_arrow.text(x + 0.4, y - 0.1, "wind sea", color='#E69F00', transform=ax_arrow.transAxes, fontsize=12, va='bottom') if arrow_dir=='pdir' else ax_arrow.text(x + 0.1, y - 0.1, "wind sea", color='#E69F00', transform=ax_arrow.transAxes, fontsize=12, va='bottom')
        if arrow_dir == 'pdir': 
            ax_arrow.text(x, y - 0.2, "direction + mean", color='black', transform=ax_arrow.transAxes, fontsize=12, va='bottom') if arrow_dir=='pdir' else ax_arrow.text(x + 0.6, y - 0.1, "wind", color=color, transform=ax_arrow.transAxes, fontsize=12, va='bottom')
            ax_arrow.text(x + 0.1, y - 0.3, "wind", color=color, transform=ax_arrow.transAxes, fontsize=12, va='bottom') if arrow_dir=='pdir' else ax_arrow.text(x + 0.6, y - 0.1, "wind", color=color, transform=ax_arrow.transAxes, fontsize=12, va='bottom')
            ax_arrow.text(x + 0.35, y - 0.3, " direction", color='black', transform=ax_arrow.transAxes, fontsize=12, va='bottom') if arrow_dir=='pdir' else ax_arrow.text(x + 0.2, y - 0.2, " direction", color='black', transform=ax_arrow.transAxes, fontsize=12, va='bottom')
        else: 
            ax_arrow.text(x + 0.55, y - 0.1, "/", color='black', transform=ax_arrow.transAxes, fontsize=12, va='bottom')
            ax_arrow.text(x + 0.1, y - 0.2, "wind", color=color, transform=ax_arrow.transAxes, fontsize=12, va='bottom') if arrow_dir=='pdir' else ax_arrow.text(x + 0.6, y - 0.1, "wind", color=color, transform=ax_arrow.transAxes, fontsize=12, va='bottom')
            ax_arrow.text(x + 0.35, y - 0.2, " direction", color='black', transform=ax_arrow.transAxes, fontsize=12, va='bottom') if arrow_dir=='pdir' else ax_arrow.text(x + 0.2, y - 0.2, " direction", color='black', transform=ax_arrow.transAxes, fontsize=12, va='bottom')

    # Color of arrow box
    rect = patches.Rectangle(
        (0, 0), 1.1 if partition==True else 1, 1.1 if partition==True else 1,            
        transform=ax_arrow.transAxes,
        facecolor=color,     
        alpha=alpha,                 
        edgecolor='none'
    )
    ax_arrow.add_patch(rect)
    rect.set_clip_on(False)


    ############ Title of the figure ############
    fig_main.text(0.2, 0.98, rf'$\mathbf{{Variance}}$ $\mathbf{{Density}}$ $\mathbf{{Spectrum}}$',
    transform=fig_main.transFigure,
    fontsize=24,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(facecolor='none', edgecolor='none', boxstyle='square,pad=0.5'))


    ############ Create label with position, period, method, and Hm0 for the plot ############

    month_map = {                                                                                   # Define month names, with 'Average' as a label for the overall mean
        'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
        'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12,
        'Average':0}

    lat_str = ", ".join(
        f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}"                                # Format latitudes with N/S
        for lat in np.unique(data['latitude'].values[0] if len(data['latitude'].values) > 1 else data['latitude'].values)
    )

    lon_str = ", ".join(
        f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"                                # Format longitudes with E/W
        for lon in np.unique(data['longitude'].values[0] if len(data['longitude'].values) > 1 else data['longitude'].values)
    )

    label = (
        rf"$\mathbf{{Position:}}$" + f" {lon_str}, {lat_str}"                   # Position label
        + "\u2007" * max(0, 27 - len(f"Lon: {lon_str}, Lat: {lat_str}")) + "\n"
    )

    label += (
        rf"$\mathbf{{Method:}}$ {method}"                                           # Method label
        + "\u2007" * max(0, 31 - len(f"Method: {method}"))
    )

    fig_main.text(0.02, 0.025 if method == 'hm0_top3_mean' else 0.02, 
        label,
        transform=fig_main.transFigure,
        fontsize=10 if method == 'hm0_top3_mean' else 12,
        color=textcolor,
        horizontalalignment='left',
        bbox=dict(facecolor='none', edgecolor='none', boxstyle='square,pad=0.5')   
    )

    if method == 'hm0_max':
        period_label = pd.to_datetime(data_aggregated_spec.time.values).to_pydatetime().strftime('%Y-%m-%dT%H') + 'Z'
    elif method == 'hm0_top3_mean':
        times = [pd.to_datetime(t).strftime('%Y-%m-%dT%H') + 'Z' for t in data_aggregated_spec.attrs['hm0_top3_timesteps']]
        period_label = ', '.join(times[:3])

    if month is None:
        label2 = rf"$\mathbf{{Period:}}$ {period_label}" + "\n"                                  # Show full period label if no month is selected
    else:
        min_year = filtered_data.time.dt.year.min().item()
        max_year = filtered_data.time.dt.year.max().item()
        year_str = f"{min_year}" if min_year == max_year else f"{min_year} - {max_year}"
        month_name = next((key for key, val in month_map.items() if val == month), None)
        label2 = (
            rf"$\mathbf{{Year:}}$ {year_str}"                                                    # Show year range and selected month
            + " " * 15  
            + rf"$\mathbf{{Month:}}$ {month_name}" + "\n"
        )
    label2 += (rf"$\mathbf{{H_{{m_0}}:}}$ {hm0_aggregated:.1f} m")                               # Hm0 label    

    fig_main.text(0.4 if method == 'hm0_top3_mean' else 0.47, 0.025 if method == 'hm0_top3_mean' else 0.02, 
        label2,
        transform=fig_main.transFigure,
        fontsize=10 if method == 'hm0_top3_mean' else 12,
        color=textcolor,
        horizontalalignment='left',
        bbox=dict(facecolor='none', edgecolor='none', boxstyle='square,pad=0.5')   
    )


    ############ Layout ############
    rect = patches.Rectangle(
        (0, 0), 1, 0.1,
        transform=fig_main.transFigure,
        facecolor=color,
        alpha=alpha,
        edgecolor='none'
    )
    fig_main.patches.append(rect)

    ax_2D_spectra.set_zorder(2)   # Bring 2D spectrum plot to front
    ax_1D_spectra.set_zorder(1)   # Push 1D inset to back

    if output_file is not None:
        plt.savefig(output_file, dpi=200, bbox_inches=None, pad_inches=0)
    plt.close()
    return fig_main


def plot_partition_peak_dir_freq_2d(data, period = None, month=None, hm0_threshold = 1, windsea_freq_mask_percentile=None, swell_freq_mask_percentile=None, output_file = 'partition_peak_dir_freq_2d.png'):
    '''
    Partition the wave spectra into wind sea and swell. Generate 2D direction-frequency polar plots showing the number 
    of occurrences of peak direction-frequency pairs for each case.

    Parameters
    - data : xarray.Dataset
        Dataset with 2D directional-frequency wave spectra.
    - period : list of two str, optional, default=None
        A list specifying the desired time range.
        - [start_time, end_time]: Filters between start_time and end_time, e.g., ('2021-01-01T00', '2021-12-31T23').
        - [start_time]: Filters to a single timestamp.
        - None: Uses the full time range available in data.
        Both start_time and end_time may be strings or datetime-like objects.
    - month : int, optional, default=None
        Filter by month (1 = January, ..., 12 = December). If set, selects data from that month across all years.
        If None, no filtering is applied.
    - hm0_threshold : int, optional, default=1
        Filters the data to only include timestamps where hm0 is over the threshold.
    - windsea_freq_mask_percentile, swell_freq_mask_percentile : int, optional, default=None
        Percentile of peak frequencies used as an upper frequency limit for plotting.
        If None, the maximum peak frequency is used.
    - output_file : str, optional, default='partition_peak_dir_freq_2d.png'
        Output filename for saving the figure. 

    Returns
    fig_main : matplotlib.figure.Figure
        The generated plot figure.
    '''

    # Style preferences — change these to update all colors in the figure consistently
    cmap = cm.ocean_r                                                              
    color="#0463d7ff"                                                                 
    alpha = 0.22

    ################## Data handling ##################
    data = spec_funcs.standardize_wave_dataset(data)                                                            # Standardize the 2D wave spectrum dataset to match the WINDSURFER/NORA3 format.

    filtered_data, period_label = spec_funcs.filter_period(data=data, period=period)                            # Filters the dataset to the specified time period.

    filtered_data['wind_direction'] = ((450 - np.rad2deg(filtered_data['wind_direction']))%360)                 # Convert to mathematical convention (radians, pointing to East counterclockwise)

    sp = spec_funcs.Spectral_Partition_wind(filtered_data, beta=1.3)                                            # Partition wave spectrum into wind sea and swell

    sp_windsea = sp[[v for v in sp.data_vars if 'windsea' in v.lower()]]
    sp_swell   = sp[[v for v in sp.data_vars if 'swell' in v.lower()]]

    # Selects data where Hm0 > threshold
    sp_windsea_aggregated = spec_funcs.aggregate_spectrum(sp_windsea, sp_windsea['Hm0_windsea'], method = 'hm0_threshold', month = month, hm0_threshold = hm0_threshold)
    sp_swell_aggregated = spec_funcs.aggregate_spectrum(sp_swell, sp_swell['Hm0_swell'], method = 'hm0_threshold', month = month, hm0_threshold = hm0_threshold)

    fig_main = plt.figure(figsize=(10, 5))

    ################## plot windsea waves ##################
    ax_2D_spectra_windsea = fig_main.add_axes([0.01, 0.25, 0.45, 0.6], projection='polar')     
    plot_partition_spec_on_ax(ax=ax_2D_spectra_windsea, sp=sp_windsea, sp_aggregated=sp_windsea_aggregated, wave_partition='windsea', freq_mask_percentile=windsea_freq_mask_percentile, cmap=cmap)   

    ################## Plot swell waves ##################
    ax_2D_spectra_swell = fig_main.add_axes([0.45, 0.25, 0.52, 0.6], projection='polar')  
    plot_partition_spec_on_ax(ax=ax_2D_spectra_swell, sp=sp_swell, sp_aggregated=sp_swell_aggregated, wave_partition='swell', freq_mask_percentile=swell_freq_mask_percentile, cmap=cmap)

    ############ Create label with position, period, and Hm0 for the plot ############
    mean_hm0_windsea = sp_windsea_aggregated['Hm0_windsea'].mean(dim='time') 
    mean_hm0_swell = sp_swell_aggregated['Hm0_swell'].mean(dim='time') 

    month_map = {                                                                               
        'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
        'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}

    lat_str = ", ".join(
            f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}"                            # Format latitudes with N/S
            for lat in np.unique(data['latitude'].values)
        )

    lon_str = ", ".join(
        f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"                                # Format longitudes with E/W
        for lon in np.unique(data['longitude'].values)
    )


    label = (
        rf"$\mathbf{{Position:}}$ {lon_str}, {lat_str}"                             # Position label
        + "\u2007" * max(0, 27 - len(f"Lon: {lon_str}, Lat: {lat_str}")) + "\n"
    )

    label += (
        rf"$\mathbf{{Hm0\ threshold:}}$ {hm0_threshold} m"                          # Method label
        )
    

    fig_main.text(0.02, 0.02, 
        label,
        transform=fig_main.transFigure,
        fontsize=12,
        color='black',
        horizontalalignment='left',
        bbox=dict(facecolor='none', edgecolor='none', boxstyle='square,pad=0.5')   
    )

    label1 = (rf"$\mathbf{{Mean\ H_{{m_0}}\ wind sea:}}$ {mean_hm0_windsea:.1f} m" + '\n')     # Hm0 label    
    label1 += (rf"$\mathbf{{Mean\ H_{{m_0}}\ swell:}}$ {mean_hm0_swell:.1f} m") 

    fig_main.text(0.33 if month is None else 0.4, 0.02, 
        label1,
        transform=fig_main.transFigure,
        fontsize=12,
        color='black',
        horizontalalignment='left',
        bbox=dict(facecolor='none', edgecolor='none', boxstyle='square,pad=0.5')   
    )

    if month is None:
        label2 = '\n' + rf"$\mathbf{{Period:}}$ {period_label}"                                  # Show full period label if no month is selected
    else:
        min_year = filtered_data.time.dt.year.min().item()
        max_year = filtered_data.time.dt.year.max().item()
        year_str = f"{min_year}" if min_year == max_year else f"{min_year} - {max_year}"
        month_name = next((key for key, val in month_map.items() if val == month), None)
        label2 = (
            rf"$\mathbf{{Year:}}$ {year_str}"                                                    # Show year range and selected month
            + '\n'
            + rf"$\mathbf{{Month:}}$ {month_name}"
        )
    

    fig_main.text(0.6 if month is None else 0.8, 0.02, 
        label2,
        transform=fig_main.transFigure,
        fontsize=12,
        color='black',
        horizontalalignment='left',
        bbox=dict(facecolor='none', edgecolor='none', boxstyle='square,pad=0.5')   
    )

    ############ Layout ############
    rect = patches.Rectangle(
        (0, 0), 1, 0.11,
        transform=fig_main.transFigure,
        facecolor=color,
        alpha=alpha,
        edgecolor='none'
    )
    fig_main.patches.append(rect)

    if output_file is not None:
        plt.savefig(output_file, dpi=200, bbox_inches=None, pad_inches=0)
    plt.close()
    return fig_main


def plot_spectra_2d(data,var='SPEC', period=None, method='monthly_mean', plot_type='pcolormesh', cbar='single', radius='frequency',dir_letters=False,output_file='spectra_2d.png'):
    '''
    Creates a 12-panel plot of aggregated 2D spectra based on method. 

    Parameters
    - data : xarray.Dataset
        Dataset with 2D directional-frequency wave spectra.
    - var : str, optional, default='SPEC'
        Spectral variable name.
    - period : tuple of two str, optional, default=None
        Time range to filter the data as (start_time, end_time), e.g., ('2021-01-01T00', '2021-12-31T23').
        If None, the full time range in the dataset is used.
        Both start_time and end_time may be strings or datetime-like objects.
    -method : str, optional, default='mean'
        Aggregation method:  
        - 'monthly mean' : Average over month 
        - 'direction' : average of data with peak direction within the specified sector.
    - plot_type : str, optional, default='pcolormesh'
        2D spectrum plotting method. Choose 'pcolormesh' or 'contour'.
    - cbar : str, optional, default='single'
        Number of colorbars
            - 'single' : one colorbar for all 12 subplots
            - 'multiple' : one colorbar per subplot
    - radius : str, optional, default='frequency'
        Units for radial axis, period or frequency.
    - dir_letters : bool, optional, default=False
        Use compass labels instead of degrees.
    - output_file : str, optional, default='spectra_2d.png'
        Output filename for saving the figure.

    Returns
    fig_main : matplotlib.figure.Figure
        The generated plot figure.

    Notes:
    - If method = 'direction' each panel represents a 30° directional sector and displays the average 2D wave spectrum computed 
      from all time steps where the peak wave wave direction falls within that sector. The percentage 
      shown in each panel indicates the fraction of total spectra contributing to that sector's average.
    '''

    valid_methods = ['monthly_mean', 'direction']
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Available methods are: {', '.join(valid_methods)}.")

    # Standardize the 2D wave spectrum dataset to match the WINDSURFER/NORA3 format.
    data = spec_funcs.standardize_wave_dataset(data)

    # Filters the dataset to the specified time period.
    if isinstance(period, str):
        raise ValueError("Period must include both start and end times, e.g. ('2024-01-01T00', '2024-12-31T23').")
    filtered_data, period_label = spec_funcs.filter_period(data=data, period=period)

    datspec=filtered_data[var]
    frequencies = datspec.coords['freq']
    directions = datspec.coords['direction']

    # Normalize and wrap directional spectral data to 0-360°.
    spec, dirc = spec_funcs.wrap_directional_spectrum(data, var=var)
    
    cmap = plt.cm.hot_r 
    
    if method == 'monthly_mean':
        max_spec = np.round(spec.groupby('time.month').mean(dim='time').max().item(),1) 
        norm = plt.Normalize(vmin=0, vmax=max_spec)

        months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 16), subplot_kw=dict(projection="polar"))
        mon=1
        for r in range(4):
            for c in range(3):
                data_spec=spec.sel(time=spec.time.dt.month.isin([mon])).mean(dim='time')
                axs[r,c].set_theta_zero_location('N')
                axs[r,c].set_theta_direction(-1)

                if radius=='period':
                    rad_data=1/frequencies.values
                elif radius=='frequency':
                    rad_data=frequencies.values

                if plot_type == 'contour':
                    cmap = cm.hot_r(np.linspace(0,1,10))
                    cmap = mcol.ListedColormap(cmap)
                    cp=axs[r,c].contourf(np.radians(dirc.values), rad_data, data_spec.values, cmap=cmap, vmin=0, vmax=max_spec if cbar=='single' else None)
                elif plot_type == 'pcolormesh':
                    cp = axs[r,c].pcolormesh(np.radians(dirc.values), rad_data, data_spec.values, cmap=cmap, shading='auto', vmin=0, vmax=max_spec if cbar=='single' else None)
                
                if dir_letters:
                    ticks_loc = axs[r,c].get_xticks().tolist()
                    axs[r,c].xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                    axs[r,c].set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
                
                if cbar == 'multiple':
                    plt.colorbar(cp, label=r'[' + re.sub(r'\*\*(\d+)', r'$^\1$', datspec.units) + ']', pad=0.15) 
                
                axs[r,c].text(0.015,0.98,months[mon-1],transform=axs[r,c].transAxes,fontsize=14,weight='bold')
                mon=mon+1
                axs[r,c].grid(True)

    
    elif method == 'direction':
            idir=data[var][:,:,:].argmax(dim=['freq','direction'])['direction'].values
            dir_mean_spec=np.zeros((len(directions),len(frequencies),len(directions)+1))
            count=np.zeros((len(directions)))

            for i in range(np.shape(datspec)[0]):
                id=int(idir[i])
                dir_mean_spec[id,:,:]=dir_mean_spec[id,:,:]+spec[i,:,:].values
                count[id]=count[id]+1
            for i in range(len(directions)):
                if count[i]!=0:
                    dir_mean_spec[i,:,:]=dir_mean_spec[i,:,:]/count[i]
                else:
                    dir_mean_spec[i,:,:]=np.full((len(frequencies),len(directions)+1),np.nan)
            
            # Combines the directions into 30 deg sectors
            dir_mean_spec_1=np.zeros((int(len(directions)/2),len(frequencies),len(directions)+1))
            count1=np.zeros((int(len(directions)/2)))
            i1=0
            for i in range(-1,len(directions)-1,2):
                dir_mean_spec_1[i1,:,:]=(dir_mean_spec[i,:,:]+dir_mean_spec[i+1,:,:])/2
                count1[i1]=count[i]+count[i+1]
                i1=i1+1

            labels=['0$^{\circ}$','30$^{\circ}$','60$^{\circ}$','90$^{\circ}$','120$^{\circ}$','150$^{\circ}$','180$^{\circ}$','210$^{\circ}$',
                    '240$^{\circ}$','270$^{\circ}$','300$^{\circ}$','330$^{\circ}$']
            
            fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 16), subplot_kw=dict(projection="polar"))
            i1=0

            max_spec = np.round(np.nanmax(dir_mean_spec_1),1)
            norm = plt.Normalize(vmin=0, vmax=max_spec)
            for r in range(4):
                for c in range(3):
                    axs[r,c].set_theta_zero_location('N')
                    axs[r,c].set_theta_direction(-1)

                    if radius=='period':
                        rad_data=1/frequencies.values
                    elif radius=='frequency':
                        rad_data=frequencies.values
                    
                    if plot_type == 'contour':
                        cmap = cm.hot_r(np.linspace(0,1,10))
                        cmap = mcol.ListedColormap(cmap)
                        cp=axs[r,c].contourf(np.radians(dirc.values), rad_data, dir_mean_spec_1[i1,:,:], cmap=cmap, vmin=0, vmax=max_spec if cbar=='single' else None)
                    elif plot_type == 'pcolormesh':
                        cp = axs[r,c].pcolormesh(np.radians(dirc.values), rad_data, dir_mean_spec_1[i1,:,:], cmap=cmap, shading='auto', vmin=0, vmax=max_spec if cbar=='single' else None)
                    
                    if dir_letters:
                        ticks_loc = axs[r,c].get_xticks().tolist()
                        axs[r,c].xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                        axs[r,c].set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
                    
                    if cbar == 'multiple':
                        plt.colorbar(cp, label=r'[' + re.sub(r'\*\*(\d+)', r'$^\1$', datspec.units) + ']', pad=0.15) 
                    
                    axs[r,c].text(0,1,labels[i1],transform=axs[r,c].transAxes,fontsize=14,weight='bold')
                    axs[r,c].text(1,1,str(int(np.rint(count1[i1]/np.sum(count1)*100)))+'%',transform=axs[r,c].transAxes,fontsize=14)
                    i1=i1+1
                    axs[r,c].grid(True)
       

    if cbar == 'single':
        cbar = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(cp, cax=cbar) if plot_type=='pcolormesh' else fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar, ticks=np.linspace(0, max_spec, 11))
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.set_title(r'[' + re.sub(r'\*\*(\d+)', r'$^\1$', datspec.units) + ']', pad=10, fontsize=14)

    fig.text(0.1,0.07,rf'$\mathbf{{Period}}$: {period_label}',transform=fig.transFigure,fontsize=16, ha='left')

    lat_str = ", ".join(
        f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}"                                # Format latitudes with N/S
        for lat in np.unique(data['latitude'].round(1).values))

    lon_str = ", ".join(
        f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"                                # Format longitudes with E/W
        for lon in np.unique(data['longitude'].round(1).values))

    label_position = (
        rf"$\mathbf{{Position:}}$" + f" Lon: {lon_str}, Lat: {lat_str}"                   # Position label
        + "\u2007" * max(0, 27 - len(f"Lon: {lon_str}, Lat: {lat_str}")) + "\n"
    )

    fig.text(0.1,0.035,label_position,transform=fig.transFigure,fontsize=16, ha='left')

    if output_file is not None:
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    return fig


def plot_1d_spectrum_on_ax(data, ax, var='SPEC', color='#0463d7ff', alpha=0.22):
    '''
    Plots a 1D spectrum for a specified month on the given Matplotlib axis. Data is aggregated based on method.
    If month=None the full period is selected. 

    Parameters:
    - data : xarray.Dataset
        Dataset containing 2D wave spectra over time.
    - ax : matplotlib.axes.Axes
        Axes object on which to plot the spectrum.
    - var : str, optional, defaul = 'SPEC'
        Name of the spectral variable to plot.
    - color : str, optional, default = '#0463d7ff'
        Color of the plotted line (any Matplotlib-compatible color).
    - alpha : float, optional, default = 0.22
        Transparency level of the line (between 0 and 1). An offset of +0.3 is applied but capped at 1.
    '''

    spec_values = spec_funcs.from_2dspec_to_1dspec(data, var=var, dataframe=False, hm0=False)
    freq = spec_values.freq.values           

    ax.plot(freq,spec_values,color=color,linewidth=1.5, alpha=min(alpha + 0.55, 1.0))

    for spine in ax.spines.values():                                                                 # Remove all spines to start clean
        spine.set_visible(False)

    ax.set_xlim(left=0)                                                                              # Set limits so origin is visible
    ax.set_ylim(bottom=0)

    x_min = np.min(freq)
    y_min = np.min(spec_values)
    x_max = np.max(freq)
    y_max = np.max(spec_values)  
    x_ext = x_max*0.05
    y_ext = y_max*0.05
    ax.set_xlim(0, x_max + x_ext)
    ax.set_ylim(0, y_max + y_ext)

    list_yticks=ax.get_yticks(minor=False)
    ax.set_yticks(list_yticks[0:-1])

    list_xticks=ax.get_xticks(minor=False)
    ax.set_xticks(list_xticks[0:-2])

    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.yaxis.get_offset_text().set_fontsize(9)

    arrow_props = dict(arrowstyle='->', linewidth=1, color='black')
    ax.annotate('', xy=(x_max + x_ext, 0), xytext=(0, 0), arrowprops=arrow_props)                    # Draw arrows slightly beyond max data values
    ax.annotate('', xy=(0, y_max + y_ext), xytext=(0, 0), arrowprops=arrow_props)

    ax.text(x_max + x_ext * 1.05, 0, 'Frequency [Hz]', va='center', ha='left', fontsize=10)          # Axis labels just beyond arrows
    ax.text(0.3*(x_max - x_min), (y_max + y_ext * 1.05)*1.1, r'Variance Density [m$^2$/Hz]', va='bottom', ha='center', fontsize=10, color=None)


def plot_direction_arrow_on_ax(ax, direction, x0, y0, color='black', draw_ticks=False, speed_data=None, arrow_length=0.25, style='arrow'):
    '''
    Plots the direction as an arrow or line on the specified axes. 

    Parameters:
    - ax : matplotlib.axes.Axes
        Axis to plot on.
    - direction : xarray.DataArray or ndarray
        Direction data in oceanographic degrees (0° = North, increasing clockwise).
    - x0, y0 : float
        Base position of arrow in axes fraction coordinates (0 to 1).
    - color : str, optional, default='black'
        Color of arrow or line.
    - draw_ticks : bool, optional, default=False
        Whether to draw wind speed ticks (barbs) along the arrow.
    - speed_data : xarray.DataArray, optional, default=None
        Wind speed data in m/s, required if draw_ticks is True.
    - arrow_length : float, optional, default=0.25
        Length of the arrow or line in axes fraction units.
    - style : str, optional, default='arrow'
        'arrow' to plot arrow with head, or 'line' for plain line.
    '''

    if direction.max() > 2 * np.pi:  
        mean_rad = np.deg2rad(((450 - direction) % 360)+180)                                                # Convert meteorological degrees (0°=N, clockwise) to mathematical radians (0°=E, counter clockwise)
        

    else: mean_rad = direction

    dx, dy = np.cos(mean_rad) * arrow_length, np.sin(mean_rad) * arrow_length                               # Calculate arrow vector components
    start = (x0 + dx, y0 + dy)
    end = (x0, y0)

    if style == 'arrow':
        ax.annotate('', xy=start, xytext=end,
                    arrowprops=dict(facecolor=color, edgecolor='none', width=1, headwidth=3),
                    xycoords='axes fraction', zorder=5)
    elif style == 'line':
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color,
                linewidth=1, transform=ax.transAxes, zorder=5)

    # Draw wind speed ticks along the arrow if requested
    if draw_ticks and speed_data is not None:
        speed_knots = speed_data * 1.94384
        vec = -np.array([dx, dy]) / np.linalg.norm([dx, dy])                                                  # Unit vector along arrow direction (from arrow tip back)
        perp = np.array([-vec[1], vec[0]])                                                                    # Perpendicular unit vector for ticks

        full_ticks = int(speed_knots // 10)                                                                   # Number of full 10-knot ticks
        half_tick = (speed_knots % 10) >= 5                                                                   # Half tick if remainder ≥ 5 knots
        tick_len = 0.1                                                                                        # Tick length in axes fraction units
        spacing = 0.025                                                                                       # Distance between ticks along arrow

        # Draw full ticks
        for i in range(full_ticks):
            s = np.array(start) + vec * i * spacing
            e = s + perp * tick_len
            ax.plot([s[0], e[0]], [s[1], e[1]], color=color, linewidth=1, transform=ax.transAxes)

        # Draw half tick if applicable
        if half_tick:
            s = np.array(start) + vec * full_ticks * spacing
            e = s + perp * tick_len / 2
            ax.plot([s[0], e[0]], [s[1], e[1]], color=color, linewidth=1, transform=ax.transAxes)

        # Uncomment the following lines to print the mean direction
        # mean_deg = (450 - np.rad2deg(mean_rad)) % 360
        # print(f"Mean {label} direction:", mean_deg.values if hasattr(mean_deg, 'values') else mean_deg)

        if direction.max() > 2 * np.pi: 
            return mean_rad
        

def plot_partition_spec_on_ax(ax, sp, sp_aggregated, wave_partition, freq_mask_percentile, cmap):  
    '''
    Generates a 2D direction-frequency polar plot showing the number of occurrences of peak direction-frequency 
    pairs on the given Matplotlib axis.

    Parameters:
    - ax : matplotlib.axes.Axes
        Axes object on which to plot the spectrum.
    - sp : xarray.Dataset
        Dataset containing partitioned wave spectra. 
    - sp_aggregated : xarray.Dataset
        Dataset of partitioned wave spectra aggregated over times with Hm0 above a threshold.
        Must include peak direction and peak frequency.
    - wave_partition : str
        Wave partition (windsea or swell) used to select peak direction-frequency data.
    - freq_mask_percentile : int or None
        Percentile of peak frequencies used as an upper frequency limit for plotting.
        If None, the maximum peak frequency is used.
    - cmap : matplotlib.colors.Colormap
        Colormap used to visualize occurrence counts.
    '''

    peak_dirc = sp_aggregated[f'pdir_{wave_partition}']
    peak_freq = sp_aggregated[f'fp_{wave_partition}'] 

    # Limit frequencies to those below the max (or percentile-based) peak frequency to avoid empty regions in the plot
    r_edges = sp.freq[sp.freq <= np.nanpercentile(peak_freq.values, freq_mask_percentile) if freq_mask_percentile else sp.freq <= peak_freq.values.max()]

    theta_edges = sp.direction.values

    # Normalize and wrap directions to 0-360°.
    sorted_idx = np.argsort(theta_edges)                  
    directions = theta_edges[sorted_idx] % 360

    if directions[0] != directions[-1]:
            if directions[0] < 360:
                dirc = np.concatenate([directions, directions[0:1] + 360])      # Add 360° copy of first value
            else:
                dirc = np.concatenate([directions, directions[0:1]])            # Fallback: just repeat first value
 
    theta_edges = np.deg2rad(dirc)

    # Compute a 2D histogram counting occurrences of peak direction–frequency pairs within each bin
    H, theta_edges, r_edges = np.histogram2d(x=np.deg2rad(peak_dirc.values),  y=peak_freq.values, bins=[theta_edges, r_edges])

    # Create 2D grids of direction (theta) and frequency (r) bin edges for plotting
    Theta, R = np.meshgrid(theta_edges, r_edges)

    pc = ax.pcolormesh(Theta, R, H.T, cmap=cmap, shading='auto')

    cbar = plt.colorbar(pc, ax=ax, pad=0.1, label = 'no. of occurences')

    # Set up polar plot style and tick appearance
    max_freq = ax.get_rmax()
    ax.grid(True)
    ax.set_rlabel_position(315)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.tick_params(axis='y', labelsize=10) 
    ax.tick_params(axis='x', labelsize=10)
    ax.set_rticks(np.linspace(max_freq / 5, max_freq, 5))
    ax.set_rmax(max_freq * 1.1)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f' if max_freq > 0.5 else '%.2f'))
    ax.set_title('Wind sea waves' if wave_partition=='windsea' else 'Swell waves')
    ax.set_xlabel('Peak dir')
    ax.text(np.deg2rad(230), ax.get_rmax() * 0.75, 'Peak freq',
                            rotation=45, ha='center', va='center', fontsize=10)