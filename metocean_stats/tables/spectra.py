import pandas as pd
import numpy as np
from ..stats import spec_funcs 

def table_monthly_freq_1dspectrum(data, var='SPEC', method='mean', month=None, average_over='month', output_file='monthly_table_spectrum.csv'):
    '''
    Aggregate 1D wave spectra by month or by year for a specific month, and optionally save to CSV.

    Parameters
    - data : xarray.Dataset
        Dataset containing 2D wave spectra over time.
    - var : str, optional, default='SPEC'
        Name of the spectral variable in the input data.
    - month : int or None, optional, default=None
        - If None: aggregate by calendar month across all years (e.g. all Januaries averaged together).
        - If 1-12: aggregate by year for the given calendar month (e.g. January for each year).
    - method : str, optional, default='mean'
        Aggregation method:  
        - 'mean'            : Average over time.  
        - 'top_1_percent_mean'   : Average over times where Hm0 ≥ 99th percentile.  
        - 'hm0_max'         : Use time step with maximum Hm0.  
        - 'hm0_top3_mean'   : Average of Hm0 over the three time steps with the highest values.
    - average_over : str, optional, default='month'
        How to compute the final 'Average' row:
        - 'month': takes the mean of the previously computed monthly (or specified-month) aggregates.
        - 'whole_dataset': aggregates the full dataset directly using the selected method.
    - output_file : str or None, default='monthly_table_spectrum.csv'
        Path to save the output CSV file. If None, no file is saved.

    Returns
    - pandas.DataFrame
        Aggregated spectra with frequency columns and Hm0 values.
        Includes one row per month/year and a final 'Average' row.
    '''
    spec = spec_funcs.standardize_wave_dataset(data)                                                # Standardize the 2D wave spectrum dataset to match the WINDSURFER/NORA3 format.                                     # Standardize dataset       
    
    if not isinstance(spec, pd.DataFrame):                                                          # If not a preprocessed 1D spectrum (pandas.DataFrame)
        spec = spec_funcs.from_2dspec_to_1dspec(spec, var=var, dataframe=True, hm0=True)            # Calculate 1D frequency spectrum by integrating over directions from a 2D directional spectrum.


    freq_cols = [c for c in spec.columns if c not in ['time', 'Hm0']]                               # Define frequency columns (exclude 'time' and 'Hm0')

    if 'Hm0' not in spec.columns:
        spec['Hm0'] = spec_funcs.integrated_parameters_dict(data[var], data.freq, data.direction)['Hs']

    month_map = {                                                                                   # Define month names, with 'Average' as a label for the overall mean
        'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
        'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12,
        'Average':0}


    ######## Monthly aggregation across all years ########
    if month is None:
        months_idx = np.unique(spec['time'].dt.month)                                               # Extract unique month numbers from the time dimension (e.g., [2, 3, 4, ...])
        months = [k for k,v in month_map.items() if v in months_idx]                                # Get corresponding month names for the months present in the data
        months.append('Average')

        df = pd.DataFrame(columns=freq_cols)                                                        # Create empty DataFrame to store aggregated spectra
        list_hm0 = []

        # Loop over each month, aggregate data, and append to df
        for i, m in enumerate(months_idx):
            df_month = spec[spec['time'].dt.month == m]                                             # Select the subset of the data corresponding to the current month 'm'
            spec_mean, hm0_mean = aggregate_group(df_month, freq_cols, method)                      # Aggregate the spectral data and Hm0 for this month using the chosen method
            df.loc[i] = spec_mean.tolist()                                                        
            list_hm0.append(hm0_mean)

        # Compute the final 'Average' row depending on the average_over method
        if average_over == 'whole_dataset':                                                         # Aggregate over the entire dataset directly using the chosen method
            spec_mean, hm0_mean = aggregate_group(spec, freq_cols, method=method)
            avg_vals = spec_mean
            avg_hm0 = hm0_mean
        else:                                                                                       # Take the average of the previously computed yearly aggregates
            if len(df) > 1:
                avg_vals = df[freq_cols].mean()
                avg_hm0 = np.mean(list_hm0)
            else:
                avg_vals = df[freq_cols].iloc[0]                                                    # If only one year present, use its values directly
                avg_hm0 = list_hm0[0]

        # Create the last row
        df.loc[len(df)] = pd.Series(data=list(avg_vals) + [avg_hm0], index=freq_cols + ['Hm0'])
        list_hm0.append(avg_hm0)

        # Add Hm0 values and Month labels to the DataFrame
        df['Hm0'] = list_hm0
        df.insert(0, 'Month', months)
        df.insert(1, 'Month_no', df['Month'].map(month_map).fillna(0).astype(int))


    ######## Yearly aggregation for a specific month ########
    else:
        if not (1 <= month <= 12):
            raise ValueError("month must be between 1 and 12")
        
        spec_month = spec[spec['time'].dt.month == month].copy()                                    # Filter data for the given month only
        spec_month[f'Year'] = spec_month['time'].dt.year                                            # Extract the years
        years = sorted(spec_month['Year'].unique())                                                 # Sorted list of unique years in data for that month

        df = pd.DataFrame(columns=freq_cols)                                                        # Create empty DataFrame to store aggregated spectra per year for the specified month
        list_hm0 = []                                                                               # Create empty list to store aggregated hm0 per year for the specified month

        # Loop over each year, aggregate data, and append to df and list_hm0
        for i, y in enumerate(years):
            df_year = spec_month[spec_month['Year'] == y]                                           # Select data for the current year 'y'                          
            spec_mean, hm0_mean = aggregate_group(df_year, freq_cols, method)                       # Aggregate the spec data and Hm0 for this year using the specified method
            df.loc[i] = spec_mean.tolist()                                          
            list_hm0.append(hm0_mean)                                              

        # Compute the final 'Average' row depending on the average_over method
        if average_over == 'whole_dataset':
            spec_mean, hm0_mean = aggregate_group(spec, freq_cols, method=method)                   # Aggregate over the entire dataset directly using the chosen method
            avg_vals = spec_mean
            avg_hm0 = hm0_mean
        else:                                                                                       # Take the average of the previously computed yearly aggregates
            if len(df) > 1:
                avg_vals = df[freq_cols].mean()
                avg_hm0 = np.mean(list_hm0)
            else:
                avg_vals = df[freq_cols].iloc[0]                                                    # If only one year present, use its values directly
                avg_hm0 = list_hm0[0]
        
        # Create the last row
        df.loc[len(df)] = pd.Series(data=list(avg_vals) + [avg_hm0], index=freq_cols + ['Hm0'])
        list_hm0.append(avg_hm0)

        # Add Hm0 values and Year labels (including 'Average') to the DataFrame
        df['Hm0'] = list_hm0
        df.insert(0, 'Year', years + ['Average'])

        # Set the index name to the specified month’s abbreviation
        df.index.name = next((key for key, value in month_map.items() if value == month), None)

    # Write output CSV if requested
    if output_file is not None:
        if not output_file.endswith('.csv'):
            output_file += '.csv'
        with open(output_file, 'w') as f:
            f.write('# Frequencies (in Hz) are given in the header\n')
            f.write('# Spectra in m**2 s\n')
            df.to_csv(f, index=False)

    return df

def aggregate_group(df, freq_cols, method):
    '''
    Aggregate frequency columns and the 'Hm0' column from a 1D frequency spectrum DataFrame based on the specified method.

    Parameters:
    - df : pandas.DataFrame
        Input DataFrame containing frequency columns and an 'Hm0' column.
    - freq_cols : list of str
        List of column names in df that contain frequency data to aggregate.
    - method : str
        Aggregation method to apply. Supported methods are:
        - 'mean': average all rows.
        - 'top_1_percent_mean': average only rows where 'Hm0' is in the top 1% quantile.
        - 'hm0_max': return the row with the maximum 'Hm0' value.
        - 'hm0_top3_mean': average the top 3 rows with highest 'Hm0' values.

    Returns:
    - tuple of (aggregated_freq, aggregated_hm0)
        - aggregated_freq: pandas Series with aggregated frequency columns.
        - aggregated_hm0: float, aggregated 'Hm0' value.
    '''

    if method == 'mean':
        return df[freq_cols].mean(), df['Hm0'].mean()               # Average all rows for frequency columns and 'Hm0'
    
    elif method == 'top_1_percent_mean':
        p99 = df['Hm0'].quantile(0.99)                              # Calculate 99th percentile of 'Hm0'
        top = df[df['Hm0'] >= p99]                                  # Filter rows where 'Hm0' >= 99th percentile
        return top[freq_cols].mean(), top['Hm0'].mean()             # Average frequency columns and 'Hm0' over top 1% rows
    
    elif method in ('hm0_max'):
        max_row = df.loc[df['Hm0'].idxmax()]                        # Find row with maximum 'Hm0' value
        return max_row[freq_cols], max_row['Hm0']                   # Return frequency columns and 'Hm0' from that row
    
    elif method == 'hm0_top3_mean':                                 
        top3 = df.nlargest(3, 'Hm0')                                # Select top 3 rows with highest 'Hm0' values
        return top3[freq_cols].mean(), top3['Hm0'].mean()           # Average frequency columns and 'Hm0' over top 3 rows
    
    else:
        raise ValueError(f"Unknown method: {method}")