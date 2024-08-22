import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
from math import floor,ceil

from ..stats.aux_funcs import *
from ..stats.general import *


def scatter_diagram(data: pd.DataFrame, var1: str, step_var1: float, var2: str, step_var2: float,number_of_events=False, output_file='scatter_diagram.csv'):
    """
    The function is written by dung-manh-nguyen and KonstantinChri.
    Plot scatter diagram (heatmap) of two variables (e.g, var1='hs', var2='tp')
    step_var: size of bin e.g., 0.5m for hs and 1s for Tp
    cmap: colormap, default is 'Blues'
    outputfile: name of output file with extrensition e.g., png, eps or pdf 
     """

    sd = calculate_scatter(data, var1, step_var1, var2, step_var2)
    start_date = data.index.min()
    end_date = data.index.max() 
    # Calculate the difference in years
    number_of_years = (end_date - start_date).days / 365.25
    dt = (data.index.to_series().diff().dropna().dt.total_seconds() / 3600).mean()

    # Convert to percentage
    tbl = sd.values
    var1_data = data[var1]
    tbl = tbl/len(var1_data)*100

    # Then make new row and column labels with a summed percentage
    sumcols = np.sum(tbl, axis=0)
    sumrows = np.sum(tbl, axis=1)

    sumrows = np.around(sumrows, decimals=2)
    sumcols = np.around(sumcols, decimals=2)

    bins_var1 = sd.index
    bins_var2 = sd.columns
    lower_bin_1 = bins_var1[0] - step_var1
    lower_bin_2 = bins_var2[0] - step_var2

    rows = []
    rows.append(f'{lower_bin_1:04.1f}-{bins_var1[0]:04.1f} | {sumrows[0]:04.2f}%')
    for i in range(len(bins_var1)-1):
        rows.append(f'{bins_var1[i]:04.1f}-{bins_var1[i+1]:04.1f} | {sumrows[i+1]:04.2f}%')

    cols = []
    cols.append(f'{int(lower_bin_2)}-{int(bins_var2[0])} | {sumcols[0]:04.2f}%')
    for i in range(len(bins_var2)-1):
        cols.append(f'{int(bins_var2[i])}-{int(bins_var2[i+1])} | {sumcols[i+1]:04.2f}%')

    #breakpoint()
    #cols.insert(0,var1+' / '+var2 )
    rows = rows[::-1]
    tbl = tbl[::-1,:]
    
    if number_of_events==True:
        #number of events 
        tbl = np.flip(sd.values,axis=0)
    else:
        pass

    dfout = pd.DataFrame(data=np.round(tbl,2), index=rows, columns=cols)
    if output_file.split('.')[-1]=='csv':
        dfout.to_csv(output_file,index_label=var1+'/'+var2)
    elif output_file.split('.')[-1]=='png':
        hi = sns.heatmap(data=dfout.where(dfout>0), cbar=True, cmap='Blues', fmt=".1f")
        plt.ylabel(var1)
        plt.xlabel(var2)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    return dfout

def table_var_sorted_by_hs(data, var, var_hs='hs', output_file='var_sorted_by_Hs.csv'):
    """
    The function is written by dung-manh-nguyen and KonstantinChri.
    This will sort variable var e.g., 'tp' by 1 m interval of hs
    then calculate min, percentile 5, mean, percentile 95 and max
    data : pandas DataFrame
    var  : variable to analyze
    var_hs : variable for binning
    output_file: CSV file to save the results
    """
    Hs = data[var_hs]
    Var = data[var]
    binsHs = np.arange(0., math.ceil(np.max(Hs)) + 0.1)  # +0.1 to get the last one

    Var_binsHs = {}
    for j in range(len(binsHs) - 1):
        Var_binsHs[str(int(binsHs[j])) + '-' + str(int(binsHs[j + 1]))] = []

    N = len(Hs)
    for i in range(N):
        for j in range(len(binsHs) - 1):
            if binsHs[j] <= Hs.iloc[i] < binsHs[j + 1]:
                Var_binsHs[str(int(binsHs[j])) + '-' + str(int(binsHs[j + 1]))].append(Var.iloc[i])

    # Collecting data in a list of dictionaries
    data_list = []

    for j in range(len(binsHs) - 1):
        Var_binsHs_temp = Var_binsHs[str(int(binsHs[j])) + '-' + str(int(binsHs[j + 1]))]

        Var_min = round(np.min(Var_binsHs_temp), 1)
        Var_P5 = round(np.percentile(Var_binsHs_temp, 5), 1)
        Var_mean = round(np.mean(Var_binsHs_temp), 1)
        Var_P95 = round(np.percentile(Var_binsHs_temp, 95), 1)
        Var_max = round(np.max(Var_binsHs_temp), 1)

        hs_bin_temp = str(int(binsHs[j])) + '-' + str(int(binsHs[j + 1]))
        data_list.append({
            'Hs': hs_bin_temp,
            'Entries': len(Var_binsHs_temp),
            'Min': Var_min,
            '5%': Var_P5,
            'Mean': Var_mean,
            '95%': Var_P95,
            'Max': Var_max
        })

    # Adding the last empty row
    hs_bin_temp = str(int(binsHs[-1])) + '-' + str(int(binsHs[-1] + 1))
    data_list.append({
        'Hs': hs_bin_temp,
        'Entries': 0,
        'Min': '-',
        '5%': '-',
        'Mean': '-',
        '95%': '-',
        'Max': '-'
    })

    # Annual row
    Var_min = round(np.min(Var), 1)
    Var_P5 = round(np.percentile(Var, 5), 1)
    Var_mean = round(np.mean(Var), 1)
    Var_P95 = round(np.percentile(Var, 95), 1)
    Var_max = round(np.max(Var), 1)

    hs_bin_temp = str(int(binsHs[0])) + '-' + str(int(binsHs[-1] + 1))
    data_list.append({
        'Hs': hs_bin_temp,
        'Entries': len(Var),
        'Min': Var_min,
        '5%': Var_P5,
        'Mean': Var_mean,
        '95%': Var_P95,
        'Max': Var_max
    })

    # Creating DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)

    # Save DataFrame to CSV
    df.to_csv(output_file, index=False)

    return df

def table_monthly_percentile(data,var,output_file='var_monthly_percentile.txt'):  
    """
    The function is written by dung-manh-nguyen and KonstantinChri.
    this function will sort variable var e.g., hs by month and calculate percentiles 
    data : panda series 
    var  : variable 
    output_file: extension .txt for latex table or .csv for csv table
    """

    Var = data[var]
    varName = data[var].name
    Var_month = Var.index.month
    M = Var_month.values
    temp_file = output_file.split('.')[0]
    
    months = calendar.month_name[1:] # eliminate the first insane one 
    for i in range(len(months)) : 
        months[i] = months[i][:3] # get the three first letters 
    
    monthlyVar = {}
    for i in range(len(months)) : 
        monthlyVar[months[i]] = [] # create empty dictionaries to store data 
    
    
    for i in range(len(Var)) : 
        m_idx = int(M[i]-1) 
        monthlyVar[months[m_idx]].append(Var.iloc[i])  
        
        
    with open(temp_file, 'w') as f:
        f.write('\\begin{tabular}{l | p{1.5cm} p{1.5cm} p{1.5cm} p{1.5cm} p{1.5cm}}' + '\n')
        f.write('& \multicolumn{5}{c}{' + varName + '} \\\\' + '\n')
        f.write('Month & 5\% & 50\% & Mean & 95\% & 99\% \\\\' + '\n')
        f.write('\hline' + '\n')
    
        for j in range(len(months)) : 
            Var_P5 = round(np.percentile(monthlyVar[months[j]],5),1)
            Var_P50 = round(np.percentile(monthlyVar[months[j]],50),1)
            Var_mean = round(np.mean(monthlyVar[months[j]]),1)
            Var_P95 = round(np.percentile(monthlyVar[months[j]],95),1)
            Var_P99 = round(np.percentile(monthlyVar[months[j]],99),1)
            f.write(months[j] + ' & '+str(Var_P5)+' & '+str(Var_P50)+' & '+str(Var_mean)+' & '+str(Var_P95)+' & '+str(Var_P99)+' \\\\' + '\n')
        
        # annual row 
        f.write('Annual & '+str(Var_P5)+' & '+str(Var_P50)+' & '+str(Var_mean)+' & '+str(Var_P95)+' & '+str(Var_P99)+' \\\\' + '\n')
    
        f.write('\hline' + '\n')
        f.write('\end{tabular}' + '\n')
    
    if output_file.split('.')[1] == 'csv':
        convert_latexTab_to_csv(temp_file, output_file)
        os.remove(temp_file)
    else:
        os.rename(temp_file, output_file)
    
    return   


def table_monthly_min_mean_max(data, var,output_file='montly_min_mean_max.txt') :  
    """
    The function is written by dung-manh-nguyen and KonstantinChri.
    It calculates monthly min, mean, max based on monthly maxima. 
    data : panda series
    var  : variable 
    output_file: extension .txt for latex table or .csv for csv table
    """
        
    var = data[var]
    temp_file  = output_file.split('.')[0]
    months = calendar.month_name[1:] # eliminate the first insane one 
    for i in range(len(months)) : 
        months[i] = months[i][:3] # get the three first letters 
    
    monthly_max = var.resample('M').max() # max in every month, all months 
    minimum = monthly_max.groupby(monthly_max.index.month).min() # min, sort by month
    mean = monthly_max.groupby(monthly_max.index.month).mean() # mean, sort by month
    maximum = monthly_max.groupby(monthly_max.index.month).max() # max, sort by month
    
    with open(temp_file, 'w') as f :
        f.write('\\begin{tabular}{l | c c c }' + '\n')
        f.write('Month & Minimum & Mean & Maximum \\\\' + '\n')
        f.write('\hline' + '\n')
        for i in range(len(months)):
            f.write(months[i] + ' & ' + str(minimum.values[i]) + ' & ' + str(round(mean.values[i],1)) + ' & ' + str(maximum.values[i]) + ' \\\\' + '\n')
        
        ## annual row 
        annual_max = var.resample('Y').max()
        min_year = annual_max.min()
        mean_year = annual_max.mean()
        max_year = annual_max.max()
        f.write('Annual Max. & ' + str(min_year) + ' & ' + str(round(mean_year,1)) + ' & ' + str(max_year) + ' \\\\' + '\n')
            
        f.write('\hline' + '\n')
        f.write('\end{tabular}' + '\n')


    if output_file.split('.')[1] == 'csv':
        convert_latexTab_to_csv(temp_file, output_file)
        os.remove(temp_file)
    else:
        os.rename(temp_file, output_file)

    return


def table_monthly_non_exceedance(data: pd.DataFrame, var: str, step_var: float, output_file: str = None):
    """
    Calculate monthly non-exceedance table for a given variable.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing the data.
        var (str): Name of the column in the DataFrame representing the variable.
        step_var (float): Step size for binning the variable.
        output_file (str, optional): File path to save the output CSV file. Default is None.

    Returns:
        pd.DataFrame: Monthly non-exceedance table with percentage of time each data level occurs in each month.
    """

# Define  bins
    bins = np.arange(int(data[var].min()), data[var].max() + step_var, step_var).tolist()
    labels =  [f'<{num}' for num in [round(bin, 2) for bin in bins]]

    # Categorize data into bins
    data[var+'-level'] = pd.cut(data[var], bins=bins, labels=labels[1:])
    
    # Group by month and var bin, then count occurrences
    grouped = data.groupby([data.index.month, var+'-level'], observed=True).size().unstack(fill_value=0)


    # Calculate percentage of time each data bin occurs in each month
    percentage_by_month = grouped.div(grouped.sum(axis=1), axis=0) * 100

    # Calculate cumulative percentage for each bin across all months
    cumulative_percentage = percentage_by_month.T.cumsum()

    # Insert 'Annual', 'Mean', 'P99', 'Maximum' rows
    cumulative_percentage.loc['Minimum'] = data.groupby(data.index.month, observed=True)[var].min()
    cumulative_percentage.loc['Mean'] = data.groupby(data.index.month, observed=True)[var].mean()
    cumulative_percentage.loc['P50'] = data.groupby(data.index.month, observed=True)[var].quantile(0.50)
    cumulative_percentage.loc['P75'] = data.groupby(data.index.month, observed=True)[var].quantile(0.75)
    cumulative_percentage.loc['P95'] = data.groupby(data.index.month, observed=True)[var].quantile(0.95)
    cumulative_percentage.loc['P99'] = data.groupby(data.index.month, observed=True)[var].quantile(0.99)
    cumulative_percentage.loc['Maximum'] = data.groupby(data.index.month, observed=True)[var].max()
    cumulative_percentage['Year'] = cumulative_percentage.mean(axis=1)[:-6]
    #cumulative_percentage['Year'].iloc[-7] = data[var].min()    
    #cumulative_percentage['Year'].iloc[-6] = data[var].mean()    
    #cumulative_percentage['Year'].iloc[-5] = data[var].quantile(0.50)
    #cumulative_percentage['Year'].iloc[-4] = data[var].quantile(0.75)
    #cumulative_percentage['Year'].iloc[-3] = data[var].quantile(0.95)
    #cumulative_percentage['Year'].iloc[-2] = data[var].quantile(0.99)
    #cumulative_percentage['Year'].iloc[-1] = data[var].max()

    cumulative_percentage.loc[cumulative_percentage.index[-7], 'Year'] = data[var].min()
    cumulative_percentage.loc[cumulative_percentage.index[-6], 'Year'] = data[var].mean()
    cumulative_percentage.loc[cumulative_percentage.index[-5], 'Year'] = data[var].quantile(0.50)
    cumulative_percentage.loc[cumulative_percentage.index[-4], 'Year'] = data[var].quantile(0.75)
    cumulative_percentage.loc[cumulative_percentage.index[-3], 'Year'] = data[var].quantile(0.95)
    cumulative_percentage.loc[cumulative_percentage.index[-2], 'Year'] = data[var].quantile(0.99)
    cumulative_percentage.loc[cumulative_percentage.index[-1], 'Year'] = data[var].max()

    # Round 2 decimals
    cumulative_percentage = round(cumulative_percentage,2)

    rename_mapping = {
    1: 'JAN',
    2: 'FEB',
    3: 'MAR',
    4: 'MAY',
    5: 'APR',
    6: 'JUN',
    7: 'JUL',
    8: 'AUG',
    9: 'SEP',
    10: 'OCT',
    11: 'NOV',
    12: 'DEC'
}
    # Rename the columns
    cumulative_percentage.rename(columns=rename_mapping, inplace=True)

    # Write to CSV file if output_file parameter is provided
    if output_file:
        cumulative_percentage.to_csv(output_file)
    
    return cumulative_percentage

def table_directional_non_exceedance(data: pd.DataFrame, var: str, step_var: float, var_dir: str, output_file: str = None):
    """
    Calculate directional non-exceedance table for a given variable.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing the data.
        var (str): Name of the column in the DataFrame representing the variable.
        step_var (float): Step size for binning the variable.
        var_dir (str): Name of the column in the DataFrame representing the direction.
        output_file (str, optional): File path to save the output CSV file. Default is None.

    Returns:
        pd.DataFrame: Directional non-exceedance table with percentage of time each data level occurs in each direction.
    """

# Define  bins
    bins = np.arange(0, data[var].max() + step_var, step_var).tolist()
    labels =  [f'<{num}' for num in [round(bin, 2) for bin in bins]]
    
    add_direction_sector(data=data,var_dir=var_dir)

    # Categorize data into bins
    data[var+'-level'] = pd.cut(data[var], bins=bins, labels=labels[1:])

    data = data.sort_values(by='direction_sector')
    data = data.set_index('direction_sector')
    data.index.name = 'direction_sector'
    # Group by direction and var bin, then count occurrences
    # Calculate percentage of time each var bin occurs in each month
    percentage_by_dir = 100*data.groupby([data.index, var+'-level'], observed=True)[var].count().unstack()/len(data[var])
    cumulative_percentage = np.cumsum(percentage_by_dir,axis=1).T
    cumulative_percentage = cumulative_percentage.fillna(method='ffill')

    # Calculate cumulative percentage for each bin across all months
    # Insert 'Omni', 'Mean', 'P99', 'Maximum' rows
    cumulative_percentage.loc['Mean'] = data.groupby(data.index, observed=True)[var].mean()
    cumulative_percentage.loc['P50'] = data.groupby(data.index, observed=True)[var].quantile(0.50)
    cumulative_percentage.loc['P75'] = data.groupby(data.index, observed=True)[var].quantile(0.75)
    cumulative_percentage.loc['P95'] = data.groupby(data.index, observed=True)[var].quantile(0.95)
    cumulative_percentage.loc['P99'] = data.groupby(data.index, observed=True)[var].quantile(0.99)
    cumulative_percentage.loc['Maximum'] = data.groupby(data.index, observed=True)[var].max()
    cumulative_percentage['Omni'] = cumulative_percentage.sum(axis=1)[:-6]
    cumulative_percentage.loc[cumulative_percentage.index[-6], 'Omni'] = data[var].mean()
    cumulative_percentage.loc[cumulative_percentage.index[-5], 'Omni'] = data[var].quantile(0.50)
    cumulative_percentage.loc[cumulative_percentage.index[-4], 'Omni'] = data[var].quantile(0.75)
    cumulative_percentage.loc[cumulative_percentage.index[-3], 'Omni'] = data[var].quantile(0.95)
    cumulative_percentage.loc[cumulative_percentage.index[-2], 'Omni'] = data[var].quantile(0.99)
    #cumulative_percentage['Omni'].iloc[-1] = data[var].max()
    cumulative_percentage.loc[cumulative_percentage.index[-1], 'Omni'] = data[var].max()
    # Round 2 decimals
    cumulative_percentage = round(cumulative_percentage,2)

    rename_mapping = {
        0.0: '0°',
        30.0: '30°',
        60.0: '60°',
        90.0: '90°',
        120.0: '120°',
        150.0: '150°',
        180.0: '180°',
        210.0: '210°',
        240.0: '240°',
        270.0: '270°',
        300.0: '300°',
        330.0: '330°'
    }

    # Rename the columns
    cumulative_percentage.rename(columns=rename_mapping, inplace=True)


    # Write to CSV file if output_file parameter is provided
    if output_file:
        cumulative_percentage.to_csv(output_file)

    return cumulative_percentage


def table_monthly_weather_window(data: pd.DataFrame, var: str,threshold=5, window_size=12,output_file: str = None):
    results = []
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month in range(1, 13):
        avg_duration, p10, p50, p90 = weather_window_length(time_series=data[var],month=month ,threshold=threshold,op_duration=window_size,timestep=3)
        results.append((p10,p50, avg_duration, p90))
    results_df = pd.DataFrame(results, columns=['P10', 'P50', 'Mean', 'P90'], index=months).T.round(2)
    if output_file:
        # Save results to CSV
        results_df.to_csv('monthly_weather_window_results.csv')
    
    return results_df

def table_profile_stats(data: pd.DataFrame, var: str, z=[10, 20, 30], var_dir=None, output_file='table_profile_stats.csv'):
    # Initialize an empty list to store the results
    results = []
    
    # Iterate over each height in z
    if var_dir is None:
        for i in range(len(z)):
            results.append([z[i], np.round(data[var[i]].mean(),2), np.round(data[var[i]].std(),2), 
                            data[var[i]].quantile(0.05), data[var[i]].quantile(0.10),
                            data[var[i]].quantile(0.50), data[var[i]].quantile(0.90),
                            data[var[i]].quantile(0.95), data[var[i]].quantile(0.99),
                            data[var[i]].max(), str(data[var[i]].idxmax())   ])
        results.insert(0,['[m]', '[m/s]', '[m/s]', '[m/s]', '[m/s]', '[m/s]', '[m/s]', '[m/s]', '[m/s]', '[m/s]', 'YYYY-MM-DD hh:mm:ss'])
        df = pd.DataFrame(results, columns=['z', 'Mean', 'Std.dev', 'P5', 'P10', 'P50', 'P90', 'P95', 'P99', 'Max', 'Max Speed Event'])

    else:
        for i in range(len(z)):
            results.append([z[i], np.round(data[var[i]].mean(),2), np.round(data[var[i]].std(),2), 
                            data[var[i]].quantile(0.05), data[var[i]].quantile(0.10),
                            data[var[i]].quantile(0.50), data[var[i]].quantile(0.90),
                            data[var[i]].quantile(0.95), data[var[i]].quantile(0.99),
                            data[var[i]].max(),data[var_dir[i]].loc[data[var[i]].idxmax()] ,  str(data[var[i]].idxmax())   ])   
        results.insert(0,['[m]', '[m/s]', '[m/s]', '[m/s]', '[m/s]', '[m/s]', '[m/s]', '[m/s]', '[m/s]', '[m/s]','[°]' ,'YYYY-MM-DD hh:mm:ss'])
        df = pd.DataFrame(results, columns=['z', 'Mean', 'Std.dev', 'P5', 'P10', 'P50', 'P90', 'P95', 'P99', 'Max','Dir. Max Event', 'Time Max Event'])
     
    # Save the DataFrame to a CSV file
    if output_file:
        df.to_csv(output_file, index=False)

    return df



def table_profile_monthly_stats(data: pd.DataFrame, var: str, z=[10, 20, 30], method = 'mean' , output_file='table_profile_monthly_stats.csv'):
    params = []
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec', 'Year']

    for month in range(1,len(months)):
        month_data = data[data.index.month == month]
        if method == 'mean':
            params.append(np.round(month_data[var].mean().values,2))
        elif method == 'std.dev':
            params.append(np.round(month_data[var].std().values,2))
        elif method == 'minimum':
            params.append(np.round(month_data[var].min().values,2))
        elif method == 'maximum':
            params.append(np.round(month_data[var].max().values,2))

    #add annual
    if method == 'mean':
        params.append(np.round(data[var].mean().values,2))
    elif method == 'std.dev':
        params.append(np.round(data[var].std().values,2))
    elif method == 'minimum':
        params.append(np.round(data[var].min().values,2))
    elif method == 'maximum':
        params.append(np.round(data[var].max().values,2))

    # Create DataFrame
    df = pd.DataFrame(np.transpose(params), columns=months,  index=z)
    df.index.name = 'z[m]'
    # Save the DataFrame to a CSV file
    if output_file:
        file_extension = output_file.split('.')[1] 
        if file_extension == 'csv':
            df.to_csv(output_file, index=True)
        elif file_extension == 'png' or 'pdf':
            import seaborn as sns
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(df, annot=True, cmap="RdBu_r", cbar=False, yticklabels=z, fmt='.2f')
            #plt.title(f'Return Period {period} Years')
            plt.ylabel('z[m]')
            ax.xaxis.tick_top()  # Move x-axis to the top
            ax.xaxis.set_label_position('top')  # Move x-axis label to the top
            plt.xticks(rotation=45, ha='left')  # Rotate x-axis labels by 45 degrees
            plt.yticks(rotation=45, ha='right')  # Rotate y-axis labels by 45 degrees
            plt.tight_layout()
            plt.savefig(output_file,dpi=100)
        else:
            print('File format is not supported')

    return df

import pandas as pd
import numpy as np

def table_tidal_levels(data: pd.DataFrame, var: str, output_file='tidal_levels.csv'):
    """
    Estimate various tidal levels from tidal data, including:
    - Mean Sea Level (MSL)
    - Highest Astronomical Tide (HAT)
    - Lowest Astronomical Tide (LAT)
    
    Parameters:
    data (pd.DataFrame): A dataframe with index 'time' and 
    var: tidal elevation e.g., 'tidal_elevation'.
    
    Returns:
    pd.DataFrame: A dataframe containing the estimated tidal levels.
    """
     # Ensure the 'time' column is a datetime type and set as index
    data[var].index = pd.to_datetime(data[var].index)
    
    # Extract tidal elevations
    tidal_elevations = data[var]
    
    # Calculate High Astr. Tide (HAT)
    hat = tidal_elevations.max()

    # Calculate Low Astr. Tide (LAT)
    lat = tidal_elevations.min()
    
    # Calculate Mean Sea Level (MSL)
    msl = tidal_elevations.mean()
    
    # Create a DataFrame for the results
    results = {
        'Tidal Level': ['HAT', 'MSL', 'LAT'],
        '[m]': [hat, msl, lat]
    }
    
    results_df = pd.DataFrame(results).round(2)
    # Save the DataFrame to a CSV file
    if output_file:
        results_df.to_csv(output_file, index=False)

    return results_df

def table_max_min_water_level(data: pd.DataFrame, var_total_water_level: str,var_tide: str,var_surge: str, var_mslp: str, output_file='table_max_min_water_level.csv'):
    """
    Creates a pandas DataFrame with max and min values for water level components.

    Args:
        data: Pandas DataFrame containing the data.
        var_total_water_level: Column name for total water level.
        var_tide: Column name for tide component.
        var_surge: Column name for surge component.
        var_mslp: Column name for mean sea level pressure for the estimation of pressure surge
        output_file: Output file path for the DataFrame (default: 'table_max_min_water_level.csv').

    Returns:
        Pandas DataFrame with max and min values for specified variables.
    """
    min_pressure_surge, max_pressure_surge = pressure_surge(data,var=var_mslp)
    var_list = [var_total_water_level, var_tide, var_surge]
    max_values = data[var_list].max()
    min_values = data[var_list].min()

    df = pd.DataFrame({'variable': var_list, 'max': max_values, 'min': min_values})
    new_row = {'variable': 'pressure_surge', 'max': max_pressure_surge, 'min': min_pressure_surge}  
    df = df._append(new_row, ignore_index=True)
    df = df.T
    df.columns = df.iloc[0]
    df = df.iloc[1:]   
    df.to_csv(output_file, index=True)
    
    return df


def table_nb_hours_below_threshold(df,var='hs',threshold=[0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10,12.5,15,17.5,20],output_file='table_hours_below_thresh.csv'):
    # Inputs
    # 1) df
    # 2) var: a string
    # 3) thr_arr: list of thresholds (should be a lot for smooth curve)
    # 4) thresholds_chosen: list of thresholds to be included in the table
    # 5) output_file: String with filename without extension
    thr_arr=(np.arange(0.05,20.05,0.05)).tolist()
    nbhr_arr=nb_hours_below_threshold(df,var,thr_arr)
    # Create file
    threshold=np.array(threshold)
    arr1=np.zeros((len(threshold),3))
    rows=[]
    for j in range(len(threshold)):
        # Needs to add/subtract 0.001 because of precision problem
        t=np.where((thr_arr>=threshold[j]-0.001) & (thr_arr<=threshold[j]+0.001))[0]
        arr1[j,0]=np.min(nbhr_arr[t,:])
        arr1[j,1]=np.round(np.mean(nbhr_arr[t,:]),0)
        arr1[j,2]=np.max(nbhr_arr[t,:])
        del t
        rows.append('<='+str(threshold[j]))
    # Convert numpy array to dataframe
    df_out = pd.DataFrame()
    df_out[var] = ['<'+str(threshold[j]) for j in range(len(threshold))]
    #df_out = pd.DataFrame(data=np.round(arr1,0), index=rows, columns=cols)
    df_out['Minimum'] = np.round(arr1[:,0],0).astype('int')
    df_out['Mean'] = np.round(arr1[:,1],0).astype('int')
    df_out['Maximum'] = np.round(arr1[:,2],0).astype('int')
    # Save to csv format
    df_out.to_csv(output_file, index=False)
    return df_out

def table_weather_window_thresholds(ds,var,threshold,op_duration=[6,12,24,48],output_file='table_ww.csv'):
    # ds: dataframe
    # var: string with var name
    # threshold: list of thresholds appropriate for the variable
    # op_duration: list, default is 6h, 12, 24h, and 48h
    # output_file: string with file name
    ds = ds[var]
    timestep = (ds.index.to_series().diff().dropna().dt.total_seconds()/3600).mean()
    arr_o=np.zeros((len(op_duration),len(threshold)))
    t1=0
    for t in threshold:
        op1=0
        for op in op_duration:
            arr_o[op1,t1],p1,p2,p3 = weather_window_length(ds,threshold=t,op_duration=op,timestep=timestep,month=None)
            op1=op1+1
            del p1,p2,p3
        t1=t1+1
    # Convert numpy array to dataframe
    cols=[var+'<'+str(th)+'m' for th in threshold]
    df_tmp1 = pd.DataFrame(data=op_duration, columns=['Operation duration [h]'])
    df_tmp2 = pd.DataFrame(data=np.round(arr_o,2), columns=cols)
    df_out = pd.concat([df_tmp1, df_tmp2], axis=1, sort=True)
    del df_tmp1,df_tmp2
    # Save to csv format
    df_out.to_csv(output_file, index=False)
    return df_out
