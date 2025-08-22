import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
from pathlib import Path

from ..stats import aux_funcs
from .. import stats

def scatter_diagram(data: pd.DataFrame, var1: str, step_var1: float, var2: str, step_var2: float,number_of_events=False, output_file='scatter_diagram.csv'):
    """
    The function is written by dung-manh-nguyen and KonstantinChri.
    Plot scatter diagram (heatmap) of two variables (e.g, var1='hs', var2='tp')
    step_var: size of bin e.g., 0.5m for hs and 1s for Tp
    cmap: colormap, default is 'Blues'
    outputfile: name of output file with extrensition e.g., png, eps or pdf 
     """

    sd = stats.calculate_scatter(data, var1, step_var1, var2, step_var2)
    # start_date = data.index.min()
    # end_date = data.index.max() 
    # Calculate the difference in years
    # number_of_years = (end_date - start_date).days / 365.25
    # dt = (data.index.to_series().diff().dropna().dt.total_seconds() / 3600).mean()

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
    
    if number_of_events is True:
        #number of events 
        tbl = np.flip(sd.values,axis=0)
    else:
        pass

    dfout = pd.DataFrame(data=np.round(tbl,2), index=rows, columns=cols)
    output_file = Path(output_file)
    if output_file.suffix == '.csv':
        dfout.to_csv(output_file, index_label=var1+'/'+var2)
    elif output_file.suffix == '.png':
        sns.heatmap(data=dfout.where(dfout>0), cbar=True, cmap='Blues', fmt=".1f")
        plt.ylabel(var1)
        plt.xlabel(var2)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
    else:
        print(f"Unsupported file type: {output_file.suffix}")

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

# def table_monthly_percentile(data,var,output_file='var_monthly_percentile.txt'):  
#     """
#     The function is written by dung-manh-nguyen and KonstantinChri.
#     this function will sort variable var e.g., hs by month and calculate percentiles 
#     data : panda series 
#     var  : variable 
#     output_file: extension .txt for latex table or .csv for csv table
#     """

#     Var = data[var]
#     varName = data[var].name
#     Var_month = Var.index.month
#     M = Var_month.values
#     temp_file = output_file.split('.')[0]
    
#     months = calendar.month_name[1:] # eliminate the first insane one 
#     for i in range(len(months)) : 
#         months[i] = months[i][:3] # get the three first letters 
    
#     monthlyVar = {}
#     for i in range(len(months)) : 
#         monthlyVar[months[i]] = [] # create empty dictionaries to store data 
    
    
#     for i in range(len(Var)) : 
#         m_idx = int(M[i]-1) 
#         monthlyVar[months[m_idx]].append(Var.iloc[i])  
        
        
#     with open(temp_file, 'w') as f:
#         f.write(r'\\begin{tabular}{l | p{1.5cm} p{1.5cm} p{1.5cm} p{1.5cm} p{1.5cm}}' + '\n')
#         f.write(r'& \multicolumn{5}{c}{' + varName + '} \\\\' + '\n')
#         f.write(r'Month & 5\% & 50\% & Mean & 95\% & 99\% \\\\' + '\n')
#         f.write(r'\hline' + '\n')
    
#         for j in range(len(months)) : 
#             Var_P5 = round(np.percentile(monthlyVar[months[j]],5),1)
#             Var_P50 = round(np.percentile(monthlyVar[months[j]],50),1)
#             Var_mean = round(np.mean(monthlyVar[months[j]]),1)
#             Var_P95 = round(np.percentile(monthlyVar[months[j]],95),1)
#             Var_P99 = round(np.percentile(monthlyVar[months[j]],99),1)
#             f.write(months[j] + ' & '+str(Var_P5)+' & '+str(Var_P50)+' & '+str(Var_mean)+' & '+str(Var_P95)+' & '+str(Var_P99)+' \\\\' + '\n')
        
#         # annual row 
#         f.write('Annual & '+str(Var_P5)+' & '+str(Var_P50)+' & '+str(Var_mean)+' & '+str(Var_P95)+' & '+str(Var_P99)+' \\\\' + '\n')
    
#         f.write(r'\hline' + '\n')
#         f.write(r'\end{tabular}' + '\n')
    
#     if output_file.split('.')[1] == 'csv':
#         aux_funcs.convert_latexTab_to_csv(temp_file, output_file)
#         os.remove(temp_file)
#     else:
#         os.rename(temp_file, output_file)
    
#     return   

def _percentile_str_to_pd_format(percentiles):
    '''
    Utility to convert various types of percentile/stats strings to pandas compatible format.
    '''

    mapping = {"Maximum":"max",
               "Max":"max",
               "Minimum":"min",
               "Min":"min",
               "Mean":"mean"}

    def strconv(name:str):
        if name.startswith("P"):
            return name.replace("P","")+"%"
        if name in mapping:
            return mapping[name]
        else: return name

    if type(percentiles) is str: return strconv(percentiles)
    else: return [strconv(p) for p in percentiles]

def table_daily_percentile(data, 
                           var, 
                           percentiles = ["5%","mean","99%","max"],
                           divide_months = False):
    '''
    Calculate daily stats/percentiles via pandas .describe().
    
    Arguments
    ---------
    data : pd.DataFrame
        Dataframe containing the var column with dataframe index.
    var : str
        Column name.
    percentiles : list[str]
        A list of strings such as count, mean, std, min, max or any percentile from 0% to 100%. [] will return everything.
    divide_months : bool
        If true, divide the year into months and return a dict of monthly dataframes. If not, return one dataframe with 365/366 entries.

    Returns
    -------
    data : DataFrame or list of DataFrames
    '''
    
    perc_cols = [p for p in percentiles] # copy original names before formatting
    percentiles = _percentile_str_to_pd_format(percentiles)
    
    # Uses pandas describe() method to create a dataframe of daily stats.
    daily_table = data[var].groupby(data.index.dayofyear).describe(percentiles=np.arange(0,1,0.01))

    # Select input percentiles
    if percentiles != []:
        daily_table = daily_table[percentiles]
        daily_table.columns = perc_cols

    if divide_months:
        # Cut 366th day if exists and re-create datetime index from integer days
        daily_table = daily_table.loc[:365]
        daily_table.index = pd.to_datetime(daily_table.index,origin="2024-12-31",unit="D")

        # Get month keywords and group table by month
        monthlabels = list(pd.date_range("2024","2024-12",freq="MS").strftime("%b"))
        groups = daily_table.groupby(pd.Grouper(level="time",freq="MS"))
        monthly_tables = {}
        
        for i,(_,g) in enumerate(groups):
            mtab = g
            mtab.index = mtab.index.day
            mtab.index.names = ["Day"]
            monthly_tables[monthlabels[i]] = mtab
        return monthly_tables

    else:
        daily_table.index.names = ["Day"]
        return daily_table

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
        f.write(r'\hline' + '\n')
        for i in range(len(months)):
            f.write(months[i] + ' & ' + str(minimum.values[i]) + ' & ' + str(round(mean.values[i],1)) + ' & ' + str(maximum.values[i]) + ' \\\\' + '\n')
        
        ## annual row 
        annual_max = var.resample('Y').max()
        min_year = annual_max.min()
        mean_year = annual_max.mean()
        max_year = annual_max.max()
        f.write('Annual Max. & ' + str(min_year) + ' & ' + str(round(mean_year,1)) + ' & ' + str(max_year) + ' \\\\' + '\n')
            
        f.write(r'\hline' + '\n')
        f.write(r'\end{tabular}' + '\n')


    if output_file.split('.')[1] == 'csv':
        aux_funcs.convert_latexTab_to_csv(temp_file, output_file)
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
    4: 'APR',
    5: 'MAY',
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
    
    aux_funcs.add_direction_sector(data=data,var_dir=var_dir)

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

def table_monthly_percentile(data:pd.DataFrame,
                             var:str,
                             percentiles:list[str]=["P25","mean","P75","P99","max"],
                             output_file="table_monthly_percentile.csv"):
    """
    Group dataframe by month and calculate percentiles.

    Parameters
    ------------
    data : pd.DataFrame
        The data containing column var, and with a datetime index.
    var : str
        The column to calculate percentiles of.
    percentiles : list of strings
        List of percentiles, e.g. P25, P50, P75, 30%, 40% etc.
        Some others are also allowed: count (number of data points), and min, mean, max.

    Returns
    -------
    pd.DataFrame
    """
    series = data[var]
    percentiles = _percentile_str_to_pd_format(percentiles)
    series.index = pd.to_datetime(series.index)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec","Year"]
    table = []
    for m,g in series.groupby(series.index.month):
        table.append(g.describe(percentiles=np.arange(0,1,0.01))[percentiles])
    table.append(series.describe(percentiles=np.arange(0,1,0.01))[percentiles])
    table = pd.DataFrame(table,month_labels)
    if output_file != "":
        table.to_csv(output_file)
    return table

def monthly_directional_percentiles(
        data: pd.DataFrame, 
        var_dir: str,
        var: str,
        percentiles: list[str] = ["P25","mean","P75","P99","max"],
        nsectors: int = 4,
        compass_point_names = True,
        output_file: str = ""):
    
    """
    Calculate monthly directional percentile tables for a given variable.

    Parameters
    ----------
    data : DataFrame
        the data
    var : str
        Name of variable magnitude column
    var_dir : str
        Name of variable direction column
    percentiles : list[str]
        A list of strings such as count, mean, std, min, max 
        or any (integer) percentile from P0 to P100. 
        percentiles = [] will return all columns.
    nsectors : int
        Number of directional sectors, typically 4, 8 or 16.
    compass_point_names : bool
        Replace degree range of sector (from, to) 
        with a compass point system label such as N, NE, etc.
        Only implemented for nsectors 4, 8 or 16.
        
    Returns
    -------
    tables : dict
        A dictionary of monthly percentile tables.
    """
    perc_cols = [p for p in percentiles] # copy original percentile format
    percentiles = _percentile_str_to_pd_format(percentiles)
    
    # Define sector bins
    bins = np.linspace(0, 360, nsectors+1)
    dir_offset = (bins[1]-bins[0])/2

    # Compass point labels if appropriate
    labels = []
    if nsectors == 2:
        labels = ['N', 'S']
    elif nsectors == 4:
        labels = ['N', 'E', 'S', 'W']
    elif nsectors == 8:
        labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    elif nsectors == 16:
        labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    elif nsectors == 32:
        labels = ['N', 'NbE', 'NNE', 'NEbN', 'NE', 'NEbE', 'ENE', 'EbN', 
                  'E', 'EbS', 'ESE', 'SEbE', 'SE', 'SEbS', 'SSE', 'SbE', 
                  'S', 'SbW', 'SSW', 'SWbS', 'SW', 'SWbW', 'WSW', 'WbS', 
                  'W', 'WbN', 'WNW', 'NWbW', 'NW', 'NWbN', 'NNW', 'NbW']
    
    # Otherwise, use labels [from, to)
    if (not labels) or (not compass_point_names):
        labels = ["["+str((bins[i]-dir_offset+360)%360)+", "+str(bins[i+1]-dir_offset)+")" for i in range(nsectors)]
        omni_label = "[0, 360)"
    else:
        omni_label = "Omni"
    
    all_percentiles = np.arange(0,1,0.01)

    # Define directional bins
    data["_dir_bin"] = pd.cut((data[var_dir]+dir_offset)%360, bins=bins, labels=labels, right=False)

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Year"]
    monthly_tables = {}
    for i,m in enumerate(month_labels):
        # Select month, group by direction, and calculate statistics for selected variable
        if i==12: # yearly
            monthly = data
        else:
            monthly = data[data.index.month == i+1]
        month_dir_stats = monthly.groupby("_dir_bin",observed=True)
        month_dir_stats = month_dir_stats.describe(percentiles=all_percentiles)[var]
        month_dir_stats.loc[omni_label] = monthly[var].describe(percentiles=all_percentiles)
        month_dir_stats.index.name = None
        
        # Calculate relative frequency (divide total by 2, since omni is included)
        n_total = np.sum(month_dir_stats["count"])//2
        month_dir_stats.insert(1, "%", [100*c/n_total for c in month_dir_stats["count"]])
        month_dir_stats.loc[omni_label,"%"] = 100

        # Select input percentiles
        if percentiles != []:
            month_dir_stats = month_dir_stats[["%"]+percentiles]
            month_dir_stats.columns = ["%"]+perc_cols

        monthly_tables[m] = month_dir_stats
        if output_file != "":
            if "." in output_file:
                fname = ".".join(output_file.split(".")[:-1])+m+".csv"
            else:
                fname = output_file+m+".csv"
            month_dir_stats.to_csv(fname)
    
    return monthly_tables
    

def table_monthly_weather_window(data: pd.DataFrame, var: str, threshold: float, window_size=12, timestep=3, output_file: str = None):
    # var should be a list of variables, and threshold should be a list of thresholds
    # more outputs than table_monthly_weather_window
    # Written by clio-met
    results = []
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month in range(1, 13):
        avg_duration, p10, p50, p90, p95, max = stats.weather_window_length_MultipleVariables(data,vars=var,threshold=threshold,op_duration=window_size,timestep=timestep,month=month)
        results.append((avg_duration, p10, p50, p90, p95, max))
    results_df = pd.DataFrame(results, columns=['Mean', 'P10', 'P50', 'P90', 'P95', 'Max'], index=months).T.round(1)
    if output_file:
        # Save results to CSV
        results_df.to_csv('monthly_weather_window_results_mv.csv')
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



def table_profile_monthly_stats(data: pd.DataFrame, 
                                var: str, z=[10, 20, 30], 
                                method = 'mean' , 
                                output_file='table_profile_monthly_stats.csv',
                                rounding:int=2):
    params = []
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec', 'Year']

    for month in range(1,len(months)):
        month_data = data[data.index.month == month]
        if method == 'mean':
            params.append(month_data[var].mean().values)
        elif method == 'std.dev':
            params.append(month_data[var].std().values)
        elif method == 'minimum':
            params.append(month_data[var].min().values)
        elif method == 'maximum':
            params.append(month_data[var].max().values)

    #add annual
    if method == 'mean':
        params.append(data[var].mean().values)
    elif method == 'std.dev':
        params.append(data[var].std().values)
    elif method == 'minimum':
        params.append(data[var].min().values)
    elif method == 'maximum':
        params.append(data[var].max().values)
    
    if rounding:
        params = [np.round(p,rounding) for p in params]

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
    min_pressure_surge, max_pressure_surge = stats.pressure_surge(data,var=var_mslp)
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
    nbhr_arr=stats.nb_hours_below_threshold(df,var,thr_arr)
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
            arr_o[op1,t1],p1,p2,p3 = stats.weather_window_length(ds,threshold=t,op_duration=op,timestep=timestep,month=None)
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
