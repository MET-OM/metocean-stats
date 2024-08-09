import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import windrose
import matplotlib.cm as cm
import os
from ..stats.aux_funcs import convert_latexTab_to_csv, add_direction_sector, consecutive_indices


def table_directional_min_mean_max(data, direction, intensity, output_file) : 
    direction = data[direction]
    intensity = data[intensity]
    temp_file  = output_file.split('.')[0]

    time = intensity.index  
    
    # sorted by sectors/directions, keep time for the next part 
    bins_dir = np.arange(0,360,30) # 0,30,...,300,330
    dic_Hs = {}
    dic_time = {}
    for i in range(len(bins_dir)) : 
        dic_Hs[str(int(bins_dir[i]))] = [] 
        dic_time[str(int(bins_dir[i]))] = [] 
    
    for i in range(len(intensity)): 
        if 345 <= direction.iloc[i] :
            dic_time[str(int(bins_dir[0]))].append(time[i])
            dic_Hs[str(int(bins_dir[0]))].append(intensity.iloc[i]) 
        else: 
            for j in range(len(bins_dir)): 
                if bins_dir[j]-15 <= direction.iloc[i] < bins_dir[j] + 15 : # -15 --> +345 
                    dic_time[str(int(bins_dir[j]))].append(time[i])
                    dic_Hs[str(int(bins_dir[j]))].append(intensity.iloc[i]) 
                    
    # write to file 
    with open(temp_file, 'w') as f :
        f.write('\\begin{tabular}{l | c c c }' + '\n')
        f.write('Direction & Minimum & Mean & Maximum \\\\' + '\n')
        f.write('\hline' + '\n')
        
        # sorted by years, get max in each year, and statistical values 
        for j in range(len(bins_dir)):
            df_dir = pd.DataFrame()
            df_dir.index = dic_time[str(int(bins_dir[j]))]
            df_dir['Hs'] = dic_Hs[str(int(bins_dir[j]))]
            annual_max_dir = df_dir.resample('Y').max()
            mind = round(annual_max_dir.min()['Hs'],1)
            meand = round(annual_max_dir.mean()['Hs'],1)
            maxd = round(annual_max_dir.max()['Hs'],1)
            start = bins_dir[j] - 15
            if start < 0 : 
                start = 345 
            f.write(str(start) + '-' + str(bins_dir[j]+15) + ' & ' + str(mind) + ' & ' + str(round(meand,1)) + ' & ' + str(maxd) + ' \\\\' + '\n')
            
        ## annual row 
        annual_max = intensity.resample('Y').max()
        mind = round(annual_max.min(),1)
        meand = round(annual_max.mean(),1)
        maxd = round(annual_max.max(),1)
        f.write('Annual & ' + str(mind) + ' & ' + str(meand) + ' & ' + str(maxd) + ' \\\\' + '\n')
        f.write('\hline' + '\n')
        f.write('\end{tabular}' + '\n')

    if output_file.split('.')[1] == 'csv':
        convert_latexTab_to_csv(temp_file, output_file)
        os.remove(temp_file)
    else:
        os.rename(temp_file, output_file)

    return

def table_directional_non_exceedance(data: pd.DataFrame, var1: str, step_var1: float, var_dir: str, output_file: str = None):
    """
    Calculate directional non-exceedance table for a given variable.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing the data.
        var1 (str): Name of the column in the DataFrame representing the variable.
        step_var1 (float): Step size for binning the variable.
        var_dir (str): Name of the column in the DataFrame representing the direction.
        output_file (str, optional): File path to save the output CSV file. Default is None.

    Returns:
        pd.DataFrame: Directional non-exceedance table with percentage of time each data level occurs in each direction.
    """

# Define  bins
    bins = np.arange(int(data[var1].min()), data[var1].max() + step_var1, step_var1).tolist()
    labels =  [f'<{num}' for num in bins]
    
    add_direction_sector(data=data,var_dir=var_dir)

    # Categorize data into bins
    data[var1+'-level'] = pd.cut(data[var1], bins=bins, labels=labels[1:])

    data = data.sort_values(by='direction_sector')
    data = data.set_index('direction_sector')
    data.index.name = 'direction_sector'
    # Group by direction and var1 bin, then count occurrences
    # Calculate percentage of time each var1 bin occurs in each month
    percentage_by_dir = 100*data.groupby([data.index, var1+'-level'], observed=True)[var1].count().unstack()/len(data[var1])
    cumulative_percentage = np.cumsum(percentage_by_dir,axis=1).T
    cumulative_percentage = cumulative_percentage.fillna(method='ffill')

    # Calculate cumulative percentage for each bin across all months
    # Insert 'Omni', 'Mean', 'P99', 'Maximum' rows
    cumulative_percentage.loc['Mean'] = data.groupby(data.index, observed=True)[var1].mean()
    cumulative_percentage.loc['P50'] = data.groupby(data.index, observed=True)[var1].quantile(0.50)
    cumulative_percentage.loc['P75'] = data.groupby(data.index, observed=True)[var1].quantile(0.75)
    cumulative_percentage.loc['P95'] = data.groupby(data.index, observed=True)[var1].quantile(0.95)
    cumulative_percentage.loc['P99'] = data.groupby(data.index, observed=True)[var1].quantile(0.99)
    cumulative_percentage.loc['Maximum'] = data.groupby(data.index, observed=True)[var1].max()
    cumulative_percentage['Omni'] = cumulative_percentage.sum(axis=1)[:-6]
    cumulative_percentage['Omni'].iloc[-6] = data[var1].mean()    
    cumulative_percentage['Omni'].iloc[-5] = data[var1].quantile(0.50)
    cumulative_percentage['Omni'].iloc[-4] = data[var1].quantile(0.75)
    cumulative_percentage['Omni'].iloc[-3] = data[var1].quantile(0.95)
    cumulative_percentage['Omni'].iloc[-2] = data[var1].quantile(0.99)
    cumulative_percentage['Omni'].iloc[-1] = data[var1].max()


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

