import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
from math import floor,ceil

from .aux_funcs import convert_latexTab_to_csv, add_direction_sector, consecutive_indices

def calculate_scatter(data: pd.DataFrame, var1: str, step_var1: float, var2: str, step_var2: float) -> pd.DataFrame:
    """
    Create scatter table of two variables (e.g, var1='hs', var2='tp')
    step_var: size of bin e.g., 0.5m for hs and 1s for Tp
    The rows are the upper bin edges of var1 and the columns are the upper bin edges of var2
    """
    dvar1 = data[var1]
    v1min = np.min(dvar1)
    v1max = np.max(dvar1)
    if step_var1 > v1max:
        raise ValueError(f"Step size {step_var1} is larger than the maximum value of {var1}={v1max}.")

    dvar2 = data[var2]
    v2min = np.min(dvar2)
    v2max = np.max(dvar2)
    if step_var2 > v2max:
        raise ValueError(f"Step size {step_var2} is larger than the maximum value of {var2}={v2max}.")

    # Find the the upper bin edges
    max_bin_1 = ceil(v1max / step_var1)
    max_bin_2 = ceil(v2max / step_var2)
    min_bin_1 = ceil(v1min / step_var1)
    min_bin_2 = ceil(v2min / step_var2)

    offset_1 = min_bin_1 - 1
    offset_2 = min_bin_2 - 1

    var1_upper_bins = np.arange(min_bin_1, max_bin_1 + 1, 1) * step_var1
    var2_upper_bins = np.arange(min_bin_2, max_bin_2 + 1, 1) * step_var2

    row_size = len(var1_upper_bins)
    col_size = len(var2_upper_bins)
    occurences = np.zeros([row_size, col_size])

    for v1, v2 in zip(dvar1, dvar2):
        # Find the correct bin and sum up
        row = floor(v1 / step_var1) - offset_1
        col = floor(v2 / step_var2) - offset_2
        occurences[row, col] += 1

    return pd.DataFrame(data=occurences, index=var1_upper_bins, columns=var2_upper_bins)

def scatter_diagram(data: pd.DataFrame, var1: str, step_var1: float, var2: str, step_var2: float, output_file):
    """
    The function is written by dung-manh-nguyen and KonstantinChri.
    Plot scatter diagram (heatmap) of two variables (e.g, var1='hs', var2='tp')
    step_var: size of bin e.g., 0.5m for hs and 1s for Tp
    cmap: colormap, default is 'Blues'
    outputfile: name of output file with extrensition e.g., png, eps or pdf 
     """

    sd = calculate_scatter(data, var1, step_var1, var2, step_var2)

    # Convert to percentage
    tbl = sd.values
    var1_data = data[var1]
    tbl = tbl/len(var1_data)*100

    # Then make new row and column labels with a summed percentage
    sumcols = np.sum(tbl, axis=0)
    sumrows = np.sum(tbl, axis=1)

    sumrows = np.around(sumrows, decimals=1)
    sumcols = np.around(sumcols, decimals=1)

    bins_var1 = sd.index
    bins_var2 = sd.columns
    lower_bin_1 = bins_var1[0] - step_var1
    lower_bin_2 = bins_var2[0] - step_var2

    rows = []
    rows.append(f'{lower_bin_1:04.1f}-{bins_var1[0]:04.1f} | {sumrows[0]:04.1f}%')
    for i in range(len(bins_var1)-1):
        rows.append(f'{bins_var1[i]:04.1f}-{bins_var1[i+1]:04.1f} | {sumrows[i+1]:04.1f}%')

    cols = []
    cols.append(f'{int(lower_bin_2)}-{int(bins_var2[0])} | {sumcols[0]:04.1f}%')
    for i in range(len(bins_var2)-1):
        cols.append(f'{int(bins_var2[i])}-{int(bins_var2[i+1])} | {sumcols[i+1]:04.1f}%')

    rows = rows[::-1]
    tbl = tbl[::-1,:]
    dfout = pd.DataFrame(data=tbl, index=rows, columns=cols)
    hi = sns.heatmap(data=dfout.where(dfout>0), cbar=True, cmap='Blues', fmt=".1f")
    plt.ylabel(var1)
    plt.xlabel(var2)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    return hi

def table_var_sorted_by_hs(data, var, var_hs='hs', output_file='var_sorted_by_Hs.txt'):
    """
    The function is written by dung-manh-nguyen and KonstantinChri.
    This will sort variable var e.g., 'tp' by 1 m interval og hs
    then calculate min, percentile 5, mean, percentile 95 and max
    data : panda series 
    var  : variable 
    output_file: extension .txt for latex table or .csv for csv table
    """
    Hs = data[var_hs]
    Var = data[var]
    binsHs = np.arange(0.,math.ceil(np.max(Hs))+0.1) # +0.1 to get the last one   
    temp_file = output_file.split('.')[0]

    Var_binsHs = {}
    for j in range(len(binsHs)-1) : 
        Var_binsHs[str(int(binsHs[j]))+'-'+str(int(binsHs[j+1]))] = [] 
    
    N = len(Hs)
    for i in range(N):
        for j in range(len(binsHs)-1) : 
            if binsHs[j] <= Hs.iloc[i] < binsHs[j+1] : 
                Var_binsHs[str(int(binsHs[j]))+'-'+str(int(binsHs[j+1]))].append(Var.iloc[i])
    
    with open(temp_file, 'w') as f:
        f.write('\\begin{tabular}{l p{1.5cm}|p{1.5cm} p{1.5cm} p{1.5cm} p{1.5cm} p{1.5cm}}' + '\n')
        f.write('& & \multicolumn{5}{c}{'+var+'} \\\\' + '\n')
        f.write('Hs & Entries & Min & 5\% & Mean & 95\% & Max \\\\' + '\n')
        f.write('\hline' + '\n')
    
        for j in range(len(binsHs)-1) : 
            Var_binsHs_temp = Var_binsHs[str(int(binsHs[j]))+'-'+str(int(binsHs[j+1]))]
            
            Var_min = round(np.min(Var_binsHs_temp), 1)
            Var_P5 = round(np.percentile(Var_binsHs_temp , 5), 1)
            Var_mean = round(np.mean(Var_binsHs_temp), 1)
            Var_P95 = round(np.percentile(Var_binsHs_temp, 95), 1)
            Var_max = round(np.max(Var_binsHs_temp), 1)
            
            hs_bin_temp = str(int(binsHs[j]))+'-'+str(int(binsHs[j+1]))
            f.write(hs_bin_temp + ' & '+str(len(Var_binsHs_temp))+' & '+str(Var_min)+' & '+str(Var_P5)+' & '+str(Var_mean)+' & '+str(Var_P95)+' & '+str(Var_max)+' \\\\' + '\n')
        hs_bin_temp = str(int(binsHs[-1]))+'-'+str(int(binsHs[-1]+1)) # +1 for one empty row 
        f.write(hs_bin_temp + ' & 0 & - & - & - & - & - \\\\' + '\n')
        
        # annual row 
        Var_min = round(np.min(Var), 1)
        Var_P5 = round(np.percentile(Var , 5), 1)
        Var_mean = round(np.mean(Var), 1)
        Var_P95 = round(np.percentile(Var, 95), 1)
        Var_max = round(np.max(Var), 1)
        
        hs_bin_temp = str(int(binsHs[0]))+'-'+str(int(binsHs[-1]+1)) # +1 for one empty row 
        f.write(hs_bin_temp + ' & '+str(len(Var))+' & '+str(Var_min)+' & '+str(Var_P5)+' & '+str(Var_mean)+' & '+str(Var_P95)+' & '+str(Var_max)+' \\\\' + '\n')
        f.write('\hline' + '\n')
        f.write('\end{tabular}' + '\n')
    
    if output_file.split('.')[1] == 'csv':
        convert_latexTab_to_csv(temp_file, output_file)
        os.remove(temp_file)
    else:
        os.rename(temp_file, output_file)
    

    return    

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


def Cmax(Hs,Tm,depth):
    
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import find_peaks
    
    # calculate k1
    g = 9.80665 # m/s2, gravity 
    Tm01 = Tm # second, period 
    d = depth # m, depth
    lamda1 = 1 # wave length from 8.5 to 212 
    lamda2 = 500 # wave length from 8.5 to 212 
    k1_temp=np.linspace(2*np.pi/lamda2,2*np.pi/lamda1,10000)
    F = (2*np.pi)**2/(g*Tm01**2*np.tanh(k1_temp*d)) - k1_temp
    #plt.plot(k1_temp,F)
    #plt.grid()
    
    
    epsilon = abs(F)
    try:
        param = find_peaks(1/epsilon)
        index = param[0][0]
    except:
        param = np.where(epsilon == epsilon.min())
        index = param[0][0]
        
    k1 = k1_temp[index]
    #lamda = 2*np.pi/k1
    #print ('wave length (m) =', round(lamda))
    
    #Hs=10
    Urs = Hs/(k1**2*d**3)
    S1 = 2*np.pi/g*Hs/Tm01**2
    
    sea_state='long-crested'
    #sea_state='short-crested'
    if sea_state == 'long-crested' :
        AlphaC = 0.3536 + 0.2892*S1 + 0.1060*Urs
        BetaC = 2 - 2.1597*S1 + 0.0968*Urs
    elif sea_state == 'short-crested' :
        AlphaC = 0.3536 + 0.2568*S1 + 0.08*Urs
        AlphaC = 2 - 1.7912*S1 - 0.5302*Urs + 0.2824*Urs
    else:
        print ('please check sea state')
        
        
    #x = np.linspace(0.1,10,1000)
    #Fc = 1-np.exp(-(x/(AlphaC*Hs))**BetaC)
    #
    #plt.plot(x,Fc)
    #plt.grid()
    #plt.title(sea_state)
    
    
    Tm02 = Tm
    #t=5
    #C_MPmax = AlphaC*Hs*(np.log(Tm02/t))**(1/BetaC) # This is wrong, anyway it is not important 
    
    t=10800 # 3 hours 
    p = 0.85
    C_Pmax = AlphaC*Hs*(-np.log(1-p**(Tm02/t)))**(1/BetaC)
    Cmax = C_Pmax*1.135 # this is between 1.13 and 1.14 
    
    return Cmax
    
    
    
def pdf_all(data, var, bins=70, output_file='pdf_all.png'): #pdf_all(data, bins=70)
    
    import matplotlib.pyplot as plt
    from scipy.stats import expon
    from scipy.stats import genextreme
    from scipy.stats import gumbel_r 
    from scipy.stats import lognorm 
    from scipy.stats import weibull_min
    
    data = data[var].values
    
    x=np.linspace(min(data),max(data),100)
    
    param = weibull_min.fit(data) # shape, loc, scale
    pdf_weibull = weibull_min.pdf(x, param[0], loc=param[1], scale=param[2])
    
    param = expon.fit(data) # loc, scale
    pdf_expon = expon.pdf(x, loc=param[0], scale=param[1])
    
    param = genextreme.fit(data) # shape, loc, scale
    pdf_gev = genextreme.pdf(x, param[0], loc=param[1], scale=param[2])
    
    param = gumbel_r.fit(data) # loc, scale
    pdf_gumbel = gumbel_r.pdf(x, loc=param[0], scale=param[1])
    
    param = lognorm.fit(data) # shape, loc, scale
    pdf_lognorm = lognorm.pdf(x, param[0], loc=param[1], scale=param[2])
    
    fig, ax = plt.subplots(1, 1)
    #ax.plot(x, pdf_expon, label='pdf-expon')
    ax.plot(x, pdf_weibull,label='pdf-Weibull')
    ax.plot(x, pdf_gev, label='pdf-GEV')
    ax.plot(x, pdf_gumbel, label='pdf-GUM')
    ax.plot(x, pdf_lognorm, label='pdf-lognorm')
    ax.hist(data, density=True, bins=bins, color='tab:blue', label='histogram', alpha=0.5)
    #ax.hist(data, density=True, bins='auto', color='tab:blue', label='histogram', alpha=0.5)
    ax.legend()
    ax.grid()
    ax.set_xlabel('Wave height')
    ax.set_ylabel('Probability density')
    plt.savefig(output_file)
    
    return 


def scatter(df,var1,var2,location,regression_line,qqplot=True):
    x=df[var1].values
    y=df[var2].values
    fig, ax = plt.subplots()
    ax.scatter(x,y,marker='.',s=10,c='g')
    dmin, dmax = np.min([x,y])*0.9, np.max([x,y])*1.05
    diag = np.linspace(dmin, dmax, 1000)
    plt.plot(diag, diag, color='r', linestyle='--')
    plt.gca().set_aspect('equal')
    plt.xlim([0,dmax])
    plt.ylim([0,dmax])
    
    if qqplot :    
        percs = np.linspace(0,100,101)
        qn_x = np.nanpercentile(x, percs)
        qn_y = np.nanpercentile(y, percs)    
        ax.scatter(qn_x,qn_y,marker='.',s=80,c='b')

    if regression_line:  
        m,b,r,p,se1=stats.linregress(x,y)
        cm0="$"+('y=%2.2fx+%2.2f'%(m,b))+"$";   
        plt.plot(x, m*x + b, 'k--', label=cm0)
        plt.legend(loc='best')
        
    rmse = np.sqrt(((y - x) ** 2).mean())
    bias = np.mean(y-x)
    mae = np.mean(np.abs(y-x))
    corr = np.corrcoef(y,x)[0][1]
    std = np.std(x-y)/np.mean(x)
    plt.annotate('rmse = '+str(round(rmse,3))
                 +'\nbias = '+str(round(bias,3))
                 +'\nmean = '+str(round(mae,3))
                 +'\ncorr = '+str(round(corr,3))
                 +'\nstd = '+str(round(std,3)), xy=(dmin+1,0.6*(dmin+dmax)))
        
    plt.xlabel(var1, fontsize=15)
    plt.ylabel(var2, fontsize=15)

    plt.title("$"+(location +', N=%1.0f'%(np.count_nonzero(~np.isnan(x))))+"$",fontsize=15)
    plt.grid()
    plt.close()
    
    return fig


def table_monthly_non_exceedance(data: pd.DataFrame, var1: str, step_var1: float, output_file: str = None):
    """
    Calculate monthly non-exceedance table for a given variable.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing the data.
        var1 (str): Name of the column in the DataFrame representing the variable.
        step_var1 (float): Step size for binning the variable.
        output_file (str, optional): File path to save the output CSV file. Default is None.

    Returns:
        pd.DataFrame: Monthly non-exceedance table with percentage of time each data level occurs in each month.
    """

# Define  bins
    bins = np.arange(0, data[var1].max() + step_var1, step_var1).tolist()
    labels =  [f'<{num}' for num in bins]

    # Categorize data into bins
    data[var1+'-level'] = pd.cut(data[var1], bins=bins, labels=labels[1:])
    
    # Group by month and var1 bin, then count occurrences
    grouped = data.groupby([data.index.month, var1+'-level'], observed=True).size().unstack(fill_value=0)


    # Calculate percentage of time each wind speed bin occurs in each month
    percentage_by_month = grouped.div(grouped.sum(axis=1), axis=0) * 100

    # Calculate cumulative percentage for each bin across all months
    cumulative_percentage = percentage_by_month.T.cumsum()

    # Insert 'Annual', 'Mean', 'P99', 'Maximum' rows
    cumulative_percentage.loc['Mean'] = data.groupby(data.index.month, observed=True)[var1].mean()
    cumulative_percentage.loc['P50'] = data.groupby(data.index.month, observed=True)[var1].quantile(0.50)
    cumulative_percentage.loc['P75'] = data.groupby(data.index.month, observed=True)[var1].quantile(0.75)
    cumulative_percentage.loc['P95'] = data.groupby(data.index.month, observed=True)[var1].quantile(0.95)
    cumulative_percentage.loc['P99'] = data.groupby(data.index.month, observed=True)[var1].quantile(0.99)
    cumulative_percentage.loc['Maximum'] = data.groupby(data.index.month, observed=True)[var1].max()
    cumulative_percentage['Year'] = cumulative_percentage.mean(axis=1)[:-6]
    cumulative_percentage['Year'].iloc[-6] = data[var1].mean()    
    cumulative_percentage['Year'].iloc[-5] = data[var1].quantile(0.50)
    cumulative_percentage['Year'].iloc[-4] = data[var1].quantile(0.75)
    cumulative_percentage['Year'].iloc[-3] = data[var1].quantile(0.95)
    cumulative_percentage['Year'].iloc[-2] = data[var1].quantile(0.99)
    cumulative_percentage['Year'].iloc[-1] = data[var1].max()

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
    bins = np.arange(0, data[var1].max() + step_var1, step_var1).tolist()
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


def plot_monthly_stats(data: pd.DataFrame, var1: str, step_var1: float, title: str='Variable [units] location',output_file: str = 'monthly_stats.png'):
    """
    Plot monthly statistics of a variable from a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        var1 (str): The name of the variable to plot.
        step_var1 (float): The step size for computing cumulative statistics.
        title (str, optional): Title of the plot. Default is 'Variable [units] location'.
        output_file (str, optional): File path to save the plot. Default is 'monthly_stats.png'.

    Returns:
        matplotlib.figure.Figure: The Figure object of the generated plot.

    Notes:
        This function computes monthly statistics (Maximum, P99, Mean) of a variable and plots them.
        It uses the 'table_monthly_non_exceedance' function to compute cumulative statistics.

    Example:
        plot_monthly_stats(data, 'temperature', 0.1, title='Monthly Temperature Statistics', output_file='temp_stats.png')
    """
    fig, ax = plt.subplots()
    cumulative_percentage = table_monthly_non_exceedance(data,var1,step_var1)
    cumulative_percentage.loc['Maximum'][:-1].plot(marker = 'o')
    cumulative_percentage.loc['P99'][:-1].plot(marker = 'o')    
    cumulative_percentage.loc['Mean'][:-1].plot(marker = 'o')

    plt.title(title,fontsize=16)
    plt.xlabel('Month',fontsize=15)
    plt.legend()
    plt.grid()
    plt.savefig(output_file)
    return fig

def plot_directional_stats(data: pd.DataFrame, var1: str, step_var1: float, var_dir: str, title: str='Variable [units] location',output_file: str = 'directional_stats.png'):
    """
    Plot directional statistics of a variable from a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        var1 (str): The name of the variable to plot.
        step_var1 (float): The step size for computing cumulative statistics.
        var_dir (str): The name of the directional variable.
        title (str, optional): Title of the plot. Default is 'Variable [units] location'.
        output_file (str, optional): File path to save the plot. Default is 'directional_stats.png'.

    Returns:
        matplotlib.figure.Figure: The Figure object of the generated plot.

    Notes:
        This function computes directional statistics (Maximum, P99, Mean) of a variable and plots them against a directional variable.
        It uses the 'table_directional_non_exceedance' function to compute cumulative statistics.

    Example:
        plot_directional_stats(data, 'hs', 0.1, 'Pdir', title='Directional Wave Statistics', output_file='directional_hs_stats.png')
    """    
    fig, ax = plt.subplots()
    cumulative_percentage = table_directional_non_exceedance(data,var1,step_var1,var_dir)
    cumulative_percentage.loc['Maximum'][:-1].plot(marker = 'o')
    cumulative_percentage.loc['P99'][:-1].plot(marker = 'o')    
    cumulative_percentage.loc['Mean'][:-1].plot(marker = 'o')
    
    plt.title(title,fontsize=16)
    plt.xlabel('Direction[$⁰$]',fontsize=15)
    plt.legend()
    plt.grid()
    plt.savefig(output_file)
    return fig

def calculate_monthly_weather_window(data: pd.DataFrame, var: str,threshold=5, window_size=12,output_file: str = None):
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

def plot_monthly_weather_window(data: pd.DataFrame, var: str,threshold=5, window_size=12, title: str='Var < ... m for ... hours',add_table=True, output_file: str = 'monthly_weather_window_plot.png'):
    results_df = calculate_monthly_weather_window(data=data, var=var, threshold=threshold, window_size=window_size)
    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 6))
    results_df.T.plot(marker='o')
    lines = results_df.T.plot(marker='o')
    plt.title(title)
    plt.xlabel('Month')
    plt.ylabel('Duration [days]')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    
    if add_table:
        # Get legend colors and labels
        legend_colors = [line.get_color() for line in lines.get_lines()]
        # Add the table of results_df under the plot
        plt.xticks([])
        plt.xlabel('')
        plt.legend('',frameon=False)
        cell_text = []
        for row in range(len(results_df)):
            cell_text.append(results_df.iloc[row].values)
        table = plt.table(cellText=cell_text, colLabels=results_df.columns, rowLabels=results_df.index, loc='bottom')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        # Make the blue color value
        cell_dict = table.get_celld()
        for i in range(1,len(legend_colors)+1):
            cell_dict[(i, -1)].set_facecolor(legend_colors[i-1])
    plt.tight_layout()
    plt.savefig(output_file)

    return fig



def calculate_weather_window(data: pd.DataFrame, var: str,threshold=5, window_size=12):
    """
    Calculate the mean and percentiles of consecutive periods where a variable
    stays below a certain threshold within a given window size.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the time series data.
        var (str): The name of the variable in the DataFrame.
        threshold (float, optional): The threshold value for the variable.
            Defaults to 5.
        window_size (int, optional): The size of the window in hours.
            Defaults to 12.

    Returns:
        tuple: A tuple containing the mean and percentiles of consecutive periods
            where the variable stays below the threshold.
    """
    #data=data['1958-02-04' :'1958-02-23']
    df = (data[var]<threshold).astype(int)
    dt = (data.index[1]-data.index[0]).total_seconds()/3600 # in hours # 1 hour for NORA3
    #consecutive_periods = df.rolling(window=str(window_size+dt)+'H').sum()
    # Mark consecutive periods of window_size (hours) with 1
    #consecutive_periods[consecutive_periods < window_size] = 0
    #consecutive_periods = consecutive_periods > 0 
    #consecutive_periods = consecutive_periods.astype(int)*df  
    # Find consecutive sequences of ones
    consecutive_periods = (df == 1).astype(int).groupby((df != 1).cumsum()).cumsum()

    # Find indices of zeros between sequences of five or more consecutive ones
    #indices_zeros = df[df == 0].groupby(consecutive_periods).filter(lambda x: (x == 0).sum() >= window_size/dt).index
    
    
    plt.plot(data[var][:],'r')
    plt.axhline( y=threshold, color='k', linestyle='--')
    plt.plot(consecutive_periods,'go')
    #plt.plot(data[var].where(consecutive_periods==0)[:],'ro')
    #plt.plot(data[var].where(consecutive_periods==1)[:],'go')
    plt.grid()
    plt.show()    
    #breakpoint()
    # add all periods with zero waiting time (consecutive_periods==1)
    counter_list_no_waiting = [window_size]*int((np.sum(consecutive_periods)*dt)//window_size)
    # add all periods with waiting time (consecutive_periods==0)
    counter = 0
    counter_list_with_waiting =   [[] for _ in range(12)]
    for i in range(len(consecutive_periods)-1):
        if consecutive_periods.iloc[i]==0 and consecutive_periods.iloc[i+1]==0:
            counter = counter + dt
        elif consecutive_periods.iloc[i]==0 and consecutive_periods.iloc[i+1]==1:
            counter = counter + dt
            counter_list_with_waiting[consecutive_periods.index[i].month-1].append(counter)
            counter = 0
    counter_list =  counter_list_with_waiting #+ counter_list_no_waiting 
    
    mean = np.zeros(12)
    p10 = np.zeros(12)
    p50 = np.zeros(12)
    p90 = np.zeros(12)

    for j in range(len(counter_list)): # 12 months
        counter_list_days = np.array(counter_list[j])/24 
        mean[j] = np.mean(counter_list_days)
        p10[j] = np.percentile(counter_list_days, 10)
        p50[j] = np.percentile(counter_list_days, 50)
        #p75 = np.percentile(counter_list_days, 75)
        p90[j] = np.percentile(counter_list_days, 90)
    #breakpoint()
    return mean, p10, p50,p90

def weather_window_length(time_series,month,threshold,op_duration,timestep):
    print(month)
    # time_series: input timeseries (numpy array)
    # threshold over which operation is possible (same unit as timeseries)
    # op_duration: duration of operation in hours
    # timestep: time resolution of time_series in hours
    # returns an array with all weather windows duration in hours
    month_ts = time_series.index.month
    ts_mask=np.where(time_series<threshold,1,0)
    od=int(op_duration/timestep)
    ts=np.zeros((len(ts_mask)-od+1))
    mon=np.zeros((len(ts_mask)-od+1))
    for i in range(len(ts_mask)-od+1):
        ts[i]=np.sum(ts_mask[i:i+od])
    s0=np.where(ts==od)[0].tolist()
    mon_s0=month_ts[0:s0[-1]+1]
    wt=[]
    for s in range(len(s0)):
        if s0[s]==0:
            wt.append(0)
        elif s==0:
            diff=s0[s]
            a=0
            wt.append(diff)
            while (diff!=0):
                diff=s0[s]-a-1
                wt.append(diff*timestep)
                a=a+1
        else:
            diff=s
            a=0
            while (diff!=0):
                diff=s0[s]-s0[s-1]-a-1
                wt.append(diff*timestep)
                a=a+1
    wt1=(np.array(wt)+op_duration)/24
    # Note that we this subroutine, we stop the calculation of the waiting time
    # at the first timestep of the last operating period found in the timeseries
    mean = np.mean(wt1[mon_s0==month])
    p10 = np.percentile(wt1[mon_s0==month],10)
    p50 = np.percentile(wt1[mon_s0==month],50)
    p90 = np.percentile(wt1[mon_s0==month],90)
    return mean, p10, p50, p90



