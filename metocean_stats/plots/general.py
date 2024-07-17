import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
from math import floor,ceil
from ..stats.general import *
from ..tables.general import *


def plot_scatter_diagram(data: pd.DataFrame, var1: str, step_var1: float, var2: str, step_var2: float, output_file):
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
    
    
    
def plot_pdf_all(data, var, bins=70, output_file='pdf_all.png'): #pdf_all(data, bins=70)
    
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


def plot_scatter(df,var1,var2,location,regression_line,qqplot=True):
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


def plot_monthly_stats(data: pd.DataFrame, var1: str, show: str, title: str='Variable [units] location',output_file: str = 'monthly_stats.png'):
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
    cumulative_percentage = table_monthly_non_exceedance(data,var1,step_var1=0.5)
    for i in show:
        cumulative_percentage.loc[i][:-1].plot(marker = 'o')
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

def plot_monthly_weather_window(data: pd.DataFrame, var: str,threshold=5, window_size=12,add_table=True, output_file: str = 'monthly_weather_window_plot.png'):
    results_df = calculate_monthly_weather_window(data=data, var=var, threshold=threshold, window_size=window_size)
    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 6))
    results_df.T.plot(marker='o')
    lines = results_df.T.plot(marker='o')
    plt.title(str(var)+' < '+str(threshold)+' for ' + str(window_size)+' hours')
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

    return fig, table


def plot_monthly_max_mean_min(df,var='T2m',out_file='plot_monthly_max_min_min.png'):
    out = sort_by_month(df)
    df2 = pd.DataFrame()
    months = calendar.month_name[1:] # eliminate the first insane one 
    df2['month']=[x[:3] for x in months] # get the three first letters 
    df2['Min']=[np.min(out[m][var]) for m in out]
    df2['Mean']=[np.mean(out[m][var]) for m in out]
    df2['Max']=[np.max(out[m][var]) for m in out]
    
    plt.plot(df2['month'],df2['Min'])
    plt.plot(df2['month'],df2['Mean'])
    plt.plot(df2['month'],df2['Max'])
    plt.ylabel('Temperature [℃]') 
    plt.grid()
    plt.savefig(out_file,dpi=100,facecolor='white',bbox_inches='tight')
    
    return df2 


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import windrose
import matplotlib.cm as cm
import os

def rose(wd,ws,max_ws,step_ws,min_percent, max_percent, step_percent):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="windrose")
    ax.bar(wd, ws, bins=np.arange(0, max_ws, step_ws), cmap=cm.rainbow, normed=True, opening=0.9, edgecolor='white')
    ax.set_yticks(np.arange(min_percent, max_percent, step_percent))
    ax.set_yticklabels(np.arange(min_percent, max_percent,step_percent))
    ax.legend(bbox_to_anchor=(0.90,-0.05),framealpha=0.5)
    return fig


def var_rose(data, direction,intensity, output_file, method='overall'):

    direction2 = data[direction]
    intensity2 = data[intensity]
    size = 5
    bins_range = np.array([0, np.percentile(intensity2,40),
                   np.percentile(intensity2,60),
                   np.percentile(intensity2,80),
                   np.percentile(intensity2,99)])
    
    if method == 'overall':
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(111, projection="windrose")
        ax.bar(direction2, intensity2, normed=True, bins=bins_range, opening=0.8, nsector=12)
        ax.set_legend()
        ax.figure.set_size_inches(size, size)
        plt.savefig(output_file,dpi=100,facecolor='white',bbox_inches='tight')

    elif method == 'monthly':
        monthly_var_rose(data,direction,intensity,output_file)
    
    plt.close()
    return fig

def monthly_var_rose(data, direction,intensity,output_file) : 

    # this function make monthly wind/wave rose
    # direction, intensity: panda series 
    # get month from panda series 
    direction = data[direction]
    intensity = data[intensity]
    M = intensity.index.month.values
    
    # get month names 
    import calendar
    months = calendar.month_name[1:] # eliminate the first insane one 
    for i in range(len(months)) : 
        months[i] = months[i][:3] # get the three first letters 
    
    # sort them outh by months 
    dic_intensity = {} # dic_intensity
    dic_direction = {} # dic_direction
    for i in range(len(months)) : 
        dic_intensity[months[i]] = [] 
        dic_direction[months[i]] = [] 
        
    for i in range(len(intensity)) : 
        m_idx = int(M[i]-1)
        dic_intensity[months[m_idx]].append(intensity.iloc[i])
        dic_direction[months[m_idx]].append(direction.iloc[i])
        
    for j in range(12):
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(111, projection="windrose")
        ax.bar(dic_direction[months[j]], dic_intensity[months[j]], normed=True, opening=0.8, nsector=12)
        ax.set_legend()
        ax.set_title(months[j])
        size = 5
        ax.figure.set_size_inches(size, size)
        plt.savefig(months[j]+'_'+output_file,dpi=100,facecolor='white',bbox_inches='tight')
        plt.close()
    return fig
