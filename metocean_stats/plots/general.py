import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt

from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from cycler import cycler

from .. import stats
from .. import tables


def plot_scatter_diagram(
        data: pd.DataFrame, 
        var1: str, step_var1: float, 
        var2: str, step_var2: float, 
        output_file='', 
        log_color=False, 
        percentage_values=True,
        annot=False, 
        significant_digits = 2,
        annot_nonzero_only = True,
        range_including = True, 
        from_origin = True,
        marginal_values = True,
        cmap = plt.get_cmap('Blues'),
        use_cbar = True,
        square = False,
        linewidths:float = 0,
        linecolor: str = 'black',
        **kwargs
        ):
    """
    Plot scatter diagram (heatmap) of two variables (e.g, var1='hs', var2='tp')
    
    Parameters
    -----------
    data : pd.DataFrame
        The data containing the variables as columns.
    var1 : str
        The first variable, plotted on y-axis
    step_var1 : float
        Interval of bins of the first variable.
    var2 : str
        The second variable, plotted on x-axis.
    step_var2 : float
        Interval of bins of the second variable.
    output_file : str, default = ''
        Output filename. Empty string will not save a file.
    log_color : bool, default False
        Use logarithmic colorscale.
    percentage_values : bool, default True
        Whether to use percentage of total sum in margins and cell annotations, as opposed to number of events.
    annot : bool, default False
        Write the numeric value of each cell in the figure.
    significant_digits : int, default 2
        Number of digits used in annotated cells and/or marginal values.
    annot_nonzero_only : bool, default True
        If false, zero-valued cells will still be annotated.
    range_including : bool, default True
        Include the end of the range in the tick labels, e.g. 0.0-0.5 instead of just 0.0
    from_origin : bool, default True
        Start the scatter plot in origin, even if there are no values in the first bin(s).
    marginal_values : bool, default True
        Include the marginal sum/fraction on ticks, e.g. 0.0-0.5 | 5.8%
    cmap : matplotlib colormap, default "Blues"
        Any matplotlib colormap.
    use_cbar : bool, default True
        Whether to include the colorbar.
    square : bool, default False
        Make the figure a square.
    **kwargs : other arguments
        Any other keyword arguments accepted by seaborn heatmap.
        
    Returns
    ----------
    matplotlib figure

    Notes
    -------
    The function is written by dung-manh-nguyen, KonstantinChri, and efvik.
    """

    sd = stats.calculate_scatter(data, var1, step_var1, var2, step_var2,from_origin=from_origin)

    # Convert to percentage
    tbl = sd.values
    var1_data = data[var1]
    if percentage_values:
        tbl = tbl/len(var1_data)*100

    # Then make new row and column labels with a summed percentage
    sumcols = np.sum(tbl, axis=0)
    sumrows = np.sum(tbl, axis=1)

    bins_var1 = sd.index
    bins_var2 = sd.columns
    lower_bin_1 = bins_var1[0] - step_var1
    lower_bin_2 = bins_var2[0] - step_var2

    def _tick_writer(a,b,c):
        a = int(a) if np.isclose(int(a),a) else np.round(a,1)
        b = int(b) if np.isclose(int(b),b) else np.round(b,1)
        c = str(int(c)) if not percentage_values else f'{c:.{significant_digits}f}'

        tick = f'{a}'
        if range_including: tick = tick + f'-{b}'
        if marginal_values:
            tick = tick + f' | {c}'
            if percentage_values: tick = tick + '%'
        return tick

    rows = []
    rows.append(_tick_writer(lower_bin_1,bins_var1[0],sumrows[0]))
    for i in range(len(bins_var1)-1):
        rows.append(_tick_writer(bins_var1[i],bins_var1[i+1],sumrows[i+1]))

    cols = []
    cols.append(_tick_writer(lower_bin_2,bins_var2[0],sumcols[0]))
    for i in range(len(bins_var2)-1):
        cols.append(_tick_writer(bins_var2[i],bins_var2[i+1],sumcols[i+1]))

    rows = rows[::-1]
    tbl = tbl[::-1,:]
    dfout = pd.DataFrame(data=tbl, index=rows, columns=cols)
    fig,ax = plt.subplots()
    norm = mcolors.LogNorm() if log_color else None
    mask = (dfout.round(significant_digits) != 0) if annot_nonzero_only else np.ones_like(dfout,dtype='bool')
    fmt = f".{significant_digits}f" if percentage_values else f'.{significant_digits}g'
    sns.heatmap(ax=ax, data=dfout.where(mask), cbar=use_cbar, cmap=cmap, fmt=fmt, norm=norm, annot=annot, square=square, linewidths=linewidths, linecolor=linecolor, **kwargs)

    plt.ylabel(var1)
    plt.xlabel(var2)
    plt.tight_layout()
    if output_file != '': plt.savefig(output_file)
    return fig
    
    
    
def plot_pdf_all(data, var, bins=70, output_file='pdf_all.png'): #pdf_all(data, bins=70)
    
    data = data[var].values
    
    x=np.linspace(min(data),max(data),100)
    
    param = st.weibull_min.fit(data) # shape, loc, scale
    pdf_weibull = st.weibull_min.pdf(x, param[0], loc=param[1], scale=param[2])
    
    # param = expon.fit(data) # loc, scale
    # pdf_expon = expon.pdf(x, loc=param[0], scale=param[1])
    
    param = st.genextreme.fit(data) # shape, loc, scale
    pdf_gev = st.genextreme.pdf(x, param[0], loc=param[1], scale=param[2])
    
    param = st.gumbel_r.fit(data) # loc, scale
    pdf_gumbel = st.gumbel_r.pdf(x, loc=param[0], scale=param[1])
    
    param = st.lognorm.fit(data) # shape, loc, scale
    pdf_lognorm = st.lognorm.pdf(x, param[0], loc=param[1], scale=param[2])
    
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
    if output_file != "": plt.savefig(output_file)
    
    return fig

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

def plot_monthly_stats(data: pd.DataFrame,
                       var:str,
                      show=["min","25%","max"],
                      fill_between:list[str]=[],
                      fill_color_like:str="",
                      title:str="",
                      cmap = plt.get_cmap("viridis"),
                      output_file:str="monthly_stats.png",
                      month_xticks=True):
    """
    Plot monthly statistics of a variable from a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        var (str): The name of the variable to plot.
        show (list[str]): List of percentiles/statistics to include. Options are: "min","mean","0%","1%",...,"100%","max". 
        fill_between (list[str]): Two percentiles to create a shaded area between. Same options as show.
        fill_color_like (str,optional): Use this to set color shade equal to a stat/percentile from show.
        month_xticks (bool): Set months as xticklabels.
        title (str, optional): Title of the plot. Default is 'Variable [units] location'.
        output_file (str, optional): File path to save the plot. Default is 'monthly_stats.png'.

    Returns:
        matplotlib.figure.Figure: The Figure object of the generated plot.

    Example:
        plot_monthly_stats(df, 'temperature', show=["min","mean","99%"], fill_between = ["25%","75%"], 
        fill_color_like = "mean", title = "Monthly Temperature Statistics", output_file = "temp_stats.png")
    """
    
    fig,ax = plt.subplots()
    percentiles = data[var].groupby(data[var].index.month).describe(percentiles=np.arange(0,1,0.01))
    xaxis = np.arange(0,data[var].index.month.max())
    
    labels = [s for s in show]
    if fill_between: labels += [fill_between[0]+"-"+fill_between[1]]
    show = _percentile_str_to_pd_format(show)
    fill_between = _percentile_str_to_pd_format(fill_between)
    fill_color_like = _percentile_str_to_pd_format(fill_color_like)
    
    colors = cmap(np.linspace(0,1,len(show)))
    for i,v in enumerate(show):
        percentiles[v].plot(color=colors[i],marker='o')

    if fill_between != []:
        if fill_color_like != "":
            fill_color = colors[np.where(fill_color_like==np.array(show))[0][0]]
            plt.fill_between(xaxis+1,percentiles[fill_between[0]],percentiles[fill_between[1]],alpha=0.25,color=fill_color)
        else:
            plt.fill_between(xaxis+1,percentiles[fill_between[0]],percentiles[fill_between[1]],alpha=0.25)

    if month_xticks:
        monthlabels = list(pd.date_range("2024","2024-12",freq="MS").strftime("%b"))
        ax.set_xticks(np.arange(1,13))
        ax.set_xticklabels(monthlabels)

    plt.title(title,fontsize=16)
    plt.xlabel('Month',fontsize=15)
    plt.legend(labels)
    plt.grid()
    if output_file != "": plt.savefig(output_file)
    return fig

def plot_daily_stats(data:pd.DataFrame,
                     var:str,
                     show=["min","25%","max"],
                     fill_between:list[str]=[],
                     fill_color_like = "",
                     title = "",
                     cmap = plt.get_cmap("viridis"),
                     output_file:str="daily_stats.png",
                     month_xticks=True):
    '''
    Plot daily statistics of a DataFrame variable.
    
    Arguments
    ---------
    data : pd.DataFrame
        The dataframe.
    var : str
        A column of the dataframe.
    show : list[str]
        List of percentiles/statistics to include. Options are: "min","mean","0%","1%",...,"100%","max".
    fill_between : [str,str]
        Optional: Set a shaded area between two percentiles (any options from the "show" argument).
    fill_color_like : str
        Optional: Color the shaded area like any item from "show" argument, e.g. fill_color_like = "mean".
    month_xticks : bool
        Set months as xtick labels.
    title : str
        Title of the plot.
    output_file : str
        File path for saved figure.    
    
    Returns
    --------
    fig : Figure
        Matplotlib figure object.
    '''
    
    fig,ax = plt.subplots()
    percentiles = data[var].groupby(data[var].index.dayofyear).describe(percentiles=np.arange(0,1,0.01))
    xaxis = np.arange(0,data[var].index.dayofyear.max())
    
    labels = [s for s in show]
    if fill_between: labels += [fill_between[0]+"-"+fill_between[1]]
    show = _percentile_str_to_pd_format(show)
    fill_between = _percentile_str_to_pd_format(fill_between)
    fill_color_like = _percentile_str_to_pd_format(fill_color_like)

    colors = cmap(np.linspace(0,1,len(show)))
    for i,v in enumerate(show):
        percentiles[v].plot(color=colors[i])

    if fill_between != []:
        if fill_color_like != "":
            fill_color = colors[np.where(fill_color_like==np.array(show))[0][0]]
            plt.fill_between(xaxis+1,percentiles[fill_between[0]],percentiles[fill_between[1]],alpha=0.25,color=fill_color)
        else:
            plt.fill_between(xaxis+1,percentiles[fill_between[0]],percentiles[fill_between[1]],alpha=0.25)
    
    if month_xticks:
        ax.xaxis.set_major_locator(MonthLocator(bymonthday=1,bymonth=range(1,13)))
        ax.xaxis.set_major_formatter(DateFormatter('%b'))

    plt.title(title,fontsize=14)
    plt.xlabel('Month',fontsize=12)
    plt.legend(labels)
    plt.grid()
    if output_file != "": plt.savefig(output_file)
    return fig

def plot_directional_stats(data: pd.DataFrame, var: str, step_var: float, var_dir: str,show=['Mean','P99','Maximum'], title: str='Variable [units] location',output_file: str = 'directional_stats.png'):
    """
    Plot directional statistics of a variable from a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        var (str): The name of the variable to plot.
        step_var (float): The step size for computing cumulative statistics.
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
    cumulative_percentage = tables.table_directional_non_exceedance(data,var,step_var,var_dir)
    for i in show:
        cumulative_percentage.loc[i][:-1].plot(marker = 'o')
    #cumulative_percentage.loc['Maximum'][:-1].plot(marker = 'o')
    #cumulative_percentage.loc['P99'][:-1].plot(marker = 'o')    
    #cumulative_percentage.loc['Mean'][:-1].plot(marker = 'o')
    
    plt.title(title,fontsize=16)
    plt.xlabel('Direction[$⁰$]',fontsize=15)
    plt.legend()
    plt.grid()
    if output_file != "": plt.savefig(output_file)
    return fig

def plot_monthly_weather_window(data: pd.DataFrame, var: str,threshold=5, window_size=12,add_table=True, output_file: str = 'monthly_weather_window_plot.png'):
    results_df = stats.calculate_monthly_weather_window(data=data, var=var, threshold=threshold, window_size=window_size)
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
    if output_file != "": plt.savefig(output_file)

    return fig, table

# This function doesn't work, because sort_by_month is not defined. 
# Therefore it's commented out. Maybe plot_monthly_stats above is a replacement.

# def plot_monthly_max_mean_min(df,var='T2m',out_file='plot_monthly_max_min_min.png'):
#     out = sort_by_month(df)
#     df2 = pd.DataFrame()
#     months = calendar.month_name[1:] # eliminate the first insane one 
#     df2['month']=[x[:3] for x in months] # get the three first letters 
#     df2['Min']=[np.min(out[m][var]) for m in out]
#     df2['Mean']=[np.mean(out[m][var]) for m in out]
#     df2['Max']=[np.max(out[m][var]) for m in out]
    
#     plt.plot(df2['month'],df2['Min'])
#     plt.plot(df2['month'],df2['Mean'])
#     plt.plot(df2['month'],df2['Max'])
#     plt.ylabel('Temperature [℃]') 
#     plt.grid()
#     plt.savefig(out_file,dpi=100,facecolor='white',bbox_inches='tight')
    
#     return df2 


def plot_profile_stats(data,var=['W10','W50','W80','W100','W150'], z=[10, 50, 80, 100, 150],reverse_yaxis=False, output_file='stats_profile.png'):
    df = tables.table_profile_stats(data=data, var=var, z=z, output_file=None)
    df = df.drop(['Std.dev', 'Max Speed Event'],axis=1)
    fig, ax = plt.subplots()
    plt.yticks(z)  # Set yticks to be the values in z
    ax.yaxis.set_major_locator(mticker.MultipleLocator(int(max(z)/4)))  # Set major y-ticks at intervals of 10

    for column in df.columns[1:]:
        plt.plot(df[column][1:],z,marker='.', label=column)
    plt.ylabel('z[m]')
    plt.xlabel('[m/s]')
    if reverse_yaxis:
        plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    if output_file != "": plt.savefig(output_file)

    return fig


def plot_profile_monthly_stats(
        data: pd.DataFrame, 
        var: str, z=[10, 20, 30], 
        months:list[str]=[],
        method='mean',
        title='Sea Temperature [°C]', 
        reverse_yaxis=True, 
        output_file='table_profile_monthly_stats.png',
        include_year=True):

    # Custom color cycle
    custom_cycler = cycler(color=['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    
    # Get the data
    df = tables.table_profile_monthly_stats(data=data, var=var, z=z, method=method, output_file=None, rounding=None)
    # Create a plot
    fig, ax = plt.subplots()
    # Set the custom color cycle
    ax.set_prop_cycle(custom_cycler)
    # Set yticks to be the values in z
    plt.yticks(z)
    
    # Set major y-ticks at intervals of max(z)/4
    ax.yaxis.set_major_locator(mticker.MultipleLocator(int(max(z)/4)))
    
    # Only plot specified variables.
    if months == []:
        months = df.columns
    if not include_year:
        months = [m for m in months if m!="Year"]
    
    # Plot each column with alternating line styles
    for idx, column in enumerate(months):
        linestyle = '-' if idx % 2 == 0 else '--'
        plt.plot(df[column], z, marker='.', linestyle=linestyle, label=column)
    
    plt.ylabel('z[m]')
    plt.title(title)
    
    # Reverse the y-axis if needed
    if reverse_yaxis:
        plt.gca().invert_yaxis()

    plt.legend()
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    # Save the figure
    if output_file != "": plt.savefig(output_file)
    return fig


def plot_tidal_levels(data, var='tide',start_time=None , end_time=None ,output_file='tidal_levels.png'):
    df = tables.table_tidal_levels(data=data, var=var, output_file=None)
    if start_time and end_time is None:
       data = data[var]
    else:
        data = data[var][start_time:end_time]


    fig, ax = plt.subplots()
    plt.plot(data,color='lightgrey')
    plt.axhline(y=df.loc[df['Tidal Level'] == 'HAT'].values[0][1], color='r', linestyle='--',label='HAT')
    plt.axhline(y=df.loc[df['Tidal Level'] == 'MSL'].values[0][1], color='k', linestyle='-',label='MSL')
    plt.axhline(y=df.loc[df['Tidal Level'] == 'LAT'].values[0][1], color='r', linestyle='--',label='LAT')

    plt.ylabel('tidal elevation [m]')
    plt.grid()

    plt.legend(loc='lower right')
    plt.xticks(rotation=45)

    # Set major locator for x-axis to avoid overlap
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))  

    plt.tight_layout()
    # Save the figure
    plt.savefig(output_file)

    return fig


def plot_nb_hours_below_threshold(df,var='hs',thr_arr=(np.arange(0.05,20.05,0.05)).tolist(),output_file='plot_name.png'):
    # Inputs
    # 1) ds
    # 2) var: a string
    # 3) thr_arr: list of thresholds (should be a lot for smooth curve)
    # 4) String with filename without extension
    nbhr_arr=stats.nb_hours_below_threshold(df,var,thr_arr)
    years=df.index.year.to_numpy()
    years_unique=np.unique(years)
    yr1=int(years_unique[0])
    yr2=int(years_unique[-1])
    del df,years,years_unique

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.fill_between(thr_arr,np.min(nbhr_arr,axis=1),np.max(nbhr_arr,axis=1),color='lightgray')
    ax.plot(thr_arr,np.mean(nbhr_arr,axis=1),linewidth=2,color='k')
    ax.grid(axis='both', color='gray',linestyle='dashed')
    ax.set_ylim(0,9000)
    ax.set_xlim(0,20)
    ax.set_xlabel(var+' threshold [m]',fontsize=20)
    ax.set_ylabel('Number of hours',fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_title('Mean number of hours per year with '+var+' < threshold\n'+str(yr1)+'-'+str(yr2),fontsize=20)
    legend_elements = [mpatches.Patch(facecolor='lightgray', edgecolor=None, label='Min-Max range')]
    ax.legend(handles=legend_elements, loc='lower right', prop={'size': 20})
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    del ax
    return fig

