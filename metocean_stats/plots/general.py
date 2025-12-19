import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from cycler import cycler
import calendar


from .. import stats
from .. import tables


def plot_scatter_diagram(
        data: pd.DataFrame, 
        var1 = "TP",
        step_var1 = 1,
        var2 = "HS",
        step_var2 = 1,
        density_joint = False,
        density_marginal = True,
        format_joint = ".0f",
        format_marginal = ".0f",
        format_xticks = ".1f",
        format_yticks = ".1f",
        from_origin = True,
        xlim = None,
        ylim = None,
        annot_cells = "nonzero",
        annot_margin = True,
        percent_sign_cells = False,
        percent_sign_margin = True,
        norm = mcolors.LogNorm(),
        cbar = False,
        cmap = "Blues",
        **kwargs
        ):
    """
    Plot scatter diagram (heatmap) of two variables (e.g, var1='hs', var2='tp')
    
    Parameters
    -----------
    data : pd.DataFrame
        The data containing the variables as columns.
    var1 : str
        The first (y-axis) variable as a column name.
    step_var1 : float
        Interval of bins of the first variable.
    var2 : str
        The second (x-axis) variable as a column name.
    step_var2 : float
        Interval of bins of the second variable.
    density_joint : bool, default False
        Display the joint distribution as percentage values.
    density_marginal : bool, default True
        Display the marginal distribution as percentage values. 
    format_joint : str
        Formatting string of the joint (cell) values
    format_marginal : str
        Formatting string of the marginal values
    format_xticks : str
        Formatting string of the x-tick values
    format_yticks : str
        Formatting string of the y-tick values
    from_origin : bool
        Control whether the histogram should start at the origin
        (default) or at the first observed data points.
    xlim : tuple(float,float)
        Manually specify start and end of the x-axis.
        This should be a multiple of step_var2.
    ylim : tuple(float,float)
        Manually specify start and end of the y-axis.
        This should be a multiple of step_var1.
    annot_cells : str
        Controls which cells will be annotated with a value:
         - "all": all cells are annotated
         - "nonzero": all cells with any occurance are annotated
         - "rounded": only cells which still have a nonzero value 
         after applying the cell number formatting are annotated
         - "off": no cells are annotated
    annot_margin : bool, default True
        Add the marginal distributions of values.
    percent_sign_cells : bool, default False
        Add a percent sign after the cell values, if using density.
    percent_sign_marginal: bool, default True
        Add a percent sign after the marginal values, if using density.
    norm : matplotlib color norm, default LogNorm()
        A colormap norm.
    cbar : bool, default False
        Include a colorbar.
    **kwargs
        Any keyword arguments for seaborn heatmap.
        For example: cbar_kws = {"anchor":(x, y)} 
        will adjust position of the colorbar.

    Returns
    ----------
    matplotlib axis

    Notes
    -------
    The function is written by efvik.
    """

    valid_annot = ["all","nonzero","rounded","off"]
    if annot_cells not in valid_annot:
        raise ValueError(f"Keyword annot_cells must be one of {valid_annot}.")

    data = data[[var1,var2]]
    if np.any(np.isnan(data)):
        print("Warning: Removing NaN rows.")
        data = data.dropna(how="any")

    # Initial min and max
    xmin = data[var2].values.min()
    ymin = data[var1].values.min()
    xmax = data[var2].values.max()
    ymax = data[var1].values.max()

    # Change min (max) to zero only if all values are above (below) and from_origin=True
    xmin = 0 if (from_origin and (xmin>0)) else np.floor(xmin/step_var2)*step_var2
    ymin = 0 if (from_origin and (ymin>0)) else np.floor(ymin/step_var1)*step_var1

    xmax = 0 if (from_origin and (xmax<0)) else np.ceil(xmax/step_var2)*step_var2
    ymax = 0 if (from_origin and (ymax<0)) else np.ceil(ymax/step_var1)*step_var1

    # ylim and xlim can be manually specified
    if xlim is not None:
        xmin = xlim[0] if xlim[0] is not None else xmin
        xmax = xlim[1] if xlim[1] is not None else xmax
        if (l:=data[(data[var2]<xmin)|(data[var2]>xmax)].size):
            print(f"WARNING: {l} points excluded by chosen xlim {xlim}")
        diff = xmax-xmin
    if ylim is not None:
        ymin = ylim[0] if ylim[0] is not None else ymin
        ymax = ylim[1] if ylim[1] is not None else ymax
        if (l:=data[(data[var1]<ymin)|(data[var1]>ymax)].size):
            print(f"WARNING: {l} points excluded by chosen ylim {ylim}")

    # Define bins and get histogram
    n_bins_x = int(np.round((xmax - xmin) / step_var2)) + 1
    n_bins_y = int(np.round((ymax - ymin) / step_var1)) + 1
    x = np.linspace(xmin,xmax,n_bins_x)
    y = np.linspace(ymin,ymax,n_bins_y)

    hist, y, x = np.histogram2d(data.values[:,0],data.values[:,1],bins=(y,x))

    xlabels = [f"{i:{format_xticks}}" for i in x]
    ylabels = [f"{i:{format_yticks}}" for i in y]

    sum_x = hist.sum(axis=0)
    sum_y = hist.sum(axis=1)

    suffix_cell = ""
    if density_joint:
        hist = 100*hist/hist.sum()
        if percent_sign_cells:
            suffix_cell = "%"

    suffix_margin = ""
    if density_marginal:
        sum_x = 100*sum_x/sum_x.sum()
        sum_y = 100*sum_y/sum_y.sum()
        if percent_sign_margin:
            suffix_margin = "%"

    if annot_cells == "rounded":
        text = [[f"{h:{format_joint}}"+suffix_cell for h in row] for row in hist]
        zero = f"{0:{format_joint}}"
        text = [[t if t!=zero else "" for t in row] for row in text]
    if annot_cells == "nonzero":
        text = [[f"{h:{format_joint}}"+suffix_cell if h>0 else "" for h in row] for row in hist]
    if annot_cells == "all":
        text = [[f"{h:{format_joint}}"+suffix_cell for h in row] for row in hist]
    if annot_cells == "off":
        text =  [["" for h in row] for row in hist]

    if hasattr(norm,"vmin") and (norm.vmin == None):
        norm.vmin = hist[hist>0].min()/10

    ax = sns.heatmap(data=np.where(hist,hist, 1e-16),annot=text,fmt="",
                    xticklabels=False,yticklabels=False,
                    cbar=cbar,norm=norm,cmap=cmap,**kwargs)

    sum_x = [[f"{i:{format_marginal}}"+suffix_margin for i in sum_x]]
    sum_y = [[f"{i:{format_marginal}}"+suffix_margin] for i in sum_y[::-1]]

    if annot_margin:
        ax.table(sum_x,loc="top",cellLoc="center")
        ax.table(sum_y,loc="right",cellLoc="center",bbox=(1,0,(1/hist.shape[1]),1))

    xticks = np.arange(0,hist.shape[1]+1)
    yticks = np.arange(0,hist.shape[0]+1)

    _=ax.set_xticks(xticks,xlabels)
    _=ax.set_yticks(yticks,ylabels)

    ax.set_xlabel(var2)
    ax.set_ylabel(var1)
    ax.invert_yaxis()
    return ax
    
    
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
        monthlabels= ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
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
        monthlabels= ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.xaxis.set_major_locator(MonthLocator(bymonthday=1,bymonth=range(1,13)))
        ax.set_xticklabels(monthlabels)
        #ax.xaxis.set_major_formatter(DateFormatter('%b')) # Writes the months in Norwegian, not English

    plt.title(title,fontsize=14)
    plt.xlabel('Month',fontsize=12)
    plt.legend(labels)
    plt.grid()
    if output_file != "": plt.savefig(output_file)
    return fig


def plot_hourly_stats(data:pd.DataFrame,
                     var:str,
                     show=["min","25%","max"],
                     fill_between:list[str]=[],
                     fill_color_like = "",
                     title = "",
                     cmap = plt.get_cmap("viridis"),
                     output_file:str="daily_stats.png"):
    '''
    Plot hourly statistics of a DataFrame variable.
    
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
    title : str
        Title of the plot.
    output_file : str
        File path for saved figure.    
    
    Returns
    --------
    fig : Figure
        Matplotlib figure object.

    Authors
    -------
    Contribution by jlopez1979
    '''
    
    fig,ax = plt.subplots()
    percentiles = data[var].groupby(data[var].index.hour).describe(percentiles=np.arange(0,1,0.01))
    xaxis = np.arange(0,data[var].index.hour.max())
    xaxis = percentiles.index.values
    
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
            plt.fill_between(xaxis,percentiles[fill_between[0]],percentiles[fill_between[1]],alpha=0.25,color=fill_color)
        else:
            plt.fill_between(xaxis,percentiles[fill_between[0]],percentiles[fill_between[1]],alpha=0.25)
    
    plt.title(title,fontsize=14)
    plt.xlabel('Hours',fontsize=12)
    plt.ylabel(var,fontsize=12)
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

def plot_monthly_weather_window(data: pd.DataFrame, var: str, threshold=5, window_size=12, timestep=3, add_table=True, output_file: str = 'monthly_weather_window_plot.png'):
    results_df = tables.table_monthly_weather_window(data=data, var=var, threshold=threshold, window_size=window_size, timestep=timestep)
    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 6))
    results_df.T.plot(marker='o',cmap='viridis')
    lines = results_df.T.plot(marker='o',cmap='viridis')
    plt.title(str(var[0])+' < '+str(threshold[0])+' for ' + str(window_size)+' hours')
    plt.xlabel('Month')
    plt.ylabel('Duration [days]')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    
    if add_table:
        # Get legend colors and labels
        legend_colors = [line.get_color() for line in lines.get_lines()]
        text_colors=[] #To change the text color to white if the cell color is too dark
        for i in range(len(legend_colors)):
            (r,g,b,a)=legend_colors[i]
            if ((r<0.1) | (g<0.1) | (b<0.1)):
                text_colors.append('white')
            else:
                text_colors.append('black')
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
        # Color the cells of the first column and 
        cell_dict = table.get_celld()
        for i in range(1,len(legend_colors)+1):
            cell_dict[(i, -1)].set_facecolor(legend_colors[i-1])
            cell_dict[(i, -1)].get_text().set_color(text_colors[i-1])
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


def plot_monthly_weather_window_MultipleVariables(data: pd.DataFrame, var: str, threshold=[5], window_size=12, timestep=3, add_table=True, output_file: str = 'monthly_weather_window_plot.png'):
    # var is a list of variables (max 3) as well as thresholds (one for each variable)
    # adjusted by clio-met
    results_df = tables.table_monthly_weather_window(data=data, var=var, threshold=threshold, window_size=window_size, timestep=timestep)
    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 6))
    results_df.T.plot(marker='o',cmap='viridis')
    lines = results_df.T.plot(marker='o',cmap='viridis')
    title=[]
    for i in range(len(var)):
        title.append(var[i]+' < '+str(threshold[i]))
    title=', '.join(title)
    title=title+' for ' + str(window_size)+' hours'
    plt.title(title)
    plt.xlabel('Month')
    plt.ylabel('Duration [days]')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    
    if add_table:
        # Get legend colors and labels
        legend_colors = [line.get_color() for line in lines.get_lines()]
        text_colors=[] #To change the text color to white if the cell color is too dark
        for i in range(len(legend_colors)):
            (r,g,b,a)=legend_colors[i]
            if ((r<0.1) | (g<0.1) | (b<0.1)):
                text_colors.append('white')
            else:
                text_colors.append('black')
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
            cell_dict[(i, -1)].get_text().set_color(text_colors[i-1])
    plt.tight_layout()
    if output_file != "": plt.savefig(output_file)

    return fig, table


def plot_profile_stats(data,var=['W10','W50','W80','W100','W150'],z=[10, 50, 80, 100, 150],xlabel='[m/s]',loc='lower right',reverse_yaxis=False,output_file='stats_profile.png'):
    df = tables.table_profile_stats(data=data, var=var, z=z, output_file=None)
    df = df.drop(['Std.dev', 'Max Speed Event'],axis=1)
    fig, ax = plt.subplots()
    plt.yticks(z)  # Set yticks to be the values in z
    ax.yaxis.set_major_locator(mticker.MultipleLocator(int(max(z)/4)))  # Set major y-ticks at intervals of 10
    cmap = mpl.colormaps['viridis']
    # Take colors at regular intervals spanning the colormap.
    colors = cmap(np.linspace(0, 1, len(df.columns[1:])))
    i=0
    for column in df.columns[1:]:
        plt.plot(df[column][1:],z,marker='.', label=column,color=colors[i])
        i=i+1
    plt.ylabel('z [m]')
    plt.xlabel(xlabel)
    if reverse_yaxis:
        plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(True)
    plt.legend(loc=loc)
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
   
    # Get the data
    df = tables.table_profile_monthly_stats(data=data, var=var, z=z, method=method, output_file=None, rounding=None)
    # Create a plot
    fig, ax = plt.subplots()
    # Set yticks to be the values in z
    plt.yticks(z)
    
    # Set major y-ticks at intervals of max(z)/4
    ax.yaxis.set_major_locator(mticker.MultipleLocator(int(max(z)/4)))
    
    # Only plot specified variables.
    if months == []:
        months = df.columns
        n_col=len(months)-1
    if not include_year:
        months = [m for m in months if m!="Year"]
        n_col=len(months)
    
    cmap = mpl.colormaps['viridis']
    # Take colors at regular intervals spanning the colormap.
    colors = cmap(np.linspace(0, 1, n_col))
    if include_year:
        colors=np.vstack((colors, [0,0,0,1])) # Add black color for the whole year
    # Plot each column with alternating line styles
    for idx, column in enumerate(months):
        if column=='Year':
            plt.plot(df[column], z, marker='.', linestyle='solid', linewidth=2, color=colors[idx], label=column)
        else:
            plt.plot(df[column], z, marker='.', linestyle='solid', color=colors[idx], label=column)
    
    plt.ylabel('z [m]')
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
    plt.axhline(y=df.loc[df['Tidal Level'] == 'LAT'].values[0][1], color='r', linestyle='-.',label='LAT')

    plt.ylabel('tidal elevation [m]')
    plt.grid()

    plt.legend(loc='lower right')
    plt.xticks(rotation=45)

    # Set major locator for x-axis to avoid overlap
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))  

    plt.tight_layout()
    # Save the figure
    plt.savefig(output_file)
    plt.close()

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

