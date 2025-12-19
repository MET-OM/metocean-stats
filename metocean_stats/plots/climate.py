import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from ..tables import climate,general
from ..stats import general
import numpy as np
import calendar
import sys
import re


#################################### TO BE ADJUSTED FOR DISCRETE COLORMAP #####################################
def plot_heatmap_profiles_yearly(df, rad_colname='current_speed_', cb_label='Current speed [m/s]', yaxis_direction='down', method='mean', output_file='heatmap_profile.pdf'):
    """
    This function plot heatmap of yearly vertical profiles

    Parameters
    ----------
    df: pd.DataFrame
        Should have datetime as index and 
        column name in format of level and m at the end, for example 12.3m 
    rad_colname: string
    yaxis_directin: string
        Can be 'up' for increasing y-axis (height) and 'down' for decreasing y-axis (depth)
    method: string
        List of percentiles, e.g. P25, P50, P75, 30%, 40% etc.
        Some others are also allowed: count (number of data points), and min, mean, max.
    cb_label: string
        Label of color bar
    
    Returns
    -------
    Figure

    Authors
    -------
    Written by Dung M. Nguyen and clio-met
    """
    df1=climate.table_yearly_1stat_vertical_levels(df, rad_colname=rad_colname, method=method, output_file='')
    levs = [str(re.search(r'\d+(\.\d+)?', s).group()) for s in df1.columns]
    # Delete the last line of the dataframe (All years)
    df1 = df1.iloc[:-1,:]

    min_value = min(df1.min())
    max_value = max(df1.max())

    norm = mcolors.Normalize(vmin=min_value, vmax=max_value)
    cmap = plt.cm.get_cmap('viridis',int((max_value-min_value)*1000))

    fig, ax = plt.subplots(figsize=(12, 7))
    cax = ax.imshow(df1.T, aspect='auto', cmap=cmap, norm=norm)
    cbar = fig.colorbar(cax, ax=ax)
    cbar.ax.tick_params(labelsize=14) # Adjust labelsize as desired
    cbar.set_label(cb_label, size=14) # Adjust size as desired
    ax.set_yticks(ticks=range(len(levs)), labels=levs)
    if len(df1.index)>10:
        ax.set_xticks(ticks=range(len(df1.index))[::5], labels=df1.index[::5])
    else:
        ax.set_xticks(ticks=range(len(df1.index)), labels=df1.index)
    ax.set_xlabel('Years', fontsize=14)
    ax.set_ylabel('z [m]', fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    if yaxis_direction=='up':
        plt.gca().invert_yaxis()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.tight_layout()
    if output_file != "":
        plt.savefig(output_file,dpi=250,facecolor='white',bbox_inches='tight')
    plt.close()

    return fig




def plot_yearly_stripes(df,var_name='Hs',method='mean',ylabel='Y label',output_file='figure.pdf'):
    """
    This function calculates a yearly statistic for one variable
    And plots the result as a curve with colored stripes in the background
    
    Parameters
    ----------
    df: pd.DataFrame with index as datetime
    var_name: string
        Variable name (column name) 
    method: string
        Can be either 'min','mean', 'max', or a percentile (P99, 99%,...)
    ylabel: string
        Label of y axis
    output_file: string
        Figure name

    Returns
    -------
    Figure

    Authors
    -------
    Written by Dung M. Nguyen and clio-met
    Inspired from Ed Hawkins' warming stripes https://en.wikipedia.org/wiki/Warming_stripes
    """

    df2=climate.table_yearly_stats(df,var=var_name,percentiles=method,output_file="")
    df2=df2.iloc[:-1,0]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdBu_r((df2 - df2.min()) / (df2.max() - df2.min()))
    ax.bar(df2.index.to_numpy(), height=100, width=1.0, color=colors, bottom=df2.min()-1.5)
    ax.set_xlim(df2.index.min()-0.5, df2.index.max()+0.5)
    ax.set_ylim(df2.min()-0.5, df2.max()+0.5)
    ax.plot(df2.index,df2,'k-',linewidth=5)
    ax.plot(df2.index,df2,'w-',linewidth=2)
    ax.scatter(df2.index,df2,marker='o',s=50,c='w',edgecolors='k',zorder=100)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(.01, .97, 'Min-Max range = '+str(round(df2.max()-df2.min(),1)), ha='left', va='top', transform=ax.transAxes, bbox=props, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    xpos=df2.index[::5].to_list()
    labs=[str(int(l)) for l in xpos]
    ax.set_xticks(ticks=xpos, labels=labs)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    plt.tight_layout()
    if output_file != "":
        plt.savefig(output_file,dpi=250,facecolor='white',bbox_inches='tight')
    plt.close()

    return fig


def plot_heatmap_monthly_yearly(df,var='air_temperature_2m',method='mean',cb_label='Mean Temperature [°C]',output_file='figure.pdf'):
    """
    This function plots the monthly statistic for every year

    Parameters
    ----------
    df: pd.DataFrame with index of datetime
    var: string
        Variable name 
    method: string
        Can be either 'min','p1','p5','p50','mean','p95','p99' or 'max'
    cb_label: string
        Title of the color bar
    output_file: string
        Figure name

    Returns
    -------
    Figure

    Authors
    -------
    Written by Dung M. Nguyen and clio-met
    Inspired from Ed Hawkins' warming stripes https://en.wikipedia.org/wiki/Warming_stripes
    """
    # get month names 
    months_names=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Year']

    df2=general.stats_monthly_every_year(df,var=var,method=[method])
    df3=climate.table_yearly_stats(df,var=var,percentiles=[method],output_file="")
    print(df2)
    print(df3)

    years=np.unique(df2['year'].to_numpy())
    months=np.unique(df2['month'].to_numpy())
    
    # Select first column
    df2_1st=df2.iloc[:,0]
    print(df2_1st)
    min_value = np.floor(df2_1st.min())
    max_value = np.ceil(df2_1st.max())
    norm = mcolors.Normalize(vmin=min_value, vmax=max_value)

    arr=df2_1st.to_numpy().reshape(len(years),len(months)).T
    # Add the whole year statistic
    arr=np.concatenate([arr,df3.iloc[:-1,0].to_numpy().flatten()[np.newaxis,:]],axis=0)
    print(arr)
    print(df3[:-1])
    del df2_1st,df2,df3
    
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.get_cmap('viridis',20)
    cax = ax.imshow(arr, aspect='auto', cmap=cmap, norm=norm)
    cbar = fig.colorbar(cax, ax=ax)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(cb_label, size=14)
    ax.set_xticks(ticks=range(len(years))[::5], labels=[str(y) for y in years.tolist()][::5])
    ax.set_yticks(ticks=range(len(months_names)), labels=months_names)
    ax.set_xlabel('Years', fontsize=14)
    ax.set_ylabel('Months', fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.axhline(y=11.5,linewidth=1, color='k')
    plt.tight_layout()
    if output_file != "":
        plt.savefig(output_file,dpi=250,facecolor='white',bbox_inches='tight')
    plt.close()

    return fig



def plot_yearly_vertical_profiles(df, rad_colname='current_speed_', method='mean', yaxis_direction='down', xlabel='Current speed [m/s]', output_file='yearly_profile.pdf'):
    """
    This function plot yearly mean vertical profiles

    Parameters
    ----------
    df: pd.DataFrame with index of datetime
    rad_colname: string
        Prefix of the variable of interest (e.g. current_speed_)
    method: string
        A percentile e.g. P25, P50, P75, 30%, 40% etc.
        or min, mean, max.
    yaxis_direction: string
        Can be 'up' for increasing y-axis (height) and 'down' for decreasing y-axis (depth)
    xlabel: string
        Label of x-axis
    output_file: string
        Name of the figure file
    
    Returns
    -------
    Figure

    Authors
    -------
    Written by Dung M. Nguyen and clio-met
    """
    df1=climate.table_yearly_1stat_vertical_levels(df, rad_colname=rad_colname, method=method, output_file='')
    levs = np.array([str(re.search(r'\d+(\.\d+)?', s).group()) for s in df1.columns])
    # Delete the last line of the dataframe (All years)
    df1=df1.iloc[:-1,:]
    years = [str(yr) for yr in df1.index.to_list()]
    n_years = len(df1)

    # To get a discrete colormap
    cmap = plt.get_cmap('viridis', n_years)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(0,n_years+1,1), ncolors=n_years)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    for y in range(n_years):
        ax.plot(df1.iloc[y,:].to_numpy(),levs,c=cmap(y))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    if n_years>10:
        cbar.set_ticks(np.arange(0,n_years,1)[::5]+0.5)
        cbar.set_ticklabels(years[::5])
    else:
        cbar.set_ticks(np.arange(0,n_years,1)+0.5)
        cbar.set_ticklabels(years)
    cbar.set_label('Year', size=14)
    cbar.ax.tick_params(labelsize=14)
    if yaxis_direction=='down':
        ax.invert_yaxis()
        ax.set_ylim(levs[-1],levs[0])
    else:
        ax.set_ylim(levs[0],levs[-1])
    ax.set_ylabel('z [m]', fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.tick_params(axis='x', labelsize=14) 
    ax.tick_params(axis='y', labelsize=14) 
    plt.grid()
    plt.tight_layout()
    if output_file != "":
        plt.savefig(output_file,dpi=250,facecolor='white',bbox_inches='tight')
    plt.close()

    return fig





def plot_linear_regression(df,var='air_temperature_2m',time='Jan',stat='mean',method=['Least-Squares','Theil-Sen'],confidence_interval=0.95,ylabel='2-m T [°C]',output_figure=None):
    """
    This function plots the Ordinary Least Squares linear regression line and/or
    the median, lower and upper slopes (within the confidence interval) using the Theil-Sen method
    for monthly or yearly time series as a function of the year 

    Parameters
    ----------
    df: pd.DataFrame with index of datetime
    var: string
        Variable name
    time: string
        Can be 'Jan', 'Feb', 'January', 'Year' (for all years)
    stat: string
        Can be 'min', 'p5', 'p90', '50%', 'mean', 'max'. Default is 'mean'
    method: list of string
        By default 'Least-Squares', 'Theil-Sen'
    confidence interval: float
        To be specified for the Theil-Sen method, between 0.5 and 1. Default is 0.95
    ylabel: string
        Label of the y-axis containing unit within []
    output_file: string
        Name of the output file

    Returns
    -------
    pd.DataFrame
    For Least-Squares: slope, intercept, and coefficient of determination R^2
    For Theil-Sen: median slope, intercept, smallest, and highest slope
    For Kendall: gives the tau and p-value

    Authors
    -------
    Written by Dung M. Nguyen and clio-met
    """
    months_names = calendar.month_abbr[1:]

    df1,df_monthly,df_yearly=climate.table_linear_regression(df,var=var,stat=stat,method=method,confidence_interval=confidence_interval,intercept=True,output_file=None)
    if time=='Year':
        df2=df1.iloc[-1,:] # last row
        yval=df_yearly.to_numpy()
    else:
        t=time[0:3]
        ix=np.where(np.array(months_names)==t)[0][0]+1
        df2=df1[t]
        yval=df_monthly[df_monthly['month']==ix].iloc[:,0].to_numpy()

    del df1

    unit = (ylabel.split('['))[1].split(']')[0]

    xval=df.index.year.unique().to_numpy()

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(xval,yval,marker='.',linewidth=0.5,color='k')
    if 'Least-Squares' in method:
        y_ls=df2['LS_slope']*xval+df2['LS_intercept']
        ax.plot(xval,y_ls,color='b',linewidth=1,label='Least Squares, slope = '+str(round(df2['LS_slope'],3))+' '+unit+'/yr')
        del y_ls
        
    if 'Theil-Sen' in method:
        y_ts=df2['TS_slope']*xval+df2['TS_intercept']
        y_ts_l=df2['TS_slope_lower']*xval+df2['TS_intercept']
        y_ts_u=df2['TS_slope_upper']*xval+df2['TS_intercept']
        x0 = (xval.min() + xval.max()) / 2.0
        y0 = df2['TS_slope'] * x0 + df2['TS_intercept']
        # Compute intercepts so the lo/hi slope lines cross at (x0, y0)
        b_l = y0 - df2['TS_slope_lower'] * x0
        b_u = y0 - df2['TS_slope_upper'] * x0
        y_ts_l=df2['TS_slope_lower']*xval+b_l
        y_ts_u=df2['TS_slope_upper']*xval+b_u
        ax.plot(xval,y_ts,color='r',linewidth=1,linestyle='solid',label='Theil–Sen, slope = '+str(round(df2['TS_slope'],3))+' '+unit+'/yr')
        ax.plot(xval,y_ts_l,color='r',linewidth=1,linestyle='-.',label='Theil–Sen lower, slope = '+str(round(df2['TS_slope_lower'],3))+' '+unit+'/yr')
        ax.plot(xval,y_ts_u,color='r',linewidth=1,linestyle='--',label='Theil–Sen upper, slope = '+str(round(df2['TS_slope_upper'],3))+' '+unit+'/yr')
        del y_ts,y_ts_l,y_ts_u,x0,y0,b_u,b_l

    ax.set_xlim(xval[0],xval[-1])

    ax.set_xticks(ticks=xval[::5], labels=xval[::5],fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel('Years',fontsize=14)
    ax.set_ylabel(ylabel,fontsize=14)
    
    plt.legend()
    ax.grid(True)
    plt.tight_layout()
    if output_figure != None:
        plt.savefig(output_figure,dpi=250,facecolor='white',bbox_inches='tight')
    plt.close()

    return fig




