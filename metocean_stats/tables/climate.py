import pandas as pd
from ..stats import general as sg
from . import general
import numpy as np
import re
import calendar
from scipy.stats import theilslopes, linregress, kendalltau



def table_yearly_stats(data:pd.DataFrame,var:str,percentiles:list[str]=["P25","mean","P75","P99","max"],output_file="table_yearly_stats.csv"):
    """
    Group dataframe by years and calculate percentiles, min, mean, or max.

    Parameters
    ------------
    data: pd.DataFrame
        The data containing column var, and with a datetime index.
    var: string
        The column to calculate percentiles of.
    percentiles: list of strings
        List of percentiles, e.g. P25, P50, P75, 30%, 40% etc.
        Some others are also allowed: count (number of data points), and min, mean, max.
    output_file: string
        Name of the output file

    Returns
    -------
    pd.DataFrame and output_file if name specified

    Authors
    -------
    Modified from table_monthly_percentile by clio-met
    """
    series = data[var]
    percentiles = general._percentile_str_to_pd_format(percentiles)
    series.index = pd.to_datetime(series.index)
    year_labels=series.index.year.unique().tolist()+["All years"]
    table = []
    for m,g in series.groupby(series.index.year):
        table.append(g.describe(percentiles=np.arange(0,1,0.01))[percentiles])
    table.append(series.describe(percentiles=np.arange(0,1,0.01))[percentiles])
    table = pd.DataFrame(table,year_labels)
    if output_file != "":
        table.to_csv(output_file)
    return table


def table_yearly_1stat_vertical_levels(df, rad_colname='current_speed_', method='mean', output_file='table_yearly_stats_all_levels.csv'):
    """
    This function calculates yearly statistics for all vertical levels

    Parameters
    ----------
    df: pd.DataFrame
        Should have datetime as index and 
        column name in format of depth and m at the end, for example 12.3m 
    rad_colname: string
        Common part of the variable name of interest
    cb_label: string
        Label of color bar
    method: string
        Can be a percentile, e.g. P25, P50, P75, 30%, 40% etc.
        Some others are also allowed: count (number of data points), and min, mean, max.
        Only one should be specified.
    
    Returns
    -------
    result: pd.DataFrame

    Authors
    -------
    Written by Dung M. Nguyen and clio-met
    """
    filter_col = [col for col in df if col.startswith(rad_colname)]
    df=df[filter_col]
    result = pd.DataFrame(columns = filter_col)
    for i in range(len(filter_col)):
        result[filter_col[i]]=table_yearly_stats(df,var=filter_col[i],percentiles=[method],output_file="")
    
    result=result.dropna(axis=1, how='any')
    if output_file != "":
        result.to_csv(output_file)

    return result


def table_linear_regression(df,var='air_temperature_2m',stat='mean',method=['Least-Squares','Theil-Sen','Kendall-tau'],confidence_interval=0.95,intercept=True,output_file='table_linreg.csv'):
    
    """
    This function calculates linear regression parameters using Ordinary Least Squares,
    the Theil-Sen method, and Kendall Tau test for months and year

    Parameters
    ----------
    df: pd.DataFrame with index of datetime
    var: string
        Variable name
    stat: string
        Can be 'min', 'p5', 'P90', '50%', 'mean', 'max'. Default is 'mean'
    method: list of string
        By default 'Least-Squares', 'Theil-Sen', 'Kendall-tau'
    confidence interval: float
        To be specified for the Theil-Sen method, between 0.5 and 1. Default is 0.95
    intercept: Boolean
        True to get intercepts of the regression lines (for Least-Squares and Theil-Sen). Default is True
    output_file: string
        Name of the output file

    Returns
    -------
    3 pd.DataFrame

    DataFrame 1 is the linear regression table
        For Least-Squares: slope, intercept, and coefficient of determination R^2
        For Theil-Sen: median slope, intercept, smallest, and highest slope
        For Kendall: gives the tau and p-value
    DataFrame 2 is the statistic for all months over the whole period
    DataFrame 3 is the yearly statistic
    DataFrames 2 and 3 are the actual data from which the regressions are calculated and can be used for plotting purposes. 

    If interested only in DataFrame 1, the function's usage is df1,_,_ = table_linear_regression(...)

    Authors
    -------
    Written by Dung M. Nguyen and clio-met
    """ 
    months_names = calendar.month_abbr[1:]
    months_names=months_names+['Year']
    
    df2=sg.stats_monthly_every_year(df,var=var,method=[stat])
    df3=table_yearly_stats(df,var=var,percentiles=[stat],output_file="")

    df_out=pd.DataFrame()

    if 'Least-Squares' in method:
        slop_ls=[]
        inter_ls=[]
        r2val_ls=[]
        # Loop over the months
        for mon in range(1,13):
            y=df2[df2['month']==mon].iloc[:,0].values
            x=df2[df2['month']==mon]['year'].values
            slope, interc, r_value, p_value, std_err = linregress(x.astype(float), y)
            slop_ls.append(slope)
            inter_ls.append(interc)
            r2val_ls.append(r_value**2)
            del slope,interc,r_value,p_value,std_err,x,y
        # For the yearly means
        y=df3.iloc[:-1,0].values
        x=df3.index[:-1].values
        slope, interc, r_value, p_value, std_err = linregress(x.astype(float), y)
        del x,y
        slop_ls.append(slope)
        inter_ls.append(interc)
        r2val_ls.append(r_value**2)
        del slope,interc,r_value,p_value,std_err
        df_out['LS_slope']=slop_ls
        if intercept:
            df_out['LS_intercept']=inter_ls
        df_out['LS_r_square']=r2val_ls
        del slop_ls,inter_ls,r2val_ls

    if 'Theil-Sen' in method:
        slop_ts=[]
        inter_ts=[]
        slopl_ts=[]
        slopu_ts=[]
        # Loop over the months
        for mon in range(1,13):
            y=df2[df2['month']==mon].iloc[:,0].values
            x=df2[df2['month']==mon]['year'].values
            slope_theil_sen, interc, lower, upper = theilslopes(y, x.astype(float), confidence_interval)
            slop_ts.append(slope_theil_sen)
            inter_ts.append(interc)
            slopl_ts.append(lower)
            slopu_ts.append(upper)
            del slope_theil_sen,interc,lower,upper
        # For the yearly means
        y=df3.iloc[:-1,0].values
        x=df3.index[:-1].values
        slope_theil_sen, interc, lower, upper = theilslopes(y, x.astype(float), confidence_interval)
        slop_ts.append(slope_theil_sen)
        inter_ts.append(interc)
        slopl_ts.append(lower)
        slopu_ts.append(upper)
        del slope_theil_sen,interc,lower,upper
        df_out['TS_slope']=slop_ts
        if intercept:
            df_out['TS_intercept']=inter_ts
        df_out['TS_slope_lower']=slopl_ts
        df_out['TS_slope_upper']=slopu_ts
        del slop_ts,inter_ts,slopl_ts,slopu_ts

    if 'Kendall-tau' in method:
        tau_test=[]
        tau_p_value=[]
        # Loop over the months
        for mon in range(1,13):
            y=df2[df2['month']==mon].iloc[:,0].values
            x=df2[df2['month']==mon]['year'].values
            tau, p_value = kendalltau(x.astype(float), y)
            tau_test.append(tau)
            tau_p_value.append(p_value)
            del tau,p_value
        # For the yearly means
        y=df3.iloc[:-1,0].values
        x=df3.index[:-1].values
        tau, p_value = kendalltau(x.astype(float), y)
        tau_test.append(tau)
        tau_p_value.append(p_value)
        df_out['Kendall_tau']=tau_test
        df_out['Kendall_p_value']=tau_p_value

    df_out['index']=months_names
    df_out=df_out.set_index('index')

    if output_file != '':
        df_out1=df_out
        df_out1=df_out1.round(3)
        df_out1.to_csv(output_file, index=True)

    return df_out,df2,df3.iloc[:-1,:]