from ..stats.verification import *
import pandas as pd
import numpy as np


def table_error_metric(data,var_ref,var_comp,error_metric=['bias','mae','rmse','scatter_index','corr'],output_file='error_metric.csv'):
    """
    Calculates error metrics between two datasets and outputs the result as a table

    Parameters
    ----------
    df: pd.DataFrame
        Contains at least the two variables to compare to each other
    var_ref: string
        Name of the variable to be used as reference
    var_comp: string
        Name of the variable to compare to the reference
    error_metric: list of strings
        List of the metrics to calculate. Default is ['bias','mae','rmse','corr','scatter_index']

    Returns
    -------
    df_out: pd.DataFrame
        Contains the error metrics

    Authors
    -------
    clio-met
    """
    df_out = error_stats(data=data,var_ref=var_ref,var_comp=var_comp,error_metric=error_metric)
    if output_file!='':
        df_out1=df_out
        df_out1=df_out1.round(3)
        df_out1.to_csv(output_file, index=False)
    return df_out


def table_binned_error_metric(data,var_bin,var_bin_size,var_ref,var_comp,threshold_min=0,error_metric=['scatter_index','rmse','bias','mae','corr'],output_file='binned_error_metric.csv'):
    """
    Calculates different error/statistics between two datasets.

    Parameters
    ----------
    data: pd.DataFrame
        Contains the total dataset. Cannot include variables with same name
    var_bin: string
        The name of the variable the data should be binned into
    var_bin_size: scalar
        The bin size of the var_bin
    var_ref: string
        Name of the variable to be used as reference
    var_comp: strins
        Name of the variable to compare to the reference
    threshold_min: scalar
        Minimum number of values within a bin for the statistic to be significant
        Default is 0. If no data, NaN
    error_metric: list of strings
        List of the different error/statistic to be calculated. Default is ['scatter_index','rmse','bias','mae','corr']

    Returns
    -------
    output_file: string
        Name of the output table. Default is 'binned_error_metric.csv'

    Authors
    -------
    theaje, modified by clio-met
    """
    data = data.dropna() # Remove NaNs if any
    max_var_bin = data[var_bin].max()
    bins = np.arange(0, max_var_bin+var_bin_size, var_bin_size)
    del max_var_bin
    nb_val_in_bin=[] # number of values in each bin
    df_out=pd.DataFrame()
    for j in range(len(bins)-1):
        df_tmp =data[(data[var_bin] > bins[j]) & (data[var_bin] <= bins[j+1])]
        nb_val_in_bin.append(len(df_tmp))
        if len(df_tmp>threshold_min):
            df_out1=error_stats(data=df_tmp,var_ref=var_ref,var_comp=var_comp,error_metric=error_metric)
        else:
            df_out1=pd.DataFrame([[np.nan]*len(error_metric)],columns=error_metric)
        if j==0:
            df_out=df_out1
        else:
            df_out=pd.concat([df_out,df_out1], ignore_index=True)
        del df_out1,df_tmp

    # Add column with number of values in each bin
    df_out['nb_val']=nb_val_in_bin
    del nb_val_in_bin
    # Add column with the center of each bin and make it as the index
    bins_centre=bins[0:-1]+(var_bin_size/2)
    df_out[var_bin+'_bin']=bins_centre
    del bins_centre
    df_out.set_index(var_bin+'_bin', inplace=True)
    if output_file!='':
        df_out1=df_out
        df_out1=df_out1.round(3)
        df_out1.to_csv(output_file)

    return df_out


def table_error_metric_multiple(data, var_ref, var_comp, error_metric=['scatter_index','rmse','bias','mae','corr'],output_file='error_metric_multiple.csv'):
    """
    Calculates different error/statistics between a reference and one or more datasets.

    Parameters
    ----------
    data: pd.DataFrame
        Contains the total dataset. Cannot include variables with same name
    var_bin: string
        The variable name the data should be binned into
    var_bin_size: scalar
        The bin size of the var_bin
    var_ref: string
        The reference variable
    var_comp: list of strings
        The variables to compare to the reference, e.g. var_comp=['TP_1','TP_2']
    threshold_min: scalar
        Minimum number of values within a bin for the statistic to be significant
        If no data, NaN
    error_metric: list of strings
        List of the different error/statistic to be calculated. Default is ['scatter_index','rmse','bias','mae','corr']

    Returns
    -------
    output_file: string
        Name of the output table. Default is 'error_metric_multiple.csv'

    Authors
    -------
    theaje, modified by clio-met
    """
    df_out=pd.DataFrame()
    for var in var_comp:
        df_tmp = error_stats(data,var_ref=var_ref,var_comp=var,error_metric=error_metric)
        df_out = pd.concat([df_out, df_tmp], ignore_index=True)
        del df_tmp
    
    df_out[var_ref]=var_comp
    df_out=df_out.set_index(var_ref)
    if output_file!='':
        df_out1=df_out
        df_out1=df_out1.round(3)
        df_out1.to_csv(output_file)

    return df_out

