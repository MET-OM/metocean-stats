import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import scipy.stats as st
import pandas as pd


def table_binned_error_metric(data, var_bin, var_ref, var_comp, var_bin_size, threshold_min, error_metric=['scatter_index','rmse','bias','mae','corr','si'],output_file='binned_error_metric.csv'):
    """
    Calculates different error/statistics between two datasets.
    data: dataframe containing the total dataset. Cannot include variables with same name. 
    var_bin: the variable the data should be binned into.
    var_ref: the ref-variable
    var_comp: the comp-variable. Since the data set can contain multiple dataset that the user want to compare to the ref, multiple var_comp can be sent in,e.g. var_comp=['TP_1','TP_2'] 
    var_bin_size: the bin size of the var_bin
    threshold_min: if the length of the dataset found within the given bin_size is less than the threshold, this data is not considered in the calculations.
    error_metric: the different error/statistic that can be calculated. The user decides which of ['rmse','bias','mae','corr','si'] should be calculated.
    output_file: name of the output table, e.g. 'binned_error_metric.csv'
    """
    def error_stats(dss_temp,x,y,error_metric): #if the name is found in error_metric, calculate the error and put it into the ds
        
        if 'rmse' in error_metric:
            dss_temp['rmse'] = (np.sqrt(((y - x) ** 2).mean())).mean()
        if 'bias' in error_metric:
            dss_temp['bias'] = (np.mean(y-x)).mean()
        if 'mae' in error_metric:
            dss_temp['mae'] = (np.mean(np.abs(y-x))).mean()
        if 'corr' in error_metric:
            dss_temp['corr'] = (np.corrcoef(y,x)[0][1]).mean()
        if 'si' in error_metric:
            dss_temp['si'] = (np.std(x-y)/np.mean(x)).mean()
        return dss_temp
    
    ds = pd.DataFrame()
    all_dss = []

    for i in range(len(var_comp)):
        df_data = data.dropna() # dff[0].dropna()
        noise = np.random.uniform(0, 1, size=len(df_data[var_comp[i]]))

        max_var3 = max(df_data[var_bin]) 
        window = np.arange(0, max_var3 + var_bin_size, var_bin_size) 
        var_xx = []
        var_yy1 = []
        var_comp1 = var_comp[i]
        dss = pd.DataFrame() #temp dataframe

        for j in range(len(window)-1):
            var_3 = df_data[var_bin].where((df_data[var_bin] > window[j]) & (df_data[var_bin] <= window[j+1])).dropna() #windo = (windo1,windo2]
            if len(var_3)<threshold_min:
                continue
            var_xx = df_data[var_ref].where(var_3.notnull()).dropna() 
            var_yy1 = ((df_data[var_comp1]).where(var_3.notnull())+noise).dropna() 
            #This to be used when noise is not added:
            #var_yy1 = ((df_comp[var_comp1]).where(var_3.notnull())).dropna() #send in Tp where HS is located
            var_bin_mean = (window[j]+window[j+1])/2 
            var_yy_mean= var_yy1.mean()
            dss_temp = pd.DataFrame({var_comp[i]+'vs'+var_ref: ['-'] , var_bin+'_bin':[var_bin_mean], var_comp1:[var_yy_mean]}) #temp dataframe
            dss_temp = error_stats(dss_temp,var_xx,var_yy1,error_metric)
            dss = pd.concat([dss,dss_temp]).round(2)
        all_dss.append(dss)

    ds = pd.concat(all_dss, axis=1)
    print(ds)
    if output_file:
        ds.to_csv(output_file, index=False)

    return ds
