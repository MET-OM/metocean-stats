import pandas as pd
import numpy as np

def error_stats(data,var_ref,var_comp,error_metric=['bias','mae','rmse','scatter_index','corr']):
    """
    Calculates error metrics between two datasets

    Parameters
    ----------
    data: pd.DataFrame
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
        Contains the error metric(s)

    Authors
    -------
    theaje, modified by clio-met
    """
    data = data.dropna() # Remove NaNs if any
    df_out=pd.DataFrame()
    x=data[var_ref]
    y=data[var_comp]
    if 'bias' in error_metric:
        df_out['bias'] = [(y-x).mean()]
    if 'mae' in error_metric:
        df_out['mae'] = [((y-x).abs()).mean()]
    if 'rmse' in error_metric:
        df_out['rmse'] = [np.sqrt((((y-x)**2)/len(y)).sum())]
    if 'corr' in error_metric:
        df_out['corr'] = [y.corr(x)]
    if 'scatter_index' in error_metric:
        df_out['scatter_index'] = [np.sqrt((((y-x)**2)/len(y)).sum())/x.mean()]
    return df_out