import os
import pandas as pd
import numpy as np

from metocean_stats.tables.verification import *
from metocean_stats.plots.verification import *
from metocean_stats.stats.aux_funcs import readNora10File

# Define TimeSeries-object for NORA3
ds = readNora10File('tests/data/NORA_test.txt')
ds_ocean = pd.read_csv('tests/data/NorkystDA_test.csv',comment='#',index_col=0, parse_dates=True)
depth = ['0m', '1m', '2.5m', '5m', '10m', '15m', '20m', '25m', '30m', '40m', '50m', '75m', '100m', '150m', '200m']


def test_error_stats(ds=ds):
    df = error_stats(ds,var_ref='HS',var_comp='HS.1',error_metric=['bias','mae','rmse','scatter_index','corr'])
    if df.shape == (1, 5):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_table_error_metric(ds=ds):
    output_file = 'test_error_metric.csv'
    df = table_error_metric(ds,var_ref='HS',var_comp='HS.1',error_metric=['bias','mae','rmse','scatter_index','corr'],output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (1, 5):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_table_binned_error_metric(ds=ds):
    output_file = 'test_binned_error_metric.csv'
    df = table_binned_error_metric(ds,var_bin='TP',var_bin_size=0.5,var_ref='HS',var_comp='HS.1',threshold_min=0,error_metric=['bias'],output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (44, 2):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_table_error_metric_multiple(ds=ds):
    output_file = 'test_error_metric_multiple.csv'
    df = table_error_metric_multiple(ds,var_ref='TP',var_comp=['HS.1','HS.2'],error_metric=['scatter_index','rmse','bias','mae','corr'],output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (2, 5):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_plot_binned_error_metric(ds=ds):
    # Test the `month_xticks` option to ensure month labels are shown correctly
    fig = plot_binned_error_metric(ds,var_bin='W10',var_bin_size=0.5,var_bin_unit='m/s',var_ref='HS',var_comp=['HS.1'],var_comp_unit='m',threshold_min=100,error_metric='bias',output_file='')
    # Check that the output is a Matplotlib Figure
    assert isinstance(fig, plt.Figure), "The output is not a Matplotlib Figure."
