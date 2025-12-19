import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from metocean_stats import plots
from metocean_stats.plots.climate import *
from metocean_stats.stats.aux_funcs import readNora10File

# Define TimeSeries-object for NORA3
dirname = os.path.dirname(__file__)
ds =    readNora10File(os.path.join(dirname, 'data/NORA_test.txt'))
ds_ocean = pd.read_csv(os.path.join(dirname, 'data/NorkystDA_test.csv'),comment='#',index_col=0, parse_dates=True)
depth = ['0m', '1m', '2.5m', '5m', '10m', '15m', '20m', '25m', '30m', '40m', '50m', '75m', '100m', '150m', '200m']



def test_plot_stripes():
    fig = plots.plot_yearly_stripes(ds,var_name='HS',method='mean',ylabel='Hs [m]',output_file='')
    # Check that the output is a Matplotlib Figure
    assert isinstance(fig, plt.Figure), "The output is not a Matplotlib Figure."


def test_plot_heatmap_monthly_yearly():
    fig = plots.plot_heatmap_monthly_yearly(ds,var='T2m',method='mean',cb_label='2-m temperature [°C]',output_file='')
    # Check that the output is a Matplotlib Figure
    assert isinstance(fig, plt.Figure), "The output is not a Matplotlib Figure."


def test_plot_yearly_depth_profiles():
    fig = plots.plot_yearly_vertical_profiles(ds_ocean, rad_colname='current_speed_', method='mean', yaxis_direction='down', xlabel='Current speed [m/s]', output_file='')
    # Check that the output is a Matplotlib Figure
    assert isinstance(fig, plt.Figure), "The output is not a Matplotlib Figure."


def test_plot_yearly_depth_profiles_wind():
    fig = plots.plot_yearly_vertical_profiles(ds, rad_colname='W', method='mean', yaxis_direction='up', xlabel='Wind speed [m/s]', output_file='')
    # Check that the output is a Matplotlib Figure
    assert isinstance(fig, plt.Figure), "The output is not a Matplotlib Figure."


def test_plot_linear_regression():
    fig = plots.plot_linear_regression(ds,var='W10',time='Year',stat='P90',method=['Least-Squares','Theil-Sen'],confidence_interval=0.95,ylabel='Wind speed [m/s]',output_figure='')
    # Check that the output is a Matplotlib Figure
    assert isinstance(fig, plt.Figure), "The output is not a Matplotlib Figure."


def test_plot_heatmap_profiles_yearly():
    fig = plots.plot_heatmap_profiles_yearly(ds,rad_colname='W',cb_label='Wind speed [m/s]',yaxis_direction='up',method='P80',output_file='')
    # Check that the output is a Matplotlib Figure
    assert isinstance(fig, plt.Figure), "The output is not a Matplotlib Figure."