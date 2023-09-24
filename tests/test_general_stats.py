from metocean_api import ts
from metocean_stats.stats import general_stats
import os

# Define TimeSeries-object
ds = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-12-31' , product='NORA3_wind_wave')
# Import data from thredds.met.no and save it as csv
ds.load_data('tests/data/'+ds.datafile)

def test_scatter_diagram(ds=ds):
    general_stats.scatter_diagram(data=ds.data, var1='hs', step_var1=1, var2='tp', step_var2=1, output_file='test.png')
    os.remove('test.png')

def test_table_var_sorted_by_hs(ds=ds):
    general_stats.table_var_sorted_by_hs(data=ds.data, var='tp', output_file='test.csv')
    os.remove('test.csv')

def test_table_monthly_percentile(ds=ds):
    general_stats.table_monthly_percentile(data=ds.data, var='hs', output_file='test.csv')
    os.remove('test.csv')

def test_table_monthly_min_mean_max(ds=ds):
    general_stats.table_monthly_min_mean_max(data=ds.data, var='hs', output_file='test.csv')
    os.remove('test.csv')



