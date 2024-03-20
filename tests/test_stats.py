from metocean_api import ts
from metocean_stats.stats import general_stats, dir_stats, extreme_stats, profile_stats, ml_stats
import os

# Define TimeSeries-object for NORA3
ds = ts.TimeSeries(lon=1.32, lat=53.324,start_time='2000-01-01', end_time='2010-12-31' , product='NORA3_wind_wave')
# Import data from thredds.met.no and save it as csv
#ds.load_data('tests/data/'+ds.datafile)
ds.load_data('tests/data/'+ds.datafile)

def test_scatter_diagram(ds=ds):
    general_stats.scatter_diagram(data=ds.data, var1='hs', step_var1=1, var2='tp', step_var2=1, output_file='test.png')
    os.remove('test.png')

def test_table_var_sorted_by_hs(ds=ds):
    general_stats.table_var_sorted_by_hs(data=ds.data, var='tp',var_hs='hs', output_file='test.csv')
    os.remove('test.csv')

def test_table_monthly_percentile(ds=ds):
    general_stats.table_monthly_percentile(data=ds.data, var='hs', output_file='test.csv')
    os.remove('test.csv')

def test_table_monthly_min_mean_max(ds=ds):
    general_stats.table_monthly_min_mean_max(data=ds.data, var='hs', output_file='test.csv')
    os.remove('test.csv')

def test_rose(ds=ds):
    dir_stats.var_rose(ds.data, 'thq','hs','test.png',method='overall')
    os.remove('test.png')

def test_directional_min_mean_max(ds=ds):
    dir_stats.directional_min_mean_max(ds.data,'thq','hs','test.csv')
    os.remove('test.csv')

def test_return_levels_pot(ds=ds):
    extreme_stats.return_levels_pot(data=ds.data, var='hs', periods=[20,100], output_file=False)

def test_return_levels_annual_max(ds=ds):
    extreme_stats.return_levels_annual_max(data=ds.data, var='hs', periods=[20,100], output_file=False) 

def test_return_levels_idm(ds=ds):
    extreme_stats.return_levels_idm(data=ds.data, var='hs', periods=[20,100], output_file=False) 

def test_mean_profile(ds=ds):
    profile_stats.mean_profile(data = ds.data, vars = ['wind_speed_10m','wind_speed_20m','wind_speed_50m','wind_speed_100m','wind_speed_250m','wind_speed_500m','wind_speed_750m'],height_levels=[10,20,50,100,250,500,750], perc = [25,75], output_file=False)
    
def test_profile_shear(ds=ds):
    profile_stats.profile_shear(data = ds.data, vars = ['wind_speed_10m','wind_speed_20m','wind_speed_50m','wind_speed_100m','wind_speed_250m','wind_speed_500m','wind_speed_750m'],height_levels=[10,20,50,100,250,500,750], z=[20,250], perc = [25,75], output_file=False)

def test_diagnostic_return_level_plot_multi(ds=ds):
    extreme_stats.diagnostic_return_level_plot_multi(data=ds.data, 
                                                     var='hs',
                                                     dist_list=['GP', 
                                                                'EXP'],
                                                     yaxis='prob',
                                                     output_file=False)

def test_predict_ts_GBR(ds=ds):
    ml_stats.predict_ts(ts_origin=ds.data,var_origin=['hs','tp','Pdir'],ts_train=ds.data.loc['2000-01-01':'2000-01-10'],var_train=['hs'], model='GBR')

def test_predict_ts_SVR(ds=ds):
    ml_stats.predict_ts(ts_origin=ds.data,var_origin=['hs','tp','Pdir'],ts_train=ds.data.loc['2000-01-01':'2000-01-10'],var_train=['hs'], model='SVR_RBF')

def test_predict_ts_LSTM(ds=ds):
    ml_stats.predict_ts(ts_origin=ds.data,var_origin=['hs','tp','Pdir'],ts_train=ds.data.loc['2000-01-01':'2000-01-10'],var_train=['hs'], model='LSTM')

def test_return_level_threshold(ds=ds):
    extreme_stats.return_level_threshold(data=ds.data, var='hs', 
                                              thresholds=[1,1.5])
    
def test_joint_distribution_Hs_Tp(ds=ds):
    extreme_stats.joint_distribution_Hs_Tp(df=ds.data, file_out='test.png')
    os.remove('test.png')
    
  