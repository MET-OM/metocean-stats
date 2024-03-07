from metocean_api import ts
from metocean_stats.stats import general_stats, dir_stats, extreme_stats, profile_stats

# Define TimeSeries-object
ds = ts.TimeSeries(lon=1.32, lat=53.324,start_time='2000-01-01', end_time='2010-12-31' , product='NORA3_wind_wave')

# Import data from thredds.met.no and save it as csv
#ds.import_data(save_csv=True)
# Load data from local file
ds.load_data('/home/birgitterf/dev/github/metocean-stats/tests/data/'+ds.datafile)

# Generate Statistics
general_stats.scatter_diagram(data=ds.data, var1='hs', step_var1=1, var2='tp', step_var2=1, output_file='scatter_hs_tp_diagram.png')
general_stats.table_var_sorted_by_hs(data=ds.data, var='tp',var_hs='hs', output_file='tp_sorted_by_hs.csv')
general_stats.table_monthly_percentile(data=ds.data, var='hs', output_file='hs_monthly_perc.csv')
general_stats.table_monthly_min_mean_max(data=ds.data, var='hs',output_file='hs_montly_min_mean_max.csv')#

# Directional Statistics
dir_stats.var_rose(ds.data, 'thq','hs','windrose.png',method='overall')
dir_stats.directional_min_mean_max(ds.data,'thq','hs','hs_dir_min_mean_max.csv')

# Extreme Statistics
rl_pot = extreme_stats.return_levels_pot(data=ds.data, var='hs', periods=[20,50,100,1000], output_file='return_levels_POT.png')
rl_am = extreme_stats.return_levels_annual_max(data=ds.data, var='hs', periods=[20,50,100,1000],method='GEV',output_file='return_levels_GEV.png')

# Profile Statistics
mean_prof = profile_stats.mean_profile(data = ds.data, vars = ['wind_speed_10m','wind_speed_20m','wind_speed_50m','wind_speed_100m','wind_speed_250m','wind_speed_500m','wind_speed_750m'],height_levels=[10,20,50,100,250,500,750],perc=[5,95], output_file='wind_profile.png')
alfa = profile_stats.profile_shear(data = ds.data, vars = ['wind_speed_10m','wind_speed_20m','wind_speed_50m','wind_speed_100m','wind_speed_250m','wind_speed_500m','wind_speed_750m'],height_levels=[10,20,50,100,250,500,750],z=[20,50], perc=[5,95], output_file='wind_profile_shear.png')
