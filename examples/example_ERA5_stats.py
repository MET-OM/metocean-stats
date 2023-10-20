from metocean_api import ts
from metocean_stats.stats import general_stats, dir_stats, extreme_stats

# Define TimeSeries-object
ds = ts.TimeSeries(lon=109.94, lat=15.51,start_time='2000-01-01', end_time='2010-12-31' , product='ERA5')


# Import data using cdsapi and save it as csv
#ds.import_data(save_csv=True)
# Load data from local file
ds.load_data('tests/data/'+ds.datafile)

# Generate Statistics
general_stats.scatter_diagram(data=ds.data, var1='swh', step_var1=1, var2='pp1d', step_var2=1, output_file='scatter_hs_tp_diagram.png')
general_stats.table_var_sorted_by_hs(data=ds.data, var='pp1d',var_hs='swh', output_file='tp_sorted_by_hs.csv')
general_stats.table_monthly_percentile(data=ds.data, var='swh', output_file='hs_monthly_perc.csv')
general_stats.table_monthly_min_mean_max(data=ds.data, var='swh',output_file='hs_montly_min_mean_max.csv')#

# Directional Statistics
dir_stats.var_rose(ds.data, 'mwd','swh','windrose.png',method='overall')
dir_stats.directional_min_mean_max(ds.data,'mwd','swh','hs_dir_min_mean_max.csv')

# Extreme Statistics
rl_pot = extreme_stats.return_levels_pot(data=ds.data, var='swh', periods=[20,50,100], output_file='return_levels_POT.png')
rl_am = extreme_stats.return_levels_annual_max(data=ds.data, var='swh', periods=[20,50,100],method='GEV',output_file='return_levels_GEV.png')


