from metocean_api import ts
from metocean_stats.stats import general_stats, dir_stats

# Define TimeSeries-object
ds = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-12-31' , product='NORA3_wind_wave')

# Import data from thredds.met.no and save it as csv
ds.load_data('tests/data/'+ds.datafile)

# Generate Statistics
general_stats.scatter_diagram(data=ds.data, var1='hs', step_var1=1, var2='tp', step_var2=1, output_file='scatter_hs_tp_diagram.png')
#general_stats.table_var_sorted_by_hs(data=ds.data, var='tp', output_file='tp_sorted_by_hs.csv')
#general_stats.table_monthly_percentile(data=ds.data, var='hs', output_file='hs_monthly_perc.csv')
#general_stats.table_monthly_min_mean_max(data=ds.data, var='hs',output_file='hs_montly_min_mean_max.csv')
#dir_stats.var_rose(ds.data, 'thq','hs','windrose.png',method='overall')
#dir_stats.directional_min_mean_max(ds.data,'thq','hs','hs_dir_min_mean_max.csv')



