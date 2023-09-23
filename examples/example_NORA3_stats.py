from metocean_api import ts
from metocean_stats.stats import general_stats
from metocean_stats.stats import dir_stats

# Define TimeSeries-object
ds = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2022-11-01', end_time='2022-12-31' , product='NORA3_wind_wave')


# Import data from thredds.met.no and save it as csv
ds.import_ts(save_csv=False)


general_stats.scatter_diagram(var1=ds.data['hs'], step_var1=1, var2=ds.data['tp'], step_var2=2, output_file='scatter_hs_tp_diagram.png')
#general_stats.table_var_sorted_by_Hs(data=ts.data, var='tp', output_file='tp_sorted_by_hs.csv')
#general_stats.table_monthly_percentile(data=ts.data, var='hs', output_file='hs_monthly_perc.csv')
#general_stats.table_monthly_min_mean_max(data=ts.data, var='hs',output_file='hs_montly_min_mean_max.csv')
#dir_stats.var_rose(ts.data, 'thq','hs','windrose.png',method='monthly')
#dir_stats.directional_min_mean_max(ts.data,'thq','hs','hs_dir_min_mean_max.csv')
#dir_stats.monthly_var_rose(ts.data,'thq','hs','windrose_month.png')



