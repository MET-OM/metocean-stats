from metocean_api import ts
from metocean_stats import plots, tables, stats
#from metocean_stats.stats.aux_funcs import readNora10File

# Define TimeSeries-object
ds = ts.TimeSeries(lon=5, lat=57.7,start_time='1980-01-01', end_time='2019-12-31' , product='NORA3_wind_wave')

# Import data from thredds.met.no and save it as csv
#ds.import_data(save_csv=True)
# Load data from local file
ds.load_data('/home/konstantinosc/Downloads/'+ds.datafile)
#ds = readNora10File('NORA10_6036N_0336E.1958-01-01.2022-12-31.txt')


# New Functions
plots.plot_prob_non_exceedance_fitted_3p_weibull(ds,var='HS',output_file='prob_non_exceedance_fitted_3p_weibull.png')

tables.table_monthly_non_exceedance(ds,var1='HS',step_var1=0.5,output_file='table_monthly_non_exceedance.csv')
plots.plot_monthly_stats(ds,var1='HS',step_var1=0.5, title = '$H_s$[m] - Havstj', output_file='monthly_stats.png')
tables.table_directional_non_exceedance(ds,var1='HS',step_var1=0.5,var_dir='DIRM',output_file='table_directional_non_exceedance.csv')
plots.plot_directional_stats(ds,var1='HS',step_var1=0.5, var_dir='DIRM', title = '$H_s$[m] - Havstj', output_file='directional_stats.png')
plots.plot_joint_distribution_Hs_Tp(ds,var1='HS',var2='TP',periods=[1,10,100,10000], title='Hs-Tp joint distribution',output_file='Hs.Tp.joint.distribution.png',density_plot=True)
tables.table_monthly_joint_distribution_Hs_Tp_param(ds,var1='HS',var2='TP',periods=[1,10,100,10000],output_file='monthly_Hs_Tp_joint_param.csv')
tables.table_directional_joint_distribution_Hs_Tp_param(ds,var1='HS',var2='TP',var_dir='DIRM',periods=[1,10,100,10000],output_file='dir_Hs_Tp_joint_param.csv')
#plots.plot_monthly_weather_window(ds,var='HS',threshold=4, window_size=48, title='$H_s$ < 4 m for 48 hours',output_file= 'NORA10_monthly_weather_window4_48_plot.png')

tables.table_monthly_weibull_return_periods(ds,var='HS',periods=[1, 10, 100, 10000], units='m',output_file='monthly_extremes_weibull.csv')
tables.table_directional_weibull_return_periods(ds,var='HS',periods=[1, 10, 100, 10000], units='m',var_dir = 'DIRM',output_file='directional_extremes_weibull.csv')

plots.plot_monthly_weibull_return_periods(ds,var='HS',periods=[1, 10, 100, 10000], units='m',output_file='monthly_extremes_weibull.png')
plots.plot_directional_weibull_return_periods(ds,var='HS',var_dir='DIRM',periods=[1, 10, 100, 10000], units='m',output_file='dir_extremes_weibull.png')


tables.table_monthly_joint_distribution_Hs_Tp_return_values(ds,var1='HS',var2='TP',periods=[1,10,100,10000],output_file='monthly_Hs_Tp_joint_return_values.csv')
tables.table_directional_joint_distribution_Hs_Tp_return_values(ds,var1='HS',var2='TP',var_dir='DIRM',periods=[1,10,100,10000],output_file='directional_Hs_Tp_joint_return_values.csv')