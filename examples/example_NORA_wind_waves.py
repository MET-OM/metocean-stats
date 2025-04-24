from metocean_stats import plots, tables, maps
from metocean_stats.stats.aux_funcs import *
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

##############PROVIDE INFO BY THE USER#################
# Impot data (e.g., use metocean-api to download metocean data)
## For NORA10 data use:
df = readNora10File('../tests/data/NORA_test.txt') 
## For NORA3 data use:
#import pandas as pd
#df = pd.read_csv('../path/to/NORA3.csv', comment="#", index_col=0, parse_dates=True)

# Define names for each variable in the dataframe (df):
var_wind_dir = 'D10' # for wind direction
var_wind = 'W10' # for wind speed
var_hs = 'HS' # for significant wave height
var_wave_dir= 'DIRM' # Mean wave direction
var_tp = 'TP'  # Peak Wave Period
output_folder = 'output_wind_waves' # folder where output figures and tables will be saved 
######################################################


# Check if the output directory exists, if not, create it
folder = Path(__file__).parent / output_folder 
if not folder.exists():
    folder.mkdir(parents=True)

# The following code is used to generate various plots and tables for wind and wave data analysis.

# Wind:
plots.var_rose(df,var_dir=var_wind_dir,var=var_wind,method='overall',max_perc=40,decimal_places=1, units='m/s',output_file=folder /  'wind_omni.png')
plots.var_rose(df,var_dir=var_wind_dir,var=var_wind,method='monthly',max_perc=40,decimal_places=1, units='m/s',output_file=folder /  'wind_monthly.png')
plots.plot_directional_stats(df,var=var_wind,step_var=0.1,var_dir=var_wind_dir,title = 'W10[m/s]', output_file=folder /  'directional_wind_stats.png')
tables.table_directional_non_exceedance(df,var=var_wind,step_var=2,var_dir=var_wind_dir,output_file=folder /  'table_wind_directional_non_exceedance.csv')
plots.plot_monthly_stats(df,var=var_wind,title = 'Wind Speed at 10 m [m/s]', output_file=folder /  'monthly_wind_stats.png')
tables.table_monthly_non_exceedance(df,var=var_wind,step_var=2,output_file=folder /  'table_monthly_non_exceedance.csv')
plots.plot_prob_non_exceedance_fitted_3p_weibull(df,var=var_wind,output_file=folder /  'prob_non_exceedance_fitted_3p_weibull_wind.png')
plots.plot_monthly_return_periods(df,var=var_wind,periods=[1, 10, 100],distribution='Weibull3P_MOM', units='m/s',output_file=folder /  'W10_monthly_extremes.png')
plots.plot_directional_return_periods(df,var=var_wind,var_dir=var_wind_dir,periods=[1, 10, 100],distribution='Weibull3P_MOM', units='m/s',adjustment='NORSOK',output_file=folder /  'W10_dir_extremes_Weibull_norsok.png')
plots.plot_monthly_weather_window(df,var=var_wind,threshold=10, window_size=12,output_file=folder /  'NORA10_monthly_weather_window_wind_10_12_plot.png')
plots.plot_multi_diagnostic_return_levels(df, var=var_wind, dist_list = ['GP', 'Weibull_2P', 'EXP'], periods=np.arange(0.1, 1000, 0.1),threshold=None,output_file=folder /  'plot_wind_diagnostic_return_levels.png')


# Waves:
plots.plot_multi_diagnostic_return_levels(df, var=var_hs, dist_list = ['GP', 'Weibull_2P', 'EXP'], periods=np.arange(0.1, 1000, 0.1),threshold=None,output_file=folder /  'plot_waves_diagnostic_return_levels.png')
plots.plot_prob_non_exceedance_fitted_3p_weibull(df,var=var_hs,output_file=folder /  'prob_non_exceedance_fitted_3p_weibull_hs.png')
tables.scatter_diagram(df, var1=var_hs, step_var1=1, var2=var_tp, step_var2=1, output_file= folder /  'Hs_Tp_scatter.csv')
tables.table_var_sorted_by_hs(df, var=var_tp, var_hs=var_hs, output_file=folder /  'Tp_sorted_by_Hs.csv')
tables.table_monthly_non_exceedance(df,var=var_hs,step_var=0.5,output_file=folder /  'Hs_table_monthly_non_exceedance.csv')
plots.plot_monthly_stats(df,var=var_hs,show=['Maximum','P99','Mean'], title = 'Hs[m]', output_file=folder /  'Hs_monthly_stats.png')
tables.table_directional_non_exceedance(df,var=var_hs,step_var=0.5,var_dir=var_wave_dir,output_file=folder /  'Hs_table_directional_non_exceedance.csv')
plots.plot_directional_stats(df,var=var_hs,step_var=0.5, var_dir=var_wave_dir, title = '$H_s$[m]', output_file=folder /  'directional_waves_stats.png')
plots.plot_joint_distribution_Hs_Tp(df,var_hs=var_hs,var_tp=var_tp,periods=[1,10,100,1000], title='Hs-Tp joint distribution',output_file=folder /  'Hs.Tp.joint.distribution.png',density_plot=True)
tables.table_monthly_joint_distribution_Hs_Tp_param(df,var_hs=var_hs,var_tp=var_tp,periods=[1,10,100,10000],output_file=folder /  'monthly_Hs_Tp_joint_param.csv')
tables.table_directional_joint_distribution_Hs_Tp_param(df,var_hs=var_hs,var_tp=var_tp,var_dir=var_wave_dir,periods=[1,10,100],output_file=folder /  'dir_Hs_Tp_joint_param.csv')
plots.plot_monthly_weather_window(df,var=var_hs,threshold=4, window_size=12,output_file=folder /  '_monthly_weather_window_Hs_4_12_plot.png')
plots.plot_nb_hours_below_threshold(df,var=var_hs,thr_arr=(np.arange(0.05,20.05,0.05)).tolist(),output_file=folder /  'Hs_number_hours_per_year.png')
tables.table_monthly_return_periods(df,var=var_hs,periods=[1, 10, 100, 10000],distribution='Weibull3P_MOM', units='m',output_file=folder /  'HS_monthly_extremes_Weibull.csv')
tables.table_directional_return_periods(df,var=var_hs,periods=[1, 10, 100, 10000], units='m',var_dir = var_wave_dir,distribution='Weibull3P_MOM', adjustment='NORSOK' ,output_file=folder /  'directional_wave_extremes_weibull.csv')
plots.plot_monthly_return_periods(df,var=var_hs,periods=[1, 10, 100],distribution='Weibull3P_MOM', units='m',output_file=folder /  'HS_monthly_extremes.png')
plots.plot_directional_return_periods(df,var=var_hs,var_dir=var_wave_dir,periods=[1, 10, 100, 10000 ],distribution='GUM', units='m',output_file=folder /  'dir_extremes_GUM.png')
plots.plot_directional_return_periods(df,var=var_hs,var_dir=var_wave_dir,periods=[1, 10, 100, 10000],distribution='Weibull3P_MOM', units='m',adjustment='NORSOK',output_file=folder /  'dir_extremes_wave_Weibull_norsok.png')
tables.table_monthly_joint_distribution_Hs_Tp_return_values(df,var_hs=var_hs,var_tp=var_tp,periods=[1,10,100,10000],output_file=folder /  'monthly_Hs_Tp_joint_return_values.csv')
tables.table_directional_joint_distribution_Hs_Tp_return_values(df,var_hs=var_hs,var_tp=var_tp,var_dir=var_wave_dir,periods=[1,10,100,1000],adjustment='NORSOK',output_file=folder /  'directional_Hs_Tp_joint_return_values.csv')
tables.table_Hs_Tpl_Tph_return_values(df,var_hs=var_hs,var_tp=var_tp,periods=[1,10,100,10000],output_file=folder /  'hs_tpl_tph_return_values.csv')
plots.plot_tp_for_given_hs(df, var_hs, var_tp,output_file=folder /  'tp_for_given_hs.png')
tables.table_tp_for_given_hs(df, var_hs, var_tp,max_hs=20,output_file=folder /  'tp_for_given_hs.csv')
tables.table_tp_for_rv_hs(df, var_hs=var_hs, var_tp=var_tp,periods=[1,10,100,10000],output_file=folder /  'tp_for_rv_hs.csv')
tables.table_wave_induced_current(df, var_hs=var_hs,var_tp=var_tp,depth=200,ref_depth=200, spectrum = 'JONSWAP',output_file=folder /  'JONSWAP_wave_induced_current_depth200.csv')
tables.table_wave_induced_current(df, var_hs=var_hs,var_tp=var_tp,depth=200,ref_depth=200, spectrum = 'TORSEHAUGEN',output_file=folder /  'TORSEHAUGEN_wave_induced_current_depth200.csv')
tables.table_hs_for_given_wind(df, var_hs,var_wind, bin_width=2, max_wind=42, output_file= folder /  'table_perc_hs_for_wind.csv')
plots.plot_hs_for_given_wind(df, var_hs, var_wind,output_file=folder /  'hs_for_given_wind.png')
tables.table_hs_for_rv_wind(df, var_wind=var_wind, var_hs=var_hs,periods=[1,10,100,10000],output_file=folder /  'hs_for_rv_wind.csv')
tables.table_Hmax_crest_return_periods(df,var_hs=var_hs, var_tp=var_tp, depth=200, periods=[1, 10, 100,10000],sea_state='long-crested',output_file=folder /  'table_Hmax_crest_rp.csv')
tables.table_directional_Hmax_return_periods(df,var_hs=var_hs, var_tp = var_tp,var_dir=var_wave_dir, periods=[10, 100,10000],adjustment='NORSOK', output_file=folder /  'table_dir_Hmax_return_values.csv')
plots.plot_multi_joint_distribution_Hs_Tp_var3(df,var_hs=var_hs,var_tp=var_tp,var3=var_wind,var3_units='m/s',periods=[100],var3_bin=5,threshold_min=100,output_file=folder /  'Hs.Tp.joint.distribution.multi.binned.var3.png')



