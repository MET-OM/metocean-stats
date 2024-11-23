from metocean_stats import plots, tables, maps
from metocean_stats.stats.aux_funcs import *
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

## For NORA10 data use:
df = readNora10File('../tests/data/NORA_test.txt') 
## For NORA3 data use:
#import pandas as pd
#df = pd.read_csv('../tests/data/NORA.csv', comment="#", index_col=0, parse_dates=True)

folder = Path(__file__).parent / 'output_mini'
if not folder.exists():
    folder.mkdir(parents=True)

## Map:
#maps.plot_points_on_map(lon=[3.35,3.10], lat=[60.40,60.90],label=['NORA3','NORKYST800'],bathymetry='NORA3',output_file='map.png')
#maps.plot_extreme_wave_map(return_period=100, product='NORA3', title='100-yr return values Hs (NORA3)', set_extent = [0,30,52,73],output_file='wave_100yrs.png')
#maps.plot_extreme_wind_map(return_period=100, product='NORA3',z=10, title='100-yr return values Wind at 100 m (NORA3)', set_extent = [0,30,52,73], output_file='wind_100yrs.png')

# Wind:
plots.var_rose(df,var_dir='D10',var='W10',method='overall',max_perc=40,decimal_places=1, units='m/s',output_file='output_mini/wind_omni.png')
plots.var_rose(df,var_dir='D10',var='W10',method='monthly',max_perc=40,decimal_places=1, units='m/s',output_file='output_mini/wind_monthly.png')
plots.plot_directional_stats(df,var='W10',step_var=0.1,var_dir='D10',title = '', output_file='output_mini/directional_stats.png')
plots.table_directional_non_exceedance(df,var='W10',step_var=2,var_dir='D10',output_file='output_mini/table_directional_non_exceedance.csv')
plots.plot_monthly_stats(df,var='W10',title = 'Wind Speed at 10 m [m/s]', output_file='output_mini/monthly_stats.png')
tables.table_monthly_non_exceedance(df,var='W10',step_var=2,output_file='output_mini/table_monthly_non_exceedance.csv')
plots.plot_prob_non_exceedance_fitted_3p_weibull(df,var='W10',output_file='output_mini/prob_non_exceedance_fitted_3p_weibull.png')
plots.plot_monthly_return_periods(df,var='W10',periods=[1, 10, 100],distribution='Weibull3P_MOM', units='m/s',output_file='output_mini/W10_monthly_extremes.png')
plots.plot_directional_return_periods(df,var='W10',var_dir='D10',periods=[1, 10, 100],distribution='Weibull3P_MOM', units='m/s',adjustment='NORSOK',output_file='output_mini/W10_dir_extremes_Weibull_norsok.png')
plots.plot_monthly_weather_window(df,var='W10',threshold=10, window_size=12,output_file='output_mini/NORA10_monthly_weather_window4_12_plot.png')
plots.plot_scatter(df,var1='W10',var2='W100',var1_units='m/s',var2_units='m/s', title='Scatter',regression_line=True,qqplot=False,density=True,output_file='output_mini/scatter_plot.png')
plots.plot_multi_diagnostic_return_levels(df, var='HS', periods=[10, 100],threshold=None,output_file='output_mini/plot_diagnostic_return_levels.png')

# Waves:
plots.plot_prob_non_exceedance_fitted_3p_weibull(df,var='HS',output_file='output_mini/prob_non_exceedance_fitted_3p_weibull.png')
tables.scatter_diagram(df, var1='HS', step_var1=1, var2='TP', step_var2=1, output_file= 'output_mini/Hs_Tp_scatter.csv')
tables.table_var_sorted_by_hs(df, var='TP', var_hs='HS', output_file='output_mini/Tp_sorted_by_Hs.csv')
tables.table_monthly_non_exceedance(df,var='HS',step_var=0.5,output_file='output_mini/Hs_table_monthly_non_exceedance.csv')
plots.plot_monthly_stats(df,var='HS',show=['Maximum','P99','Mean'], title = 'Hs[m]', output_file='output_mini/Hs_monthly_stats.png')
tables.table_directional_non_exceedance(df,var='HS',step_var=0.5,var_dir='DIRM',output_file='output_mini/table_directional_non_exceedance.csv')
plots.plot_directional_stats(df,var='HS',step_var=0.5, var_dir='DIRM', title = '$H_s$[m]', output_file='output_mini/directional_stats.png')
plots.plot_joint_distribution_Hs_Tp(df,var_hs='HS',var_tp='TP',periods=[1,10,100,1000], title='Hs-Tp joint distribution',output_file='output_mini/Hs.Tp.joint.distribution.png',density_plot=True)
tables.table_monthly_joint_distribution_Hs_Tp_param(df,var_hs='HS',var_tp='TP',periods=[1,10,100,10000],output_file='monthly_Hs_Tp_joint_param.csv')
tables.table_directional_joint_distribution_Hs_Tp_param(df,var_hs='HS',var_tp='TP',var_dir='DIRM',periods=[1,10,100],output_file='output_mini/dir_Hs_Tp_joint_param.csv')
plots.plot_monthly_weather_window(df,var='HS',threshold=4, window_size=12,output_file='output_mini/_monthly_weather_window4_12_plot.png')
tables.table_monthly_return_periods(df,var='HS',periods=[1, 10, 100, 10000],distribution='Weibull3P_MOM', units='m',output_file='output_mini/HS_monthly_extremes_Weibull.csv')
tables.table_directional_return_periods(df,var='HS',periods=[1, 10, 100, 10000], units='m',var_dir = 'DIRM',distribution='Weibull3P_MOM', adjustment='NORSOK' ,output_file='output_mini/directional_extremes_weibull.csv')
plots.plot_monthly_return_periods(df,var='HS',periods=[1, 10, 100],distribution='Weibull3P_MOM', units='m',output_file='output_mini/HS_monthly_extremes.png')
plots.plot_directional_return_periods(df,var='HS',var_dir='DIRM',periods=[1, 10, 100, 10000 ],distribution='GUM', units='m',output_file='output_mini/dir_extremes_GUM.png')
plots.plot_directional_return_periods(df,var='HS',var_dir='DIRM',periods=[1, 10, 100, 10000],distribution='Weibull3P_MOM', units='m',adjustment='NORSOK',output_file='output_mini/dir_extremes_Weibull_norsok.png')
tables.table_monthly_joint_distribution_Hs_Tp_return_values(df,var_hs='HS',var_tp='TP',periods=[1,10,100,10000],output_file='output_mini/monthly_Hs_Tp_joint_return_values.csv')
tables.table_directional_joint_distribution_Hs_Tp_return_values(df,var_hs='HS',var_tp='TP',var_dir='DIRM',periods=[1,10,100,1000],adjustment='NORSOK',output_file='output_mini/directional_Hs_Tp_joint_return_values.csv')
tables.table_Hs_Tpl_Tph_return_values(df,var_hs='HS',var_tp='TP',periods=[1,10,100,10000],output_file='output_mini/hs_tpl_tph_return_values.csv')
plots.plot_tp_for_given_hs(df, 'HS', 'TP',output_file='output_mini/tp_for_given_hs.png')
tables.table_tp_for_given_hs(df, 'HS', 'TP',max_hs=20,output_file='output_mini/tp_for_given_hs.csv')
tables.table_tp_for_rv_hs(df, var_hs='HS', var_tp='TP',periods=[1,10,100,10000],output_file='output_mini/tp_for_rv_hs.csv')
tables.table_wave_induced_current(df, var_hs='HS',var_tp='TP',depth=200,ref_depth=200, spectrum = 'JONSWAP',output_file='output_mini/JONSWAP_wave_induced_current_depth200.csv')
tables.table_wave_induced_current(df, var_hs='HS',var_tp='TP',depth=200,ref_depth=200, spectrum = 'TORSEHAUGEN',output_file='output_mini/TORSEHAUGEN_wave_induced_current_depth200.csv')
tables.table_hs_for_given_wind(df, 'HS','W10', bin_width=2, max_wind=42, output_file= 'output_mini/table_perc_hs_for_wind.csv')
plots.plot_hs_for_given_wind(df, 'HS', 'W10',output_file='output_mini/hs_for_given_wind.png')
tables.table_hs_for_rv_wind(df, var_wind='W10', var_hs='HS',periods=[1,10,100,10000],output_file='output_mini/hs_for_rv_wind.csv')
tables.table_Hmax_crest_return_periods(df,var_hs='HS', var_tp='TP', depth=200, periods=[1, 10, 100,10000],sea_state='long-crested',output_file='output_mini/table_Hmax_crest_rp.csv')
tables.table_directional_Hmax_return_periods(df,var_hs='HS', var_tp = 'TP',var_dir='DIRM', periods=[10, 100,10000],adjustment='NORSOK', output_file='output_mini/table_dir_Hmax_return_values.csv')
plots.plot_multi_joint_distribution_Hs_Tp_var3(df,var_hs='HS',var_tp='TP',var3='W10',var3_units='m/s',periods=[100],var3_bin=5,threshold_min=100,output_file='output_mini/Hs.Tp.joint.distribution.multi.binned.var3.png')


# Air Temperature:
plots.plot_monthly_return_periods(df,var='T2m',periods=[1, 10, 100],distribution='GUM_L',method='minimum', units='°C',output_file='output_mini/T2m_monthly_extremes_neg.png')
tables.table_monthly_return_periods(df,var='T2m',periods=[1, 10, 100],distribution='GUM_L', method='minimum' ,units='°C',output_file='output_mini/T2m_monthly_extremes_neg.csv')
plots.plot_monthly_return_periods(df,var='T2m',periods=[1, 10, 100],distribution='GUM', method='maximum', units='°C',output_file='output_mini/T2m_monthly_extremes_pos.png')
tables.table_monthly_return_periods(df,var='T2m',periods=[1, 10, 100],distribution='GUM', method='maximum' ,units='°C',output_file='output_mini/T2m_monthly_extremes_pos.csv')
plots.plot_monthly_stats(df,var='T2m',show=['Minimum','Mean','Maximum'], title = 'T2m', output_file='output_mini/T2m_monthly_stats.png')
tables.table_monthly_non_exceedance(df,var='T2m',step_var=0.5,output_file='output_mini/T2m_table_monthly_non_exceedance.csv')

