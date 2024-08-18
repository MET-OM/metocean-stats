from metocean_stats import plots, tables, stats, maps
from metocean_stats.stats.aux_funcs import *

df = readNora10File('../tests/data/NORA_test.txt') 

# Map:
maps.plot_points_on_map(lon=[3.35,3.10], lat=[60.40,60.90],label=['NORA3','NORKYST800'],bathymetry='NORA3',output_file='map.png')
maps.plot_extreme_wave_map(return_period=100, product='NORA3', title='100-yr return values Hs (NORA3)', set_extent = [0,30,52,73],output_file='wave_100yrs.png')
maps.plot_extreme_wind_map(return_period=100, product='NORA3',z=10, title='100-yr return values Wind at 100 m (NORA3)', set_extent = [0,30,52,73], output_file='wind_100yrs.png')


# Wind:
plots.var_rose(df,var_dir='D10',var='W10',method='overall',max_perc=40,decimal_places=1, units='m/s',output_file='wind_omni.png')
plots.var_rose(df,var_dir='D10',var='W10',method='monthly',max_perc=40,decimal_places=1, units='m/s',output_file='wind_monthly.png')
plots.plot_directional_stats(df,var='W10',step_var=0.1,var_dir='D10',title = '', output_file='directional_stats.png')
plots.table_directional_non_exceedance(df,var='W10',step_var=2,var_dir='D10',output_file='table_directional_non_exceedance.csv')
plots.plot_monthly_stats(df,var='W10',title = 'Wind Speed at 10 m [m/s]', output_file='monthly_stats.png')
tables.table_monthly_non_exceedance(df,var='W10',step_var=2,output_file='table_monthly_non_exceedance.csv')
plots.plot_prob_non_exceedance_fitted_3p_weibull(df,var='W10',output_file='prob_non_exceedance_fitted_3p_weibull.png')
plots.plot_monthly_return_periods(df,var='W10',periods=[1, 10, 100],distribution='Weibull3P_MOM', units='m/s',output_file='W10_monthly_extremes.png')
plots.plot_directional_return_periods(df,var='W10',var_dir='D10',periods=[1, 10, 100],distribution='Weibull3P_MOM', units='m/s',adjustment='NORSOK',output_file='W10_dir_extremes_Weibull_norsok.png')
plots.plot_monthly_weather_window(df,var='W10',threshold=10, window_size=12,output_file= 'NORA10_monthly_weather_window4_12_plot.png')
plots.plot_scatter(df,var1='W10',var2='W100',var1_units='m/s',var2_units='m/s', title='Scatter',regression_line=True,qqplot=False,density=True,output_file='scatter_plot.png')
plots.plot_multi_diagnostic_return_levels(df, var='HS', periods=[10, 100],threshold=None,output_file='plot_diagnostic_return_levels.png')

# Waves:
plots.plot_prob_non_exceedance_fitted_3p_weibull(df,var='HS',output_file='prob_non_exceedance_fitted_3p_weibull.png')
tables.scatter_diagram(df, var1='HS', step_var1=1, var2='TP', step_var2=1, output_file='Hs_Tp_scatter.csv')
tables.table_var_sorted_by_hs(df, var='TP', var_hs='HS', output_file='Tp_sorted_by_Hs.csv')
tables.table_monthly_non_exceedance(df,var='HS',step_var=0.5,output_file='Hs_table_monthly_non_exceedance.csv')
plots.plot_monthly_stats(df,var='HS',show=['Maximum','P99','Mean'], title = 'Hs[m]', output_file='Hs_monthly_stats.png')
tables.table_directional_non_exceedance(df,var='HS',step_var=0.5,var_dir='DIRM',output_file='table_directional_non_exceedance.csv')
plots.plot_directional_stats(df,var='HS',step_var=0.5, var_dir='DIRM', title = '$H_s$[m]', output_file='directional_stats.png')
plots.plot_joint_distribution_Hs_Tp(df,var_hs='HS',var_tp='TP',periods=[1,10,100,1000], title='Hs-Tp joint distribution',output_file='Hs.Tp.joint.distribution.png',density_plot=True)
tables.table_monthly_joint_distribution_Hs_Tp_param(df,var_hs='HS',var_tp='TP',periods=[1,10,100,10000],output_file='monthly_Hs_Tp_joint_param.csv')
tables.table_directional_joint_distribution_Hs_Tp_param(df,var_hs='HS',var_tp='TP',var_dir='DIRM',periods=[1,10,100],output_file='dir_Hs_Tp_joint_param.csv')
plots.plot_monthly_weather_window(df,var='HS',threshold=4, window_size=12,output_file= 'NORA10_monthly_weather_window4_12_plot.png')
tables.table_monthly_return_periods(df,var='HS',periods=[1, 10, 100, 10000],distribution='Weibull3P_MOM', units='m',output_file='HS_monthly_extremes_Weibull.csv')
tables.table_directional_return_periods(df,var='HS',periods=[1, 10, 100, 10000], units='m',var_dir = 'DIRM',distribution='Weibull3P_MOM', adjustment='NORSOK' ,output_file='directional_extremes_weibull.csv')
plots.plot_monthly_return_periods(df,var='HS',periods=[1, 10, 100],distribution='Weibull3P_MOM', units='m',output_file='HS_monthly_extremes.png')
plots.plot_directional_return_periods(df,var='HS',var_dir='DIRM',periods=[1, 10, 100, 10000 ],distribution='GUM', units='m',output_file='dir_extremes_GUM.png')
plots.plot_directional_return_periods(df,var='HS',var_dir='DIRM',periods=[1, 10, 100, 10000],distribution='Weibull3P_MOM', units='m',adjustment='NORSOK',output_file='dir_extremes_Weibull_norsok.png')
tables.table_monthly_joint_distribution_Hs_Tp_return_values(df,var_hs='HS',var_tp='TP',periods=[1,10,100,10000],output_file='monthly_Hs_Tp_joint_return_values.csv')
tables.table_directional_joint_distribution_Hs_Tp_return_values(df,var_hs='HS',var_tp='TP',var_dir='DIRM',periods=[1,10,100,1000],adjustment='NORSOK',output_file='directional_Hs_Tp_joint_return_values.csv')
tables.table_Hs_Tpl_Tph_return_values(df,var_hs='HS',var_tp='TP',periods=[1,10,100,10000],output_file='hs_tpl_tph_return_values.csv')
plots.plot_tp_for_given_hs(df, 'HS', 'TP',output_file='tp_for_given_hs.png')
tables.table_tp_for_given_hs(df, 'HS', 'TP',max_hs=20,output_file='tp_for_given_hs.csv')
tables.table_tp_for_rv_hs(df, var_hs='HS', var_tp='TP',periods=[1,10,100,10000],output_file='tp_for_rv_hs.csv')
tables.table_wave_induced_current(df, var_hs='HS',var_tp='TP',depth=200,ref_depth=200, spectrum = 'JONSWAP',output_file='JONSWAP_wave_induced_current_depth200.csv')
tables.table_wave_induced_current(df, var_hs='HS',var_tp='TP',depth=200,ref_depth=200, spectrum = 'TORSEHAUGEN',output_file='TORSEHAUGEN_wave_induced_current_depth200.csv')
tables.table_hs_for_given_wind(df, 'HS','W10', bin_width=2, max_wind=42, output_file='table_perc_hs_for_wind.csv')
plots.plot_hs_for_given_wind(df, 'HS', 'W10',output_file='hs_for_given_wind.png')
tables.table_hs_for_rv_wind(df, var_wind='W10', var_hs='HS',periods=[1,10,100,10000],output_file='hs_for_rv_wind.csv')
tables.table_Hmax_crest_return_periods(df,var_hs='HS', var_tp='TP', depth=200, periods=[1, 10, 100,10000],sea_state='long-crested',output_file='table_Hmax_crest_rp.csv')
tables.table_directional_Hmax_return_periods(df,var_hs='HS', var_tp = 'TP',var_dir='DIRM', periods=[10, 100,10000],adjustment='NORSOK', output_file='table_dir_Hmax_return_values.csv')


# Air Temperature:
plots.plot_monthly_return_periods(df,var='T2m',periods=[1, 10, 100],distribution='GUM_L',method='minimum', units='°C',output_file='T2m_monthly_extremes_neg.png')
tables.table_monthly_return_periods(df,var='T2m',periods=[1, 10, 100],distribution='GUM_L', method='minimum' ,units='°C',output_file='T2m_monthly_extremes_neg.csv')
plots.plot_monthly_return_periods(df,var='T2m',periods=[1, 10, 100],distribution='GUM', method='maximum', units='°C',output_file='T2m_monthly_extremes_pos.png')
tables.table_monthly_return_periods(df,var='T2m',periods=[1, 10, 100],distribution='GUM', method='maximum' ,units='°C',output_file='T2m_monthly_extremes_pos.csv')
plots.plot_monthly_stats(df,var='T2m',show=['Minimum','Mean','Maximum'], title = 'T2m', output_file='T2m_monthly_stats.png')
tables.table_monthly_non_exceedance(df,var='T2m',step_var=0.5,output_file='T2m_table_monthly_non_exceedance.csv')


# Currents:
#import pandas as pd
#ds_ocean = pd.read_csv('../tests/data/NorkystDA_test.csv',comment='#',index_col=0, parse_dates=True)
#depth = ['0m', '1m', '2.5m', '5m', '10m', '15m', '20m', '25m', '30m', '40m', '50m', '75m', '100m', '150m', '200m']

#ds_all = pd.concat([ds.loc['2017-01-02 00:00:00':'2018-12-31 21:00:00'], ds_ocean.resample('3h').mean()], axis=1)
#ds_all = ds_all.dropna(how='all')


#plots.plot_monthly_return_periods(ds_ocean,var='current_speed_0m',periods=[1, 10, 100],distribution='Weibull3P_MOM',method='POT',threshold='P99', units='m/s',output_file='csp0m_monthly_extremes.png')
#plots.var_rose(df, f'current_direction_{depth}',f'current_speed_{depth}',max_perc=30,decimal_places=2, units='m/s', method='monthly', output_file='monthly_rose.png')
#plots.var_rose(df,f'current_direction_{depth}',f'current_speed_{depth}',max_perc=30,decimal_places=2, units='m/s', method='overall', output_file='overall_rose.png')
#plots.plot_monthly_stats(df,var1=f'current_speed_{depth}',show=['Mean','P99','Maximum'], title = f'Current[m/s], depth:{depth}', output_file=f'current{depth}_monthly_stats.png')
#plots.plot_directional_stats(df,var1=f'current_speed_{depth}',var_dir=f'current_direction_{depth}',step_var1=0.05,show=['Mean','P99','Maximum'], title = f'Current[m/s], depth:{depth}', output_file=f'current{depth}_dir_stats.png')
#tables.table_directional_return_periods(df,var=f'current_speed_{depth}',periods=[1, 10, 100, 10000], units='m/s',var_dir = f'current_direction_{depth}',distribution='Weibull', adjustment='NORSOK' ,output_file=f'directional_extremes_weibull_current{depth}.csv')
#tables.table_monthly_return_periods(df,var=f'current_speed_{depth}',periods=[1, 10, 100, 10000], units='m/s',distribution='Weibull3P_MOM',method='POT',threshold='P99',output_file=f'monthly_extremes_weibull_current{depth}.csv')
#df = tables.table_monthly_return_periods(df,var='HS',periods=[1, 10, 100, 10000], units='m',distribution='Weibull3P_MOM',method='POT',threshold='P99',output_file='HS_monthly_extremes_Weibull_POT_P99.csv')
#df = tables.table_monthly_return_periods(df,var='HS',periods=[1, 10, 100, 10000],distribution='Weibull3P_MOM', units='m',output_file='HS_monthly_extremes_Weibull.csv')
#df = tables.table_directional_return_periods(df,var='HS',var_dir='DIRM',periods=[1, 10, 100, 10000],distribution='Weibull3P_MOM', units='m',output_file='HS_dir_extremes_Weibull.csv')
#df = tables.table_directional_return_periods(df,var='HS',var_dir='DIRM',periods=[1, 10, 100, 10000],distribution='Weibull3P_MOM',method='POT',threshold='P99', units='m',output_file='HS_dir_extremes_Weibull_POT.csv')
# plots.plot_directional_return_periods(df,var='HS',var_dir='DIRM',periods=[1, 10, 100, 10000],distribution='Weibull3P_MOM', units='m',adjustment='NORSOK',method='POT',threshold='P99',output_file='dir_extremes_Weibull_norsok.png')
# df = tables.table_profile_return_values(df,var=['W10','W50','W80','W100','W150'], z=[10, 50, 80, 100, 150], periods=[1, 10, 100, 10000], output_file='RVE_wind_profile.csv')
#fig = plots.plot_profile_return_values(ds_ocean,var=['current_speed_' + d for d in depth], z=[float(d[:-1]) for d in depth], periods=[1, 10, 100, 10000],reverse_yaxis=True, output_file='RVE_current_profile.png')
#df = tables.table_current_for_given_wind(ds_all, var_curr='current_speed_0m', var_wind='W10', bin_width=2, max_wind=42, output_file='table_perc_current_for_wind.csv')
#plots.plot_current_for_given_wind(ds_all, var_curr='current_speed_0m', var_wind='W10',max_wind=40 ,output_file='curr_for_given_wind.png')
#df = tables.table_current_for_given_hs(ds_all, var_curr='current_speed_0m', var_hs='HS', bin_width=2, max_hs=20, output_file='table_perc_current_for_Hs.csv')
#ds['current_speed_0m'] = 0.05*ds['W10']
#df = tables.table_current_for_given_wind(df, var_curr='current_speed_0m', var_wind='W10', bin_width=2, max_wind=42, output_file='table_perc_current_for_wind.csv')
#df = tables.table_current_for_given_hs(df, var_curr='current_speed_0m', var_hs='HS', bin_width=2, max_hs=20, output_file='table_perc_current_for_Hs.csv')

#plots.plot_current_for_given_hs(ds_all, var_curr='current_speed_0m', var_hs='HS', max_hs=20, output_file='curr_for_given_hs.png')
#df = tables.table_extreme_current_profile_rv(ds_ocean, var=['current_speed_' + d for d in depth], z=[float(d[:-1]) for d in depth], periods=[1,100,1000],percentile=95, output_file='table_extreme_current_profile_rv.png')
#df = tables.table_profile_stats(ds_ocean, var=['current_speed_' + d for d in depth], z=[float(d[:-1]) for d in depth], var_dir=['current_direction_' + d for d in depth], output_file='table_profile_stats.csv')
#fig = plots.plot_profile_stats(ds_ocean,var=['current_speed_' + d for d in depth], z=[float(d[:-1]) for d in depth],reverse_yaxis=True, output_file='stats_current_profile.png')
#df = tables.table_current_for_rv_wind(ds_all, var_curr='current_speed_0m', var_wind='W10',periods=[1,10,100,10000],output_file='Uc_for_rv_wind.csv')
#df = tables.table_current_for_rv_hs(ds_all, var_curr='current_speed_0m', var_hs='HS',periods=[1,10,100,10000],output_file='Uc_for_rv_hs.csv')


# Sea temperature
# df = tables.table_profile_monthly_stats(ds_ocean, var=['temp_' + d for d in depth], z=[float(d[:-1]) for d in depth], method='mean', output_file='table_mean_temp_profile_monthly_stats.png')
# df = tables.table_profile_monthly_stats(ds_ocean, var=['temp_' + d for d in depth], z=[float(d[:-1]) for d in depth], method='std.dev', output_file='table_std_temp_profile_monthly_stats.png')
# df = tables.table_profile_monthly_stats(ds_ocean, var=['temp_' + d for d in depth], z=[float(d[:-1]) for d in depth], method='minimum', output_file='table_min_temp_profile_monthly_stats.png')
# df = tables.table_profile_monthly_stats(ds_ocean, var=['temp_' + d for d in depth], z=[float(d[:-1]) for d in depth], method='maximum', output_file='table_max_temp_profile_monthly_stats.png')

# fig = plots.plot_profile_monthly_stats(ds_ocean, var=['temp_' + d for d in depth], z=[float(d[:-1]) for d in depth], method='mean',title='Mean Sea Temperature [°C]', output_file='plot_mean_temp_profile_monthly_stats.png')
# fig = plots.plot_profile_monthly_stats(ds_ocean, var=['temp_' + d for d in depth], z=[float(d[:-1]) for d in depth], method='std.dev',title='St.Dev Sea Temperature [°C]', output_file='plot_std_temp_profile_monthly_stats.png')
# fig = plots.plot_profile_monthly_stats(ds_ocean, var=['temp_' + d for d in depth], z=[float(d[:-1]) for d in depth], method='minimum',title='Min. Sea Temperature [°C]', output_file='plot_min_temp_profile_monthly_stats.png')
# fig = plots.plot_profile_monthly_stats(ds_ocean, var=['temp_' + d for d in depth], z=[float(d[:-1]) for d in depth], method='maximum',title='Max. Sea Temperature [°C]', output_file='plot_max_temp_profile_monthly_stats.png')

# # Sainity:
# df = tables.table_profile_monthly_stats(ds_ocean, var=['salt_' + d for d in depth], z=[float(d[:-1]) for d in depth], method='mean', output_file='table_mean_sal_profile_monthly_stats.png')
# df = tables.table_profile_monthly_stats(ds_ocean, var=['salt_' + d for d in depth], z=[float(d[:-1]) for d in depth], method='std.dev', output_file='table_std_sal_profile_monthly_stats.png')
# df = tables.table_profile_monthly_stats(ds_ocean, var=['salt_' + d for d in depth], z=[float(d[:-1]) for d in depth], method='minimum', output_file='table_min_sal_profile_monthly_stats.png')
# df = tables.table_profile_monthly_stats(ds_ocean, var=['salt_' + d for d in depth], z=[float(d[:-1]) for d in depth], method='maximum', output_file='table_max_sal_profile_monthly_stats.png')

# fig = plots.plot_profile_monthly_stats(ds_ocean, var=['salt_' + d for d in depth], z=[float(d[:-1]) for d in depth], method='mean',title='Mean Salinity [PSU]', output_file='plot_mean_sal_profile_monthly_stats.png')
# fig = plots.plot_profile_monthly_stats(ds_ocean, var=['salt_' + d for d in depth], z=[float(d[:-1]) for d in depth], method='std.dev',title='St.Dev Salinity [PSU]', output_file='plot_std_sal_profile_monthly_stats.png')
# fig = plots.plot_profile_monthly_stats(ds_ocean, var=['salt_' + d for d in depth], z=[float(d[:-1]) for d in depth], method='minimum',title='Min. Salinity [PSU]', output_file='plot_min_sal_profile_monthly_stats.png')
# fig = plots.plot_profile_monthly_stats(ds_ocean, var=['salt_' + d for d in depth], z=[float(d[:-1]) for d in depth], method='maximum',title='Max. Salinity [PSU]', output_file='plot_max_sal_profile_monthly_stats.png')


# Water levels:
#ds_tide = pd.read_csv('../tests/data/GTSM_test.csv',comment='#',index_col=0, parse_dates=True)
#df = tables.table_tidal_levels(ds_tide, var='tide', output_file='tidal_levels.csv')
#fig = plots.plot_tidal_levels(ds_tide, var='tide',start_time='2010-01-01',end_time='2010-03-30', output_file='tidal_levels.png')
#df = tables.table_storm_surge_for_given_hs(ds_all, var_surge='zeta_0m', var_hs='HS', bin_width=1, max_hs=20, output_file='table_perc_surge_for_Hs.csv')
#fig = plots.plot_storm_surge_for_given_hs(df,var_surge='zeta_0m', var_hs='HS', max_hs=20, output_file='surge_for_given_hs.png')
#df = tables.table_extreme_total_water_level(df, var_hs='HS',var_tp='TP',var_surge='zeta_0m', var_tide='tide', periods=[100,10000], output_file='table_extreme_total_water_level.csv')
#df = tables.table_storm_surge_for_rv_hs(df, var_hs='HS',var_tp='TP',var_surge='zeta_0m', var_tide='tide', periods=[1,10,100,10000],depth=200, output_file='table_storm_surge_for_rv_hs.csv')
