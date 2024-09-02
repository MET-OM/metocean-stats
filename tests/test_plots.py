from metocean_stats import plots, tables, stats, maps
from metocean_stats.stats.aux_funcs import *
import pandas as pd

import os

# Define TimeSeries-object for NORA3
ds = readNora10File('tests/data/NORA_test.txt')
ds_ocean = pd.read_csv('tests/data/NorkystDA_test.csv',comment='#',index_col=0, parse_dates=True)
depth = ['0m', '1m', '2.5m', '5m', '10m', '15m', '20m', '25m', '30m', '40m', '50m', '75m', '100m', '150m', '200m']


def test_plot_scatter(ds=ds):
    output_file = 'test_plot_scatter.png'
    fig = plots.plot_scatter(ds, 'W10', 'W150', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.dpi == 100.0:
        pass
    else:
        raise ValueError("FigValue is not correct")


def test_plot_prob_non_exceedance_fitted_3p_weibull(ds=ds):
    output_file = 'test_prob_non_exceedance.png'
    fig = plots.plot_prob_non_exceedance_fitted_3p_weibull(ds, var='HS', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.axes[0].lines[0].get_xdata()[0].round(2) == -0.69:
        pass
    else:
        raise ValueError("FigValue is not correct")

#def test_scatter_diagram(ds=ds):
#    output_file = 'test_scatter_diagram.csv'
#    df = tables.scatter_diagram(ds, var1='HS', step_var1=1, var2='TP', step_var2=1, output_file=output_file)
#    if os.path.exists(output_file):
#        os.remove(output_file)
#    if df.shape[0] == 14:
#        pass
#    else:
#        raise ValueError("Shape is not correct")

def test_plot_monthly_stats(ds=ds):
    output_file = 'test_monthly_stats.png'
    fig = plots.plot_monthly_stats(ds, var='T2m', show=['Minimum', 'Mean', 'Maximum'], title='T2m', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.axes[0].lines[0].get_xdata()[0].round(2) == 0:
        pass
    else:
        raise ValueError("FigValue is not correct")

def test_plot_directional_stats(ds=ds):
    output_file = 'test_directional_stats.png'
    fig = plots.plot_directional_stats(ds, var='HS', step_var=0.5, var_dir='DIRM', title='$H_s$[m]', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.axes[0].lines[0].get_xdata()[0].round(2) == 0:
        pass
    else:
        raise ValueError("FigValue is not correct")

def test_plot_joint_distribution_Hs_Tp(ds=ds):
    output_file = 'test_joint_distribution_Hs_Tp.png'
    fig = plots.plot_joint_distribution_Hs_Tp(ds, var_hs='HS', var_tp='TP', periods=[1, 10, 100, 1000], title='Hs-Tp joint distribution', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if int(fig.axes[0].lines[0].get_xdata()[0]) == 3:
        pass
    else:
        raise ValueError("FigValue is not correct")

def test_plot_monthly_weather_window(ds=ds):
    output_file = 'test_monthly_weather_window.png'
    fig, table = plots.plot_monthly_weather_window(ds, var='HS', threshold=4, window_size=12, output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if table._loc == 17:
        pass
    else:
        raise ValueError("FigValue is not correct")

def test_plot_monthly_return_periods(ds=ds):
    output_file = 'test_monthly_return_periods.png'
    fig = plots.plot_monthly_return_periods(ds, var='HS', periods=[1, 10, 100], distribution='Weibull3P_MOM', units='m', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.axes[0].lines[0].get_xdata()[0] == 'Jan':
        pass
    else:
        raise ValueError("FigValue is not correct")

def test_plot_directional_return_periods(ds=ds):
    output_file = 'test_directional_return_periods.png'
    fig = plots.plot_directional_return_periods(ds, var='HS', var_dir='DIRM', periods=[1, 10, 100, 10000], distribution='GUM', units='m', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.axes[0].lines[0].get_xdata()[0] == '0°':
        pass
    else:
        raise ValueError("FigValue is not correct")

def test_plot_polar_directional_return_periods(ds=ds):
    output_file = 'test_polar_directional_return_periods.png'
    fig = plots.plot_polar_directional_return_periods(ds, var='HS', var_dir='DIRM', periods=[1, 10, 100, 10000], distribution='Weibull3P_MOM', units='m', adjustment='NORSOK', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.dpi == 100.0:
        pass
    else:
        raise ValueError("FigValue is not correct")

def test_plot_tp_for_given_hs(ds=ds):
    output_file = 'test_tp_for_given_hs.png'
    fig = plots.plot_tp_for_given_hs(ds, 'HS', 'TP', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.dpi == 100.0:
        pass
    else:
        raise ValueError("FigValue is not correct")

def test_plot_hs_for_given_wind(ds=ds):
    output_file = 'test_plot_hs_for_given_wind.png'
    fig = plots.plot_hs_for_given_wind(ds, 'HS', 'W10', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.dpi == 100.0:
        pass
    else:
        raise ValueError("FigValue is not correct")

def test_plot_monthly_return_periods_T2m_min(ds=ds):
    output_file = 'test_plot_monthly_return_periods_T2m_min.png'
    fig = plots.plot_monthly_return_periods(ds, var='T2m', periods=[1, 10, 100], distribution='GUM_L', method='minimum', units='°C', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.axes[0].lines[0].get_xdata()[0] == 'Jan':
        pass
    else:
        raise ValueError("FigValue is not correct")

def test_plot_monthly_return_periods_T2m_max(ds=ds):
    output_file = 'test_plot_monthly_return_periods_T2m_max.png'
    fig = plots.plot_monthly_return_periods(ds, var='T2m', periods=[1, 10, 100], distribution='GUM', method='maximum', units='°C', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.axes[0].lines[0].get_xdata()[0] == 'Jan':
        pass
    else:
        raise ValueError("FigValue is not correct")

def test_plot_var_rose(ds=ds):
    output_file = 'test_rose.png'
    fig = plots.var_rose(ds,'DIRM' ,'HS', method='monthly', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.dpi == 100.0:
        pass
    else:
        raise ValueError("FigValue is not correct")

def test_plot_directional_return_periods_POT(ds=ds):
    output_file = 'test_directional_return_periods_POT.png'
    fig = plots.plot_directional_return_periods(ds,var='HS',var_dir='DIRM',periods=[1, 10, 100, 10000],distribution='Weibull3P_MOM', units='m',adjustment='NORSOK',method='POT',threshold='P99',output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.axes[0].lines[0].get_xdata()[0] == '0°':
        pass
    else:
        raise ValueError("FigValue is not correct")


def test_plot_profile_return_values(ds=ds):
    output_file = 'test_plot_profile_return_values.png'
    fig = plots.plot_profile_return_values(ds,var=['W10','W50','W80','W100','W150'], z=[10, 50, 80, 100, 150], periods=[1, 10, 100, 10000],reverse_yaxis=True, output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.axes[0].lines[0].get_xdata()[0] == 27.31:
        pass
    else:
        raise ValueError("FigValue is not correct")

def test_plot_profile_stats(ds=ds_ocean):
    output_file = 'test_plot_profile_stats.png'
    fig = plots.plot_profile_stats(ds_ocean,var=['current_speed_' + d for d in depth], z=[float(d[:-1]) for d in depth],reverse_yaxis=True, output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.axes[0].lines[0].get_xdata()[0] == 0.17:
        pass
    else:
        raise ValueError("FigValue is not correct")

def test_plot_current_for_given_wind(ds=ds):
    output_file = 'test_current_for_given_wind.png'
    fig = plots.plot_current_for_given_wind(ds, 'HS', 'W10', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.dpi == 100.0:
        pass
    else:
        raise ValueError("FigValue is not correct")


def test_plot_current_for_given_hs(ds=ds):
    output_file = 'test_current_for_given_Hs.png'
    ds['current_speed_0m'] = 0.05*ds['W10']
    fig = plots.plot_current_for_given_hs(ds, 'current_speed_0m', 'HS', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.dpi == 100.0:
        pass
    else:
        raise ValueError("FigValue is not correct")


def test_plot_profile_monthly_stats(ds=ds_ocean):
    output_file = 'test_plot_profile_monthly_stats.png'
    fig = plots.plot_profile_monthly_stats(ds_ocean, var=['temp_' + d for d in depth], z=[float(d[:-1]) for d in depth], method='mean',title='Mean Sea Temperature [°C]', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.axes[0].lines[0].get_xdata()[0] == 7.72:
        pass
    else:
        raise ValueError("FigValue is not correct")

def test_plot_storm_surge_for_given_hs(ds=ds):
    output_file = 'test_surge_for_given_Hs.png'
    ds['zeta_0m'] = 0.02*ds['HS']  + 0.05*np.log(ds['HS'])
    fig = plots.plot_storm_surge_for_given_hs(ds, 'zeta_0m', 'HS', max_hs=20, output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.dpi == 100.0:
        pass
    else:
        raise ValueError("FigValue is not correct")


def test_plot_tidal_levels(ds=ds):
    output_file = 'test_plot_tidal_levels.png'
    ds['tide'] = 0.01*ds['HS'] 
    fig = plots.plot_tidal_levels(ds,'tide', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.dpi == 100.0:
        pass
    else:
        raise ValueError("FigValue is not correct")
    


def test_plot_monthly_return_periods_cur_pot(ds=ds_ocean):
    output_file = 'test_plot_monthly_return_periods_curr_pot.png'
    fig = plots.plot_monthly_return_periods(ds_ocean,var='current_speed_0m',periods=[1, 10, 100],distribution='Weibull3P_MOM',method='POT',threshold='P99', units='m/s',output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.axes[0].lines[0].get_xdata()[0] == 'Jan':
        pass
    else:
        raise ValueError("FigValue is not correct")

#def test_threshold_sensitivity(ds=ds):
#    extreme_stats.threshold_sensitivity(data=ds.data, var='hs', 
#                                        thresholds=[1,1.5])
                                        
#def test_joint_distribution_Hs_Tp(ds=ds):
#    extreme_stats.joint_distribution_Hs_Tp(df=ds.data, file_out='test.png')
#    os.remove('test.png')

#def test_mean_profile(ds=ds):
#    profile_stats.mean_profile(data = ds.data, vars = ['wind_speed_10m','wind_speed_20m','wind_speed_50m','wind_speed_100m','wind_speed_250m','wind_speed_500m','wind_speed_750m'],height_levels=[10,20,50,100,250,500,750], perc = [25,75], output_file=False)
    
#def test_profile_shear(ds=ds):
#    profile_stats.profile_shear(data = ds.data, vars = ['wind_speed_10m','wind_speed_20m','wind_speed_50m','wind_speed_100m','wind_speed_250m','wind_speed_500m','wind_speed_750m'],height_levels=[10,20,50,100,250,500,750], z=[20,250], perc = [25,75], output_file=False)

#def test_predict_ts_GBR(ds=ds):
#    ml_stats.predict_ts(ts_origin=ds.data,var_origin=['hs','tp','Pdir'],ts_train=ds.data.loc['2000-01-01':'2000-01-10'],var_train=['hs'], model='GBR')

#def test_predict_ts_SVR(ds=ds):
#    ml_stats.predict_ts(ts_origin=ds.data,var_origin=['hs','tp','Pdir'],ts_train=ds.data.loc['2000-01-01':'2000-01-10'],var_train=['hs'], model='SVR_RBF')

#def test_predict_ts_LSTM(ds=ds):
#    ml_stats.predict_ts(ts_origin=ds.data,var_origin=['hs','tp','Pdir'],ts_train=ds.data.loc['2000-01-01':'2000-01-10'],var_train=['hs'], model='LSTM')

def test_plot_nb_hours_below_threshold(ds=ds):
    output_file = 'test_plot_nb_hr_below_t.png'
    fig = plots.plot_nb_hours_below_threshold(ds,var='HS',thr_arr=(np.arange(0.05,20.05,0.05)).tolist(),output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.dpi == 100.0:
        pass
    else:
        raise ValueError("FigValue is not correct")