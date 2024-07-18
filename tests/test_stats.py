from metocean_api import ts
from metocean_stats import plots, tables, stats
from metocean_stats.stats.aux_funcs import *
from metocean_stats.stats.map_funcs import *
import os

# Define TimeSeries-object for NORA3
ds = readNora10File('tests/data/NORA_test.txt')

def test_plot_prob_non_exceedance_fitted_3p_weibull(ds=ds):
    output_file = 'test_prob_non_exceedance.png'
    fig = plots.plot_prob_non_exceedance_fitted_3p_weibull(ds, var='HS', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.axes[0].lines[0].get_xdata()[0].round(2) == -0.69:
        pass
    else:
        raise ValueError("FigValue is not correct")

def test_scatter_diagram(ds=ds):
    output_file = 'test_scatter_diagram.csv'
    df = tables.scatter_diagram(ds, var1='HS', step_var1=1, var2='TP', step_var2=1, output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape[0] == 14:
        pass
    else:
        raise ValueError("Shape is not correct")

def test_table_var_sorted_by_hs(ds=ds):
    output_file = 'test_var_sorted_by_hs.csv'
    df = tables.table_var_sorted_by_hs(ds, var='TP', var_hs='HS', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (16, 7):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_table_monthly_non_exceedance(ds=ds):
    output_file = 'test_monthly_non_exceedance.csv'
    df = tables.table_monthly_non_exceedance(ds, var1='HS', step_var1=0.5, output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (31, 13):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_plot_monthly_stats(ds=ds):
    output_file = 'test_monthly_stats.png'
    fig = plots.plot_monthly_stats(ds, var1='T2m', show=['Minimum', 'Mean', 'Maximum'], title='T2m', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.axes[0].lines[0].get_xdata()[0].round(2) == 0:
        pass
    else:
        raise ValueError("FigValue is not correct")

def test_table_directional_non_exceedance(ds=ds):
    output_file = 'test_directional_non_exceedance.csv'
    df = tables.table_directional_non_exceedance(ds, var1='HS', step_var1=0.5, var_dir='DIRM', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (30, 13):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_plot_directional_stats(ds=ds):
    output_file = 'test_directional_stats.png'
    fig = plots.plot_directional_stats(ds, var1='HS', step_var1=0.5, var_dir='DIRM', title='$H_s$[m]', output_file=output_file)
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

def test_table_monthly_joint_distribution_Hs_Tp_param(ds=ds):
    output_file = 'test_monthly_joint_distribution_Hs_Tp_param.csv'
    df = tables.table_monthly_joint_distribution_Hs_Tp_param(ds, var_hs='HS', var_tp='TP', periods=[1, 10, 100, 10000], output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (13, 6):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_table_directional_joint_distribution_Hs_Tp_param(ds=ds):
    output_file = 'test_directional_joint_distribution_Hs_Tp_param.csv'
    df = tables.table_directional_joint_distribution_Hs_Tp_param(ds, var_hs='HS', var_tp='TP', var_dir='DIRM', periods=[1, 10, 100, 10000], output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (13, 6):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_plot_monthly_weather_window(ds=ds):
    output_file = 'test_monthly_weather_window.png'
    fig, table = plots.plot_monthly_weather_window(ds, var='HS', threshold=4, window_size=12, output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if table._loc == 17:
        pass
    else:
        raise ValueError("FigValue is not correct")

def test_table_directional_return_periods(ds=ds):
    output_file = 'test_directional_return_periods.csv'
    df = tables.table_directional_return_periods(ds, var='HS', periods=[1, 10, 100, 10000], units='m', var_dir='DIRM', distribution='Weibull', adjustment='NORSOK', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (14, 9):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_plot_monthly_return_periods(ds=ds):
    output_file = 'test_monthly_return_periods.png'
    fig = plots.plot_monthly_return_periods(ds, var='HS', periods=[1, 10, 100], distribution='Weibull', units='m', output_file=output_file)
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
    fig = plots.plot_polar_directional_return_periods(ds, var='HS', var_dir='DIRM', periods=[1, 10, 100, 10000], distribution='Weibull', units='m', adjustment='NORSOK', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.dpi == 100.0:
        pass
    else:
        raise ValueError("FigValue is not correct")

def test_table_monthly_joint_distribution_Hs_Tp_return_values(ds=ds):
    output_file = 'test_monthly_joint_distribution_Hs_Tp_return_values.csv'
    df = tables.table_monthly_joint_distribution_Hs_Tp_return_values(ds, var_hs='HS', var_tp='TP', periods=[1, 10, 100, 10000], output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (13, 10):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_table_directional_joint_distribution_Hs_Tp_return_values(ds=ds):
    output_file = 'test_directional_joint_distribution_Hs_Tp_return_values.csv'
    df = tables.table_directional_joint_distribution_Hs_Tp_return_values(ds, var_hs='HS', var_tp='TP', var_dir='DIRM', periods=[1, 10, 100], adjustment='NORSOK', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (13, 8):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_table_Hs_Tpl_Tph_return_values(ds=ds):
    output_file = 'test_Hs_Tpl_Tph_return_values.csv'
    df = tables.table_Hs_Tpl_Tph_return_values(ds, var_hs='HS', var_tp='TP', periods=[1, 10, 100, 10000], output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (10, 12):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_plot_tp_for_given_hs(ds=ds):
    output_file = 'test_tp_for_given_hs.png'
    fig = plots.plot_tp_for_given_hs(ds, 'HS', 'TP', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.dpi == 100.0:
        pass
    else:
        raise ValueError("FigValue is not correct")

def test_table_tp_for_given_hs(ds=ds):
    output_file = 'test_table_tp_for_given_hs.csv'
    df = tables.table_tp_for_given_hs(ds, 'HS', 'TP', max_hs=20, output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (20, 8):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_table_tp_for_rv_hs(ds=ds):
    output_file = 'test_table_tp_for_rv_hs.csv'
    df = tables.table_tp_for_rv_hs(ds, 'HS', 'TP', periods=[1, 10, 100, 10000], output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (4, 5):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_table_wave_induced_current_JONSWAP(ds=ds):
    output_file = 'test_wave_induced_current_JONSWAP.csv'
    df = tables.table_wave_induced_current(ds, 'HS', 'TP', depth=200, ref_depth=200, spectrum='JONSWAP', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (20, 14):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_table_wave_induced_current_TORSEHAUGEN(ds=ds):
    output_file = 'test_wave_induced_current_TORSEHAUGEN.csv'
    df = tables.table_wave_induced_current(ds, 'HS', 'TP', depth=200, ref_depth=200, spectrum='TORSEHAUGEN', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (20, 14):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_table_tp_for_given_wind(ds=ds):
    output_file = 'test_table_tp_for_given_wind.csv'
    df = tables.table_tp_for_given_wind(ds, 'HS', 'W10', bin_width=2, max_wind=42, output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (21, 10):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_plot_hs_for_given_wind(ds=ds):
    output_file = 'test_plot_hs_for_given_wind.png'
    fig = plots.plot_hs_for_given_wind(ds, 'HS', 'W10', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.dpi == 100.0:
        pass
    else:
        raise ValueError("FigValue is not correct")

def test_table_Hmax_crest_return_periods(ds=ds):
    output_file = 'test_table_Hmax_crest_return_periods.csv'
    df = tables.table_Hmax_crest_return_periods(ds, var_hs='HS', var_tp='TP', depth=200, periods=[1, 10, 100, 10000], sea_state='long-crested', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (4, 12):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_table_directional_Hmax_return_periods(ds=ds):
    output_file = 'test_table_directional_Hmax_return_periods.csv'
    df = tables.table_directional_Hmax_return_periods(ds, var_hs='HS', var_tp='TP', var_dir='DIRM', periods=[10, 100], adjustment='NORSOK', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (13, 14):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_table_monthly_return_periods_T2m_min(ds=ds):
    output_file = 'test_table_monthly_return_periods_T2m_min.csv'
    df = tables.table_monthly_return_periods(ds, var='T2m', periods=[1, 10, 100], distribution='GUM_L', method='minimum', units='°C', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (14, 8):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_table_monthly_return_periods_T2m_max(ds=ds):
    output_file = 'test_table_monthly_return_periods_T2m_max.csv'
    df = tables.table_monthly_return_periods(ds, var='T2m', periods=[1, 10, 100], distribution='GUM', method='maximum', units='°C', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (14, 8):
        pass
    else:
        raise ValueError("Shape is not correct")

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


def test_table_hs_for_rv_wind(ds=ds):
    output_file = 'test_table_hs_for_given_wind.csv'
    df = tables.table_hs_for_given_wind(ds, var_wind='W10', var_hs='HS',periods=[1,10,100,10000],output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (4, 6):
        pass
    else:
        raise ValueError("Shape is not correct")

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

  