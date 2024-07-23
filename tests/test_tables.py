from metocean_api import ts
from metocean_stats import plots, tables, stats
from metocean_stats.stats.aux_funcs import *
from metocean_stats.stats.map_funcs import *
import pandas as pd

import os

# Define TimeSeries-object for NORA3
ds = readNora10File('tests/data/NORA_test.txt')
ds_ocean = pd.read_csv('tests/data/NorkystDA_test.csv',comment='#',index_col=0, parse_dates=True)
depth = ['0m', '1m', '2.5m', '5m', '10m', '15m', '20m', '25m', '30m', '40m', '50m', '75m', '100m', '150m', '200m']


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

def test_table_directional_non_exceedance(ds=ds):
    output_file = 'test_directional_non_exceedance.csv'
    df = tables.table_directional_non_exceedance(ds, var1='HS', step_var1=0.5, var_dir='DIRM', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (30, 13):
        pass
    else:
        raise ValueError("Shape is not correct")

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
    df = tables.table_directional_joint_distribution_Hs_Tp_param(ds,var_hs='HS',var_tp='TP',var_dir='DIRM',periods=[1,10,100],output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (13, 6):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_table_directional_return_periods(ds=ds):
    output_file = 'test_directional_return_periods.csv'
    df = tables.table_directional_return_periods(ds, var='HS', periods=[1, 10, 100, 10000], units='m', var_dir='DIRM', distribution='Weibull3P_MOM', adjustment='NORSOK', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (14, 9) and df['Return period: 10000 [years]'][13]==18.7:
        pass
    else:
        raise ValueError("Shape is not correct")
    
def test_table_monthly_return_periods(ds=ds):
    output_file = 'test_monthly_return_periods.csv'
    df = tables.table_monthly_return_periods(ds,var='HS',periods=[1, 10, 100, 10000],distribution='Weibull3P_MOM', units='m',output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (14, 9) and df['Return period: 10000 [years]'][13]==18.7:
        pass
    else:
        raise ValueError("Shape is not correct")

def test_table_directional_return_periods_POT(ds=ds):
    output_file = 'test_dir_return_periods_POT.csv'
    df = tables.table_directional_return_periods(ds,var='HS',var_dir='DIRM',periods=[1, 10, 100, 10000],distribution='Weibull3P_MOM',method='POT',threshold='P99', units='m',output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (14, 11) and df['Return period: 10000 [years]'][13]==20.73:
        pass
    else:
        raise ValueError("Shape is not correct")


def test_table_directional_return_periods(ds=ds):
    output_file = 'test_dir_return_periods.csv'
    df = tables.table_directional_return_periods(ds,var='HS',var_dir='DIRM',periods=[1, 10, 100, 10000],distribution='Weibull3P_MOM', units='m',output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (14, 9) and df['Return period: 10000 [years]'][13]==18.7:
        pass
    else:
        raise ValueError("Shape is not correct")

def test_table_monthly_return_periods_POT(ds=ds):
    output_file = 'test_monthly_return_periods_POT.csv'
    df = tables.table_monthly_return_periods(ds,var='HS',periods=[1, 10, 100, 10000],distribution='Weibull3P_MOM',method='POT',threshold='P99', units='m',output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (14, 11) and df['Return period: 10000 [years]'][13]==20.73:
        pass
    else:
        raise ValueError("Shape is not correct")

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

def test_table_hshs_for_given_wind(ds=ds):
    output_file = 'test_table_hs_for_given_wind.csv'
    df = tables.table_hs_for_given_wind(ds, 'HS', 'W10', bin_width=2, max_wind=42, output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (21, 10):
        pass
    else:
        raise ValueError("Shape is not correct")

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

def test_table_hs_for_rv_wind(ds=ds):
    output_file = 'test_table_hs_for_given_wind.csv'
    df = tables.table_hs_for_rv_wind(ds, var_wind='W10', var_hs='HS',periods=[1,10,100,10000],output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (4, 6):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_profile_return_values(ds=ds):
    output_file = 'test_profile_return_values.csv'
    df = tables.table_profile_return_values(ds,var=['W10','W50','W80','W100','W150'], z=[10, 50, 80, 100, 150], periods=[1, 10, 100, 10000], output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (6, 5) and int(df['Return period 10000 [years]'][5])==49:
        pass
    else:
        raise ValueError("Shape is not correct")

def test_table_current_for_given_wind(ds=ds):
    output_file = 'test_table_current_for_given_wind.csv'
    ds['current_speed_0m'] = 0.05*ds['W10']
    df = tables.table_current_for_given_wind(ds, var_curr='current_speed_0m', var_wind='W10', bin_width=2, max_wind=42, output_file=output_file)

    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (21, 10):
        pass
    else:
        raise ValueError("Shape is not correct")

def test_table_current_for_given_Hs(ds=ds):
    output_file = 'test_table_current_for_given_Hs.csv'
    ds['current_speed_0m'] = 0.05*ds['W10']
    df = tables.table_current_for_given_Hs(ds, var_curr='current_speed_0m', var_hs='HS', bin_width=2, max_hs=20, output_file=output_file)

    if os.path.exists(output_file):
        os.remove(output_file)
    if df.shape == (10, 10):
        pass
    else:
        raise ValueError("Shape is not correct")


def test_table_extreme_current_profile_rv(ds=ds_ocean):
    df = tables.table_extreme_current_profile_rv(ds_ocean, var=['current_speed_' + d for d in depth], z=[float(d[:-1]) for d in depth], periods=[1,100,1000],percentile=95, output_file=None)

    if df.shape == (16, 17):
        pass
    else:
        raise ValueError("Shape is not correct")


def test_table_profile_stats(ds=ds_ocean):
    df = tables.table_profile_stats(ds_ocean, var=['current_speed_' + d for d in depth], z=[float(d[:-1]) for d in depth], var_dir=['current_direction_' + d for d in depth], output_file=None)

    if df.shape == (16, 12) and df['P50'][1]== 0.16:
        pass
    else:
        raise ValueError("Shape or value are not correct")

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

  