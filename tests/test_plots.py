import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from metocean_stats import plots
from metocean_stats.plots.climate import *
from metocean_stats.stats.aux_funcs import readNora10File
from .data import synthetic_dataset

# Define TimeSeries-object for NORA3
dirname = os.path.dirname(__file__)
ds =    readNora10File(os.path.join(dirname, 'data/NORA_test.txt'))
ds_ocean = pd.read_csv(os.path.join(dirname, 'data/NorkystDA_test.csv'),comment='#',index_col=0, parse_dates=True)
depth = ['0m', '1m', '2.5m', '5m', '10m', '15m', '20m', '25m', '30m', '40m', '50m', '75m', '100m', '150m', '200m']
ds_synthetic_spectra = synthetic_dataset.synthetic_dataset_spectra()


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
    output_file = 'test_monthly_weather_window_mv.png'
    fig, table = plots.plot_monthly_weather_window(ds, var=['HS','TP'], threshold=[2,8], timestep=3, window_size=12, output_file=output_file)
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
    if round(fig.axes[0].lines[0].get_xdata()[0],2) == 7.72:
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

# #def test_threshold_sensitivity(ds=ds):
# #    extreme_stats.threshold_sensitivity(data=ds.data, var='hs', 
# #                                        thresholds=[1,1.5])
                                        
# #def test_joint_distribution_Hs_Tp(ds=ds):
# #    extreme_stats.joint_distribution_Hs_Tp(df=ds.data, file_out='test.png')
# #    os.remove('test.png')

# #def test_mean_profile(ds=ds):
# #    profile_stats.mean_profile(data = ds.data, vars = ['wind_speed_10m','wind_speed_20m','wind_speed_50m','wind_speed_100m','wind_speed_250m','wind_speed_500m','wind_speed_750m'],height_levels=[10,20,50,100,250,500,750], perc = [25,75], output_file=False)
    
# #def test_profile_shear(ds=ds):
# #    profile_stats.profile_shear(data = ds.data, vars = ['wind_speed_10m','wind_speed_20m','wind_speed_50m','wind_speed_100m','wind_speed_250m','wind_speed_500m','wind_speed_750m'],height_levels=[10,20,50,100,250,500,750], z=[20,250], perc = [25,75], output_file=False)

def test_plot_nb_hours_below_threshold(ds=ds):
    output_file = 'test_plot_nb_hr_below_t.png'
    fig = plots.plot_nb_hours_below_threshold(ds,var='HS',thr_arr=(np.arange(0.05,20.05,0.05)).tolist(),output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.dpi == 100.0:
        pass
    else:
        raise ValueError("FigValue is not correct")
    

def test_plot_multi_diagnostic_with_uncertainty(ds=ds):
    output_file = 'test_plot_multi_diagnostic.png'
    fig=plots.plot_multi_diagnostic_return_levels_uncertainty(ds, var='HS', dist_list=['GP'], yaxis='rp', threshold=5.4, uncertainty=0.95, output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if fig.dpi == 100.0:
        pass
    else:
        raise ValueError("FigValue is not correct")
    

def test_plot_multi_joint_distribution_Hs_Tp_var3(ds=ds):
    output_file = 'test_mutli_joint_distribution_Hs_Tp_var3.png'
    fig = plots.plot_multi_joint_distribution_Hs_Tp_var3(ds,var_hs='HS',var_tp='TP',var3='W10',var3_units='m/s',periods=[100],var3_bin=10,threshold_min=100,output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
#    if int(fig.axes[0].lines[0].get_xdata()[0]) == 3:
    if fig.dpi == 100.0:
        pass
    else:
        raise ValueError("FigValue is not correct")

def test_plot_daily_stats_wind_speed():
    # Test the function with wind speed data (W10)
    fig = plots.plot_daily_stats(ds, var="W10", show=["min", "mean", "max"],output_file="")

    # Check that the output is a Matplotlib Figure
    assert isinstance(fig, plt.Figure), "The output is not a Matplotlib Figure."
    
    # Ensure the figure has the correct title if provided
    assert fig.axes[0].get_title() == "", "Unexpected title for the plot."

def test_plot_daily_stats_wave_height_fill():
    # Test the function with significant wave height data (HS) and fill_between option
    fig = plots.plot_daily_stats(ds, var="HS", show=["25%", "75%", "mean"], 
                                 fill_between=["25%", "75%"], fill_color_like="mean",output_file="")
    
    # Check that the output is a Matplotlib Figure
    assert isinstance(fig, plt.Figure), "The output is not a Matplotlib Figure."
    
def test_plot_daily_stats_missing_column():
    # Test with a missing column (should raise an error or handle it gracefully)
    try:
        plots.plot_daily_stats(ds, var="non_existent_column",output_file="")
        assert False, "The function did not raise an error for a missing column."
    except KeyError:
        print("test_plot_daily_stats_missing_column passed (KeyError raised as expected).")

def test_plot_hourly_stats_wind_speed():
    # Test the function with wind speed data (W10)
    fig = plots.plot_hourly_stats(ds, var="W10", show=["min", "mean", "max"],output_file="")

    # Check that the output is a Matplotlib Figure
    assert isinstance(fig, plt.Figure), "The output is not a Matplotlib Figure."
    
    # Ensure the figure has the correct title if provided
    assert fig.axes[0].get_title() == "", "Unexpected title for the plot."

def test_plot_hourly_stats_wave_height_fill():
    # Test the function with significant wave height data (HS) and fill_between option
    fig = plots.plot_hourly_stats(ds, var="HS", show=["25%", "75%", "mean"], 
                                 fill_between=["25%", "75%"], fill_color_like="mean",output_file="")
    
    # Check that the output is a Matplotlib Figure
    assert isinstance(fig, plt.Figure), "The output is not a Matplotlib Figure."
    
def test_plot_hourly_stats_missing_column():
    # Test with a missing column (should raise an error or handle it gracefully)
    try:
        plots.plot_hourly_stats(ds, var="non_existent_column",output_file="")
        assert False, "The function did not raise an error for a missing column."
    except KeyError:
        print("test_plot_hourly_stats_missing_column passed (KeyError raised as expected).")

def test_plot_monthly_stats_wind_speed():
    # Test the function with wind speed data (W10)
    fig = plots.plot_monthly_stats(ds, var="W10", show=["min", "mean", "max"],output_file="")
    
    # Check that the output is a Matplotlib Figure
    assert isinstance(fig, plt.Figure), "The output is not a Matplotlib Figure."
    
    # Check the x-axis labels if `month_xticks=True`
    ax = fig.axes[0]
    labels = [label.get_text() for label in ax.get_xticklabels()]
    assert len(labels) == 12, f"Expected 12 x-tick labels for months, but found {len(labels)}."

    print("test_plot_monthly_stats_wind_speed passed.")

def test_plot_monthly_stats_wave_height_fill():
    # Test the function with significant wave height data (HS) and fill_between option
    fig = plots.plot_monthly_stats(ds, var="HS", show=["25%", "75%", "mean"], 
                                   fill_between=["25%", "75%"], fill_color_like="mean",output_file="")
    
    # Check that the output is a Matplotlib Figure
    assert isinstance(fig, plt.Figure), "The output is not a Matplotlib Figure."
    
def test_plot_monthly_stats_missing_column():
    # Test with a missing column (should raise an error or handle it gracefully)
    try:
        plots.plot_monthly_stats(ds, var="non_existent_column",output_file="")
        assert False, "The function did not raise an error for a missing column."
    except KeyError:
        print("test_plot_monthly_stats_missing_column passed (KeyError raised as expected).")

def test_plot_monthly_stats_month_xticks():
    # Test the `month_xticks` option to ensure month labels are shown correctly
    fig = plots.plot_monthly_stats(ds, var="W10", month_xticks=True,output_file="")
    # Check that the output is a Matplotlib Figure
    assert isinstance(fig, plt.Figure), "The output is not a Matplotlib Figure."


def test_plot_taylor_diagram(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)  # override plt.show to no-op
    fig = plots.taylor_diagram(ds,var_ref=['HS'],var_comp=['HS.1','HS.2'],norm_std=True,output_file="")
    # Check that the output is a Matplotlib Figure
    assert isinstance(fig, plt.Figure), "The output is not a Matplotlib Figure."
    plt.close(fig)

# def test_plot_environmental_contours_hs_tp():
#     figures = plot_environmental_contours(ds,'HS','TP',config='DNVGL_hs_tp',save_path='total_sea_hs_tp_')
#     assert isinstance(figures[0],plt.Figure)

# def test_plot_environmental_contours_U_hs():
#     figures = plot_environmental_contours(ds,'HS','W10',config='DNVGL_hs_U',save_path='joint_U_hs_')
#     assert isinstance(figures[0],plt.Figure)


def test_plot_cca_profile():
    fig = plots.plot_cca_profiles(ds_ocean,var='current_speed_',month=None,return_period=10,output_file="")
    # Check that the output is a Matplotlib Figure
    assert isinstance(fig, plt.Figure), "The output is not a Matplotlib Figure."

def test_plot_spectra_1d():
    output_file = 'test_plot_monthly_spectra_1d.png'
    fig = plots.plot_spectra_1d(data=ds_synthetic_spectra, var='SPEC', period=None, month=None, method='mean', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)

def test_plot_spectrum_2d():
    output_file = 'test_plot_spectrum_2d.png'
    fig = plots.plot_spectrum_2d(data=ds_synthetic_spectra, var='SPEC', period=None, month=None, method='mean', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)

def test_plot_diana_spectrum():
    output_file = 'test_plot_diana_spectrum.png'
    fig = plots.plot_diana_spectrum(data=ds_synthetic_spectra, var='SPEC', period=None, month=None, method='mean', partition=False, output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)

def test_plot_partition_peak_dir_freq_2d():
    output_file = 'test_wave_partition_peak_dir_freq_2d.png'
    fig = plots.plot_partition_peak_dir_freq_2d(data=ds_synthetic_spectra, period=None, month=None, hm0_threshold=1, windsea_freq_mask_percentile=None, swell_freq_mask_percentile=None, output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)

def test_plot_spectra_2d():
    output_file = 'test_plot_dir_mean_2dspectrum.png'
    fig = plots.plot_spectra_2d(data=ds_synthetic_spectra, var='SPEC', method='monthly_mean', output_file=output_file)
    if os.path.exists(output_file):
        os.remove(output_file)

