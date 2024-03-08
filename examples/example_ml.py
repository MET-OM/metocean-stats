from metocean_api import ts
from metocean_stats.stats import ml_stats

df_norac = ts.TimeSeries(lon=5.07, lat=62.22,start_time='2018-01-01', end_time='2019-12-31' ,product='NORAC_wave')
df_norac.load_data(local_file=df_norac.datafile)
df_nora3 = ts.TimeSeries(lon=5.07, lat=62.22,start_time='2000-01-01', end_time='2020-12-31' ,product='NORA3_wave_sub')
df_nora3.load_data(local_file=df_nora3.datafile)

model='GBR' # 'SVR_RBF', 'LSTM'
var_origin = ['hs','tp','Pdir']
ts_pred = ml_stats.predict_ts(ts_origin=df_nora3.data,var_origin=var_origin,ts_train=df_norac.data.loc['2019-01-01':'2019-01-31'],var_train=['hs'], model=model)

print(ts_pred)