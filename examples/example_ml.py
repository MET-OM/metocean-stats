import matplotlib.pyplot as plt
from metocean_api import ts
df_norac = ts.TimeSeries(lon=5.07, lat=62.22,start_time='2018-01-01', end_time='2019-12-31' ,product='NORAC_wave')
df_norac.load_data(local_file=df_norac.datafile)
df_nora3 = ts.TimeSeries(lon=5.07, lat=62.22,start_time='2000-01-01', end_time='2020-12-31' ,product='NORA3_wave_sub')
df_nora3.load_data(local_file=df_nora3.datafile)
# Sample data initialization
# Assuming you have pandas DataFrames for NORA3 and NORAC
# Example DataFrame creation
from metocean_stats.stats import ml_stats
ts_pred = ml_stats.predict_ts(ts_origin=df_nora3,var_origin='hs',ts_train=df_norac,var_train='hs', model='LinearRegression')


# Plotting the actual vs predicted values
#plt.scatter(X_test, y_test, color='black')
#plt.plot(Hs_NORA3, Hs_NORAC_pred, color='blue', linewidth=3)
#plt.title('Actual vs Predicted Values for Hs from NORAC based on NORA3')
#plt.xlabel('Hs from NORA3')
#plt.ylabel('Hs from NORAC')
#plt.show()

plt.plot(df_nora3.data['hs'],'o',label='NORA3')
plt.plot(ts_pred,'x',label='NORAC_pred')
plt.plot(df_norac.data['hs'],'.',label='NORAC_train')
plt.legend()
plt.show()