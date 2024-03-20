import matplotlib.pyplot as plt
import sklearn
from metocean_api import ts
from metocean_stats.stats import ml_stats

# Import norac data
df_norac = ts.TimeSeries(lon=6.08, lat=62.4,start_time='2018-01-01', end_time='2019-12-31' ,product='NORAC_wave')
#df_norac.import_data(save_csv=True)
df_norac.load_data(local_file=df_norac.datafile)
# Import nora3 data
df_nora3 = ts.TimeSeries(lon=6.08, lat=62.4,start_time='2015-01-01', end_time='2019-12-31' ,product='NORA3_wave_sub')
#df_nora3.import_data(save_csv=True)
df_nora3.load_data(local_file=df_nora3.datafile)

# Define training and validation period:
start_training = '2019-01-01'
end_training   = '2019-12-31'
start_valid    = '2018-01-01'
end_valid      = '2018-12-31'

# Select method and variables for ML model:
model='GBR' # 'SVR_RBF', 'LSTM', GBR
var_origin = ['hs','tp','Pdir']
var_train  = ['hs']
# Run ML model:
ts_pred = ml_stats.predict_ts(ts_origin=df_nora3.data,var_origin=var_origin,ts_train=df_norac.data.loc[start_training:end_training],var_train=var_train, model=model)


# Plotting a month of data:
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(12, 6),gridspec_kw={'top': 0.95,'bottom': 0.150,'left': 0.05,'right': 0.990,'hspace': 0.2,'wspace': 0.2})
plt.title('Model: '+model+',Training Variables: '+','.join(var_origin))
plt.plot(df_nora3.data['hs'].loc['2017-12-30':'2018-01-30'],'o',label='NORA3')
plt.plot(ts_pred.loc['2017-12-30':'2018-01-30'],'x',label='NORAC_pred')
plt.ylabel('Hs[m]',fontsize=20)
plt.plot(df_norac.data['hs'].loc['2017-12-30':'2018-01-30'].asfreq('H'),'.',label='NORAC')
plt.grid()
plt.legend()
plt.savefig(model+'-'+'_'.join(var_origin)+'ts.png')
plt.close()

#Plot all the data:
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(12, 6),gridspec_kw={'top': 0.95,'bottom': 0.150,'left': 0.05,'right': 0.990,'hspace': 0.2,'wspace': 0.2})
plt.title('Model: '+model+',Training Variables: '+','.join(var_origin))
plt.plot(df_nora3.data['hs'],'o',label='NORA3')
plt.plot(ts_pred,'x',label='NORAC_pred')
plt.ylabel('Hs[m]',fontsize=20)
plt.plot(df_norac.data['hs'],'.',label='NORAC')
plt.grid()
plt.legend()
plt.savefig(model+'-'+'_'.join(var_origin)+'ts_all.png')
plt.close()

# Scatter plot and metrics:
plt.scatter(df_norac.data['hs'].loc[start_valid:end_valid], ts_pred.loc[start_valid:end_valid], color='black')
plt.title('scatter:'+model+'-'+'_'.join(var_origin))
plt.text(0, 1.0,'ΜΑΕ:'+str(sklearn.metrics.mean_absolute_error(df_norac.data['hs'].loc[start_valid:end_valid], ts_pred.loc[start_valid:end_valid]).round(3)))
plt.text(0, 0.8,'$R²$:'+str(sklearn.metrics.r2_score(df_norac.data['hs'].loc[start_valid:end_valid], ts_pred.loc[start_valid:end_valid]).round(3)))
plt.text(0, 0.6,'RMSE:'+str((sklearn.metrics.mean_squared_error(df_norac.data['hs'].loc[start_valid:end_valid], ts_pred.loc[start_valid:end_valid])**0.5).round(3)))
plt.xlabel('Hs from NORAC')
plt.ylabel('Hs from NORAC_pred')
plt.savefig(model+'-'+'_'.join(var_origin)+'scatter.png')
plt.close()
