import matplotlib.pyplot as plt
import sklearn
from metocean_api import ts
from metocean_stats.stats import ml
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings('ignore')

ds = xr.open_dataset('ww3_spec_E#LonN#Lat_NORA3_Sula_20180101T0000-20180630T2300.nc')

# Define training and validation period:
start_training = '2018-01-01'
end_training   = '2018-05-31'
start_valid    = '2018-06-01'
end_valid      = '2018-06-30'

# Select method and variables for ML model:
model='LSTM' # 'SVR_RBF', 'LSTM', GBR
var_origin = ['efth']
var_train  = ['efth']
station_origin = 2 # A location 62.427002, 6.044994
station_train = 1 # B location 62.404, 6.079001
frequency = 0
direction = 0

# Run ML model:
ds_ml = xr.full_like(ds[var_origin[0]][:,station_train,:,:], fill_value=np.nan)
for frequency in range(len(ds_ml.frequency)):
    for direction in range(len(ds_ml.direction)):    
        ts_pred = ml.predict_ts(ts_origin=ds[var_origin[0]][:,station_origin,frequency,direction].to_dataframe(),var_origin=var_origin,
                                ts_train=ds[var_train[0]][:,station_train,frequency,direction].to_dataframe().loc[start_training:end_training],var_train=var_train, model=model)
        ds_ml[:,frequency,direction] = np.squeeze(ts_pred)

ds_ml.to_netcdf('ml_spec.nc')
ds_ml = xr.open_dataset('ml_spec.nc')
#breakpoint()
fig, axs = plt.subplots(1, 2, figsize=(10, 5)) 
ds_ml[var_train[0]][-50,:,:].plot(ax=axs[0])  
ds[var_train[0]][-50,station_train,:,:].plot(ax=axs[1])
plt.show()  

breakpoint()
