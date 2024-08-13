from metocean_api import ts
from metocean_stats import plots, tables, stats
from metocean_stats.stats.aux_funcs import *
from metocean_stats.stats import ml
import pandas as pd
import os

# Define TimeSeries-object for NORA3
ds = readNora10File('tests/data/NORA_test.txt')
#ds_ocean = pd.read_csv('tests/data/NorkystDA_test.csv',comment='#',index_col=0, parse_dates=True)
#depth = ['0m', '1m', '2.5m', '5m', '10m', '15m', '20m', '25m', '30m', '40m', '50m', '75m', '100m', '150m', '200m']


def test_predict_ts_GBR(ds=ds):
    ml.predict_ts(ts_origin=ds,var_origin=['W10','TP','DIRP'],ts_train=ds.loc['1960-01-01':'1960-12-31'],var_train=['HS'], model='GBR')

def test_predict_ts_SVR(ds=ds):
    ml.predict_ts(ts_origin=ds,var_origin=['W10','TP','DIRP'],ts_train=ds.loc['1960-01-01':'1960-12-31'],var_train=['HS'], model='SVR_RBF')

#def test_predict_ts_LSTM(ds=ds):
#    ml.predict_ts(ts_origin=ds,var_origin=['W10','TP','DIRP'],ts_train=ds.loc['1960-01-01':'1960-12-31'],var_train=['HS'], model='LSTM')

  