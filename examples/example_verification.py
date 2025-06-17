#from metocean_stats 
from metocean_stats import plots, tables, maps
from metocean_stats.stats.aux_funcs import *
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

df1 = readNora10File('../tests/data/NORA_test.txt') 

folder = Path(__file__).parent / 'output' / 'NORA'
if not folder.exists():
    folder.mkdir(parents=True)

####
var1 = 'HS'; var2 ='TP'
df_ref = df1[[var1,var2]]
df_comp = df1[[var2]].rename(columns={'TP': 'TP_1'})
df_comp2 = df1[[var2]].rename(columns={'TP': 'TP_2'})

df = pd.concat([df_ref, df_comp,df_comp2], axis=1)

tables.table_binned_error_metric(data=df,var_bin = 'HS', var_ref='TP', var_comp=['TP_1'] , var_bin_size=1,threshold_min=100, error_metric=['rmse','bias','mae','corr','si'],output_file='binned_error_metric.csv')
plots.plot_binned_error_metric(data=df,var_bin = 'HS',var_bin_unit='m', var_ref='TP', var_comp=['TP_1'], var_comp_unit='s', var_bin_size=0.5,threshold_min=100, plot_y='bias',title='Example title',output_file='plot_binned_error_metric.png')

