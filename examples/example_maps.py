from metocean_stats import plots, tables, stats, maps
from metocean_stats.stats.aux_funcs import *

# Map:
maps.plot_extreme_wave_map(return_period=100, product='NORA3', title='100-yr return values Hs (NORA3)', set_extent = [0,30,52,73],output_file='wave_100yrs.png')

maps.plot_extreme_wind_map(return_period=100, product='NORA3',z=100, title='100-yr return values Wind at 100 m (NORA3)', set_extent = [0,30,52,73], output_file='wind_100yrs_100m.png')

maps.plot_extreme_wind_map(return_period=100, product='NORA3',z=10, title='100-yr return values Wind at 10 m (NORA3)', set_extent = [0,30,52,73], output_file='wind_100yrs_10m.png')

