from pathlib import Path
from metocean_stats import maps
import warnings
warnings.filterwarnings("ignore")

folder = Path(__file__).parent / 'maps'
if not folder.exists():
    folder.mkdir(parents=True)

# Map:
maps.plot_points_on_map(lon=[3.35,3.10], lat=[60.40,60.90],label=['NORA3','NORKYST800'],bathymetry='NORA3',output_file= folder /'map.png')
maps.plot_extreme_wave_map(return_period=100, product='NORA3', title='100-yr return values Hs (NORA3)', set_extent = [0,30,52,73],output_file= folder / 'wave_100yrs.png')
maps.plot_extreme_wind_map(return_period=100, product='NORA3',z=100, title='100-yr return values Wind at 100 m (NORA3)', set_extent = [0,30,52,73], output_file=folder / 'wind_100yrs_100m.png')
maps.plot_extreme_wind_map(return_period=100, product='NORA3',z=10, title='100-yr return values Wind at 10 m (NORA3)', set_extent = [0,30,52,73], output_file= folder / 'wind_100yrs_10m.png')
