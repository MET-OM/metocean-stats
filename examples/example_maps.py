from pathlib import Path
from metocean_stats import maps
import warnings
warnings.filterwarnings("ignore")

folder = Path(__file__).parent / 'maps'
if not folder.exists():
    folder.mkdir(parents=True)

# The following functions will illustrate statistics in a map format:
maps.plot_points_on_map(lon=[3.35,3.10], lat=[60.40,60.90],label=['NORA3','NORKYST800'],bathymetry='NORA3',output_file= folder /'map.png')

maps.plot_extreme_wave_map(return_period=100, product='NORA3', title='100-Year Return Values of Significant Wave Height (NORA3)', 
                            set_extent = [0, 30, 52, 73], output_file='wave_100yrs_hexbin.png', method='hexbin', percentile_contour='min')
maps.plot_extreme_wave_map(return_period=100, product='NORA3', title='100-Year Return Values of Significant Wave Height (NORA3)', 
                            set_extent = [0, 30, 52, 73], output_file='wave_100yrs_contour.png', method='contour', percentile_contour='min')

maps.plot_extreme_wind_map(return_period=100, product='NORA3', z=100, title=f'100-Year Return Values of Wind Speed at 100 m Height (NORA3)',
                            set_extent=[0, 30, 52, 73], output_file=f'wind_100yrs_100m_hexbin.png', land_mask=True, method='hexbin', percentile_contour=40)
maps.plot_extreme_wind_map(return_period=100, product='NORA3', z=100, title=f'100-Year Return Values of Wind Speed at 100 m Height (NORA3)',
                            set_extent=[0, 30, 52, 73], output_file=f'wind_100yrs_100m_contour.png', land_mask=True, method='contour', percentile_contour=40)

maps.plot_extreme_current_map(return_period=100, z='surface', distribution='gumbel', product='NORA3', title=f'100-Year Return Values of Surface Current Speed (NORA3)',
                            set_extent = [-2, 30, 52, 76], output_file='current_100yrs_surface_hexbin.png', method='hexbin', percentile_contour=60)
maps.plot_extreme_current_map(return_period=100, z='surface', distribution='gumbel', product='NORA3', title=f'100-Year Return Values of Surface Current Speed (NORA3)',
                            set_extent = [-2, 30, 52, 76], output_file='current_100yrs_surface_contour.png', method='contour', percentile_contour=60)