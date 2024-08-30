import os
from metocean_stats import maps

#def test_plot_points_on_map():
#     output_file = 'test_points.png'
#     fig = maps.plot_points_on_map(lon=[3.35, 3.10], lat=[60.40, 60.90], label=['NORA3', 'NORKYST800'], bathymetry='NORA3', output_file=output_file)
#     if int(fig.axes[0].lines[0].get_xdata()[0]) == 3:
#         pass
#     else:
#         raise ValueError("FigValue is not correct")
#    
#     # Check if file is created
#     if os.path.isfile(output_file):
#         os.remove(output_file)
#     else:
#         raise FileNotFoundError(f"{output_file} was not created")
#     return

#def test_plot_extreme_wave_map():
#     output_file = 'test_wave.png'
#     fig = maps.plot_extreme_wave_map(return_period=50, product='NORA3', title='50-yr return values Hs (NORA3)', set_extent=[0, 30, 52, 73], output_file=output_file)
#     if fig.dpi == 100.0:
#         pass
#     else:
#         raise ValueError("FigValue is not correct")
#   
#     # Check if file is created
#     if os.path.isfile(output_file):
#         os.remove(output_file)
#     else:
#         raise FileNotFoundError(f"{output_file} was not created")
#     return

#def test_plot_extreme_wind_map():
#     output_file = 'test_wind.png'
#     fig = maps.plot_extreme_wind_map(return_period=100, product='NORA3', z=100, title='100-yr return values Wind at 10 m (NORA3)', set_extent=[0, 30, 52, 73], output_file=output_file)
#     if fig.dpi == 100.0:
#         pass
#     else:
#         raise ValueError("FigValue is not correct")
#    
#     # Check if file is created
#     if os.path.isfile(output_file):
#         os.remove(output_file)
#     else:
#         raise FileNotFoundError(f"{output_file} was not created")
#     return

#def test_plot_mean_air_temperature_map():
#     output_file = 'test_temp.png'
#     fig = maps.plot_mean_air_temperature_map(product='NORA3', title='Mean 2-m air temperature 1991-2020 (NORA3)', set_extent=[-25, -10, 60.5, 68], unit='degC', output_file=output_file)
#     if fig.dpi == 100.0:
#         pass
#     else:
#         raise ValueError("FigValue is not correct")
#    
#     # Check if file is created
#     if os.path.isfile(output_file):
#         os.remove(output_file)
#     else:
#         raise FileNotFoundError(f"{output_file} was not created")
#     return