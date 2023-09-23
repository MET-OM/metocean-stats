import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from xclim.indices.stats import frequency_analysis

def return_value(values: xr.DataArray,return_periods=[50,100], month=[1,2,3,4,5, 6, 7, 8, 9, 10,11,12], image_path=None) -> xr.DataArray:
    """
    Compute values corresponding to the given return periods. Optionally plot the return values and save the image.
    """
    time = values.time
    #time = values.index
    r_val = frequency_analysis(values, t=return_periods, dist="genextreme", mode="max", freq="YS", month=month)
    if image_path:
        color = plt.cm.rainbow(np.linspace(0, 1, len(r_val.return_period)))
        _, ax = plt.subplots()
        values.plot(color='grey')
        for i in range(len(r_val.return_period)):
            lbl = str(r_val.values[i].round(1))+r_val.units+'('+str(r_val.return_period[i].values)+'y)'
            ax.hlines(y=r_val[i], xmin=time[0], xmax=time[-1],color=color[i] ,linestyle='dashed', linewidth=2,label=lbl )
        plt.grid()
        plt.legend(loc='center left')
        plt.title('Return value')
        plt.tight_layout()
        plt.savefig(image_path)

    return r_val

