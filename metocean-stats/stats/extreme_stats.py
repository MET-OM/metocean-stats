from typing import Dict, Sequence
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from virocon import get_OMAE2020_Hs_Tz, GlobalHierarchicalModel,IFORMContour
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

def joint_2D_contour(hs: xr.DataArray,tp: xr.DataArray, return_periods=[50,100], image_path=None, image_title: str = None) -> Sequence[Dict]:
    """Compute a joint contour for the given return periods. Optionally plot the contour and save the image.
    Input:
        hs: Hs values
        tp: Tp values
        return_periods: A list of return periods in years
        image_path: Path to save the image (optional)
        image_title: Title of the image (optional)
    """
    # Define joint distribution model
    dist_descriptions, fit_descriptions, _ = get_OMAE2020_Hs_Tz()
    model = GlobalHierarchicalModel(dist_descriptions)
    # Perform fitting
    data =  np.transpose(np.array([hs,tp]))
    model.fit(data, fit_descriptions=fit_descriptions)
    # Create a list of contours for each return period
    dt = 1  # duration in hours
    contours = []
    for rp in return_periods:
        alpha = 1 / (rp * 365.25 * 24 / dt)
        contour = IFORMContour(model, alpha)
        coords = contour.coordinates
        x = coords[:, 1].tolist()
        y = coords[:, 0].tolist()
        contours.append({
            'return_period': rp,
            'x': x,
            'y': y
        })

    # Optionally plot the contours and save the image
    if image_path:
        import seaborn as sns
        _, ax = plt.subplots()
        sns.set_theme(style="ticks")
        sns.scatterplot(x=tp, y=hs, ax=ax)
        color = plt.cm.rainbow(np.linspace(0, 1, len(return_periods)))
        # Compute an IFORM contour with a return period of N years
        dt = 1  # duration in hours
        #loop over contours and also get index

        for i,contour in enumerate(contours):
            # Plot the contour
            x = []
            x.extend(contour['x'])
            x.append(x[0])

            y = []
            y.extend(contour['y'])
            y.append(y[0])

            rp = contour['return_period']
            ax.plot(x, y, color=color[i],label=str(rp)+'y')

        if hs.name and tp.name:
            ax.set_xlabel(hs.name)
            ax.set_ylabel(tp.name)
        plt.grid()
        if image_title:
            plt.title(image_title)
        plt.legend()
        plt.savefig(image_path)

    return contours
