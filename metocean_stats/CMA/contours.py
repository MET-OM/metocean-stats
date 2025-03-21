import matplotlib.pyplot as plt
import pandas as pd
import virocon
from .joint_models import JointProbabilityModel

# This module is no longer useful, but can eventually become an example / test.

def plot_environmental_contours(
        data:pd.DataFrame,
        var1:str,
        var2:str,
        config:str = 'DNVGL_hs_tp',
        save_path = 'joint_distribution',
        return_periods = [1,10,100],
        sea_state_duration:float = 1,
        plot_marginal_quantiles = True,
        plot_interval_histograms = True,
        plot_dependency_functions = True,
        plot_isodensity = True
        ):
    
    """
    Fit 2D joint probability model and produce associated plots.
    This only offers some default configurations.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing var_hs and var_T as columns.
    var1 : str
        The wave height column.
    var2 : str
        The wave period column.
    config : str
        Preset configurations for different cases:

            - DNVGL_hs_tp: weibull and lognormal
            - OMAE_hs_tp: exp. weibull and lognormal
            - DNVGL_hs_U: weibull and weibull
            - OMAE_U_hs: exp. weibull and exp. weibull
            - windsea_hs_tp: weibull and exp. weibull, 
                works well for wind sea

    """

    if type(var1) is int:
        var1 = data.columns[var1]
    if type(var2) is int:
        var2 = data.columns[var2]
    data = data[[var1,var2]]

    model = JointProbabilityModel(config)
    model.fit(data)
    semantics = model.semantics
    swap_axes = model.swap_axis

    figures = []

    if return_periods:
        fig,ax = plt.subplots()

        for i,T_return in enumerate(return_periods):
            alpha = virocon.calculate_alpha(sea_state_duration,T_return)
            contour = virocon.IFORMContour(model, alpha)
            
            if i == 0: # only plot the data once
                virocon.plot_2D_contour(contour, data, semantics=semantics, swap_axis=swap_axes, ax=ax)
            else:
                virocon.plot_2D_contour(contour, semantics=semantics, swap_axis=swap_axes, ax=ax)
        
        figures.append(fig)
        if save_path:
            fig.savefig(save_path+'_contours.png')

    if plot_marginal_quantiles:
        try:
            fig,ax = plt.subplots(1,2)
            virocon.plot_marginal_quantiles(model,data,semantics,axes=ax)
            figures.append(fig)
            if save_path:
                fig.savefig(save_path+'_quantiles.png')
        except Exception as e:
            print(f"Warning - could not create quantiles plots \n: {e}")

    if plot_interval_histograms:
        try:
            fig,ax = virocon.plot_histograms_of_interval_distributions(model,data,semantics)
            for i,f in enumerate(fig):
                figures.append(f)
                if save_path:
                    f.savefig(save_path+'_histograms.png')
        except Exception as e:
            print(f"Warning - could not create histogram plots \n: {e}")

    if plot_dependency_functions:
        try:
            fig,ax = plt.subplots(1,2)
            virocon.plot_dependence_functions(model,semantics,axes=ax)
            figures.append(fig)
            if save_path:
                fig.savefig(save_path+'_dependences.png')
        except Exception as e:
            print(f"Warning - could not create dependence plots \n: {e}")
    
    if plot_isodensity:
        try:
            fig,ax = plt.subplots()
            virocon.plot_2D_isodensity(model,data,semantics,swap_axis=swap_axes, ax=ax)
            figures.append(fig)
            if save_path:
                fig.savefig(save_path+'_isodensity.png')
        except Exception as e:
            print(f"Warning - could not create isodensity plot \n: {e}")

    return figures