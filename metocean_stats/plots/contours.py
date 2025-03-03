import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import virocon

########################
# Dependence functions
########################

def _dependence_logistic():
    '''
    4 paramter logistic (S) curve.
    '''
    def _logistic(x, a=1, b=1, c=-1, d=1):
        return a + b / (1 + np.exp(c * (x - d)))

    bounds = [(0, None), (0, None), (None, 0), (0, None)]
    weights = lambda x,y: y
    latex = "$a + b / (1 + \exp[c * (x -d)])$"

    return virocon.DependenceFunction(
        func = _logistic,
        bounds = bounds,
        weights = weights,
        latex = latex
    )


def _dependence_double_logistic():
    '''
    Superposition of two logistic (S) curves - one increasing and one decreasing.
    '''
    def _double_logistic(x, a=1, b1=1, c1=-1, d1=1, b2=1, c2=-1, d2=1):
        return a + (b1 / (1 + np.exp(c1 * (x - d1)))) - (b2 / (1 + np.exp(c2 * (x - d2))))

    bounds = [(0, None), (0, None), (None, 0), (0, None),(0, None), (None, 0), (0, None)]
    weights = lambda x,y: y
    latex = "$a + b1 / (1 + \exp[c1 * (x -d1)]) - b2 / (1 + \exp[c2 * (x -d2)])$"

    return virocon.DependenceFunction(
        func = _double_logistic,
        bounds = bounds,
        weights = weights,
        latex = latex
    )

def _dependence_power():
    '''
    Power law. 
    
    From DNV GL (2017) recommended practice, 3.6.3.
    There, it is used to describe the mu (location) parameter 
    of the wave period lognormal distribution, as dependent on wave height.
    '''
    def _power(x, a, b, c):
        return a + b * x**c

    bounds = [(0, None), (0, None), (None, None)]
    latex="$a + b * x^c$"
    return virocon.DependenceFunction(
        func = _power, 
        bounds = bounds, 
        latex = latex)

def _dependence_exp():
    '''
    Exponential dependence. 
    
    From DNV GL (2017) recommended practice, 3.6.3.
    There, it is used to describe the sigma (scale) parameter 
    of wave period lognormal distribution, as dependent on wave height.
    '''
    def _exp3(x, a, b, c):
        return a + b * np.exp(c * x)

    bounds = [(0, None), (0, None), (None, None)]
    latex="$a + b * \exp(c * x)$"
    return virocon.DependenceFunction(
        func=_exp3,
        bounds= bounds, 
        latex=latex)



#######################
# Model descriptions
#######################

def description_DNVGL_hs_tp():
    '''
    DNVGL (2017) model for hs and tp.
    '''
    dist_descriptions, fit_descriptions, _ = virocon.get_DNVGL_Hs_Tz()
    semantics = {
        'names': ['Significant wave height','Wave peak period'],
        'symbols': ['H_s','T_p'],
        'units': ['m','s']
    }
    return dist_descriptions, fit_descriptions, semantics

def description_OMAE_hs_tp():
    '''
    OMAE (2020) model for hs and tp.
    '''
    dist_descriptions, fit_descriptions, _ = virocon.get_OMAE2020_Hs_Tz()
    semantics = {
        'names': ['Significant wave height','Wave peak period'],
        'symbols': ['H_s','T_p'],
        'units': ['m','s']
    }
    return dist_descriptions, fit_descriptions, semantics

def description_DNVGL_hs_U():
    '''
    DNVGL (2017) model for hs and wind speed (dependent).
    '''
    return virocon.get_DNVGL_Hs_U()

def description_OMAE_U_hs():
    '''
    OMAE (2020) model for wind speed and hs (dependent).
    '''
    return virocon.get_OMAE2020_V_Hs()

def description_windsea_hs_tp():
    '''
    Joint model of weibull (hs) and exp. weibull (tp), 
    which should work well for wind sea.
    Parameters alpha and beta in the dependent
    distribution are modelled as power laws; a+bx**c,
    where x is the dependency wave height.
    '''

    dist_description_hs = {
        "distribution": virocon.WeibullDistribution(),
        "intervals": virocon.NumberOfIntervalsSlicer(13),
    }
    dist_description_tp = {
        "distribution": virocon.ExponentiatedWeibullDistribution(f_delta=20),
        "conditional_on": 0,
        "parameters": {
            "alpha": _dependence_power(),
            "beta": _dependence_power()
        },
    }

    dist_descriptions = [dist_description_hs, dist_description_tp]

    fit_description_hs = None
    fit_description_tp = {"method": "wlsq", "weights": "quadratic"}
    fit_descriptions = [fit_description_hs, fit_description_tp]

    semantics = {
        "names": ["Significant wave height", "Peak wave period"],
        "symbols": ["H_s", "T_p"],
        "units": ["m","s"],
    }

    return dist_descriptions, fit_descriptions, semantics



#############
# Procedures
#############

def fit_joint_model(
        data:pd.DataFrame,
        preset:str=None,
        var1:str|int=0,
        var2:str|int=1,
        ):
    '''
    Function to fit 2D distribution on a set of samples.

    This is a simplified interface to the virocon package [1]_,
    offering some commonly used configurations. For more detailed
    configuration options, use the virocon package directly [2]_.

    Parameters
    ------------
    data : pd.DataFrame
        Array of data.
    preset : str
        Preset configurations for different cases:

            - DNVGL_hs_tp: weibull and lognormal
            - OMAE_hs_tp: exp. weibull and lognormal
            - DNVGL_hs_U: weibull and weibull
            - OMAE_U_hs: exp. weibull and exp. weibull
            - windsea_hs_tp: weibull and exp. weibull, 
                works well for wind sea

    var1 : str or int, default: first column
        The primary (independent) variable, 
        as column name (str) or position (integer). 
        This corresponds to the first variable in either 
        the simplified config string or the custom descriptions.
    var2 : str or int, default: second column
        The secondary (dependent) variable,
        as column name (str) or position (integer).
        This corresponds to the second variable from either 
        the simplified config string or the custom descriptions.
    dist_descriptions : list[dict]
        Description of the distributions. 
        See virocon documentation for more info [2]_.
    fit_descriptions : list[dict]
        Description of the distribution fitting procedure.
        See virocon documentation for more info [2]_.
    semantics : dict
        Description of variables, mainly for labels on plots.
        See viricon documentation for more info [2]_.
    
    Returns
    ----------
    model : virocon.GlobalHierarchicalModel
        The fitted joint model.
    semantics : dict
        Description of the parameters, used in plotting.

    References
    -----------
    .. [1] https://github.com/virocon-organization/virocon [MIT]

    .. [2] https://virocon.readthedocs.io/en/latest/index.html .
            
    '''

    # Set table columns to [var1, var2]
    if type(var1) == int:
        var1 = data.columns[var1]
    if type(var2) == int:
        var2 = data.columns[var2]
    data = data[[var1,var2]]

    # Load preset
    if preset == 'OMAE_hs_tp':
        dist_descriptions, fit_descriptions, semantics = description_OMAE_hs_tp()
    elif preset == 'DNVGL_hs_tp':
        dist_descriptions, fit_descriptions, semantics = description_DNVGL_hs_tp()
    elif preset == 'OMAE_U_hs':
        dist_descriptions, fit_descriptions, semantics = description_OMAE_U_hs()
    elif preset == 'DNVGL_hs_U':
        dist_descriptions, fit_descriptions, semantics = description_DNVGL_hs_U()
    elif preset == 'windsea_hs_tp':
        dist_descriptions, fit_descriptions, semantics = description_windsea_hs_tp()
    else:
        raise ValueError(f'Preset {preset} not found. See docstring for available presets.')

    # Fit and return model
    model = virocon.GlobalHierarchicalModel(dist_descriptions)
    model.fit(data,fit_descriptions)
    return model, semantics



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
    For more detailed configuration options, the virocon package should be consulted directly.
    This module can then be used as an example.

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

    if type(var1) == int:
        var1 = data.columns[var1]
    if type(var2) == int:
        var2 = data.columns[var2]
    data = data[[var1,var2]]

    model, semantics = fit_joint_model(data,preset=config)

    if config in ['DNVGL_hs_tp','OMAE_hs_tp','DNVGL_hs_U','windsea_hs_tp']:
        swap_axes = True
    else:
        swap_axes = False

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