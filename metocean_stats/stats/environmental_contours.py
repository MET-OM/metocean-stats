import virocon
import pandas as pd
import numpy as np

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
        "intervals": virocon.NumberOfIntervalsSlicer(16),
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
    if preset == 'DNVGL_hs_tp':
        dist_descriptions, fit_descriptions, semantics = description_DNVGL_hs_tp()
    if preset == 'OMAE_U_hs':
        dist_descriptions, fit_descriptions, semantics = description_OMAE_U_hs()
    if preset == 'DNVGL_hs_U':
        dist_descriptions, fit_descriptions, semantics = description_DNVGL_hs_U()
    if preset == 'windsea_hs_tp':
        dist_descriptions, fit_descriptions, semantics = description_windsea_hs_tp()

    # Fit and return model
    model = virocon.GlobalHierarchicalModel(dist_descriptions)
    model.fit(data,fit_descriptions)
    return model, semantics


