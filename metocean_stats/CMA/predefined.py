import virocon
import numpy as np

from virocon.predefined import (
    get_DNVGL_Hs_Tz,
    get_OMAE2020_Hs_Tz,
    get_DNVGL_Hs_U,
    get_OMAE2020_V_Hs
)

from .distributions import LoNoWeDistribution

__all__ = [
    # Predefined in virocon
    "get_DNVGL_Hs_Tz",
    "get_OMAE2020_Hs_Tz",
    "get_DNVGL_Hs_U",
    "get_OMAE2020_V_Hs",
    # Custom implementation
    "get_LoNoWe_hs_tp",
    "get_windsea_hs_tp"
]

def semantics_hs_tp():
    return {
        'names': ['Significant wave height','Wave peak period'],
        'symbols': ['H_s','T_p'],
        'units': ['m','s']
    }

def semantics_hs_tz():
    return {
        "names": ["Significant wave height", "Zero-up-crossing period"],
        "symbols": ["H_s", "T_z"],
        "units": ["m", "s"],
    }

def _dependence_logistic():
    '''
    4 paramter logistic (S) curve.
    '''
    def _logistic(x, a=1, b=1, c=-1, d=1):
        return a + b / (1 + np.exp(c * (x - d)))

    bounds = [(0, None), (0, None), (None, 0), (0, None)]
    def weights(x, y): return y
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
    def weights(x, y): return y
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


def get_LoNoWe_hs_tp():
    """
    Similar to the DNV GL RP 2017 model, but uses the combined 
    Lognormal-Weibull model in Hs (LoNoWe) rather than just weibull.
    The model is fitted using the method of moment in scipy.
    """

    dist_description_hs = {
        "distribution": LoNoWeDistribution(),
        "intervals": virocon.NumberOfIntervalsSlicer(10),
    }

    dist_description_tz = {
        "distribution": virocon.LogNormalDistribution(),
        "conditional_on": 0,
        "parameters": {"mu": _dependence_power(), 
                       "sigma": _dependence_exp()},
    }

    dist_descriptions = [dist_description_hs, dist_description_tz]

    fit_descriptions = [{'method':'mom','weights':None},{'method':'mom','weights':None}]

    semantics = {
        "names": ["Significant wave height", "Peak wave period"],
        "symbols": ["H_s", "T_p"],
        "units": ["m", "s"],
    }
    
    return dist_descriptions,fit_descriptions,semantics

def get_windsea_hs_tp():
    """
    Joint model of weibull (hs) and exp. weibull (tp), 
    intended for wind-sea.
    """

    dist_description_hs = {
        "distribution": virocon.WeibullDistribution(),
        "intervals": virocon.NumberOfIntervalsSlicer(10),
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
