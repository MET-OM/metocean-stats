import virocon
import numpy as np

from virocon.predefined import (
    #get_DNVGL_Hs_Tz,
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
    "get_windsea_hs_tp",
    "get_virocon_V_Hs_Tz",
    "get_LiGaoMoan_U_hs_tp"
]

def get_DNVGL_Hs_Tz():
    """
    Get DNVGL significant wave height and wave period model.

    Get the descriptions necessary to create th significant wave height
    and wave period model as defined in DNVGL [1]_ in section 3.6.3.

    Returns
    -------
    dist_descriptions : list of dict
        List of dictionaries containing the dist descriptions for each dimension.
        Can be used to create a GlobalHierarchicalModel.
    fit_descriptions : None
        Default fit is used so None is returned.
        Can be passed to fit function of GlobalHierarchicalModel.
    semantics : dict
        Dictionary with a semantic description of the model.
        Can be passed to plot functions.

    References
    ----------
    .. [1] DNV GL (2017). Recommended practice DNVGL-RP-C205: Environmental
        conditions and environmental loads.

    """

    # DNVGL 3.6.3
    def _power3(x, a, b, c):
        return a + b * x**c

    def _exp3(x, a, b, c):
        return a + b * np.exp(c * x)

    bounds = [(None, None), (0, None), (None, None)]

    dist_description_hs = {
        "distribution": virocon.WeibullDistribution(),
        "intervals": virocon.NumberOfIntervalsSlicer(20,min_n_points=10), # 0.5 m is a common choice, see e.g. 10.1115/OMAE2020-18041
    }

    dist_description_tz = {
        "distribution": virocon.LogNormalDistribution(),
        "conditional_on": 0,
        "parameters": {
            "mu": virocon.DependenceFunction(_power3, bounds, latex="$a + b * x^c$"), 
            "sigma": virocon.DependenceFunction(_exp3, bounds, latex="$a + b * \exp(c * x)$")
            },
    }

    dist_descriptions = [dist_description_hs, dist_description_tz]

    fit_descriptions = [{"method": "MoM"}, None]

    semantics = {
        "names": ["Significant wave height", "Peak wave period"],
        "symbols": ["H_s", "T_p"],
        "units": ["m", "s"],
    }

    return dist_descriptions, fit_descriptions, semantics


def get_LoNoWe_hs_tp():
    """
    Similar to the DNV GL RP 2017 model, but uses the combined 
    Lognormal-Weibull model in Hs (LoNoWe) rather than just weibull.
    The model is fitted using the method of moment in scipy.
    """

    def _power3(x, a, b, c):
        return a + b * x**c

    def _exp3(x, a, b, c):
        return a + b * np.exp(c * x)

    bounds = [(None, None), (0, None), (None, None)]

    dist_description_hs = {
        "distribution": LoNoWeDistribution(),
        "intervals": virocon.NumberOfIntervalsSlicer(10),
    }

    dist_description_tz = {
        "distribution": virocon.LogNormalDistribution(),
        "conditional_on": 0,
        "parameters": {
            "mu": virocon.DependenceFunction(_power3, bounds, latex="$a + b * x^c$"), 
            "sigma": virocon.DependenceFunction(_exp3, bounds, latex="$a + b * \exp(c * x)$")
            },
    }

    dist_descriptions = [dist_description_hs, dist_description_tz]

    fit_descriptions = [{'method':'mom'},
                        {'method':'mom'}]

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

    def _power3(x, a, b, c):
        return a + b * x**c

    bounds = [(None, None), (0, None), (None, None)]


    dist_description_hs = {
        "distribution": virocon.WeibullDistribution(),
        "intervals": virocon.NumberOfIntervalsSlicer(10),
    }
    dist_description_tp = {
        "distribution": virocon.ExponentiatedWeibullDistribution(f_delta=20),
        "conditional_on": 0,
        "parameters": {
            "alpha": virocon.DependenceFunction(_power3, bounds, latex="$a + b * x^c$"), 
            "beta": virocon.DependenceFunction(_power3, bounds, latex="$a + b * x^c$"), 
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

def get_virocon_V_Hs_Tz():

    def _power3(x, a, b, c):
        return a + b * x ** c

    def _exp3(x, a, b, c):
        return a + b * np.exp(c * x)

    def _alpha3(x, a, b, c, d_of_x):
        return (a + b * x ** c) / 2.0445 ** (1 / d_of_x(x))

    def _logistics4(x, a=1, b=1, c=-1, d=1):
        return a + b / (1 + np.exp(c * (x - d)))

    bounds = [(None, None), (0, None), (None, None)]
    logistics_bounds = [(None, None), (0, None), (None, 0), (0, None)]

    power3 = virocon.DependenceFunction(_power3, bounds, latex="$a + b * x^c$")
    exp3 = virocon.DependenceFunction(_exp3, bounds, latex="$a + b * \exp(c * x)$")
    logistics4 = virocon.DependenceFunction(_logistics4, logistics_bounds,weights=lambda x, y: y,latex="$a + b / (1 + \exp[c * (x -d)])$")
    alpha3 = virocon.DependenceFunction(_alpha3, bounds, d_of_x=logistics4,weights=lambda x, y: y,latex="$(a + b * x^c) / 2.0445^{1 / F()}$")

    dist_description_0 = {
        "distribution": virocon.ExponentiatedWeibullDistribution(),
        "intervals": virocon.NumberOfIntervalsSlicer(10),
    }

    dist_description_1 = {
        "distribution": virocon.ExponentiatedWeibullDistribution(f_delta=5),
        "intervals": virocon.NumberOfIntervalsSlicer(10),
        "conditional_on": 0,
        "parameters": {
            "alpha": alpha3, 
            "beta": logistics4,
            },
    }

    dist_description_2 = {
        "distribution": virocon.LogNormalDistribution(),
        "conditional_on": 1,
        "parameters": {
            "mu": power3, 
            "sigma": exp3},
    }

    dist_descriptions = [dist_description_0,dist_description_1,dist_description_2]

    fit_descriptions = [{'method':'mom','weights':None},
                        {'method':'mom','weights':None},
                        {'method':'mom','weights':None}]

    semantics = {
        "names": ["Wind speed", "Significant wave height", "Zero-up-crossing period"],
        "symbols": ["V", "H_s", "T_z"],
        "units": ["m/s", "m", "s"],
    }

    return dist_descriptions, fit_descriptions, semantics

def get_LiGaoMoan_U_hs_tp():

    """
    A 3D joint model for wind, wave height and period.

    Notes
    ------------
    Li, L, Gao, Z, & Moan, T. "Joint Environmental Data at Five European Offshore Sites 
    for Design of Combined Wind and Wave Energy Devices." Proceedings of the ASME 2013 
    32nd International Conference on Ocean, Offshore and Arctic Engineering. 
    Volume 8: Ocean Renewable Energy. Nantes, France. June 9–14, 2013. V008T09A006. ASME. 
    https://doi.org/10.1115/OMAE2013-10156
    """

    def _power3(x, a=2.0, b=0.01, c=1.0):
        return a + b * x ** c

    def _exp3(x, a=0.001, b=0.1, c=-0.3):
        return a + b * np.exp(c * x)

    def _exp2(x,a=-0.1,b=10):
        return a*np.exp(b*x)

    def _power2(x,a=-0.1,b=2):
        return a*x**b

    def _linear(x,a,b):
        return a+b*x

    def _logistics4(x, a=1, b=1, c=-1, d=1):
        return a + b / (1 + np.exp(c * (x - d)))

    #logistics_bounds = [(None, None), (0, None), (None, 0), (0, None)]
    bounds = [(None, None), (0, None), (0, None)]
    sigma_bounds = [(None, None), (0, None), (None, 0)]

    dist_description_0 = {
        "distribution": virocon.WeibullDistribution(),
        "intervals": virocon.NumberOfIntervalsSlicer(25,min_n_points=50),
    }

    dist_description_1 = {
        "distribution": virocon.WeibullDistribution(f_gamma=0),
        "intervals": virocon.NumberOfIntervalsSlicer(15,min_n_points=20),
        "conditional_on": 0,
        "parameters": {
            "alpha": virocon.DependenceFunction(_power3, bounds, latex="$a + b * x^c$"), 
             "beta": virocon.DependenceFunction(_power3, bounds, latex="$a + b * x^c$"),
             #"gamma":virocon.DependenceFunction(_linear, [(None,None),(None,None)], latex="$a + b * \exp(c * x)$")
            },
    }

    dist_description_2 = {
        "distribution": virocon.LogNormalDistribution(),
        "conditional_on": 1,
        "parameters": {
            "mu":   virocon.DependenceFunction(_power3, bounds, latex="$a + b * x^c$"),
            "sigma":virocon.DependenceFunction(_exp3,   sigma_bounds, latex="$a + b * \exp(c * x)$")},
    }

    dist_descriptions = [dist_description_0,dist_description_1,dist_description_2]

    fit_descriptions = [{'method':"mle"},
                        {'method':"mle"},
                        {'method':"mle"}]

    semantics = {
        "names": ["Wind speed", "Significant wave height", "Zero-up-crossing period"],
        "symbols": ["V", "H_s", "T_z"],
        "units": ["m/s", "m", "s"],
    }

    return dist_descriptions, fit_descriptions, semantics