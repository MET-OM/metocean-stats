import virocon
import numpy as np

import virocon.dependencies
from virocon.predefined import (
    #get_DNVGL_Hs_Tz,
    get_OMAE2020_Hs_Tz,
    get_DNVGL_Hs_U,
    get_OMAE2020_V_Hs
)

from .distributions import (
    LoNoWeDistribution,
    VonMisesDistribution,
    GeneralizedNormalDistribution,
)

from .intervals import(
    WidthSlicer,
    NumberSlicer
)

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

def get_weiblognormal_independent_Hs_Tz():
    dist_description_hs = {
        "distribution": virocon.WeibullDistribution(),
    }

    dist_description_tz = {
        "distribution": virocon.LogNormalDistribution(),
    }
    dist_descriptions = [dist_description_hs, dist_description_tz]
    fit_descriptions = [{"method": "mom"}, {"method":"mle"}]
    semantics = {
        "names": ["Significant wave height", "Peak wave period"],
        "symbols": ["H_s", "T_p"],
        "units": ["m", "s"],
        "swap_axis":True
    }
    return dist_descriptions, fit_descriptions, semantics


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
    def _power3(x, a=0, b=1.5, c=0.5):
        return a + b * x**c

    def _exp3(x, a,b,c):
        return a + b * np.exp(c * x)


    def _linear(x,a,b):
        return a+b*x

    def _asymdecrease3(x, a, b, c):
        return a + b / (1 + c * x)

    def _lnsquare2(x, a, b, c):
        return np.log(a + b * np.sqrt(c*x))


    bounds_mu = [(0, None), (None, None), (None, None)]
    bounds_sigma = [(0, None), (0, None), (None, None)]
    
    dist_description_hs = {
        "distribution": virocon.WeibullDistribution(),
        #"intervals": virocon.NumberOfIntervalsSlicer(25,min_n_points=50), # 0.5 m is a common choice, see e.g. 10.1115/OMAE2020-18041
        "intervals": virocon.NumberOfIntervalsSlicer(n_intervals=15,min_n_points=10)
    }

    dist_description_tz = {
        "distribution": virocon.LogNormalDistribution(),
        "conditional_on": 0,
        "parameters": {
            "mu": virocon.DependenceFunction(_lnsquare2, bounds_mu, latex="$a + b * x^c$"), 
            #"mu": virocon.DependenceFunction(_power3, bounds_mu, latex="$a + b * x^c$"), 
            "sigma": virocon.DependenceFunction(_exp3, bounds_sigma, latex="$a + b * \exp(c * x)$")
            },
    }

    dist_descriptions = [dist_description_hs, dist_description_tz]

    fit_descriptions = [{"method": "mom"}, {"method":"mle"}]

    semantics = {
        "names": ["Significant wave height", "Peak wave period"],
        "symbols": ["H_s", "T_p"],
        "units": ["m", "s"],
        "swap_axis":True
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

def get_weibweib_hs_tp():
    """
    Joint model of weibull (hs) and weibull (tp), 
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
        "distribution": virocon.WeibullDistribution(f_gamma=1.2),
        "conditional_on": 0,
        "parameters": {
            "alpha": virocon.DependenceFunction(_power3, bounds, latex="$a + b * x^c$"), 
            "beta": virocon.DependenceFunction(_power3, bounds, latex="$a + b * x^c$"), 
        },
    }

    dist_descriptions = [dist_description_hs, dist_description_tp]

    fit_description_hs = {"method":"mom"}
    fit_description_tp = {"method": "mle"}
    fit_descriptions = [fit_description_hs, fit_description_tp]

    semantics = {
        "names": ["Significant wave height", "Peak wave period"],
        "symbols": ["H_s", "T_p"],
        "units": ["m","s"],
        "swap_axis":True
    }

    return dist_descriptions, fit_descriptions, semantics


def get_weibexp_hs_tp():
    """
    Joint model of weibull (hs) and exp. weibull (tp), 
    intended for wind-sea.
    """

    def _power3(x, a, b, c):
        return a + b * x**c

    def _exp3(x, a, b, c):
        return a + b * np.exp(c * x)

    def _asymdecrease3(x, a, b, c):
        return a + b / (1 + c * x)

    def _lnsquare2(x, a, b, c):
        return np.log(a + b * np.sqrt(np.divide(x, 9.81)))

    mu_bounds = [(None, None), (None, None), (None, None)]
    sigma_bounds = [(0, None), (0, None), (None, None)]


    dist_description_hs = {
        "distribution": virocon.ExponentiatedWeibullDistribution(),
        "intervals": virocon.NumberOfIntervalsSlicer(10),
    }
    dist_description_tp = {
        "distribution": virocon.LogNormalDistribution(),
        "conditional_on": 0,
        "parameters": {
            "mu": virocon.DependenceFunction(_lnsquare2, mu_bounds, latex="$a + b * x^c$"), 
            "sigma": virocon.DependenceFunction(_exp3, sigma_bounds, latex="$a + b * \exp(c * x)$")
            },
    }

    dist_descriptions = [dist_description_hs, dist_description_tp]

    fit_description_hs = {"method": "wlsq", "weights": "quadratic"}
    fit_description_tp = {"method":"mom"}
    fit_descriptions = [fit_description_hs, fit_description_tp]

    semantics = {
        "names": ["Significant wave height", "Peak wave period"],
        "symbols": ["H_s", "T_p"],
        "units": ["m","s"],
        "swap_axis":True
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
        "names": ["Wind speed", "Significant wave height", "Peak wave period"],
        "symbols": ["V", "H_s", "T_p"],
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

    def _alpha(x, a1=2.0, a2=0.01, a3=1.0):
        return a1 + a2 * x ** a3
    def _beta(x, b1=2.0, b2=0.01, b3=1.0):
        return b1 + b2 * x ** b3
    def _mu(x, c1=0, c2=1, c3=0.6):
        return c1 + c2 * x ** c3
    def _linear(x,a,b,c):
        return a+b*x+c*x*x
    def _sigma(x, d1=0.001, d2=0.1, d3=-0.3):
        return d1 + d2 * np.exp(d3 * x)
    def _lnsquare2(x, a, b, c):
        return np.log(a + b * np.sqrt(c*x))


    #logistics_bounds = [(None, None), (0, None), (None, 0), (0, None)]
    bounds = [(None, None), (0, None), (0, None)]
    mu_bounds = [(None, None), (0, None), (0, 1)]
    sigma_bounds = [(0, None), (0, None), (None, 0)]

    dist_description_0 = {
        "distribution": virocon.WeibullDistribution(),
        "intervals": virocon.NumberOfIntervalsSlicer(25,min_n_points=50),
    }

    dist_description_1 = {
        "distribution": virocon.WeibullDistribution(f_gamma=0),
        "intervals": virocon.NumberOfIntervalsSlicer(15,min_n_points=50),
        "conditional_on": 0,
        "parameters": {
            "alpha": virocon.DependenceFunction(_alpha, bounds, latex="$a1 + a2 * x^a3$"), 
             "beta": virocon.DependenceFunction(_beta, bounds, latex="$b1 + b2 * x^b3$"),
             #"gamma":virocon.DependenceFunction(_linear, [(None,None),(None,None)], latex="$a + b * \exp(c * x)$")
            },
    }

    dist_description_2 = {
        "distribution": virocon.LogNormalDistribution(),
        "conditional_on": 1,
        "parameters": {
            "mu":   virocon.DependenceFunction(_lnsquare2,bounds, latex="$c1 + c2 * x^c3$"),
            "sigma":virocon.DependenceFunction(_sigma,   sigma_bounds, latex="$d1 + d2 * \exp(d3 * x)$")},
    }

    dist_descriptions = [dist_description_0,dist_description_1,dist_description_2]

    fit_descriptions = [{'method':"mle"},
                        {'method':"mle"},
                        {'method':"mle"}]

    semantics = {
        "names": ["Wind speed", "Significant wave height", "Peak wave period"],
        "symbols": ["W_s", "H_s", "T_p"],
        "units": ["m/s", "m", "s"],
        "swap_axis":True
    }

    return dist_descriptions, fit_descriptions, semantics

def get_cheynet_wind_misalignment():
    """
    A weibull-normal model intended to describe distribution 
    of wind speed and misalignment between wind and wave direction.
    """

    def _mu(x, e1, e2, e3):
        return e1 + e2 * np.power(x,e3,out=np.zeros_like(x),where=x>0)
    def _sigma(x, f1, f2, f3):
        return f1 + f2 * np.power(x,f3,out=np.zeros_like(x),where=x>0)

    def _logistics4(x, f1=1, f2=-1, f3=-1, f4=1):
        return f1 + f2 / (1 + np.exp(f3 * (x - f4)))

    bounds = [(None, None), (0, None), (None, None)]
    logistics_bounds = [(0, None), (None, 0), (None, 0), (0, None)]

    dist_description_ws = {
        "distribution": virocon.WeibullDistribution(),
        "intervals": virocon.NumberOfIntervalsSlicer(30),
    }
    dist_description_delta = {
        "distribution": virocon.NormalDistribution(),
        "conditional_on": 0,
        "parameters": {
            #"mu": virocon.DependenceFunction(_mu,   bounds, latex="$e1 + e2 * x^e3$"),
            "mu":virocon.DependenceFunction(
                _logistics4, logistics_bounds,latex="$a + b / (1 + \exp[c * (x -d)])$"),
            #"sigma": virocon.DependenceFunction(_sigma, bounds, latex="$f1 + f2 * x^f3$"),
            "sigma": virocon.DependenceFunction(
                _logistics4, logistics_bounds,latex="$a + b / (1 + \exp[c * (x -d)])$")
        },
    }

    dist_descriptions = [dist_description_ws, dist_description_delta]

    fit_description_ws = {"method":"mom"}
    fit_description_delta = {"method": "mom"}
    fit_descriptions = [fit_description_ws, fit_description_delta]

    semantics = {
        "names": ["Wind speed", "Misalignment"],
        "symbols": ["W_s", r"\delta"],
        "units": ["m/s",r"\textdegree"],
        "swap_axis":True
    }

    return dist_descriptions, fit_descriptions, semantics

def get_vonmises_wind_misalignment():
    """
    A weibull-vonmises model intended to describe distribution 
    of wind speed and misalignment between wind and wave direction.
    """

    def _mu(x, e1, e2, e3):
        #return e1 + e2 * np.power(x,e3,out=np.zeros_like(x),where=x>0)
        return e1 + e2 * np.power(x,e3)
    def _sigma(x, f1, f2, f3):
        return f1 + f2 * np.power(x,f3,out=np.zeros_like(x),where=x>0)
    def _linear(x,a,b):
        return a+b*x
    def _constant(x,a):
        return np.ones_like(x)*a
    def _scale(x,a,b):
        return a+b*x
    def _kappa(x, f1=1, f2=-1, f3=-1, f4=1):
        return f1 + f2 / (1 + np.exp(f3 * (x - f4)))
        
    bounds = [(None, None), (0, None), (None, None)]
    logistics_bounds = [(0, None), (None, None), (None, 0), (0, None)]

    dist_description_ws = {
        "distribution": virocon.WeibullDistribution(),
        "intervals": virocon.WidthOfIntervalSlicer(
            1,value_range=(0,20),min_n_points=10),
    }
    # dist_description_delta = {
    #     "distribution": GeneralizedNormalDistribution(f_scale=np.pi),
    #     "conditional_on": 0,
    #     "parameters": {
    #         "beta":virocon.DependenceFunction(_mu,latex="$f1 + f2 / (1 + \exp[f3 * (x -f4)])$"),
    #         "loc":virocon.DependenceFunction(_linear),
    #         #"scale":virocon.DependenceFunction(_linear,latex="$e1 + e2 * x^e3$"),
            
    #     },
    # }
    dist_description_delta = {
        "distribution": VonMisesDistribution(),
        "conditional_on": 0,
        "parameters": {
            "kappa":virocon.DependenceFunction(_kappa,logistics_bounds,latex="$f1 + f2 / (1 + \exp[f3 * (x -f4)])$"),
            "mu":virocon.DependenceFunction(_linear,latex="$a+bx$"),
        },
    }

    dist_descriptions = [dist_description_ws, dist_description_delta]

    fit_description_ws = {"method":"mom"}
    fit_description_delta = {"method": "mle"}
    fit_descriptions = [fit_description_ws, fit_description_delta]

    semantics = {
        "names": ["Wind speed", "Misalignment"],
        "symbols": ["W_s", r"\Delta"],
        "units": ["m/s",r"degrees"],
        "swap_axis":True
    }

    return dist_descriptions, fit_descriptions, semantics

def get_cT_Hs_Tp():
    """
    A model describing dependency of Hs and Tp, on current aligned
    in the direction of the waves [-Cs, Cs]
    """

    def _alpha(x, a1=2.0, a2=0.01, a3=1.0):
        return a1 + a2 * x ** a3
    def _beta(x, b1=2.0, b2=0.01, b3=1.0):
        return b1 + b2 * x ** b3
    def _mu(x, c1=2.0, c2=0.01, c3=1.0):
        return c1 + c2 * x ** c3
    def _sigma(x, d1=0.001, d2=0.1, d3=-0.3):
        return d1 + d2 * np.exp(d3 * x)
    
    def _linear(x,a,b):
        return a+b*x
    
    def _piecewise_linear(x, a=-1, b=0.1, c=1, d=0.03):
        return  np.where(x <= b,
                d + a * (x - b),
                d + c * (x - b))

    def _poly4(x,a,b,c,d,e):
        x = x-e
        return a + d*x**4
    def _poly2(x,a,b,c):
        return a + b*x + c*x*x

    def _mu(x, c1=2.0, c2=0.01, c3=1.0):
        return c1 + c2 * x ** c3
    def _sigma(x, d1=0.001, d2=0.1, d3=-0.3):
        return d1 + d2 * np.exp(d3 * x)
    
    #logistics_bounds = [(None, None), (0, None), (None, 0), (0, None)]
    bounds = [(None, None), (0, None), (0, None)]
    gauss_bounds = [(-10,0),(-10,10),(-10,10)]
    sigma_bounds = [(0, None), (0, None), (None, 0)]
    poly_bounds = [(0,None),(None,None),(None,None)]
    bounds = None
    #sigma_bounds = None
    
    dist_description_0 = {
        #"distribution": (),
        "distribution":GeneralizedNormalDistribution(),
        "intervals": virocon.WidthOfIntervalSlicer(
            width=0.1,value_range=[-2,2],min_n_points=50),
    }

    dist_description_1 = {
        "distribution": virocon.WeibullDistribution(),
        "intervals": virocon.NumberOfIntervalsSlicer(15,min_n_points=20),
        "conditional_on": 0,
        "parameters": {
            "alpha": virocon.DependenceFunction(_linear, bounds, latex="$a + b * x$"),
            "beta": virocon.DependenceFunction(_linear,latex="$a + b*x + c*x^2$"),
            "gamma":virocon.DependenceFunction(_linear)
            },
    }

    dist_description_2 = {
        "distribution": virocon.LogNormalDistribution(),
        "conditional_on": 1,
        "parameters": {
            "mu":   virocon.DependenceFunction(_mu, bounds, latex="$c1 + c2 * x^c3$"),
            "sigma":virocon.DependenceFunction(_sigma,   sigma_bounds, latex="$d1 + d2 * \exp(d3 * x)$")},
    }

    dist_descriptions = [dist_description_0,dist_description_1,dist_description_2]

    fit_descriptions = [{'method':"mle"},
                        {'method':"mom"},
                        {'method':"mom"}]

    semantics = {
        "names": ["Surface current aligned", "Significant wave height", "Peak wave period"],
        "symbols": ["U", "H_s", "T_p"],
        "units": ["m/s", "m", "s"],
        "swap_axis":True
    }

    return dist_descriptions, fit_descriptions, semantics

def get_Hs_Tp_cT():
    """
    A model describing dependency of cT on Hs and Tp on Hs.
    """

    def _alpha(x, a1=2.0, a2=0.01, a3=1.0):
        return a1 + a2 * x ** a3
    def _beta(x, b1=2.0, b2=0.01, b3=1.0):
        return b1 + b2 * x ** b3
    def _mu(x, c1=2.0, c2=0.01, c3=1.0):
        return c1 + c2 * x ** c3
    def _sigma(x, d1=0.001, d2=0.1, d3=-0.3):
        return d1 + d2 * np.exp(d3 * x)
    
    def _linear(x,a,b):
        return a+b*x
    
    def _exp(x,a=1,b=-0.5,c=0.1):
        return a*np.exp(b*x)+c

    def _piecewise_linear(x, a, b, c, d):
        e = (a + b * c) - d * c
        left = a + b * x
        right = e+ d * x
        return np.where(x < c, left, right)    
    
    def _gauss(x,b=-1,c=0.5,d=1):
        return 1.5+b * np.exp(-np.divide((x-c)**2,2*d*d))

    def _poly2(x,a,b,c):
        return a + b*x + c*x*x

    def _mu(x, c1=2.0, c2=0.01, c3=1.0):
        return c1 + c2 * x ** c3
    def _sigma(x, d1=0.001, d2=0.1, d3=-0.3):
        return d1 + d2 * np.exp(d3 * x)
    def _wavy(x,a,b,c):
        return x*np.exp(-b*(x-c)**2)
    
    def _logistics4(x, a=1, b=1, c=-1, d=1):
        return a + b / (1 + np.exp(c * (x - d)))


    #logistics_bounds = [(None, None), (0, None), (None, 0), (0, None)]
    bounds = [(None, None), (0, None), (0, None)]
    gauss_bounds = [(-10,0),(-10,10),(-10,10)]
    sigma_bounds = [(0, None), (0, None), (None, 0)]
    poly_bounds = [(0,None),(None,None),(None,None)]
    bounds = None
    #sigma_bounds = None
    seagull_bounds = [(None,None),(None,None),(None,0),(None,0),(-1,1)]
    seagull_sym_bounds = [(None,None),(None,0),(None,0),(-1,1)]

    dist_description_0 = {
        "distribution": virocon.WeibullDistribution(),
        "intervals": virocon.NumberOfIntervalsSlicer(40,min_n_points=2),
    }

    dist_description_1 = {
        "distribution": virocon.LogNormalDistribution(),
        "conditional_on": 0,
        "parameters": {
            "mu":   virocon.DependenceFunction(_mu, bounds, latex="$c1 + c2 * x^c3$"),
            "sigma":virocon.DependenceFunction(_sigma,   sigma_bounds, latex="$d1 + d2 * \exp(d3 * x)$")},
    }

    dist_description_2 = {
        #"distribution": (),
        "distribution":virocon.NormalDistribution(),
        #"intervals": virocon.NumberOfIntervalsSlicer(50,min_n_points=10),
        "conditional_on": 0,
        "parameters": {
            #"gamma":virocon.DependenceFunction(_gamma,latex="a+x"),
            "mu":virocon.DependenceFunction(_linear),
            "sigma":virocon.DependenceFunction(_logistics4),
            #"beta": virocon.DependenceFunction(_linear, bounds, latex="$b1 + b2 * x^b3$"),
            #"loc": virocon.DependenceFunction(_linear, bounds, latex="$b1 + b2 * x^b3$"),
            #"scale": virocon.DependenceFunction(_linear, bounds, latex="$b1 + b2 * x^b3$"),
            },
    }
    # dist_description_2 = {
    #     #"distribution": (),
    #     "distribution":GeneralizedNormalDistribution(),
    #     #"intervals": virocon.NumberOfIntervalsSlicer(50,min_n_points=10),
    #     "conditional_on": 0,
    #     "parameters": {
    #         "beta":virocon.DependenceFunction(_linear),
    #         "loc":virocon.DependenceFunction(_linear),
    #         "scale":virocon.DependenceFunction(_linear),
    #         #"beta": virocon.DependenceFunction(_linear, bounds, latex="$b1 + b2 * x^b3$"),
    #         #"loc": virocon.DependenceFunction(_linear, bounds, latex="$b1 + b2 * x^b3$"),
    #         #"scale": virocon.DependenceFunction(_linear, bounds, latex="$b1 + b2 * x^b3$"),
    #         },
    # }

    dist_descriptions = [dist_description_0,dist_description_1,dist_description_2]

    fit_descriptions = [{'method':"mom"},
                        {'method':"mom"},
                        {'method':"mle"}]

    semantics = {
        "names": ["Significant wave height","Peak wave period","Surface current alignment"],
        "symbols": ["H_s", "T_p", "cT"],
        "units": ["m", "s", "m/s"],
        "swap_axis":True
    }

    return dist_descriptions, fit_descriptions, semantics
