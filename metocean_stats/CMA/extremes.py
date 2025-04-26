import pandas as pd
import numpy as np
import pyextremes

def get_return_values(data: pd.Series,
                      distribution: str,
                      return_period: list[float],
                      threshold: float = None,
                      r = "48h",
                      block_size = "365.244D",
                      ):
    """
    Calculate return values from return periods using pyextremes.

    Parameters
    ------------
    data : pd.Series
        The timeseries of data to analyze.
    distribution : str
        A distribution from scipy. Currently implemented are
         - genpareto (POT)
         - expon (POT)
         - genextreme (BM)
         - gumbel_r (BM)
    """

    model = pyextremes.EVA(data)
    if distribution in ["genpareto","expon"]:
        if threshold is None:
            threshold = np.percentile(data,99)
        model.get_extremes(method="POT",threshold=threshold,r=r)
    elif distribution in ["genextreme","gumbel_r"] or distribution.startswith("gumbel"):
        model.get_extremes(method="BM",block_size=block_size)
    else:
        raise ValueError("Distribution must be one of: [genpareto, expon, genextreme, gumbel_r].")
    
    model.fit_model(model="MLE",distribution=distribution)

    return model.get_return_value(return_period=return_period)[0]

def get_dependent_given_marginal(model, 
                                 given:np.ndarray,
                                 percentile:np.ndarray=0.5,
    ):
    """
    Function to get values of dependent/secondary variable from a joint model,
    given specific values of the marginal/primary variable. 
    Optionally, a percentile may be specified to get a conditional value other than the median.
    """
    # Check that percentile is 1 number.
    percentile = np.squeeze(np.array(percentile))
    if percentile.size > 1: 
        raise ValueError("Only one percentile may be specified at a time.")
    
    given = np.squeeze(np.array(given))
    if given.ndim > 1:
        raise ValueError("Given must be a 0 or 1-dimensional array.")
    given = np.array([given,np.zeros_like(given)]).T

    return model.conditional_icdf(p=percentile,dim=1,given=given)