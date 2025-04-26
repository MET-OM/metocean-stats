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
