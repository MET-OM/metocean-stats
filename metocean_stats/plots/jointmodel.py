import pandas as pd
import numpy as np
from ..CMA import JointProbabilityModel

"""
This module is just an interface to the CMA (conditional modelling approach) module (metocean_stats/CMA),
which in turn builds upon the virocon package. 

This module is for quick and simple analysis. For a full set of options,
and possibility of creating custom distributions and models,
use either the CMA module or virocon directly.
"""

def plot_joint_model_isodensity_contours(data:pd.DataFrame,
                                         var1="hs",
                                         var2="tp",
                                         model="DNVGL_HS_TP",
                                         levels=np.logspace(-10,-1,10),
                                         cmap=None):
    """
    Fit a pre-defined conditional joint model to the data,
    and plot the contours of constant probability density.

    Parameters
    -----------
    data : pd.DataFrame
        Dataframe containing columns var1 and var2.
    var1 : str or int
        The first variable. If a string, a column name, if integer, a column number.
    var2 : str or int
        The second variable. If a string, a column name, if integer, a column number.
    model : str
        The name of a predefined model. Options:

             - DNVGL_Hs_Tp
             - DNVGL_Hs_U
             - OMAE_Hs_Tp
             - OMAE2020_V_Hs
             - LoNoWe_Hs_Tp
             - windsea_hs_tp
    levels : list of floats
        A list of probability densities.

    Returns
    -------
    matplotlib axes object.
    """
    model = JointProbabilityModel(model)
    model.fit(data,var1,var2)
    return model.plot_2D_isodensity(levels=levels,cmap=cmap)

def plot_joint_model_iform_contours(data:pd.DataFrame,
                                    var1="hs",
                                    var2="tp",
                                    model="DNVGL_HS_TP",
                                    periods=[1,10,100]):
    """
    Fit a pre-defined conditional joint model to the data,
    and plot the IFORM contours.

    Parameters
    -----------
    data : pd.DataFrame
        Dataframe containing columns var1 and var2.
    var1 : str or int
        The first variable. If a string, a column name, if integer, a column number.
    var2 : str or int
        The second variable. If a string, a column name, if integer, a column number.
    model : str
        The name of a predefined model. Options:

             - DNVGL_Hs_Tp
             - DNVGL_Hs_U
             - OMAE_Hs_Tp
             - OMAE2020_V_Hs
             - LoNoWe_Hs_Tp
             - windsea_hs_tp
    periods : list of floats
        A list of return periods (in years).

    Returns
    -------
    matplotlib axes object.
    """
    model = JointProbabilityModel(model)
    model.fit(data,var1,var2)
    return model.plot_2D_contours(periods=periods)

def plot_joint_model_marginal_quantiles(data:pd.DataFrame,
                                        var1="hs",
                                        var2="tp",
                                        model="DNVGL_HS_TP"):
    """
    Fit a pre-defined conditional joint model to the data,
    and plot a comparison of empirical (data) quantiles 
    to theoretical (fitted model) quantiles.

    Parameters
    -----------
    data : pd.DataFrame
        Dataframe containing columns var1 and var2.
    var1 : str or int
        The first variable. If a string, a column name, if integer, a column number.
    var2 : str or int
        The second variable. If a string, a column name, if integer, a column number.
    model : str
        The name of a predefined model. Options:

             - DNVGL_Hs_Tp
             - DNVGL_Hs_U
             - OMAE_Hs_Tp
             - OMAE2020_V_Hs
             - LoNoWe_Hs_Tp
             - windsea_hs_tp

    Returns
    -------
    matplotlib axes object.
    """
    model = JointProbabilityModel(model)
    model.fit(data,var1,var2)
    return model.plot_marginal_quantiles()

def plot_joint_model_dependence_functions(data:pd.DataFrame,
                                          var1="hs",
                                          var2="tp",
                                          model="DNVGL_HS_TP"):
    """
    Fit a pre-defined conditional joint model to the data,
    and plot the functions describing the dependency of 
    the statistical variables in the joint model.

    Parameters
    -----------
    data : pd.DataFrame
        Dataframe containing columns var1 and var2.
    var1 : str or int
        The first variable. If a string, a column name, if integer, a column number.
    var2 : str or int
        The second variable. If a string, a column name, if integer, a column number.
    model : str
        The name of a predefined model. Options:

             - DNVGL_Hs_Tp
             - DNVGL_Hs_U
             - OMAE_Hs_Tp
             - OMAE2020_V_Hs
             - LoNoWe_Hs_Tp
             - windsea_hs_tp

    Returns
    -------
    matplotlib axes object.
    """
    model = JointProbabilityModel(model)
    model.fit(data,var1,var2)
    return model.plot_dependence_functions()

def plot_joint_model_interval_histograms(data:pd.DataFrame,
                                        var1="hs",
                                        var2="tp",
                                        model="DNVGL_HS_TP"):
    """
    Fit a pre-defined conditional joint model to the data,
    and plot histograms for each interval of the primary variable,
    e.g., a histogram of Tp for each bin of Hs.

    Parameters
    -----------
    data : pd.DataFrame
        Dataframe containing columns var1 and var2.
    var1 : str or int
        The first variable. If a string, a column name, if integer, a column number.
    var2 : str or int
        The second variable. If a string, a column name, if integer, a column number.
    model : str
        The name of a predefined model. Options:

             - DNVGL_Hs_Tp
             - DNVGL_Hs_U
             - OMAE_Hs_Tp
             - OMAE2020_V_Hs
             - LoNoWe_Hs_Tp
             - windsea_hs_tp

    Returns
    -------
    matplotlib axes object.
    """

    model = JointProbabilityModel(model)
    model.fit(data,var1,var2)
    return model.plot_histograms_of_interval_distributions()

def plot_joint_model_pdf_heatmap(data:pd.DataFrame,
                                var1="hs",
                                var2="tp",
                                model="DNVGL_HS_TP",
                                **kwargs):
    """
    Fit a pre-defined conditional joint model to the data,
    and plot a heatmap of the 2D probability density function.

    Parameters
    -----------
    data : pd.DataFrame
        Dataframe containing columns var1 and var2.
    var1 : str or int
        The first variable. If a string, a column name, if integer, a column number.
    var2 : str or int
        The second variable. If a string, a column name, if integer, a column number.
    model : str
        The name of a predefined model. Options:

             - DNVGL_Hs_Tp
             - DNVGL_Hs_U
             - OMAE_Hs_Tp
             - OMAE2020_V_Hs
             - LoNoWe_Hs_Tp
             - windsea_hs_tp

    Returns
    -------
    matplotlib axes object.
    """
    model = JointProbabilityModel(model)
    model.fit(data,var1,var2)
    return model.plot_2D_pdf_heatmap(**kwargs)
