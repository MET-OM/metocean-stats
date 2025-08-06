import virocon
import numpy as np
from scipy.signal import find_peaks

from virocon.distributions import (
    Distribution,
    ScipyDistribution,
    NormalDistribution,
    WeibullDistribution,
    VonMisesDistribution,
    LogNormalDistribution,
    GeneralizedGammaDistribution,
    LogNormalNormFitDistribution,
    ExponentiatedWeibullDistribution
)

__all__ = [
    # Virocon's models
    "ScipyDistribution",
    "NormalDistribution",
    "WeibullDistribution",
    "VonMisesDistribution",
    "LogNormalDistribution",
    "GeneralizedGammaDistribution",
    "LogNormalNormFitDistribution",
    "ExponentiatedWeibullDistribution",
    # Custom models
    "LoNoWeDistribution",
]

class LoNoWeDistribution(Distribution):
    """
    A LoNoWe distribution (Lognormal + Weibull)

    Parameters
    ----------
    alpha : float, default 1
        Scale parameter of the weibull distribution.
    beta : float, default 1
        Shape parameter of the weibull distribution.
    gamma : float
        Location parameter of the weibull distribution (3-parameter weibull). Defaults to 0.

    f_alpha : float, default None
        Fixed scale parameter of the weibull distribution.
    f_beta : float, default None
       Fixed shape parameter of the weibull distribution.
    f_gamma : float, default None
        Fixed location parameter of the weibull distribution.
    
    mu : float, default 0
        Location (mean) parameter of the normal distribution.
        Defaults to 0.
    sigma : float, default 1
        Standard deviation of the normal distribution.
        Defaults to 1.

    f_mu : float, default None
        Fix the mu parameter of the lognormal distribution.
    f_sigma : float, default None
       Fix the sigma parameter of the lognormal distribution.
    """

    def __init__(
        self, shifting_point=1, f_shifting_point=None, alpha=1, beta=1, gamma=0, f_alpha=None, f_beta=None, f_gamma=None, mu=0, sigma=1, f_mu=None, f_sigma=None
    ):
        self.weibull = virocon.WeibullDistribution(alpha=alpha,beta=beta,gamma=gamma,f_alpha=f_alpha,f_beta=f_beta,f_gamma=f_gamma)
        self.lognormal =virocon.LogNormalDistribution(mu=mu,sigma=sigma,f_mu=f_mu,f_sigma=f_sigma)

        self.shifting_point = shifting_point if f_shifting_point is None else f_shifting_point
        self.f_shifting_point = f_shifting_point

    @property
    def parameters(self):

        return {"alpha": self.weibull.alpha, 
                "beta": self.weibull.beta, 
                "gamma": self.weibull.gamma,
                "mu":self.lognormal.mu,
                "sigma":self.lognormal.sigma}
    
    def cdf(self, x, alpha=None, beta=None, gamma=None, mu=None, sigma=None):
        
        x = np.array(x)
        cdf = np.zeros_like(x,dtype=float)

        weibull_mask = x<=self.shifting_point
        lognormal_mask = x>self.shifting_point

        cdf[lognormal_mask] = self.lognormal.cdf(x[lognormal_mask], mu=mu, sigma=sigma)
        cdf[weibull_mask] = self.weibull.cdf(x[weibull_mask], alpha=alpha, beta=beta, gamma=gamma)

        return cdf

    def icdf(self, prob, alpha=None, beta=None, gamma=None, mu=None, sigma=None):

        weibull_x = self.weibull.icdf(prob, alpha=alpha,beta=beta,gamma=gamma)
        
        lognormal_x = self.lognormal.icdf(prob, mu=mu, sigma=sigma)

        shift_mask = np.mean([lognormal_x, weibull_x],axis=0)<=self.shifting_point
        return np.where(shift_mask,lognormal_x,weibull_x)

    def pdf(self, x, alpha=None, beta=None, gamma=None, mu=None,sigma=None):

        x_low = x[x<=self.shifting_point]
        x_high = x[x>self.shifting_point]

        lognormal_pdf = self.lognormal.pdf(x_low, mu=mu, sigma=sigma)
        weibull_pdf = self.weibull.pdf(x_high, alpha=alpha, beta=beta, gamma=gamma)

        return np.concatenate([lognormal_pdf,weibull_pdf],axis=0)

    def draw_sample(self, n, alpha=None, beta=None, gamma=None, mu=None, sigma=None, *, random_state=None):

        lognormal_fraction = self.cdf(self.shifting_point)
        lognormal_n = np.ceil(lognormal_fraction*n).astype('int')
        weibull_n = n - lognormal_n

        lognormal_sample = np.array([])
        while len(lognormal_sample) < lognormal_n:
            lognormal_draw = self.lognormal.draw_sample(n, mu, sigma, random_state=random_state)
            lognormal_draw = lognormal_draw[lognormal_draw <= self.shifting_point]
            lognormal_sample = np.concatenate([lognormal_sample,lognormal_draw],axis=0)
        lognormal_sample = lognormal_sample[:lognormal_n]

        weibull_sample = np.array([])
        while len(weibull_sample) < weibull_n:
            weibull_draw = self.weibull.draw_sample(n, alpha, beta, gamma, random_state=random_state)
            weibull_draw = weibull_draw[weibull_draw > self.shifting_point]
            weibull_sample = np.concatenate([weibull_sample,weibull_draw],axis=0)
        weibull_sample = weibull_sample[:weibull_n]

        sample = np.concatenate([weibull_sample,lognormal_sample],axis=0)
        np.random.shuffle(sample)

        return sample

    def _fit_mom(self,sample):
        self.weibull._fit_mom(sample)
        self.lognormal._fit_mom(sample)
        self._fit_shifting_point(sample)

    def _fit_mle(self,sample):
        self.weibull._fit_mle(sample)
        self.lognormal._fit_mle(sample)
        self._fit_shifting_point(sample)

    def _fit_lsq(self, data, weights):
        raise NotImplementedError()
    
    def _fit_shifting_point(self,sample):

        data = np.sort(np.squeeze(sample))
        data = data[(data>0.5*np.max(data)) & (data<0.95*np.max(data))]
        
        weib_pdf = self.weibull.pdf(data)
        lono_pdf = self.lognormal.pdf(data)
        d_pdf = np.squeeze(np.abs(weib_pdf-lono_pdf))

        peaks, _ = find_peaks(-d_pdf)
        if not peaks.size: 
            print('Warning: No optimal shifting point found.')
            peaks = [np.argmin(d_pdf)]

        self.shifting_point = np.squeeze(data[peaks[-1]])

    def __repr__(self):
        return f"\n < {self.shifting_point:.3f}: {str(self.lognormal)}, \n > {self.shifting_point:.3f}: {str(self.weibull)} \n"

class GeneralizedNormalDistribution(ScipyDistribution):
    """
    A generalization of the normal distribution which adds a shape parameter beta.
    This is a symmetric distribution to model data which is similar to normal, 
    but has e.g. heavier or lighter tails.
    """
    scipy_dist_name = "gennorm"