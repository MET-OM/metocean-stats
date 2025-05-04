from virocon.contours import Contour
from virocon import GlobalHierarchicalModel, TransformedModel
import numpy as np
import scipy.stats as sts
"""
Classes for environmental contours.

This is a temporary simplification of the virocon respective module, 
to work with until changes can eventually be merged there.
"""

__all__ = [
    "IFORMContour",
    "ISORMContour",
]

class NSphere:

    def __init__(self, dim, n_samples, distribute_evenly=False):
        """
        Parameters
        ----------
        dim : int
            The number of dimensions. (i.e. the n in n-sphere plus 1)
        n_samples : int
            The number of points to distribute on the n-sphere.
        distribute_evenly : bool, default True
            Use N-dimensional spacing algorithm to distribute points evenly on the sphere.

        """
        self.dim = dim
        self.n_samples = n_samples
        if distribute_evenly is True: raise NotImplementedError("Use virocon for even spacing algorithm.")
        self.unit_sphere_points = self._random_unit_sphere_points()

    def _random_unit_sphere_points(self):
        """
        Generates equally distributed points on the sphere's surface.

        Note
        ----
        Used algorithm:

        - Use a N(0,1) standard normal distribution to generate cartesian coordinate vectors.
        - Normalize the vectors to length 1.
        - The points then are uniformly distributed on the sphere's surface.

        """
        # create pseudorandom number generator with seed for reproducability
        prng = np.random.RandomState(seed=43)
        #  draw normally distributed samples
        rand_points = prng.normal(size=(self.n_samples, self.dim))
        # calculate lengths of vectors
        radii = np.linalg.norm(rand_points, axis=1, keepdims=True)
        # normalize
        return rand_points / radii



class IFORMContour(Contour):
    """
    Contour based on the inverse first-order reliability method.

    This method was proposed by Winterstein et. al (1993) [3]_

    Parameters
    ----------
    model :  MultivariateModel | TransformedModel
        The model to be used to calculate the contour.
    alpha : float
        The exceedance probability. The probability that an observation falls
        outside the environmental contour.
    n_points : int, optional
        Number of points on the contour. Defaults to 180.

    Attributes
    ----------
    coordinates :
        Coordinates of the calculated contour.
    beta :
        Reliability index.
    sphere_points :
          Points of the sphere in U space [3]_ .

    References
    ----------
    .. [3] Winterstein, S.R.; Ude, T.C.; Cornell, C.A.; Bjerager, P.; Haver, S. (1993)
        Environmental parameters  for extreme response: Inverse FORM with omission
        factors. ICOSSAR 93, Innsbruck, Austria.


    """

    def __init__(self, model, alpha, n_points=180, distribute_evenly = None):
        allowed_model_types = (GlobalHierarchicalModel, TransformedModel)
        if isinstance(model, allowed_model_types):
            self.model = model
        else:
            raise TypeError(
                f"Type of model was {type(model).__name__} but expected one of {allowed_model_types}"
            )
        self.alpha = alpha
        self.n_points = n_points

        if distribute_evenly is None:
            distribute_evenly = True if n_points > 1000 else False
        self.distribute_evenly = bool(distribute_evenly)

        super().__init__()

    def _compute(
        self,
    ):
        """
        Calculates coordinates using IFORM.

        """
        n_dim = self.model.n_dim
        n_points = self.n_points

        # A GlobalHierarchicalModel has the attributes distributions and conditional_on
        # but a TransformedModel not. Consequently, contour calculation requries different
        # algorithms for these two cases. TransformedModel is used the EW models defined in
        # predefined.py.

        if isinstance(self.model, GlobalHierarchicalModel):
            distributions = self.model.distributions
            conditional_on = self.model.conditional_on
        elif isinstance(self.model, TransformedModel):
            distributions = None
            conditional_on = None
        else:
            raise TypeError()

        beta = sts.norm.ppf(1 - self.alpha)
        self.beta = beta

        # TODO Update NSphere to handle n_dim case with order
        # Create sphere
        if n_dim == 2:
            _phi = np.linspace(0, 2 * np.pi, num=n_points, endpoint=False)
            _x = np.cos(_phi)
            _y = np.sin(_phi)
            _circle = np.stack((_x, _y), axis=1)
            sphere_points = beta * _circle

        else:
            sphere = NSphere(dim=n_dim, n_samples=n_points, distribute_evenly = self.distribute_evenly)
            sphere_points = beta * sphere.unit_sphere_points

        # Get probabilities for coordinates
        norm_cdf = sts.norm.cdf(sphere_points)

        # Inverse procedure. Get coordinates from probabilities.
        p = norm_cdf
        coordinates = np.empty_like(p)

        if distributions:
            coordinates[:, 0] = distributions[0].icdf(p[:, 0])
        else:
            coordinates[:, 0] = self.model.marginal_icdf(
                p[:, 0], 0, precision_factor=self.model.precision_factor
            )

        for i in range(1, n_dim):
            if distributions:
                if conditional_on[i] is None:
                    coordinates[:, i] = distributions[i].icdf(p[:, i])
                else:
                    cond_idx = conditional_on[i]
                    coordinates[:, i] = distributions[i].icdf(
                        p[:, i], given=coordinates[:, cond_idx]
                    )
            else:
                given = coordinates[:, np.arange(n_dim) != i]
                coordinates[:, i] = self.model.conditional_icdf(
                    p[:, i], i, given, random_state=self.model.random_state
                )

        self.sphere_points = sphere_points
        self.coordinates = coordinates


class ISORMContour(Contour):
    """
    Contour based on the inverse second-order reliability method.

    This method was proposed by Chai and Leira (2018) [4]_

    Parameters
    ----------
    model : MultivariateModel
        The model to be used to calculate the contour.
    alpha : float
        The exceedance probability. The probability that an observation falls
        outside the environmental contour.
    n_points : int, optional
        Number of points on the contour. Defaults to 180.

    Attributes
    ----------
    coordinates :
        Coordinates of the calculated contour.
    beta :
        Reliability index.
    sphere_points :
          Points of the sphere in U space [4]_ .

    References
    ----------
    .. [4] Chai, W.; Leira, B.J. (2018)
        Environmental contours based on inverse SORM. Marine Structures Volume 60,
        pp. 34-51. DOI: 10.1016/j.marstruc.2018.03.007 .

    """

    def __init__(self, model, alpha, n_points=180, distribute_evenly=None):
        self.model = model
        self.alpha = alpha
        self.n_points = n_points

        if distribute_evenly is None:
            distribute_evenly = True if n_points > 1000 else False
        self.distribute_evenly = bool(distribute_evenly)

        super().__init__()

    def _compute(
        self,
    ):
        """
        Calculates coordinates using ISORM.

        """

        n_dim = self.model.n_dim
        n_points = self.n_points

        distributions = self.model.distributions
        conditional_on = self.model.conditional_on

        # Use the ICDF of a chi-squared distribution with n dimensions. For
        # reference see equation 20 in Chai and Leira (2018).
        beta = np.sqrt(sts.chi2.ppf(1 - self.alpha, n_dim))

        # Create sphere.
        if n_dim == 2:
            _phi = np.linspace(0, 2 * np.pi, num=n_points, endpoint=False)
            _x = np.cos(_phi)
            _y = np.sin(_phi)
            _circle = np.stack((_x, _y)).T
            sphere_points = beta * _circle

        else:
            sphere = NSphere(dim=n_dim, n_samples=n_points, distribute_evenly=self.distribute_evenly)
            sphere_points = beta * sphere.unit_sphere_points

        # Get probabilities for coordinates of shape.
        norm_cdf_per_dimension = [
            sts.norm.cdf(sphere_points[:, dim]) for dim in range(n_dim)
        ]

        # Inverse procedure. Get coordinates from probabilities.
        data = np.zeros((n_points, n_dim))

        for i in range(n_dim):
            dist = distributions[i]
            cond_idx = conditional_on[i]
            if cond_idx is None:
                data[:, i] = dist.icdf(norm_cdf_per_dimension[i])
            else:
                conditioning_values = data[:, cond_idx]
                for j in range(n_points):
                    data[j, i] = dist.icdf(
                        norm_cdf_per_dimension[i][j], given=conditioning_values[j]
                    )

        self.beta = beta
        self.sphere_points = sphere_points
        self.coordinates = data
