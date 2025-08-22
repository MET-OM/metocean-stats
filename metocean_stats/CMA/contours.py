"""
Procedures relating to environmental contours.

The IFORM and ISORM procedures are slighly modified from virocon, under the following licence:

-----------------------------------------------------------------------------

MIT License

Copyright (c) 2017-2021 virocon developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

------------------------------------------------------------------------------
"""

import virocon
from virocon.contours import Contour
from virocon import GlobalHierarchicalModel, TransformedModel
from scipy.spatial.distance import cdist
import numpy as np
import scipy.stats as sts

__all__ = [
    "IFORMContour",
    "ISORMContour",
    "get_contour",
    "sort_contour",
    "split_contour",
]

def get_contour(
        model:GlobalHierarchicalModel,
        return_period:float=100,
        state_duration:float=1,
        method:str="IFORM",
        point_distribution="equal",
        n_samples = 360,
        **kwargs
        ):
    """
    Get one contour object from virocon.

    Parameters
    ----------
    model : GHM
        The joint model.
    return_period : float
        The return period in years
    state_duration : float
        The state duration in hours
    method : str
        The contour method.
    point_distribution : str, default "equal"
        How to distribute points on the sphere:
        "equal", "gridded", or "random".
    n_samples : int, default 360
        Number of points on the contour.
    """
    # Check point distribution
    point_distribution = point_distribution.lower()
    if point_distribution not in ["equal","gridded","random"]:
        raise ValueError("Unknown points distribution.")
    if point_distribution == "gridded" and model.n_dim != 3:
        raise ValueError("Gridded point distribution is only meaningful for a 3-dimensional model.")

    # Check contour method
    contour_method = method.lower()
    if contour_method == "iform":
        ContourMethod = IFORMContour
    elif contour_method == "isorm":
        ContourMethod = ISORMContour
    elif point_distribution == "gridded":
        raise ValueError("Gridded/surface contour only possible with IFORM or ISORM.")
    elif contour_method in ["highestdensity","highestdensitycontour","hdc"]:
        ContourMethod = virocon.HighestDensityContour
    elif contour_method == ["directsampling","montecarlo"]:
        ContourMethod = virocon.DirectSamplingContour
    elif contour_method in ["constantandexceedance","and"]:
        ContourMethod = virocon.AndContour
    elif contour_method in ["constantorexceedance","or"]:
        ContourMethod = virocon.OrContour
    else:
        raise ValueError(f"Unknown contour method: {contour_method}")            

    # Calculate alpha and contour
    alpha = virocon.calculate_alpha(state_duration,return_period)
    contour = ContourMethod(model,alpha,n_points=n_samples,point_distribution=point_distribution)

    # Return either reshaped xyz for gridded, otherwise just the contour.
    if point_distribution == "gridded":
        return np.array(contour.coordinates).reshape(n_samples,n_samples,3).transpose([2,0,1])
    return contour

def sort_contour(contour):
    """
    Sort a 2D contour by a greedy approach of iteratively adding the nearest point to the list.
    This will not work at all if the points are not too noisy or too many, so always check the result.
    Thanks to ChatGPT.
        
    Parameters
    ------------
        contour : np.ndarray
            (N, 2) array of 2D points, which together constitute a closed contour.
    """
    N = len(contour)
    
    # Compute the pairwise distance matrix (Euclidean distances)
    dist_matrix = cdist(contour, contour, metric='euclidean')

    # To keep track of visited points
    visited = np.zeros(N, dtype=bool)
    
    # Start with the first point
    visited[0] = True
    ordered_indices = [0]
    
    # Loop through the points to create the tour
    for i in range(1, N+1):
        last_idx = ordered_indices[-1]
        unvisited_idxs = np.where(~visited)[0]  # Find unvisited points
        
        # Get the distances from the last point to all unvisited points
        distances_to_unvisited = dist_matrix[last_idx, unvisited_idxs]
        
        # Find the index of the nearest unvisited point
        nearest_idx = unvisited_idxs[np.argmin(distances_to_unvisited)]
        
        # Mark the nearest point as visited
        visited[nearest_idx] = True
        
        # Add the nearest point to the ordered indices
        ordered_indices.append(nearest_idx)

        if i==N//2: # Unvisit start, to be able to add it at the end.
            visited[0] = False

        # Eventually we should arrive back at the start.
        if nearest_idx == 0:
            return contour[ordered_indices]

def split_contour(contour,split_dim):
    """
    Where "contour" is a Nx2 array, 
    where each point in the array is placed to the previous, 
    divide into left hand side and right hand side on the second coordinate (e.g., Tp),
    and sort from low to high on the first coordinate (e.g., Hs).
    """
    range_dim = 0 if split_dim else 1

    bot_idx = np.argmin(contour[:,range_dim])
    top_idx = np.argmax(contour[:,range_dim])
    i = 0
    state = "none"
    lhs = []
    rhs = []
    while True:
        if i == len(contour): i = 0
        
        # We start counting at the bottom of the contour
        if state == "none" and i == bot_idx:
            if contour[i-1,split_dim] < contour[i,split_dim]:
                state = "rhs"
            else:
                state = "lhs"

        # If we reached top, switch side
        elif i == top_idx:
            if state == "lhs": 
                state = "rhs"
            elif state == "rhs":
                state = "lhs"
                
        # If we reached bottom again, we stop
        elif i == bot_idx: break

        # Adding values depending on state
        if state == "lhs": lhs.append(contour[i])
        if state == "rhs": rhs.append(contour[i])

        # Increment
        i += 1

    lhs = np.array(lhs)
    lhs = lhs[np.argsort(lhs[:,range_dim])]
    rhs = np.array(rhs)
    rhs = rhs[np.argsort(rhs[:,range_dim])]

    return lhs,rhs

def get_random_unit_sphere_points(n_dim,n_samples):
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
    rand_points = prng.normal(size=(n_samples, n_dim))
    # calculate lengths of vectors
    radii = np.linalg.norm(rand_points, axis=1, keepdims=True)
    # normalize
    return rand_points / radii

def get_3D_gridded_unit_sphere_points(primary_dim,n_polar,n_azimuthal):
    """
    Generates gridded unit sphere points.

    Parameters
    ----------
    primary_dim : int
        The dimension to align the unit sphere poles after.
    n_polar : int
        Number of circles along the primary dimension.
    n_azimuthal : int
        Number of points on each circle.
    """

    theta = np.linspace(0,2*np.pi,n_azimuthal)
    phi = np.linspace(0,np.pi,n_polar)

    x = np.outer(np.cos(theta),np.sin(phi))
    y = np.outer(np.sin(theta),np.sin(phi))
    z = np.outer(np.ones_like(theta),np.cos(phi))

    # Rotate axes to match primary dimension
    if primary_dim == 0:
        return z,x,y
    elif primary_dim == 1:
        return y,z,x
    elif primary_dim == 2:
        return x,y,z
    else:
        raise ValueError(f"primary_dim must be an integer in [0,1,2], got {primary_dim}.")


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

    def __init__(self, model, alpha, n_points=180, point_distribution="random"):
        allowed_model_types = (GlobalHierarchicalModel, TransformedModel)
        if isinstance(model, allowed_model_types):
            self.model = model
        else:
            raise TypeError(
                f"Type of model was {type(model).__name__} but expected one of {allowed_model_types}"
            )
        self.alpha = alpha
        self.n_points = n_points
        self.point_distribution = point_distribution

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

        # Create sphere
        if n_dim == 2:
            _phi = np.linspace(0, 2 * np.pi, num=n_points, endpoint=False)
            _x = np.cos(_phi)
            _y = np.sin(_phi)
            unit_sphere_points = np.stack((_x, _y), axis=1)
        elif self.point_distribution == "equal":
            from virocon._nsphere import NSphere
            sphere = NSphere(self.model.n_dim,self.n_points)
            unit_sphere_points = sphere.unit_sphere_points
        elif self.point_distribution == "random":
            unit_sphere_points = get_random_unit_sphere_points(n_dim=self.model.n_dim,n_samples=self.n_points)
        elif self.point_distribution == "gridded":
            unit_sphere_points = get_3D_gridded_unit_sphere_points(primary_dim=0,n_polar=self.n_points,n_azimuthal=self.n_points)

        sphere_points = beta * unit_sphere_points

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

    def __init__(self, model, alpha, n_points=180, point_distribution="random"):
        self.model = model
        self.alpha = alpha
        self.n_points = n_points
        self.point_distribution = point_distribution

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

        # Create sphere
        if n_dim == 2:
            _phi = np.linspace(0, 2 * np.pi, num=n_points, endpoint=False)
            _x = np.cos(_phi)
            _y = np.sin(_phi)
            unit_sphere_points = np.stack((_x, _y), axis=1)
        elif self.point_distribution == "equal":
            from virocon._nsphere import NSphere
            sphere = NSphere(self.model.n_dim,self.n_points)
            unit_sphere_points = sphere.unit_sphere_points
        elif self.point_distribution == "random":
            unit_sphere_points = get_random_unit_sphere_points(n_dim=self.model.n_dim,n_samples=self.n_points)
        elif self.point_distribution == "gridded":
            unit_sphere_points = get_3D_gridded_unit_sphere_points(primary_dim=0,n_polar=self.n_points,n_azimuthal=self.n_points)
        sphere_points = beta * unit_sphere_points

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
