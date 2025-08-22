import os
from metocean_stats.stats.aux_funcs import readNora10File
from metocean_stats.CMA import JointProbabilityModel,predefined
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = True

data = readNora10File(os.path.join(os.path.dirname(__file__),'../tests/data/NORA_test.txt'))

# Initialize and fit 2D model of Hs (wave height) and U (wind speed)
model = JointProbabilityModel(predefined.get_DNVGL_Hs_U)
model.fit(data,"HS","W10") # fit to HS and W10

# Create the 2D contours
ax=model.plot_contours(periods=[1,10,100,1000],method="IFORM")
model.plot_data_scatter(ax,s=2)
model.plot_dependent_percentiles(ax)
model.plot_legend(ax)
model.reset_labels()

# Fitted distributions and histograms per interval
plt.rcParams["figure.figsize"] = (20,20)
fig,ax=model.plot_histograms_of_interval_distributions()
fig[1].subplots_adjust(hspace=0.3)

# Dependency functions of U on Hs
plt.rcParams["figure.figsize"] = (15,7)
fig,ax = plt.subplots(1,2)
model.plot_dependence_functions(ax)

# Analytical vs empirical quantiles
fig,ax = plt.subplots(1,2)
model.plot_marginal_quantiles(ax)

# Initialize 3D model of wind, wave height and wave period
model = JointProbabilityModel(predefined.get_LiGaoMoan_U_hs_tp)
model.fit(data,"W10","HS","TP")

# Full 3D contour obtained with IFORM
model.plot_3D_contour(return_period=100,n_samples=720)

# 2D slices of the 3D contour at a range of wind levels
plt.rcParams["figure.figsize"] = (20,12)
ax=model.plot_3D_contour_slices(subplots=False,slice_values=[i for i in range(1,40)])
model.plot_legend(ax)
model.reset_labels()

# Isodensity contour - defined by a marginal wind return value
model.plot_3D_isodensity_contour(marginal_value=30)

# 2D slices of isodensity contours, at a range of probability densities.
plt.rcParams["figure.figsize"] = (20,7)
model.reset_labels()
ax=model.plot_3D_isodensity_contour_slice(
    density_levels=[1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,2e-3,4e-3,6e-3],slice_value=10)
ax.set_xlim([0,40])
ax.set_ylim([0,10])
model.plot_legend(ax)
model.reset_labels()

# Get a dataframe describing the joint model.
params = model.parameters()
print(params)

# Show figures
plt.show()
