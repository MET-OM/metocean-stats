import matplotlib.pyplot as plt

plt.rcParams["figure.constrained_layout.use"] = True

from .timeseries import TimeSeries, Variable, CMAConfig
from .profile import ProfileTimeSeries