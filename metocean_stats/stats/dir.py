import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import windrose
import matplotlib.cm as cm
import os
from .aux_funcs import convert_latexTab_to_csv, Tp_correction


def rose(wd,ws,max_ws,step_ws,min_percent, max_percent, step_percent):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="windrose")
    ax.bar(wd, ws, bins=np.arange(0, max_ws, step_ws), cmap=cm.rainbow, normed=True, opening=0.9, edgecolor='white')
    ax.set_yticks(np.arange(min_percent, max_percent, step_percent))
    ax.set_yticklabels(np.arange(min_percent, max_percent,step_percent))
    ax.legend(bbox_to_anchor=(0.90,-0.05),framealpha=0.5)
    return fig


