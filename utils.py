
from jax import random, numpy as np
from matplotlib import pyplot as plt
from scipy.stats.kde import gaussian_kde


def plot_kde(ax, data, xlim, resolution=1000, **kwargs):
    if xlim is None:
        xlim = [np.min(data), np.max(data)]
    linsp = np.linspace(*xlim, resolution)
    dens = gaussian_kde(data)
    ax.plot(linsp, dens(linsp), **kwargs)


def dens_clean_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.xaxis.set_ticks_position('none')
    plt.tight_layout()

