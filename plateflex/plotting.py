import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde as kde
import seaborn as sns
sns.set()


def plot_real_grid(grid, log=False):

    if log:
        grid = np.log(grid)

    plt.figure()
    plt.imshow(grid, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.show()


def plot_trace_stats(trace):

    data = np.array([trace['Te'], trace['F']]).transpose()
    data = pd.DataFrame(data, columns=['Te (km)', 'F'])

    with sns.axes_style('white'):
        sns.jointplot('Te (km)', 'F', data, kind='kde')


def plot_fitted(k, adm, eadm, coh, ecoh, padm, pcoh):

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.errorbar(k*1.e3,adm,yerr=eadm)
    ax1.plot(k*1.e3,padm)
    ax1.set_ylabel('Admittance (mGal/m)')

    ax2.errorbar(k*1.e3,coh,yerr=ecoh)
    ax2.plot(k*1.e3,pcoh)
    ax2.set_ylabel('Coherence)')
    ax2.set_xlabel('Wavenumber (rad/km)')

    ax1.set_xscale('log')
    ax2.set_xscale('log')

    plt.show()
