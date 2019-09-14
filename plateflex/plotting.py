# Copyright 2019 Pascal Audet

# This file is part of Telewavesim.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde as kde
import seaborn as sns
sns.set()


def plot_real_grid(grid, log=False, mask=None, title=None, save=None, clabel=None):

    # Take log of real values
    if log:
        if not np.all(grid==np.absolute(grid)):
            raise(Exception('cannot plot log of grid containing \
                negative values'))
        grid = np.log(grid)

    # Apply mask
    if mask is not None:
        grid = np.ma.masked_where(mask, grid)

    # Plot figure and add colorbar
    plt.figure()
    plt.imshow(grid, origin='lower', cmap='viridis')
    cbar = plt.colorbar()

    # Add units on colorbar label
    if clabel is not None:
        cbar.set_label(clabel)

    # Add title
    if title:
        plt.title(title)
    
    # Save figure
    if save:
        plt.savefig(save+'.png')

    # Show
    plt.show()


def plot_stats(trace, summary, map_estimate, title=None):

    import plateflex.estimate as est

    results = est.get_estimates(summary, map_estimate)

    keys = []
    for var in trace.varnames:
        if var[-1]=='_':
            continue
        keys.append(var)

    # this means we searched for Te and F only
    if len(keys)==2:

        data = np.array([trace['Te'], trace['F']]).transpose()
        data = pd.DataFrame(data, columns=['Te (km)', 'F'])

        g = sns.PairGrid(data)
        g.map_diag(plt.hist, lw=1)
        g.map_lower(sns.kdeplot)

        ax = g.axes[0][1]
        ax.set_visible(False)

        tetext = '\n'.join((
            r'$\mu$ = {0:.0f} km'.format(results[0]),
            r'$\sigma$ = {0:.0f} km'.format(results[1]),
            r'$95\%$ CI = [{0:.0f}, {1:.0f}] km'.format(results[2], results[3]),
            r'MAP = {0:.0f} km'.format(results[4])))

        ax1 = g.axes[0][0]
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(1.05, 0.9, tetext, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

        Ftext = '\n'.join((
            r'$\mu$ = {0:.2f}'.format(results[5]),
            r'$\sigma$ = {0:.2f}'.format(results[6]),
            r'$95\%$ CI = [{0:.2f}, {1:.2f}]'.format(results[7], results[8]),
            r'MAP = {0:.2f}'.format(results[9])))

        ax2 = g.axes[1][1]
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.135, 1.4, Ftext, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

    elif len(keys)==3:

        data = np.array([trace['Te'], trace['F'], trace['alpha']]).transpose()
        data = pd.DataFrame(data, columns=['Te (km)', 'F', r'$\alpha$'])

        g = sns.PairGrid(data)
        g.map_diag(plt.hist, lw=1)
        g.map_lower(sns.kdeplot)

        # g.axes[0][0].set_ylim(0.,0.1)
        ax = g.axes[0][1]
        ax.set_visible(False)
        ax = g.axes[0][2]
        ax.set_visible(False)
        ax = g.axes[1][2]
        ax.set_visible(False)

        tetext = '\n'.join((
            r'$\mu$ = {0:.0f} km'.format(results[0]),
            r'$\sigma$ = {0:.0f} km'.format(results[1]),
            r'$95\%$ CI = [{0:.0f}, {1:.0f}] km'.format(results[2], results[3]),
            r'MAP = {0:.0f} km'.format(results[4])))

        ax1 = g.axes[0][0]
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(1.05, 0.9, tetext, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

        Ftext = '\n'.join((
            r'$\mu$ = {0:.2f}'.format(results[5]),
            r'$\sigma$ = {0:.2f}'.format(results[6]),
            r'$95\%$ CI = [{0:.2f}, {1:.2f}]'.format(results[7], results[8]),
            r'MAP = {0:.2f}'.format(results[9])))

        ax2 = g.axes[1][1]
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.135, 1.4, Ftext, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

        atext = '\n'.join((
            r'$\mu$ = {0:.2f}'.format(results[10]),
            r'$\sigma$ = {0:.2f}'.format(results[11]),
            r'$95\%$ CI = [{0:.2f}, {1:.2f}]'.format(results[12], results[13]),
            r'MAP = {0:.2f}'.format(results[14])))

        ax3 = g.axes[2][2]
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax3.text(0.135, 1.4, atext, transform=ax3.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

    else:

        raise(Exception('there are less than 2 or more than 3 variables in pymc3 chains'))

    if title:
        plt.suptitle(title)

    plt.show()


def plot_fitted(k, adm, eadm, coh, ecoh, summary, map_estimate, est='MAP', title=None):

    import plateflex.flexure as flex

    if est=='mean':
        mte = summary.loc['Te',est]
        mF = summary.loc['F',est]
        if sum(summary.index.isin(['alpha']))==1:
            ma = summary.loc['alpha',est]
        else:
            ma = np.pi/2.

    elif est=='MAP':
        mte = np.float(map_estimate['Te'])
        mF = np.float(map_estimate['F'])
        if 'alpha' in map_estimate:
            ma = np.float(map_estimate['alpha'])
        else:
            ma = np.pi/2.

    else:
        raise(Exception('estimate does not exist. Choose among: "mean" or "MAP"'))

    padm, pcoh = flex.real_xspec_functions(k, mte, mF, ma)

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.errorbar(k*1.e3,adm,yerr=eadm)
    ax1.plot(k*1.e3,padm)
    ax1.set_ylabel('Admittance (mGal/m)')

    ax2.errorbar(k*1.e3,coh,yerr=ecoh)
    ax2.plot(k*1.e3,pcoh)
    ax2.set_ylabel('Coherence')
    ax2.set_xlabel('Wavenumber (rad/km)')

    ax1.set_xscale('log')
    ax2.set_xscale('log')

    if title:
        plt.suptitle(title)

    plt.show()
